import streamlit as st
import os
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any

# Required imports
import qdrant_client
from qdrant_client import models
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from supabase import create_client
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="📚 Smart Research Assistant",
    page_icon="📚",
    layout="wide"
)

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def initialize_system():
    """Initialize all components"""
    
    # Qdrant client
    client = qdrant_client.QdrantClient(
        QDRANT_HOST,
        api_key=QDRANT_API_KEY
    )
    
    # Create collection if it doesn't exist
    try:
        client.get_collection(QDRANT_COLLECTION_NAME)
        print(f"✅ Collection {QDRANT_COLLECTION_NAME} exists")
    except:
        print(f"🔧 Creating collection {QDRANT_COLLECTION_NAME}")
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,  # Size for all-MiniLM-L6-v2 embeddings
                distance=models.Distance.COSINE
            )
        )
        print(f"✅ Collection {QDRANT_COLLECTION_NAME} created")
    
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Vector store
    vector_store = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embeddings
    )
    
    # Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=GEMINI_API_KEY
    )
    
    return vector_store, supabase, llm

class SupaBaseChatHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, supabase_client):
        self.session_id = session_id
        self.supabase = supabase_client
    
    @property
    def messages(self):
        try:
            res = self.supabase.table('ez-history').select("*").eq("session_id", self.session_id).order("created_at").execute()
            return [HumanMessage(content=msg['content']) if msg['role'] == 'user'
                    else AIMessage(content=msg['content'])
                    for msg in res.data]
        except:
            return []
    
    def add_message(self, message: BaseMessage):
        if isinstance(message, HumanMessage):
            self._store_message("user", message.content)
        elif isinstance(message, AIMessage):
            self._store_message("ai", message.content)
    
    def _store_message(self, role: str, content: str):
        try:
            self.supabase.table("ez-history").insert({
                "id": str(uuid.uuid4()),
                "session_id": self.session_id,
                "role": role,
                "content": content,
                "created_at": datetime.utcnow().isoformat() + "Z"
            }).execute()
        except Exception as e:
            st.error(f"Error storing message: {e}")
    
    def clear(self):
        try:
            self.supabase.table("ez-history").delete().eq("session_id", self.session_id).execute()
        except Exception as e:
            st.error(f"Error clearing history: {e}")

# Initialize session state
def init_session_state():
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store, st.session_state.supabase, st.session_state.llm = initialize_system()
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False

# Document processing functions
def process_document(uploaded_file):
    """Process uploaded document"""
    try:
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        st.session_state.current_document_id = document_id
        
        # Save uploaded file temporarily
        with open("temp_document.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and process
        loader = PyPDFLoader("temp_document.pdf")
        content = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, 
            chunk_overlap=300
        )
        data = text_splitter.split_documents(content)
        
        # Extract texts and metadata - ADD DOCUMENT ID
        texts = [doc.page_content for doc in data]
        metadatas = []
        for doc in data:
            metadata = doc.metadata.copy()
            metadata['document_id'] = document_id  # Add unique document ID
            metadata['filename'] = uploaded_file.name  # Add filename
            metadatas.append(metadata)
        
        # Add to vector store with document-specific metadata
        st.session_state.vector_store.add_texts(texts=texts, metadatas=metadatas)
        
        # Store document info
        st.session_state.current_document_name = uploaded_file.name
        
        # Generate and store summary
        summary = generate_document_summary(texts[:5])  # Use first 5 chunks
        st.session_state.document_summary = summary
        
        # Clean up
        os.remove("temp_document.pdf")
        
        return len(texts)
        
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return 0
    
def generate_document_summary(text_chunks):
    """Generate document summary"""
    try:
        # Combine first few chunks
        combined_text = "\n\n".join(text_chunks)[:4000]
        
        summary_prompt = f"""
        Please provide a concise summary of this document in exactly 150 words or less.
        Focus on main topics, key findings, and important concepts.
        
        Document content:
        {combined_text}
        
        Summary (150 words max):
        """
        
        response = st.session_state.llm.invoke(summary_prompt)
        summary = response.content.strip()
        
        # Ensure word limit
        words = summary.split()
        if len(words) > 150:
            summary = " ".join(words[:150]) + "..."
        
        return summary
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def ask_question(question: str):
    """Ask a question using RAG - Only search current document"""
    try:
        # Create QA prompt
        qa_prompt_template = """
        You are a helpful research assistant. Use the following context to answer the question.
        
        IMPORTANT:
        1. Answer ONLY based on the provided context
        2. If you don't know from the context, say so clearly
        3. Always cite page numbers when available
        4. Be specific and detailed
        
        Context:
        {context}
        
        Question: {question}
        
        Answer with citations:"""
        
        # Get relevant documents filtered by current document
        if hasattr(st.session_state, 'current_document_id'):
            # Search all documents first
            all_docs = st.session_state.vector_store.similarity_search(question, k=15)
            
            # Filter by current document ID
            relevant_docs = [
                doc for doc in all_docs 
                if doc.metadata.get('document_id') == st.session_state.current_document_id
            ][:5]  # Take top 5 from current document
            
            if not relevant_docs:
                return {
                    'answer': 'No relevant information found in the current document for this question.',
                    'sources': []
                }
            
            # Build context manually
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create full prompt
            full_prompt = qa_prompt_template.format(context=context, question=question)
            
            # Get answer from LLM
            response = st.session_state.llm.invoke(full_prompt)
            answer = response.content
            
        else:
            # Fallback for old documents without document_id
            QA_PROMPT = PromptTemplate(
                template=qa_prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": QA_PROMPT},
                return_source_documents=True
            )
            
            result = qa_chain({"query": question})
            answer = result['result']
            relevant_docs = result.get('source_documents', [])
        
        # Get chat history
        chat_history = SupaBaseChatHistory(st.session_state.session_id, st.session_state.supabase)
        
        # Store conversation
        chat_history.add_message(HumanMessage(content=question))
        chat_history.add_message(AIMessage(content=answer))
        
        # Format sources
        sources_info = []
        for i, doc in enumerate(relevant_docs):
            sources_info.append({
                'content': doc.page_content[:300] + "...",
                'metadata': doc.metadata,
                'page': doc.metadata.get('page', 'Unknown')
            })
        
        return {
            'answer': answer,
            'sources': sources_info
        }
        
    except Exception as e:
        st.error(f"Error answering question: {e}")
        return {'answer': 'Error processing question', 'sources': []}

def generate_challenge_questions():
    """Generate challenge questions"""
    try:
        # Get sample content
        sample_docs = st.session_state.vector_store.similarity_search("main concepts key points", k=5)
        combined_content = "\n\n".join([doc.page_content for doc in sample_docs])[:5000]
        
        question_prompt = f"""
        Based on this document, generate 3 challenging questions that test comprehension.
        
        Document content:
        {combined_content}
        
        Format each question as:
        Q1: [question]
        A1: [expected answer approach]
        D1: [Easy/Medium/Hard]
        
        Q2: [question]  
        A2: [expected answer approach]
        D2: [Easy/Medium/Hard]
        
        Q3: [question]
        A3: [expected answer approach] 
        D3: [Easy/Medium/Hard]
        """
        
        response = st.session_state.llm.invoke(question_prompt)
        return parse_questions(response.content)
        
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

def parse_questions(response_text):
    """Parse generated questions"""
    questions = []
    lines = response_text.split('\n')
    current_q = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith('Q') and ':' in line:
            if current_q:
                questions.append(current_q)
                current_q = {}
            current_q['question'] = line.split(':', 1)[1].strip()
        elif line.startswith('A') and ':' in line:
            current_q['expected'] = line.split(':', 1)[1].strip()
        elif line.startswith('D') and ':' in line:
            current_q['difficulty'] = line.split(':', 1)[1].strip()
    
    if current_q:
        questions.append(current_q)
    
    return questions

def evaluate_answer(question, user_answer, expected):
    """Evaluate user's answer"""
    try:
        eval_prompt = f"""
        Question: {question}
        Expected: {expected}
        User Answer: {user_answer}
        
        Evaluate the answer and give:
        1. Score (0-100)
        2. Feedback on strengths and improvements needed
        
        Evaluation:
        """
        
        response = st.session_state.llm.invoke(eval_prompt)
        
        # Extract score
        score_match = re.search(r'(\d+)', response.content)
        score = int(score_match.group(1)) if score_match else 50
        
        return {
            'score': score,
            'feedback': response.content
        }
        
    except Exception as e:
        return {'score': 50, 'feedback': f'Evaluation error: {e}'}

# Main app
def main():
    init_session_state()
    
    st.title("📚 Smart Research Assistant")
    st.write("Upload documents and interact with them using AI!")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Controls")
        
        # Document upload section
        if not st.session_state.document_processed:
            st.subheader("📤 Upload Document")
            uploaded_file = st.file_uploader("Choose PDF file", type=['pdf'])
            
            if uploaded_file and st.button("🚀 Process Document"):
                with st.spinner("Processing document..."):
                    chunks = process_document(uploaded_file)
                    if chunks > 0:
                        st.session_state.document_processed = True
                        st.success(f"✅ Processed {chunks} chunks!")
                        st.rerun()
        else:
            st.success("✅ Document loaded!")
            if st.button("📤 Upload New Document"):
                st.session_state.document_processed = False
                st.session_state.chat_history = []
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()
        
        st.divider()
        
        # Session controls
        if st.session_state.document_processed:
            st.subheader("💬 Session")
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                chat_history = SupaBaseChatHistory(st.session_state.session_id, st.session_state.supabase)
                chat_history.clear()
                st.rerun()
            
            if st.button("🆕 New Session"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.rerun()
    
    # Main content
    if not st.session_state.document_processed:
        st.info("👆 Please upload a PDF document to get started!")
        st.write("### Features:")
        st.write("- 🤔 **Ask Anything**: Natural language Q&A with sources")
        st.write("- 🎯 **Challenge Mode**: AI-generated comprehension questions")
        st.write("- 💾 **Memory**: Maintains conversation context")
        st.write("- 📚 **Citations**: Shows source references")
        
    else:
        if hasattr(st.session_state, 'document_summary'):
            st.subheader("📋 Document Summary")
            st.write(st.session_state.document_summary)
            st.divider()
        # Tabs for different modes
        tab1, tab2 = st.tabs(["🤔 Ask Anything", "🎯 Challenge Me"])
        
        with tab1:
            st.subheader("Ask Questions About Your Document")
            
            # Display chat history
            if st.session_state.chat_history:
                st.write("**Conversation History:**")
                for i, item in enumerate(st.session_state.chat_history):
                    with st.expander(f"Q{i+1}: {item['question'][:60]}..."):
                        st.write(f"**Question:** {item['question']}")
                        st.write(f"**Answer:** {item['answer']}")
                        if item.get('sources'):
                            st.write(f"**Sources:** {len(item['sources'])} found")
                            for j, source in enumerate(item['sources'][:2]):
                                st.write(f"  {j+1}. Page {source['page']}: {source['content'][:100]}...")
                st.divider()
            
            # Question input
            question = st.text_area("Your question:", height=100, 
                                   placeholder="e.g., What are the main findings of this research?")
            
            if st.button("🔍 Ask Question", type="primary"):
                if question.strip():
                    with st.spinner("Analyzing your question..."):
                        result = ask_question(question)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': result['answer'],
                            'sources': result['sources']
                        })
                        
                        # Display result
                        st.success("✅ Question answered!")
                        st.write("**Answer:**")
                        st.write(result['answer'])
                        
                        if result['sources']:
                            st.write("**Sources:**")
                            for i, source in enumerate(result['sources'][:3]):
                                with st.expander(f"Source {i+1} - Page {source['page']}"):
                                    st.write(source['content'])
                        
                        st.rerun()
                else:
                    st.warning("Please enter a question!")
        
        with tab2:
            st.subheader("Test Your Understanding")
            
            if 'challenge_questions' not in st.session_state:
                if st.button("🎲 Generate Challenge Questions", type="primary"):
                    with st.spinner("Generating questions..."):
                        questions = generate_challenge_questions()
                        if questions:
                            st.session_state.challenge_questions = questions
                            st.success(f"✅ Generated {len(questions)} questions!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to generate questions")
            else:
                st.write("**Answer these questions based on the document:**")
                
                for i, q in enumerate(st.session_state.challenge_questions):
                    st.write(f"**Question {i+1}** ({q.get('difficulty', 'Medium')}):")
                    st.write(q['question'])
                    
                    # Answer input
                    user_answer = st.text_area(
                        f"Your answer:",
                        key=f"answer_{i}",
                        height=80
                    )
                    
                    if st.button(f"📝 Submit Answer {i+1}", key=f"submit_{i}"):
                        if user_answer.strip():
                            with st.spinner("Evaluating..."):
                                evaluation = evaluate_answer(
                                    q['question'], 
                                    user_answer, 
                                    q.get('expected', '')
                                )
                                
                                score = evaluation['score']
                                if score >= 80:
                                    st.success(f"🎉 Excellent! Score: {score}/100")
                                elif score >= 60:
                                    st.info(f"👍 Good! Score: {score}/100")
                                else:
                                    st.warning(f"📚 Needs improvement. Score: {score}/100")
                                
                                st.write("**Detailed Feedback:**")
                                st.write(evaluation['feedback'])
                        else:
                            st.warning("Please provide an answer!")
                    
                    st.divider()
                
                if st.button("🔄 Generate New Questions"):
                    del st.session_state.challenge_questions
                    st.rerun()

if __name__ == "__main__":
    main()

# To run this app:
# 1. Save this code as app.py
# 2. Install requirements: pip install streamlit langchain langchain-google-genai langchain-community qdrant-client supabase sentence-transformers pypdf2
# 3. Run: streamlit run app.py