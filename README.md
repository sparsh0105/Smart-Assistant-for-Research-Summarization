# Smart-Assistant-for-Research-Summarization

AI-powered document analysis tool with contextual understanding and logical reasoning

## Overview

The Smart Research Assistant is an intelligent document analysis tool that enables users to upload documents and interact with them through natural language queries and AI-generated comprehension questions. This implementation demonstrates advanced RAG (Retrieval-Augmented Generation) capabilities with strict document grounding and contextual memory.

### Key Features

-  **Document Upload**: PDF support with automatic processing and chunking
-  **Auto Summary**: Generates concise summaries (≤150 words) immediately after upload
-  **Ask Anything Mode**: Natural language Q&A with source attribution and conversation memory
-  **Challenge Me Mode**: AI-generated comprehension questions with evaluation and detailed feedback
-  **Memory Management**: Maintains conversation context across sessions using Supabase
-  **Source Citations**: Every answer includes document references with page numbers
-  **No Hallucination**: Responses are strictly grounded in uploaded document content


##  Architecture & Reasoning Flow

### System Architecture
```
User Uploads PDF/TXT
         │
         ▼
 Extract & Chunk Text
         │
         ▼
   Auto-Summarize (≤ 150 Words)
         │
         ▼
   User Chooses:
 ┌─────────────┬─────────────┐
 │ Ask Anything│ Challenge Me│
 └─────────────┴─────────────┘
         │                │
     User Q → RAG → LLM   │
         │                │
  Answer + Justification  │
                          │
  LLM Generates Qs → User Answers
                          │
      Evaluate → Feedback + Citation
```

### Processing Flows

#### 1. Document Processing Pipeline
```
PDF Upload → Text Extraction (PyPDF2) → 
Text Chunking (3000 chars, 300 overlap) → 
Embedding Generation (all-MiniLM-L6-v2) → 
Vector Storage (Qdrant with document ID) → 
Summary Generation (Gemini) → Ready for Q&A
```

#### 2. Question Answering Flow
```
User Question → Query Embedding → 
Semantic Search (filtered by document ID) → 
Context Assembly → Prompt Engineering → 
Gemini Generation → Source Attribution → 
Response Display → Memory Storage (Supabase)
```

#### 3. Challenge Mode Flow
```
Document Analysis → Content Sampling → 
Question Generation (Gemini) → User Answer Collection → 
Evaluation Prompt → AI Scoring & Feedback → 
Results Display with Improvement Suggestions
```

### Technical Implementation

- **Frontend**: Streamlit (single-file web application)
- **LLM**: Google Gemini 1.5 Flash (text generation and reasoning)
- **Embeddings**: Sentence Transformers all-MiniLM-L6-v2 (384 dimensions)
- **Vector Database**: Qdrant Cloud (semantic search with document isolation)
- **Memory**: Supabase (conversation history and session management)
- **Document Processing**: LangChain + PyPDF2 (text extraction and chunking)

##  Setup Instructions

### Prerequisites

- Python 3.8+
- API keys for: Google Gemini, Qdrant Cloud, Supabase

### Step 1: Get API Keys for Google Gemini , Qdrant cloud and Supabase

**IMPORTANT**: Run this SQL in Supabase SQL Editor:
```sql
CREATE TABLE "ez-history" (
    id uuid PRIMARY KEY,
    session_id uuid NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Step 2: Setup Files

1. **Download/Copy `app.py`** (contains complete application)

2. **Install dependencies:**
```
pip install streamlit langchain langchain-google-genai langchain-community qdrant-client supabase sentence-transformers PyPDF2 python-dotenv
```

3. **Create `.env` file** with your API keys:
```bash
QDRANT_HOST=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=EZ-data
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
GEMINI_API_KEY=your_gemini_api_key
```

### Step 3: Run Application

```bash
streamlit run app.py
```

## How to Use

### Document Upload & Summary
1. Click **"Choose PDF file"** in sidebar
2. Select your document
3. Click **"🚀 Process Document"**
4. View auto-generated summary (≤150 words)

### Ask Anything Mode
- Ask questions like:
  - "What are the main findings?"
  - "Explain the methodology used"
  - "What are the conclusions?"
- Get answers with source citations and page numbers
- Continue with follow-up questions (maintains context)

### Challenge Me Mode
1. Click **"🎲 Generate Challenge Questions"**
2. Answer 3 AI-generated comprehension questions
3. Get detailed evaluation with:
   - Score (0-100)
   - Strengths and areas for improvement
   - Document references

### Session Management
- **🗑️ Clear Chat**: Reset conversation
- **🆕 New Session**: Start fresh
- **📤 Upload New Document**: Process different file

## Technical Specifications

### Core Parameters
- **Chunk Size**: 3000 characters
- **Chunk Overlap**: 300 characters
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Retrieval**: Top-5 relevant chunks per query
- **Document Isolation**: Unique IDs prevent cross-document confusion
- **Summary**: Maximum 150 words
- **LLM Temperature**: 0.2 (focused, less creative)

### Quality Features
- **Zero Hallucination**: Responses only from document content
- **Source Attribution**: Page numbers and section references
- **Context Memory**: Conversation history across questions
- **Document Grounding**: Every answer justified with sources
- **Auto-Collection Creation**: Qdrant collection created automatically

## Implementation Details

### Document Processing
- **Text Extraction**: PyPDF2 for PDF reading
- **Chunking Strategy**: Recursive splitting with overlap
- **Embedding Generation**: Sentence Transformers for semantic vectors
- **Storage**: Qdrant with metadata (document_id, page, filename)

### Question Answering
- **Retrieval**: Semantic search filtered by current document
- **Context Assembly**: Top-5 chunks combined for LLM
- **Prompt Engineering**: Structured prompts for grounded responses
- **Memory**: Supabase storage for conversation history

### Challenge System
- **Question Generation**: Gemini analyzes document content
- **Evaluation**: AI scoring with detailed feedback
- **Grading Criteria**: Accuracy, completeness, understanding, evidence use

## Requirements Fulfilled

✅ **Document Upload (PDF)**: Streamlit file upload with processing  
✅ **Ask Anything Mode**: Free-form Q&A with contextual understanding  
✅ **Challenge Me Mode**: 3 AI-generated questions with evaluation  
✅ **Contextual Understanding**: All answers grounded in document content  
✅ **Auto Summary (≤150 words)**: Generated immediately after upload  
✅ **Web-based Interface**: Clean, intuitive Streamlit application  
✅ **Memory Handling**: Follow-up questions with conversation context  
✅ **Source Justification**: Every answer includes document references  


## 🚀 Quick Start

1. Get API keys (Gemini, Qdrant, Supabase)
2. Create `.env` file with keys
3. Run Supabase SQL to create table
4. Install dependencies
5. Run `streamlit run app.py`
6. Upload PDF and start asking questions!

---

