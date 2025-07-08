# AI Chatbot with Vector Memory

A sophisticated AI chatbot that uses vector embeddings to remember and learn from past conversations, providing contextual responses based on previous interactions.

## Features

- **Vector Memory**: Stores conversations as embeddings using Faiss for similarity search
- **WebSocket Communication**: Real-time chat interface
- **LangChain Integration**: Advanced language model capabilities
- **Session Management**: Multiple conversation threads
- **Streamlit Frontend**: User-friendly web interface
- **FastAPI Backend**: High-performance API server

## Architecture

- **Backend**: FastAPI + WebSocket + LangChain + Faiss
- **Frontend**: Streamlit
- **Vector Store**: Faiss with sentence-transformers
- **Memory**: Conversation embeddings for contextual retrieval

## Setup

### 1. Install Dependencies

```bash
# Backend dependencies
cd backend
pip install -r requirements.txt

# Frontend dependencies
pip install streamlit