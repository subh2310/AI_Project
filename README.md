🚀 Local PDF Chatbot (RAG + Ollama)
An AI-powered document assistant that lets you chat with your PDF files locally. It uses RAG (Retrieval-Augmented Generation) to fetch relevant context from your documents and answer questions using a local LLM.

✨ Features
100% Local: Your data never leaves your machine.

Vector Search: Uses ChromaDB for fast, semantic retrieval of information.

Streaming Responses: Real-time text generation for a smooth UI experience.

Smart Chunking: Breaks down large PDFs into manageable, context-aware pieces.

🛠️ Prerequisites
Before running the app, you must have Ollama installed and the necessary models pulled:

Install Ollama: Download from ollama.com.

Pull Embedding Model:

Bash
ollama pull nomic-embed-text
Pull LLM (Llama 3 or Llama 2):

Bash
ollama pull llama3
🚀 Getting Started
1. Clone the repository
Bash
git clone <your-repo-url>
cd <your-repo-folder>
2. Install Python dependencies
Make sure you have Python 3.9+ installed.

Bash
pip install -r requirements.txt
3. Run the application
Bash
streamlit run app.py
📖 How it Works
Upload: You upload a PDF file via the sidebar.

Process: The app extracts text and splits it into small "chunks."

Embed: Those chunks are converted into mathematical vectors using nomic-embed-text.

Store: Vectors are stored in ChromaDB.

Query: When you ask a question, the app finds the most relevant chunks and sends them to the LLM (Llama 3) to generate an answer based only on that context.

📦 Dependencies
The core libraries used in this project are:

streamlit - Web Interface

langchain - RAG Framework

chromadb - Vector Database

pdfplumber - PDF Parsing

requests - Ollama API communication
