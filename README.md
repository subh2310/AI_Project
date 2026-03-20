🛡️ NexusIQ: Enterprise Agentic Document Intelligence
NexusIQ is a high-performance, local Agentic RAG (Retrieval-Augmented Generation) platform. Unlike standard generative chatbots, NexusIQ operates as an autonomous document analyst that reasons over unstructured data using a local-first, privacy-shielded architecture.

Feature,Generative AI (The Engine),Agentic AI (The Brain)
Primary Action,Generates human-like text based on a prompt.,Reasons whether the provided context is sufficient to answer.
Knowledge Source,Uses internal weights (Llama 3) to draft responses.,"Actively retrieves and filters specific ""knowledge shards"" from ChromaDB."
Logic Goal,Focused on Fluency and Creativity.,"Focused on Factuality, Source Attribution, and Contextual Constraints."

✨ Key Features
100% Privacy-Preserving: Operates entirely within your local infrastructure. Your data never touches the cloud.

Autonomous Context Synthesis: Uses Recursive Character Splitting to ensure the Agent understands the semantic "intent" behind a paragraph.

Stateful Session Memory: Maintains a rolling window of 10 interaction cycles for sophisticated multi-turn reasoning.

Secure Multi-User Isolation: Uses UUID-based session sharding to ensure document intelligence remains strictly siloed between users.


📖 System ArchitectureNexusIQ follows a sophisticated Agentic RAG Pipeline designed for production stability:Ingestion & Distillation: Raw text is extracted via pdfplumber and distilled into semantic chunks ($1000$ tokens with a $200$ token overlap).Vectorization: Each chunk is mapped into a high-dimensional vector space using the Nomic-Embed model.Autonomous Retrieval: Upon a user query, the Agent performs a Similarity Search ($k=5$) to pull the most statistically relevant "evidence."Augmented Reasoning: The LLM is constrained by a System Instruction Layer that forces it to act as an analyst, strictly preventing hallucinations by anchoring it to the retrieved evidence.


🛠️ Prerequisites
NexusIQ requires the Ollama orchestration layer to be active on your machine:

Install Ollama: ollama.com

Pull Semantic Engine:

Bash
ollama pull nomic-embed-text

Pull Reasoning Engine:

Bash
ollama pull llama3
🚀 Getting Started
1. Install Python dependencies
Recommended: Python 3.10+

Bash
pip install -r requirements.txt
2. Launch the Application
Bash
streamlit run app.py
📦 Technical Dependency Stack
Orchestration: langchain (The logic bridge between the Agent and its tools).

Vector Engine: chromadb (The Agent's long-term semantic memory).

Inference Engine: ollama (The local LLM provider).

Interface: streamlit (The professional-grade command center).

2. File: requirements.txt
Create a file named requirements.txt in your project folder and paste this list. I have pinned the versions to ensure your app doesn't break during future updates.

Plaintext
streamlit==1.32.0
pdfplumber==0.11.0
langchain==0.1.12
langchain-community==0.0.28
langchain-ollama==0.0.1
chromadb==0.4.24
pydantic==2.6.4
typing-extensions==4.10.0

Or pip install -r requirements.txt

3. File: TECHNICAL_SPECS.md (Optional)
If you want to impress a client or recruiter, add this file to define your function parameters.

Function,Definition,Parameters,Returns
get_embeddings_model,Loads the Nomic-Embed-Text model.,None,OllamaEmbeddings
ingest_document_pipeline,Converts PDF to vector shards.,pdf_stream (Binary),bool (Success)
query_knowledge_base,Semantic Similarity search.,"query (str), k=5",str (Context)
