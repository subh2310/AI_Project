# 🛡️ NexusIQ: Enterprise Agentic Document Intelligence

NexusIQ is a high-performance, Agentic RAG (Retrieval-Augmented Generation) platform. Unlike standard chatbots, NexusIQ operates as an autonomous analyst that reasons over unstructured data using a hybrid architecture—combining local embeddings with high-speed cloud inference.

## ✨ Key Features
* **Hybrid Intelligence:** Uses `Llama-3.1-8b` (via Groq) for lightning-fast reasoning and `Nomic-Embed` (via Ollama) for local privacy.
* **Multi-User Sharding:** Unique UUID-based FAISS indexing ensures User A’s data remains strictly siloed from User B.
* **Rate-Limit Shield:** Integrated **Global Thread Locking** and **RPM Throttling** to ensure stability on shared API tiers.
* **Autonomous Web Augmentation:** The Agent automatically triggers a `WebSearch` if the PDF lacks sufficient detail.



## 🛠️ Prerequisites
1.  **Install Ollama:** [ollama.com](https://ollama.com)
2.  **Pull Semantic Engine:**
    ```bash
    ollama pull nomic-embed-text
    ollama serve
    ```
3.  **Environment Setup:** Create a `.env` file:
    ```plaintext
    GROQ_API_KEY=your_groq_key_here
    TAVILY_API_KEY=your_tavily_key_here
    ```

## 🚀 Getting Started
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt (for generative AI) --> nexus_iq_analyst.py
   pip install -r agentic_ai_requirements.txt (for Agenic AI) --> agentic_nexus_iq_analyst.py

Launch Application:
  streamlit run nexus_iq_analyst.py
  streamlit run agentic_nexus_iq_analyst.py
---

### 2. File: `requirements.txt`
*Use this to install all necessary libraries in one command.*

```plaintext
streamlit==1.32.0
pdfplumber==0.11.0
langchain==0.1.12
langchain-community==0.0.28
langchain-ollama==0.0.1
langchain-groq==0.1.3
faiss-cpu==1.8.0
crewai==0.28.8
tavily-python==0.3.3
python-dotenv==1.0.1

1. File: README.md
This is the main documentation for your GitHub or project folder.

Markdown
# 🛡️ NexusIQ: Enterprise Agentic Document Intelligence

NexusIQ is a high-performance, Agentic RAG (Retrieval-Augmented Generation) platform. Unlike standard chatbots, NexusIQ operates as an autonomous analyst that reasons over unstructured data using a hybrid architecture—combining local embeddings with high-speed cloud inference.

## ✨ Key Features
* **Hybrid Intelligence:** Uses `Llama-3.1-8b` (via Groq) for lightning-fast reasoning and `Nomic-Embed` (via Ollama) for local privacy.
* **Multi-User Sharding:** Unique UUID-based FAISS indexing ensures User A’s data remains strictly siloed from User B.
* **Rate-Limit Shield:** Integrated **Global Thread Locking** and **RPM Throttling** to ensure stability on shared API tiers.
* **Autonomous Web Augmentation:** The Agent automatically triggers a `WebSearch` if the PDF lacks sufficient detail.



## 🛠️ Prerequisites
1.  **Install Ollama:** [ollama.com](https://ollama.com)
2.  **Pull Semantic Engine:**
    ```bash
    ollama pull nomic-embed-text
    ollama serve
    ```
3.  **Environment Setup:** Create a `.env` file:
    ```plaintext
    GROQ_API_KEY=your_groq_key_here
    TAVILY_API_KEY=your_tavily_key_here
    ```

## 🚀 Getting Started
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r agentic_ai_requirements.txt
Launch Application:

Bash
streamlit run nexus_iq_analyst.py
streamlit run agentic_nexus_iq_analyst.py

---

### 2. File: `requirements.txt` and `agentic_ai_requirements.txt`
*Use this to install all necessary libraries in one command.*

```plaintext
streamlit==1.32.0
pdfplumber==0.11.0
langchain==0.1.12
langchain-community==0.0.28
langchain-ollama==0.0.1
langchain-groq==0.1.3
faiss-cpu==1.8.0
crewai==0.28.8
tavily-python==0.3.3
python-dotenv==1.0.1

3. File: TECHNICAL_SPECS.md
# NexusIQ Technical Specifications

## 📖 System Pipeline
1. **Ingestion:** Extracts text via `pdfplumber`, chunks it into 600-character segments.
2. **Vectorization:** Maps segments into vector space using `nomic-embed-text`.
3. **Retrieval:** Uses Similarity Search ($k=2$) to provide context while staying under Groq's 6,000-12,000 TPM limit.
4. **Agent Logic:** A CrewAI `Analyst` agent evaluates the context and performs `WebSearch` if required.

## 📑 Function Reference

| Function | Definition | Parameters | Returns |
| :--- | :--- | :--- | :--- |
| `get_user_faiss_path` | Generates unique folder path for user isolation. | None | `str` (Path) |
| `get_embeddings_model` | Loads Nomic-Embed-Text model via Ollama. | None | `OllamaEmbeddings` |
| `ingest_document_pipeline`| Converts PDF to vector shards and saves locally. | `pdf_stream` | `bool` (Success) |
| `query_vector_store` | Searches isolated index for relevant snippets. | `query` (str) | `str` (Context) |
| `run_agentic_research` | Orchestrates Agent, Task, and Global Lock. | `user_query` | `Any` (Report) |

## 🛡️ Security & Stability
* **Concurrency:** Uses `threading.Lock()` to prevent simultaneous API calls that trigger RateLimitErrors.
* **Privacy:** UUID-based directory naming prevents cross-user data contamination.



   
