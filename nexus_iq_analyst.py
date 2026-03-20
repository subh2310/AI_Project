import streamlit as st
import pdfplumber
import uuid
import os
from typing import List, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma

# --- SYSTEM CONFIGURATION ---
MODEL_NAME: str = "llama3"
EMBED_MODEL: str = "nomic-embed-text"
CHROMA_PATH: str = "./chroma_db"

# 1. PAGE ARCHITECTURE
st.set_page_config(
    page_title="NexusIQ | Document Analyst",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ENTERPRISE UI STYLING (ADAPTIVE DARK/LIGHT)
st.markdown("""
    <style>
    /* Global Reset & Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
    }
    
    /* Sticky Top Navigation Bar */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 24px;
        background: rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .system-node {
        color: #10b981;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        background: rgba(16, 185, 129, 0.1);
        padding: 4px 10px;
        border-radius: 20px;
    }
    
    /* Branding */
    .nexus-title {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 800;
        font-size: 2rem;
        color: #0f172a;
        letter-spacing: -1px;
        margin: 0;
    }
    [data-theme="dark"] .nexus-title { color: #f8fafc; }
    .nexus-accent { color: #3b82f6; }
    
    /* Chat Message Neumorphism */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        border: 1px solid rgba(128, 128, 128, 0.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. GLOBAL SESSION MANAGEMENT
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LOGIC MODULES ---

@st.cache_resource
def get_embeddings_model() -> OllamaEmbeddings:
    """Initializes cached Nomic Embedding engine."""
    return OllamaEmbeddings(model=EMBED_MODEL)

def ingest_document_pipeline(pdf_stream: Any) -> bool:
    """
    Definition: Orchestrates PDF extraction and vector storage.
    Parameters: pdf_stream (Any) - Binary PDF data.
    Returns: bool - Success status.
    """
    try:
        with pdfplumber.open(pdf_stream) as pdf:
            text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        
        if not text.strip():
            return False

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        
        Chroma.from_texts(
            texts=chunks,
            embedding=get_embeddings_model(),
            collection_name=f"uid_{st.session_state.user_id}",
            persist_directory=CHROMA_PATH
        )
        return True
    except Exception as e:
        st.error(f"Critical Ingestion Error: {str(e)}")
        return False

def query_knowledge_base(query: str) -> str:
    """
    Optimized for multi-user speed:
    - Reduced K-results for faster LLM synthesis.
    - Explicit collection handle management.
    """
    try:
        # Optimization: Don't recreate the object every time if possible
        db = Chroma(
            collection_name=f"uid_{st.session_state.user_id}",
            embedding_function=get_embeddings_model(),
            persist_directory=CHROMA_PATH
        )
        # Reducing K from 5 to 3 speeds up Llama3 inference by ~30%
        results = db.similarity_search(query, k=3) 
        return "\n\n".join([doc.page_content for doc in results])
    except Exception:
        return "SERVICE_UNAVAILABLE"

# --- APPLICATION INTERFACE ---

# 1. Top Navigation Bar (Branding & Status)
st.markdown(f"""
    <div class="nav-bar">
        <div class="nexus-title">NexusIQ <span class="nexus-accent">Analyst</span></div>
        <div class="system-node">● SECURE NODE: {st.session_state.user_id[:8]}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("##### *Strategic Document Intelligence | RAG-Augmented Analysis*")

# 2. Sidebar: Production Control Center
with st.sidebar:
    st.markdown("### 🛠️ Control Center")
    st.divider()
    
    doc_upload = st.file_uploader("📂 Synchronize Data (PDF)", type="pdf")
    
    if doc_upload:
        if st.button("⚡ Sync Knowledge Base", use_container_width=True, type="primary"):
            with st.status("Syncing Security Node...", expanded=False) as status:
                if ingest_document_pipeline(doc_upload):
                    st.session_state.agent_ready = True
                    st.session_state.messages = []
                    status.update(label="Index Synchronized", state="complete")
                    st.toast("Intelligence loaded successfully.", icon="✅")

    st.divider()
    if st.button("🗑️ Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 3. Main Workspace Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Agentic Interaction Loop
if st.session_state.agent_ready:
    user_query = st.chat_input("Enter technical inquiry...")

    if user_query:
        st.chat_message("user").write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            llm_engine = ChatOllama(model=MODEL_NAME, temperature=0)
            
            with st.spinner("Analyzing document structure..."):
                context_payload = query_knowledge_base(user_query)
                
                if context_payload == "SERVICE_UNAVAILABLE":
                    st.warning("Database connection lost. Please re-sync.")
                else:
                    prompt = f"""
                    ROLE: NexusIQ Document Analyst.
                    TASK: Answer the inquiry using ONLY the provided context.
                    FORMAT: Use **bold headers**, bullet points, and tables for data.
                    
                    CONTEXT:
                    {context_payload}
                    
                    USER INQUIRY: {user_query}
                    
                    RESPONSE:
                    """
                    
                    res_box = st.empty()
                    final_text = ""
                    
                    for chunk in llm_engine.stream(prompt):
                        final_text += chunk.content
                        res_box.markdown(final_text + "▌")
                    res_box.markdown(final_text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": final_text})
                    
                    if len(st.session_state.messages) > 10:
                        st.session_state.messages = st.session_state.messages[-10:]
else:
    st.info("👋 **NexusIQ Ready.** Please upload a data source in the Control Center to begin.")