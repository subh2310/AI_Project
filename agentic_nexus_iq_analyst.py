import streamlit as st
import pdfplumber
import uuid
import os
import threading
import time
from typing import List, Any, Optional
from dotenv import load_dotenv

# LangChain & Vector Store
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults

# CrewAI for Agentic Workflows
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# 1. INITIAL SYSTEM SETUP
load_dotenv() 

# Force ReAct pattern to prevent tool-calling loops
os.environ["CREWAI_USE_NATIVE_TOOLS"] = "false" 
os.environ["OPENAI_API_KEY"] = "NA"

# Access API Keys Safely
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

if not os.environ["GROQ_API_KEY"]:
    st.error("⚠️ GROQ_API_KEY not found in .env file!")
    st.stop()

EMBED_MODEL: str = "nomic-embed-text"

# --- MULTI-USER INFRASTRUCTURE ---

if 'global_lock' not in globals():
    global_lock = threading.Lock()

def get_user_faiss_path():
    """
    Generates a unique local directory path for the current user's vector store index.
    
    Returns:
        str: A string representing the folder path (e.g., 'faiss_index_a1b2c3d4').
    """
    return f"faiss_index_{st.session_state.user_id[:8]}"

# 2. PAGE CONFIGURATION
st.set_page_config(
    page_title="NexusIQ | Agentic Analyst",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem !important; }
    .nav-bar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 12px 24px; background: rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(128, 128, 128, 0.15); border-radius: 12px; margin-bottom: 1rem;
    }
    .system-node {
        color: #10b981; font-weight: 700; font-size: 0.75rem;
        text-transform: uppercase; background: rgba(16, 185, 129, 0.1);
        padding: 4px 10px; border-radius: 20px;
    }
    .nexus-title { font-weight: 800; font-size: 2rem; color: #3b82f6; }
    </style>
    """, unsafe_allow_html=True)

# 3. SESSION STATE
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LOGIC MODULES ---

@st.cache_resource
def get_embeddings_model():
    """
    Initializes and caches the Ollama embeddings model to avoid redundant loading.
    
    Returns:
        OllamaEmbeddings: The initialized embeddings model instance.
    """
    return OllamaEmbeddings(model=EMBED_MODEL)

def ingest_document_pipeline(pdf_stream: Any) -> bool:
    """
    Extracts text from a PDF, splits it into chunks, and saves it to a unique FAISS index.
    
    Parameters:
        pdf_stream (Any): The file-like object uploaded via Streamlit.
        
    Returns:
        bool: True if ingestion was successful, False otherwise.
    """
    try:
        with pdfplumber.open(pdf_stream) as pdf:
            text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        
        if not text.strip(): return False

        # Small chunks help stay under the 6k/12k token limit
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_text(text)
        
        vector_db = FAISS.from_texts(chunks, get_embeddings_model())
        vector_db.save_local(get_user_faiss_path()) 
        return True
    except Exception as e:
        st.error(f"Ingestion Error: {str(e)}")
        return False

def query_vector_store(query: str) -> str:
    """
    Searches the user-specific FAISS index for document snippets relevant to the query.
    
    Parameters:
        query (str): The search term or user question.
        
    Returns:
        str: A concatenated string of the top relevant document chunks.
    """
    path = get_user_faiss_path()
    if not os.path.exists(path): return ""
    try:
        vector_db = FAISS.load_local(path, get_embeddings_model(), allow_dangerous_deserialization=True)
        # Pull only 2 results to keep the prompt small
        results = vector_db.similarity_search(query, k=2) 
        return "\n\n".join([doc.page_content for doc in results])
    except Exception:
        return ""

def run_agentic_research(user_query: str):
    """
    Orchestrates a CrewAI agent to research a query using PDF context and Web Search.
    Uses a global lock to manage multi-user traffic and prevent API rate limits.
    
    Parameters:
        user_query (str): The question asked by the user.
        
    Returns:
        Any: The final text output from the CrewAI research task.
    """
    search_llm = LLM(
        model="groq/llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.0
    )
    
    tavily_instance = TavilySearchResults(k=2) # Keep search results small
    
    @tool("WebSearch")
    def web_search_tool(query: str):
        """Useful for searching the internet for details not in the PDF."""
        return tavily_instance.run(query)

    # 1. Get Context and TRUNCATE it to stay under token limits
    raw_context = query_vector_store(user_query)
    local_context = raw_context[:3000] if raw_context else "No local info."

    # 2. Define Lean Agent (Minimize backstory to save tokens)
    researcher = Agent(
        role='Analyst',
        goal=f'Answer: {user_query}',
        # ADD THIS LINE: It tells the model to follow the standard CrewAI format
        backstory="You are a technical researcher. Always use the following format: Thought: [your reasoning], Action: [tool name], Action Input: [query].",
        tools=[web_search_tool],
        llm=search_llm,
        verbose=True,
        allow_delegation=False,
        # SET THIS TO TRUE: This helps the agent "self-correct" if the tool output looks weird
        max_iter=5 
    )

    # 3. Define Task
    research_task = Task(
        description=f"Query: {user_query}\n\nContext: {local_context}",
        expected_output="A structured technical report under 400 words.",
        agent=researcher
    )

    # 4. Execute with Global Lock for Multi-User safety
    with global_lock:
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            process=Process.sequential,
            memory=False, # Avoids SQLite conflicts
            max_rpm=2     # Throttles requests to respect Groq limits
        )
        result = crew.kickoff()
        time.sleep(1) # Small cool-down for the next user
        return result

# --- UI INTERFACE ---
st.markdown(f"""
    <div class="nav-bar">
        <div class="nexus-title">NexusIQ <span style="color:white">Analyst</span></div>
        <div class="system-node">● NODE: {st.session_state.user_id[:8]}</div>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛠️ Control Center")
    doc_upload = st.file_uploader("📂 Sync PDF Data", type="pdf")
    
    if doc_upload:
        if st.button("⚡ Synchronize Node", use_container_width=True, type="primary"):
            with st.status("Ingesting Intelligence...", expanded=False):
                if ingest_document_pipeline(doc_upload):
                    st.session_state.agent_ready = True
                    st.toast("Sync Complete")

    if st.button("🗑️ Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Chat Display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.agent_ready:
    user_input = st.chat_input("Ask about the document...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant"):
            with st.spinner("Queueing request (Multi-user protection active)..."):
                try:
                    result = run_agentic_research(user_input)
                    final_response = str(result)
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                except Exception as e:
                    st.error(f"Rate Limit or Token Error: {str(e)}")
                    st.info("Try a shorter question or wait 10 seconds.")
else:
    st.info("👋 **System Standby.** Upload a PDF to start.")