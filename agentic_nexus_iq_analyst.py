import streamlit as st
import pdfplumber
import uuid
import os
import threading
import time
from typing import Any
from dotenv import load_dotenv

# LangChain & Vector Store
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch

# CrewAI
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# --- INITIAL SETUP ---
load_dotenv()

os.environ["CREWAI_USE_NATIVE_TOOLS"] = "false"
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ.pop("OPENAI_API_KEY", None)

if not os.environ["GROQ_API_KEY"]:
    st.error("⚠️ GROQ_API_KEY not found!")
    st.stop()

EMBED_MODEL = "nomic-embed-text"

# Multi-user lock
if 'global_lock' not in globals():
    global_lock = threading.Lock()

# --- HELPERS ---

def get_user_faiss_path():
    return f"faiss_index_{st.session_state.user_id[:8]}"

@st.cache_resource
def get_embeddings_model():
    return OllamaEmbeddings(model=EMBED_MODEL)

# --- PDF INGESTION ---

def ingest_document_pipeline(pdf_stream: Any) -> bool:
    try:
        with pdfplumber.open(pdf_stream) as pdf:
            text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])

        if not text.strip():
            return False

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)

        vector_db = FAISS.from_texts(chunks, get_embeddings_model())
        vector_db.save_local(get_user_faiss_path())

        return True

    except Exception as e:
        st.error(f"Ingestion Error: {str(e)}")
        return False

# --- PDF QUERY ---

def query_vector_store(query: str) -> str:
    path = get_user_faiss_path()

    if not os.path.exists(path):
        return ""

    try:
        vector_db = FAISS.load_local(
            path,
            get_embeddings_model(),
            allow_dangerous_deserialization=True
        )

        results = vector_db.max_marginal_relevance_search(
            query,
            k=3,
            fetch_k=6
        )

        return "\n\n".join([doc.page_content for doc in results])

    except Exception:
        return ""

# --- AGENT ENGINE ---

def run_agentic_research(user_query: str):

    llm = LLM(
        model="groq/llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        temperature=0.0
    )

    tavily_instance = TavilySearch(max_results=2)

    # ✅ PDF TOOL
    @tool("PDFSearch")
    def pdf_search_tool(query: str):
        """
        Search inside the uploaded PDF document.
        """
        result = query_vector_store(query)
        return result if result else "PDF_SEARCH_FAILED"

    # ✅ WEB TOOL
    @tool("WebSearch")
    def web_search_tool(query: str):
        """
        Search the internet for general knowledge.
        """
        return tavily_instance.run(query)

    # ✅ AGENT
    researcher = Agent(
        role='Analyst',
        goal=f'Answer: {user_query}',
        backstory="""
            You are an intelligent assistant.

            RULES:
            - Use PDFSearch first
            - If it fails, use WebSearch
            - ALWAYS return a final answer
            - NEVER show Thought/Action steps
        """,
        tools=[pdf_search_tool, web_search_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=5
    )

    # ✅ TASK
    research_task = Task(
        description=f"""
        Answer the query: {user_query}

        Follow rules:
        1. Try PDFSearch first
        2. If it fails → use WebSearch
        3. Use WebSearch directly for general queries
        4. Keep answer short (2-3 sentences)
        """,
        expected_output="""
        Give ONLY the final answer.
        Do NOT include Thought, Action, or tool steps.
        """,
        agent=researcher
    )

    # ✅ EXECUTION
    with global_lock:
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            process=Process.sequential,
            memory=False,
            max_rpm=2
        )

        result = crew.kickoff()
        time.sleep(1)
        return str(result)

# --- UI ---

st.set_page_config(page_title="NexusIQ Analyst", layout="wide")

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False

# Sidebar
with st.sidebar:
    st.title("🛠️ Control Center")

    doc_upload = st.file_uploader("Upload PDF", type="pdf")

    if doc_upload:
        if st.button("Process PDF"):
            with st.spinner("Processing..."):
                if ingest_document_pipeline(doc_upload):
                    st.session_state.agent_ready = True
                    st.success("PDF Loaded!")

    if st.button("Reset"):
        st.session_state.messages = []
        st.rerun()

# Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.agent_ready:
    user_input = st.chat_input("Ask anything...")

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = run_agentic_research(user_input)
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                except Exception as e:
                    st.error(str(e))
else:
    st.info("👋 Upload a PDF to start.")