import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb
import requests
import json

# Configuration
MODEL_NAME = "llama3" # llama3 is significantly faster/smarter than llama2
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="Ultra-Fast RAG", layout="wide")

# 1. Cache the Embedding Model so it doesn't reload
@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)

# 2. Cache the Vector DB Client
@st.cache_resource
def get_vector_client():
    return chromadb.Client()

def process_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    
    # Smaller chunks = faster retrieval and better accuracy
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    client = get_vector_client()
    # Reset collection for the new file
    try:
        client.delete_collection("temp_pdf")
    except:
        pass
    
    collection = client.create_collection(name="temp_pdf")
    
    # Embed and add
    embeddings_obj = load_embeddings()
    vectors = embeddings_obj.embed_documents(chunks)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors
    )
    return collection

# ---------------------------------------------------------
# UI UI UI
# ---------------------------------------------------------
st.title("⚡ Instant PDF Chat")

if "collection" not in st.session_state:
    st.session_state.collection = None

with st.sidebar:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Index Document"):
        with st.spinner("Processing..."):
            st.session_state.collection = process_pdf(uploaded_file)
            st.success("Ready!")

if st.session_state.collection:
    query = st.chat_input("Ask a question...")

    if query:
        st.chat_message("user").write(query)
        
        # 3. Fast Retrieval
        emb_model = load_embeddings()
        query_vec = emb_model.embed_query(query)
        results = st.session_state.collection.query(query_embeddings=[query_vec], n_results=4)
        context = "\n\n".join(results['documents'][0])

        # 4. Streamed Generation (No more waiting!)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_ans = ""
            
            # Direct Ollama Stream API
            url = "http://127.0.0.1:11434/api/generate"
            payload = {
                "model": MODEL_NAME,
                "prompt": f"Context: {context}\n\nQuestion: {query}\n\nAnswer briefly:",
                "stream": True
            }
            
            with requests.post(url, json=payload, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        full_ans += token
                        response_placeholder.markdown(full_ans + "▌")
            
            response_placeholder.markdown(full_ans)