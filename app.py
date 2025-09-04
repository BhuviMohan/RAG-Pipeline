import streamlit as st
import PyPDF2
import docx
import nltk
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
nltk.download("punkt")

# Extraction Functions
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Chunking Function
def chunk_text(text, max_tokens=500):
    sentences = nltk.sent_tokenize(text)
    chunks, chunk = [], []
    num_tokens = 0
    for sent in sentences:
        tokens = len(sent.split())
        if num_tokens + tokens > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, num_tokens = [], 0
        chunk.append(sent)
        num_tokens += tokens
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# Embedding & Vector DB setup
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    return embedder.encode(chunks)

def store_embeddings(embeddings):
    embeddings = np.array(embeddings).astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve(query, chunks, index, embedder, top_k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, top_k)
    indices = I[0]
    valid_indices = [int(i) for i in indices if int(i) >= 0 and int(i) < len(chunks)]
    return [chunks[i] for i in valid_indices]

def generate_answer_gemini(query, context, gemini_api_key):
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = model.generate_content(prompt)
    return response.text.strip()

st.title("AI Document Q/A")
gemini_api_key = os.getenv("GEMINI_API_KEY")
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if uploaded_file and uploaded_file.name != st.session_state.uploaded_file_name:
    st.session_state.query = ""
    st.session_state.uploaded_file_name = uploaded_file.name

if uploaded_file and gemini_api_key:
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        text = extract_text_from_docx(uploaded_file)
    elif file_type == "txt":
        text = extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file type!")
        st.stop()
    st.success("Document loaded! Processing...")
    chunks = chunk_text(text)
    st.info(f"Document split into {len(chunks)} chunks.")
    embeddings = embed_chunks(chunks)
    index = store_embeddings(embeddings)
    st.success("Document indexed.")

    # Show question input only after document processed
    query = st.text_input("Ask me a question about your document:", value=st.session_state.get("query", ""))
    st.session_state.query = query
    if query:
        retrieved_chunks = retrieve(query, chunks, index, embedder)
        context = "\n\n".join(retrieved_chunks)
        answer = generate_answer_gemini(query, context, gemini_api_key)
        st.markdown(f"**Answer:** {answer}")