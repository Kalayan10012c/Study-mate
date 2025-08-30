import os
import torch
import faiss
import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from sentence_transformers import SentenceTransformer

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = "ibm-granite/granite-3.2-2b-instruct"  # light Granite model for 4GB GPU
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Select GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"✅ Using device: {device}")

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")
    return model, tokenizer, embedding_model

model, tokenizer, embedding_model = load_models()

# -------------------------------
# TEXT EXTRACTION HELPERS
# -------------------------------
def read_txt(file):
    return file.read().decode("utf-8")

def read_pdf(file):
    """Extract text from PDF using PyMuPDF"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# -------------------------------
# FAISS INDEX
# -------------------------------
def create_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def search_index(query, chunks, index, embeddings, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# -------------------------------
# AI ANSWER GENERATION
# -------------------------------
def generate_answer_ai(query, context_chunks):
    context_text = "\n".join(context_chunks)
    prompt = f"Answer the following question based on the provided context.\n\nContext:\n{context_text}\n\nQuestion: {query}\nAnswer:"

    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    set_seed(42)
    output = model.generate(
        **input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )

    prediction = tokenizer.decode(
        output[0, input_ids["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return prediction.strip()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📚 StudyMate - Granite AI Powered")
st.write("Upload study material (TXT or PDF), ask questions, and get AI-powered answers!")

uploaded_file = st.file_uploader("Upload a .txt or .pdf study file", type=["txt", "pdf"])

if uploaded_file:
    if uploaded_file.type == "text/plain":
        text = read_txt(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    # simple chunking (could be improved with overlap)
    chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
    index, embeddings = create_faiss_index(chunks)

    query = st.text_input("Ask a question:")
    if query:
        top_chunks = search_index(query, chunks, index, embeddings)
        ai_answer = generate_answer_ai(query, top_chunks)

        st.subheader("📖 Answer")
        st.write(ai_answer)

        st.subheader("🔍 Context Used")
        for chunk in top_chunks:
            st.write(f"- {chunk}")
