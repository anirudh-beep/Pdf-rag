import streamlit as st
import tempfile
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

st.set_page_config(page_title="PDF RAG Chatbot")
st.title("📄 PDF Question Answering (RAG)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Safety check
    if len(documents) == 0:
        st.error("❌ No readable text found in PDF (image/scanned PDF not supported).")
        st.stop()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    if len(chunks) == 0:
        st.error("❌ Could not split PDF text.")
        st.stop()

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Local FREE LLM (Ollama)
    llm = Ollama(model="mistral")

    # User question
    question = st.text_input("Ask a question:")

    if question:
        # Retrieve relevant chunks
        docs = retriever.invoke(question)

        # Build context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Prompt
        prompt = f"""
You are a helpful assistant.
Answer the question ONLY using the context below.
If answer not found, say "Not found in document".

Context:
{context}

Question: {question}
Answer:
"""

        # Generate response
        response = llm.invoke(prompt)

        st.subheader("Answer:")
        st.write(response)

        st.subheader("Source Pages:")
        for doc in docs:
            st.write("Page:", doc.metadata.get("page", "N/A"))

    os.remove(pdf_path)
