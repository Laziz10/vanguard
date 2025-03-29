import numpy as np
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float

import os
import streamlit as st
import fitz  # PyMuPDF
import re

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load CSS
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

# Page setup
st.set_page_config(page_title="📊 Earnings Call Summarizer", layout="wide")
load_css()

# OpenAI API key securely from Streamlit secrets
openai_key = st.secrets.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

# Sidebar
with st.sidebar:
    st.markdown("### 📁 **Upload Earnings Call PDF**", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"])

    st.markdown("### 📊 **Chunks to summarize**", unsafe_allow_html=True)
    chunk_size = st.slider("Chunk size", 300, 1500, 1000, step=100)

    st.markdown("### 🎤 **Filter Q&A by speaker**", unsafe_allow_html=True)
    selected_speaker = st.selectbox("Speaker", ["All"])

# Main
st.image("vanguard_logo.png", width=180)
st.markdown("## **📄 Earnings Call Summarizer**")

if uploaded_file:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()

    st.markdown("### 🔍 Transcript Preview")
    with st.expander("Show Raw Text"):
        st.text(raw_text[:3000] + "...")

    st.markdown("### 🧠 Generating Summary...")

    # Text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    chunks = splitter.create_documents([raw_text])

    try:
        # Embedding
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # LLM for summary
        llm = ChatOpenAI(temperature=0)
        summary_prompt = "Summarize the earnings call including major highlights, risks, opportunities, and sentiment:"
        summary = llm.predict(summary_prompt + "\n\n" + raw_text[:3000])

        st.markdown("### 📌 Summary")
        st.markdown(summary)

        # Ask a question
        st.markdown("### 💬 Ask a Question")
        question = st.text_input("Ask something about this call:")
        if question:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                chain_type="stuff"
            )
            answer = qa_chain.run(question)
            st.success(answer)

    except Exception as e:
        st.error(f"⚠️ Vectorstore creation failed: {e}")

