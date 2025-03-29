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
st.set_page_config(page_title="Earnings Call Summarizer", layout="wide")
load_css()

# OpenAI API key securely from Streamlit secrets
openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = openai_key

# Sidebar
with st.sidebar:
    st.markdown("### **Upload Earnings Call PDF**", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"])

# Main UI
st.image("vanguard_logo.png", width=180)
st.markdown("## **Earnings Call Summarizer**")

if uploaded_file:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()

    # Extract speaker segments with name and title (support – or - as dash)
    speaker_blocks = re.findall(
        r"(?<=\n)([A-Z][A-Za-z\s\-]+[–\-]{1,2}\s+[A-Z][A-Za-z\s]+):([\s\S]*?)(?=\n[A-Z][A-Za-z\s\-]+[–\-]{1,2}\s+[A-Z][A-Za-z\s]+:|\Z)",
        raw_text
    )

    speaker_dict = {}
    for name_title, speech in speaker_blocks:
        key = name_title.strip()
        speaker_dict.setdefault(key, "")
        speaker_dict[key] += speech.strip() + "\n"

    all_speakers = ["All"] + sorted(speaker_dict.keys())

    with st.sidebar:
        st.markdown("### **Filter Q&A by speaker**", unsafe_allow_html=True)
        selected_speaker = st.selectbox("Speaker", all_speakers, index=0)

    if selected_speaker != "All":
        raw_text = speaker_dict.get(selected_speaker, "")
    else:
        raw_text = "\n".join(speaker_dict.values())

    # Set fixed chunk size
    chunk_size = 500
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    chunks = splitter.create_documents([raw_text])

    try:
        # Embedding
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # LLM for summary
        llm = ChatOpenAI(temperature=0)
        summary_prompt = (
            "Summarize the earnings call into four main sections:\n"
            "1. Key financial highlights\n"
            "2. Risks and concerns\n"
            "3. Opportunities or forward-looking statements\n"
            "4. General sentiment\n"
            "Format each as a section title followed by 1–2 bullet points."
        )
        response = llm.predict(summary_prompt + "\n\n" + raw_text[:3000])

        # Format sectioned summary
        styled_summary = ""
        raw_lines = response.split("\n")

        lines = [
            line.strip() for line in raw_lines
            if line.strip()
            and not line.lower().startswith("transcript of")
            and "sec filings" not in line.lower()
            and "risks and uncertainties" not in line.lower()
        ]

        section_titles = [
            "Key financial highlights",
            "Risks and concerns",
            "Opportunities or forward-looking statements",
            "General sentiment"
        ]

        bullet_group = ""

        for line in lines:
            normalized_line = re.sub(r"^\d+\.\s*", "", line).rstrip(":").strip()
            if any(normalized_line.lower().startswith(title.lower()) for title in section_titles):
                if bullet_group:
                    styled_summary += f"<ul>{bullet_group}</ul>"
                    bullet_group = ""
                styled_summary += f"<p style='color:black; font-weight:bold; font-size:16px'>{normalized_line}:</p>"
            else:
                clean_line = re.sub(r"^[-•\s]+", "", line)
                bullet_group += f"<li><span style='color:black;'>{clean_line}</span></li>"

        if bullet_group:
            styled_summary += f"<ul>{bullet_group}</ul>"

        st.markdown("### Summary", unsafe_allow_html=True)
        st.markdown(styled_summary, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Vectorstore creation failed: {e}")

    # Chat-style Q&A section
    st.markdown("### Ask a Question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def handle_question():
        user_input = st.session_state.chat_input.strip()
        if not user_input:
            return

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff"
        )
        answer = qa_chain.run(user_input)
        st.session_state.chat_history.append({"role": "ai", "content": answer})

        st.session_state.chat_input = ""

    st.text_input("", key="chat_input", on_change=handle_question)

    qa_pairs = []
    temp = {}

    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            temp["question"] = entry["content"]
        elif entry["role"] == "ai" and "question" in temp:
            temp["answer"] = entry["content"]
            qa_pairs.append(temp)
            temp = {}

    for pair in reversed(qa_pairs):
        st.markdown(f"""
        <div style="display: flex; gap: 2rem; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
            <div style="flex: 1; color: black; font-weight: bold;">Q: {pair['question']}</div>
            <div style="flex: 2; color: black; font-weight: bold;">A: {pair['answer']}</div>
        </div>
        """, unsafe_allow_html=True)
