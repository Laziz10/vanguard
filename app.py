import numpy as np
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float

import os
import re
import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO

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

# OpenAI API key
openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = openai_key

# Session state defaults
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "selected_speaker" not in st.session_state:
    st.session_state.selected_speaker = "All"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Speaker list
speaker_titles = {
    "Christopher Locke Peirce": "Executive VP & CFO",
    "Hamid Talal Mirza": "Executive VP, President of US Retail Markets & Director",
    "Neeti Bhalla Johnson": "Executive VP, President of Global Risk Solutions & Director",
    "Robert Pietsch": "",
    "Timothy Michael Sweeney": "President, CEO & Director",
    "Vlad Yakov Barbalat": "Chief Investment Officer, Executive VP, President of Liberty Mutual Investments & Director",
    "Chad Stogel": "Spectrum Asset Management, Inc."
}
speakers = ["All"] + list(speaker_titles.keys())

# Sidebar
with st.sidebar:
    # Speaker dropdown always shown
    st.markdown("### **Call Participants**", unsafe_allow_html=True)
    selected_speaker = st.selectbox("Select a speaker to analyze their speech:", options=speakers, index=speakers.index(st.session_state.selected_speaker))
    st.session_state.selected_speaker = selected_speaker

    if selected_speaker != "All":
        title = speaker_titles.get(selected_speaker, "")
        if title:
            st.markdown(f"<p style='color: white; font-style: italic; margin-top: 0.25rem;'>{title}</p>", unsafe_allow_html=True)

    # File uploader only shown if no file is uploaded
    if not st.session_state.uploaded_file:
        st.markdown("### **Upload Earnings Call PDF**", unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["pdf"])
        if uploaded:
            st.session_state.uploaded_file = uploaded

# Main UI
uploaded_file = st.session_state.uploaded_file
selected_speaker = st.session_state.selected_speaker

st.image("vanguard_logo.png", width=180)
st.markdown("## **Earnings Call Summarizer**")

if uploaded_file:
    # Fix for PyMuPDF EmptyFileError
    pdf_bytes = BytesIO(uploaded_file.getvalue())
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()

    # Filter transcript by selected speaker
    if selected_speaker != "All":
        pattern = re.compile(
            rf"{selected_speaker}\s*\n(.*?)(?=\n[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s*\n|$)",
            re.DOTALL
        )
        matches = pattern.findall(raw_text)
        if matches:
            raw_text = "\n".join(matches).strip()
        else:
            st.warning(f"No speech found for {selected_speaker}. Displaying empty result.")
            raw_text = ""

    if not raw_text.strip():
        st.warning("No transcript text available for summarization.")
    else:
        # Text chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        chunks = splitter.create_documents([raw_text])

        try:
            # Embeddings & LLM setup
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            llm = ChatOpenAI(temperature=0)

            # Summary generation
            summary_prompt = (
                "Summarize the earnings call into four main sections:\n"
                "1. Key financial highlights\n"
                "2. Risks and concerns\n"
                "3. Opportunities or forward-looking statements\n"
                "4. General sentiment\n"
                "Format each as a section title followed by 1–2 bullet points."
            )
            response = llm.predict(summary_prompt + "\n\n" + raw_text[:3000])

            lines = [
                line.strip() for line in response.split("\n")
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

            styled_summary, bullet_group = "", ""
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

        # Q&A Section
        st.markdown("### Ask a Question")
        st.text_input("", key="chat_input", on_change=lambda: handle_question(vectorstore, llm))

        # Display Q&A
        for pair in reversed([
            {"question": q["content"], "answer": a["content"]}
            for q, a in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2])
            if q["role"] == "user" and a["role"] == "ai"
        ]):
            st.markdown(f"""
            <div style="display: flex; gap: 2rem; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                <div style="flex: 1; color: black; font-weight: bold;">Q: {pair['question']}</div>
                <div style="flex: 2; color: black; font-weight: bold;">A: {pair['answer']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Suggested Follow-Up Questions
        st.markdown("### Suggested Follow-Up Questions")
        if raw_text.strip():
            followup_prompt = (
                f"Based on the following earnings call transcript, suggest 3 insightful follow-up questions "
                f"that an analyst might ask to better understand the discussion.\n\n"
                f"---\n\n{raw_text[:2000]}\n\n---\n\n"
                f"List each question on a new line, without numbering."
            )
            try:
                followup_response = llm.predict(followup_prompt)
                followup_questions = [q.strip("-• ").strip() for q in followup_response.strip().split("\n") if q.strip()]

                for i, question in enumerate(followup_questions):
                    if st.button(question, key=f"followup_q_{i}"):
                        st.session_state.chat_history.append({"role": "user", "content": question})
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=vectorstore.as_retriever(),
                            chain_type="stuff"
                        )
                        answer = qa_chain.run(question)
                        st.session_state.chat_history.append({"role": "ai", "content": answer})
                        st.rerun()
            except Exception as e:
                st.warning(f"Could not generate follow-up questions: {e}")
        else:
            st.info("Transcript not available for follow-up generation.")

# Question handler
def handle_question(vectorstore, llm):
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
