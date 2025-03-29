import numpy as np
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float

import streamlit as st
import fitz  # PyMuPDF
import openai
import os
import re
from textblob import TextBlob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 🔐 Use API key from Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config("Earnings Call Assistant", layout="wide")

# Load custom CSS
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css()

# Sidebar
with st.sidebar:
    st.markdown("## 📄 Upload Earnings Call PDF", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="pdf")
    st.markdown("## 🧩 Chunks to summarize", unsafe_allow_html=True)
    chunk_count = st.slider("", 1, 6, 4)
    st.markdown("## 🎤 Filter Q&A by speaker", unsafe_allow_html=True)

# Initialize session state
if "question_history" not in st.session_state:
    st.session_state.question_history = []
if "answer_history" not in st.session_state:
    st.session_state.answer_history = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

# Vanguard logo and title
st.image("vanguard_logo.png", width=200)
st.markdown("<h1 style='color:#8B0000;'>Earnings Call Summarizer</h1>", unsafe_allow_html=True)

if uploaded_file:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = "".join([page.get_text() for page in doc])
    text = re.sub(r"\s{2,}", " ", text)

    speaker_roles = {
        "Timothy Michael Sweeney": "President, CEO & Director",
        "Christopher Locke Peirce": "EVP & CFO",
        "Neeti Bhalla Johnson": "EVP, President of Global Risk Solutions",
        "Hamid Talal Mirza": "EVP, President of US Retail Markets",
        "Vlad Yakov Barbalat": "CIO, President of Liberty Mutual Investments",
        "Robert Pietsch": "Executive Director of IR"
    }
    speaker_list = ["All"] + [f"{name} — {role}" for name, role in speaker_roles.items()]
    selected_speaker_label = st.sidebar.selectbox("", speaker_list)
    selected_speaker = selected_speaker_label.split(" — ")[0] if " — " in selected_speaker_label else "All"

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key)

    # Summary
    st.subheader("📌 Summary")
    joined_summary = ""
    for chunk in chunks[:chunk_count]:
        prompt = f"Summarize this part of the earnings call:\n\n{chunk.page_content}"
        result = llm.predict(prompt)
        joined_summary += f"- {result.strip()}\n"
    st.markdown(joined_summary)

    # Sentiment
    st.subheader("📊 Overall Sentiment")
    polarity = TextBlob(joined_summary).sentiment.polarity
    sentiment = "Positive" if polarity > 0.2 else "Negative" if polarity < -0.2 else "Neutral"
    st.markdown(f"**{sentiment}**")

    # Ask a question
    st.subheader("💬 Ask a Question About the Call")
    question = st.text_input("Your question")
    if question:
        st.session_state.question_history.append(question)

    # Q&A history (chat format)
    if st.session_state.question_history:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff"
        )
        for i, q in enumerate(st.session_state.question_history[len(st.session_state.answer_history):]):
            result = qa_chain.run(q)
            st.session_state.answer_history.append(result)

        st.subheader("🧠 Q&A History")
        for q, a in zip(st.session_state.question_history, st.session_state.answer_history):
            if selected_speaker == "All" or selected_speaker in a:
                st.markdown(f"<div class='chat-bubble user'><strong>You:</strong> {q}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-bubble bot'><strong>Bot:</strong> {a}</div>", unsafe_allow_html=True)

    # Suggested Questions
    st.subheader("🎯 Suggested Questions")
    if not st.session_state.suggested_questions:
        prompt = f"Based on this summary, suggest 4 insightful follow-up questions:\n\n{joined_summary}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        lines = response["choices"][0]["message"]["content"].split("\n")
        clean = [re.sub(r"^\s*(Q\d[:\.\-]|\d+[\.\:])?\s*", "", q.strip("-• ")) for q in lines if q.strip()]
        st.session_state.suggested_questions = [f"Q{i+1}: {q}" for i, q in enumerate(clean)]

    for q in st.session_state.suggested_questions:
        if st.button(q):
            st.session_state.question_history.append(q)
