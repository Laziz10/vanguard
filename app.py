import numpy as np
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float

import os
import re
import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
import yfinance as yf  # For financial data fetching

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ✅ MUST BE FIRST: Set page layout
st.set_page_config(page_title="Earnings Call Summarizer", layout="wide")

# ✅ Refined spacing to align Speaker & Vanguard logo
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.3rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Optional external CSS
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass
load_css()

# Load OpenAI key
openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = openai_key

# Initialize session
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "selected_speaker" not in st.session_state:
    st.session_state.selected_speaker = "All"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Updated Speakers List with positions in braces
speaker_titles = {
    "Brett Iversen": "CVP",
    "Satya Nadella": "CEO",
    "Amy Hood": "CFO",
    "Alice Jolla": "CAO"
}
speakers = ["All"] + [f"{speaker} ({title})" for speaker, title in speaker_titles.items()]

# --- Sidebar for Navigation ---
page = st.sidebar.selectbox("Select Page", ["Earnings Call Summary", "Microsoft Financial Performance"])

# --- Microsoft Financial Performance Page ---
if page == "Microsoft Financial Performance":
    st.title("Microsoft Financial Performance (Past Quarter)")

    # Sidebar for financial performance
    quarter = st.selectbox("Select the Quarter", ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"])
    
    # Fetch Microsoft financial data
    ticker = "MSFT"
    data = yf.Ticker(ticker)
    
    # Get financial reports
    financials = data.financials
    earnings = data.earnings
    balance_sheet = data.balance_sheet
    
    # Display the metrics for the selected quarter
    st.markdown(f"**Showing financial performance for: {quarter}**")

    st.markdown("### **Key Financial Metrics**")
    try:
        # Display key metrics from the earnings report and financial data
        revenue = financials.loc["Total Revenue"].iloc[0]
        operating_income = financials.loc["Operating Income"].iloc[0]
        net_income = financials.loc["Net Income"].iloc[0]
        eps = earnings.loc["EPS"].iloc[0]
        
        st.write(f"**Revenue**: ${revenue:,}")
        st.write(f"**Operating Income**: ${operating_income:,}")
        st.write(f"**Net Income**: ${net_income:,}")
        st.write(f"**Earnings Per Share (EPS)**: ${eps}")
    except KeyError:
        st.warning("Unable to retrieve all financial data for the selected quarter.")
    
    # Optionally, show a stock price chart for Microsoft
    st.markdown("#### **Stock Price Chart**")
    stock_data = data.history(period="1y")
    st.line_chart(stock_data['Close'])

# --- Earnings Call Summary Page ---
elif page == "Earnings Call Summary":
    st.title("Earnings Call Summarizer")
    
    # Sidebar for uploading file
    uploaded_file = st.file_uploader("Upload Earnings Call PDF", type=["pdf"])
    if uploaded_file:
        pdf_bytes = BytesIO(uploaded_file.getvalue())
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            raw_text = "".join([page.get_text() for page in doc])

        # Process the transcript text and display a summary
        st.markdown("### Summary of Earnings Call")
        st.write(raw_text[:1000])  # Display first 1000 characters of the transcript for review

        # Optional: Speaker filtering & displaying Q&A section
        selected_speaker = st.selectbox("Select Speaker", ["All", "Brett Iversen", "Satya Nadella", "Amy Hood", "Alice Jolla"])
        
        if selected_speaker != "All":
            speaker_pattern = re.compile(rf"{selected_speaker}:(.*?)(?=\n[A-Z][a-z]+:|\Z)", re.DOTALL)
            matches = speaker_pattern.findall(raw_text)
            filtered_speech = "\n".join(matches)
            st.write(filtered_speech)

        # Chat-like Q&A section for user interactions
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
            st.session_state.chat_input = ""  # Clear input

        # Input box with callback
        st.text_input("", key="chat_input", on_change=handle_question)

        # Display Q&A side-by-side (most recent first)
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

        # --- Follow-Up Questions --- 
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
            st.info("Transcript not available for generating follow-up questions.")
