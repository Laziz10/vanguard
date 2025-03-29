import numpy as np
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float

import os
import re
import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        "<div style='color:white; font-weight:bold; font-size:18px; margin-bottom:0.25rem;'>Speaker Analysis</div>",
        unsafe_allow_html=True
    )

    # Ensure session state is initialized
    if "selected_speaker" not in st.session_state:
        st.session_state.selected_speaker = "All"

    # Safe dropdown handling with a fallback if index is out of range
    selected_speaker = st.selectbox(
        label="Speaker Dropdown",
        options=speakers,
        index=speakers.index(st.session_state.selected_speaker) if st.session_state.selected_speaker in speakers else 0,
        label_visibility="collapsed"
    )
    st.session_state.selected_speaker = selected_speaker

    # Do not show the name of the speaker or title under the dropdown after selection
    if st.session_state.uploaded_file is None:
        st.markdown("### **Upload Earnings Call PDF**", unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["pdf"], key="uploader")
        if uploaded is not None:
            st.session_state.uploaded_file = uploaded
            st.rerun()

    # Add an option to download the summary and Q&A as a PDF
    if st.session_state.uploaded_file and len(st.session_state.chat_history) > 0:
        if st.button("Download Summary and Q&A as PDF"):
            pdf_filename = "Earnings_Call_Summary_and_QA.pdf"
            generate_pdf(pdf_filename)

# Session values
uploaded_file = st.session_state.uploaded_file
selected_speaker = st.session_state.selected_speaker

# --- Main Area ---
st.image("vanguard_logo.png", width=180)
st.markdown("## **Earnings Call Summarizer**")

if uploaded_file:
    pdf_bytes = BytesIO(uploaded_file.getvalue())
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        raw_text = "".join([page.get_text() for page in doc])

    if selected_speaker != "All":
        # Extracting name without title for matching
        speaker_name_for_matching = selected_speaker.split(" (")[0] if selected_speaker != "All" else "All"

        # Updated regex for case-insensitive matching and handling the colon after the name
        pattern = re.compile(
            rf"{re.escape(speaker_name_
