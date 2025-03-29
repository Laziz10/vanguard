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
import pandas as pd
import matplotlib.pyplot as plt

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ✅ MUST BE FIRST: Set page layout
st.set_page_config(page_title="Earnings Call Summarizer", layout="wide")

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
    st.markdown("### 🎤 Speaker Analysis")

    selected_speaker = st.selectbox(
        label="Speaker Dropdown",
        options=speakers,
        index=speakers.index(st.session_state.selected_speaker) if st.session_state.selected_speaker in speakers else 0,
        label_visibility="collapsed"
    )
    st.session_state.selected_speaker = selected_speaker

    if st.session_state.uploaded_file is None:
        st.markdown("### **Upload Earnings Call PDF**")
        uploaded = st.file_uploader("", type=["pdf"], key="uploader")
        if uploaded is not None:
            st.session_state.uploaded_file = uploaded
            st.rerun()

    if st.session_state.uploaded_file and len(st
