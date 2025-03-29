import numpy as np
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float

import os
import re
import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ✅ MUST BE FIRST: Set page layout
st.set_page_config(page_title="Earnings Call Summarizer", layout="wide")

# Styling tweaks
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0.3rem !important; }
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

# API key
openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = openai_key

# --- Session State ---
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "selected_speaker" not in st.session_state:
    st.session_state.selected_speaker = "All"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_benchmark" not in st.session_state:
    st.session_state.show_benchmark = False

# --- Speaker Titles ---
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
        "<div style='color:white; font-weight:bold; font-size:18px; margin-bottom:0.25rem;'>Select View</div>",
        unsafe_allow_html=True
    )

    if st.button("Speaker Analysis", use_container_width=True):
        st.session_state.show_benchmark = False

    if st.button("Benchmark Analysis", use_container_width=True):
        st.session_state.show_benchmark = True

    if not st.session_state.show_benchmark:
        selected_speaker = st.selectbox(
            label="Speaker Dropdown",
            options=speakers,
            index=speakers.index(st.session_state.selected_speaker) if st.session_state.selected_speaker in speakers else 0,
            label_visibility="collapsed"
        )
        st.session_state.selected_speaker = selected_speaker

    if st.session_state.uploaded_file is None:
        st.markdown("### **Upload Earnings Call PDF**", unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["pdf"], key="uploader")
        if uploaded:
            st.session_state.uploaded_file = uploaded
            st.rerun()

# --- Main Content ---
st.image("vanguard_logo.png", width=180)
st.markdown("## **Earnings Call Summarizer**")

uploaded_file = st.session_state.uploaded_file
selected_speaker = st.session_state.selected_speaker

# --- Benchmark Analysis Section ---
def get_10yr_annual_return_comparison():
    years = list(range(2014, 2024))
    msft = [24.16, 19.44, 12.00, 37.66, 18.74, 55.26, 41.04, 51.21, -28.69, 56.80]
    goog = [13.89, 44.56, -1.84, 35.58, -0.80, 28.18, 30.85, 65.17, -38.68, 47.38]
    aapl = [40.00, -4.64, 10.03, 48.24, -5.39, 88.96, 82.31, 34.65, -26.40, 48.00]
    vgt  = [18.53, 4.50, 15.89, 36.20, 1.99, 51.60, 44.74, 29.87, -29.78, 51.26]
    return pd.DataFrame({
        "Year": years,
        "MSFT (%)": msft,
        "GOOG (%)": goog,
        "AAPL (%)": aapl,
        "VGT (%)": vgt
    })

def plot_10yr_stock_returns(df):
    plt.figure(figsize=(10, 5))
    for col in df.columns[1:]:
        plt.plot(df["Year"], df[col], marker='o', label=col)
    plt.title("10-Year Annual Return Comparison (2014–2023)")
    plt.xlabel("Year")
    plt.ylabel("Return (%)")
    plt.axhline(0, linestyle='--', color='gray')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def generate_return_insights():
    return [
        "VGT closely mirrors the average performance of its top holdings like MSFT and AAPL.",
        "Microsoft delivered consistent returns, ending 2023 strong at +56.80%.",
        "Apple peaked in 2019 (+88.96%) and 2020 (+82.31%) but dipped in 2022.",
        "All assets experienced significant drops in 2022 during tech sector correction."
    ]

# --- Display Main Content ---
if st.session_state.show_benchmark:
    st.markdown("## Benchmark Analysis")
    df_returns = get_10yr_annual_return_comparison()
    st.dataframe(df_returns, use_container_width=True)
    plot_10yr_stock_returns(df_returns)
    st.markdown("### Insights")
    for insight in generate_return_insights():
        st.markdown(f"- {insight}")
else:
    if uploaded_file:
        pdf_bytes = BytesIO(uploaded_file.getvalue())
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            raw_text = "".join([page.get_text() for page in doc])

        if selected_speaker != "All":
            speaker_name = selected_speaker.split(" (")[0]
            pattern = re.compile(
                rf"{re.escape(speaker_name)}\s*:\s*(.*?)(?=\n[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s*\n|$)",
                re.DOTALL | re.IGNORECASE
            )
            matches = pattern.findall(raw_text)
            raw_text = "\n".join(matches).strip() if matches else ""
            if not matches:
                st.warning(f"No speech found for {selected_speaker}.")

        if not raw_text.strip():
            st.warning("No transcript text available.")
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            chunks = splitter.create_documents([raw_text])
            try:
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(chunks, embeddings)
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
                lines = [line.strip() for line in response.split("\n") if line.strip()]

                section_titles = [
                    "Key financial highlights",
                    "Risks and concerns",
                    "Opportunities or forward-looking statements",
                    "General sentiment"
                ]

                styled_summary, bullets = "", ""
                for line in lines:
                    normalized = re.sub(r"^\d+\.\s*", "", line).rstrip(":").strip()
                    if any(normalized.lower().startswith(t.lower()) for t in section_titles):
                        if bullets:
                            styled_summary += f"<ul>{bullets}</ul>"
                            bullets = ""
                        styled_summary += f"<p style='font-weight:bold;'>{normalized}:</p>"
                    else:
                        bullets += f"<li>{line.strip('-• ')}</li>"

                if bullets:
                    styled_summary += f"<ul>{bullets}</ul>"

                st.markdown("### Summary", unsafe_allow_html=True)
                st.markdown(styled_summary, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Vectorstore error: {e}")

            st.markdown("### Ask a Question")
            st.text_input("", key="chat_input", on_change=lambda: handle_question(vectorstore, llm))

            for pair in reversed([
                {"question": q["content"], "answer": a["content"]}
                for q, a in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2])
                if q["role"] == "user" and a["role"] == "ai"
            ]):
                st.markdown(f"**Q: {pair['question']}**\n\n**A:** {pair['answer']}")

            st.markdown("### Suggested Follow-Up Questions")
            try:
                followup_prompt = (
                    f"Based on the following earnings call transcript, suggest 3 insightful follow-up questions:\n\n"
                    f"{raw_text[:2000]}"
                )
                followup_response = llm.predict(followup_prompt)
                followups = [q.strip("-• ").strip() for q in followup_response.split("\n") if q.strip()]
                for i, q in enumerate(followups):
                    if st.button(q, key=f"followup_{i}"):
                        st.session_state.chat_history.append({"role": "user", "content": q})
                        answer = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=vectorstore.as_retriever(),
                            chain_type="stuff"
                        ).run(q)
                        st.session_state.chat_history.append({"role": "ai", "content": answer})
                        st.rerun()
            except Exception as e:
                st.warning(f"Follow-up question generation failed: {e}")

# --- Q&A Handler ---
def handle_question(vectorstore, llm):
    user_input = st.session_state.chat_input.strip()
    if not user_input:
        return
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff"
    )
    answer = qa_chain.run(user_input)
    st.session_state.chat_history.append({"role": "ai", "content": answer})
    st.session_state.chat_input = ""
