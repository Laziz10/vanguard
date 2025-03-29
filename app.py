import numpy as np
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float

import streamlit as st
import fitz  # PyMuPDF
import re
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fpdf import FPDF
import os
import time

# ✅ Required by Streamlit
st.set_page_config(page_title="Earnings Call Assistant", layout="wide")



# OpenAI API key securely from Streamlit secrets
openai_key = st.secrets.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_key

# 🎨 Load Vanguard-style CSS
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css()

# --- SIDEBAR ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload Earnings Call PDF", type="pdf")
    chunk_count = st.slider("Chunks to summarize", 1, 6, 4)

# --- MAIN CONTENT ---
st.image("vanguard_logo.png", width=180)
st.title("Earnings Call Summarizer")

if uploaded_file:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = "".join([page.get_text() for page in doc])
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    speaker_roles = {
        "Timothy Michael Sweeney": "President, CEO & Director",
        "Christopher Locke Peirce": "EVP & CFO",
        "Neeti Bhalla Johnson": "EVP, President of Global Risk Solutions",
        "Hamid Talal Mirza": "EVP, President of US Retail Markets",
        "Vlad Yakov Barbalat": "CIO, President of Liberty Mutual Investments",
        "Robert Pietsch": "Executive Director of IR"
    }

    speaker_list = ["All"] + [f"{name} — {title}" for name, title in speaker_roles.items()]
    selected_speaker_label = st.sidebar.selectbox("Filter Q&A by speaker", speaker_list)
    selected_speaker = selected_speaker_label.split(" — ")[0] if " — " in selected_speaker_label else "All"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    def summarize_text(text_block):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Summarize this excerpt:\n{text_block}"}],
            temperature=0.5,
            max_tokens=300
        )
        return response["choices"][0]["message"]["content"]

    st.subheader("Summary")
    summary_bullets = []
    with st.spinner("Summarizing..."):
        for chunk in chunks[:chunk_count]:
            summary_bullets.append(summarize_text(chunk))

    st.markdown("<ul>" + "".join(f"<li>{s}</li>" for s in summary_bullets) + "</ul>", unsafe_allow_html=True)

    joined_summary = "\n".join(summary_bullets)
    sentiment_prompt = f"Classify the overall sentiment of this earnings call summary as Positive, Negative, or Neutral:\n\n{joined_summary}"
    sentiment_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": sentiment_prompt}],
        temperature=0
    )
    sentiment = sentiment_response["choices"][0]["message"]["content"]
    emoji = "🟢" if "Positive" in sentiment else "🔴" if "Negative" in sentiment else "🟡"
    st.markdown(f"### Overall Sentiment: {emoji} **{sentiment.strip()}**")

    time.sleep(20)

    def tag_speaker(chunk):
        for name in speaker_roles:
            if name in chunk:
                return name
        return "Unknown"

    tagged_chunks = [{"text": chunk, "speaker": tag_speaker(chunk)} for chunk in chunks]
    filtered_chunks = [c["text"] for c in tagged_chunks if selected_speaker == "All" or c["speaker"] == selected_speaker]

    st.subheader("Ask a Question About the Call")
    question = st.text_input("Your question", value=st.session_state.get("question", ""), label_visibility="visible")

    if question:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        doc_embeddings = embedder.encode(filtered_chunks)
        q_embedding = embedder.encode([question])
        index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        index.add(np.array(doc_embeddings))
        D, I = index.search(np.array(q_embedding), k=3)
        top_chunks = [filtered_chunks[i] for i in I[0]]
        context = "\n\n".join(top_chunks)

        prompt = f"Use the following excerpts from the transcript to answer:\n\n{context}\n\nQ: {question}\nA:"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=400
        )
        answer = response["choices"][0]["message"]["content"]
        st.markdown(f"**{answer}**")

    # ✅ Suggested Questions now below Q&A
    def suggest_questions(summary):
        prompt = f"Suggest 3 insightful questions a financial analyst might ask based on this summary:\n\n{summary}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response["choices"][0]["message"]["content"]

    st.markdown("### Suggested Questions")
    for i, q in enumerate(suggest_questions(joined_summary).split("\n"), 1):
        question_text = re.sub(r"^[\d\.\-\•\s]+", "", q).strip()
        if question_text:
            if st.button(f"Q{i}: {question_text}"):
                st.session_state["question"] = question_text

    def generate_pdf(lines):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "Earnings Call Summary\n")
        for line in lines:
            pdf.multi_cell(0, 10, f"- {line}")
        pdf_path = "earnings_summary.pdf"
        pdf.output(pdf_path)
        return pdf_path

    with st.sidebar:
        with open(generate_pdf(summary_bullets), "rb") as f:
            st.download_button("📥 Download Summary PDF", f.read(), file_name="earnings_summary.pdf")
