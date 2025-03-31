import numpy as np
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float

import os
import re
import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
import pandas as pd

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import yfinance as yf
import datetime
import feedparser

st.set_page_config(page_title="Earnings Call Summarizer", layout="wide")

# --- Global Styling ---
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.3rem !important;
    }
    .stRadio label span {
        color: white !important;
        font-weight: bold !important;
    }
    .stSelectbox label {
        color: white !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass
load_css()

openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = openai_key

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "selected_speaker" not in st.session_state:
    st.session_state.selected_speaker = "All"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_benchmark" not in st.session_state:
    st.session_state.selected_benchmark = None

speaker_titles = {
    "Brett Iversen": "CVP",
    "Satya Nadella": "CEO",
    "Amy Hood": "CFO",
    "Alice Jolla": "CAO"
}
speakers = ["All"] + [f"{speaker} ({title})" for speaker, title in speaker_titles.items()]
benchmark_stocks = ["VGT", "GOOGL", "AAPL", "AMZN"]

# --- Sidebar ---
sidebar_header_style = "color:white; font-weight:bold; font-size:16px; margin-bottom:0.25rem;"

with st.sidebar:
    st.markdown(f"<div style='{sidebar_header_style}'>Investor Menu</div>", unsafe_allow_html=True)

    view_mode = st.radio("", ["Speaker Analysis", "Market Analysis", "Benchmark Analysis", "Risk Analysis", "Recommendations"])
    
    if view_mode == "Speaker Analysis":
        st.markdown(f"<div style='{sidebar_header_style}'>Speaker Analysis</div>", unsafe_allow_html=True)
        selected_speaker = st.selectbox(
            label="Speaker Dropdown",
            options=speakers,
            index=speakers.index(st.session_state.selected_speaker),
            label_visibility="collapsed"
        )
        st.session_state.selected_speaker = selected_speaker
        st.session_state.selected_benchmark = None

if view_mode == "Market Analysis":
    st.markdown("### 📊 Market Analysis")

    ticker = st.text_input("Enter a Stock Ticker (e.g., MSFT, AAPL, GOOGL)", value="MSFT")

    if ticker:
        try:
            stock = yf.Ticker(ticker.upper())
            data = stock.history(period="1d", interval="5m")
            info = stock.info  # fallback to old API

            if data.empty or "regularMarketPrice" not in info:
                st.error("Could not retrieve market data for this ticker. Please check the symbol or try again later.")
            else:
                current_price = info.get("regularMarketPrice")
                open_price = info.get("regularMarketOpen")
                high = info.get("dayHigh")
                low = info.get("dayLow")
                volume = info.get("volume")
                year_high = info.get("fiftyTwoWeekHigh")
                year_low = info.get("fiftyTwoWeekLow")

                # Price movement logic
                price_diff = current_price - open_price
                percent = (price_diff / open_price) * 100 if open_price else 0
                arrow = "🔼" if price_diff > 0 else "🔽"

                st.markdown(f"## **${current_price:.2f}** {arrow} {price_diff:.2f} ({percent:.2f}%) Today")
                st.line_chart(data['Close'])

                st.markdown("#### Key Metrics")
                st.markdown(f"- **Open**: {open_price}")
                st.markdown(f"- **Day Range**: {low} - {high}")
                st.markdown(f"- **52 Week Range**: {year_low} - {year_high}")
                st.markdown(f"- **Volume**: {volume}")

                # --- Latest News Headlines ---
                st.markdown("#### Latest News Headlines")
                try:
                    news_feed = feedparser.parse(f"https://news.google.com/rss/search?q={ticker}+stock")
                    headlines = [f"- {entry.title}" for entry in news_feed.entries[:3]]
                    news_summary = "\n".join(headlines)
                    for line in headlines:
                        st.markdown(line)
                except Exception as e:
                    news_summary = "No news available."
                    st.warning(f"News fetch failed: {e}")

                # --- LLM Summary ---
                if "llm" in locals():
                    market_summary_prompt = f"""
You are a financial analyst assistant. Summarize the current market status of {ticker.upper()} based on the following:

1. **Stock Price & Change**: {current_price} USD ({price_diff:+.2f}, {percent:+.2f}%)
2. **Key Metrics**:
    - Open: {open_price}
    - Day Range: {low} - {high}
    - 52 Week Range: {year_low} - {year_high}
    - Volume: {volume}
3. **Recent Financial Performance**: Revenue $245.12B (+15.67%), Earnings $88.14B (+21.8%)
4. **Recent News Headlines**:
{news_summary}

Provide a short summary (~100 words) on overall market sentiment, short-term momentum, and any risks or growth drivers based on these inputs.
                    """
                    try:
                        llm_response = llm.predict(market_summary_prompt)
                        st.markdown("#### LLM Summary")
                        st.markdown(f"<div style='color:black; font-size:16px'>{llm_response}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"LLM summarization failed: {e}")

        except Exception as e:
            st.error(f"Error fetching data: {e}")    

    if view_mode in ["Benchmark Analysis", "Risk Analysis"]:
        st.markdown(f"<div style='{sidebar_header_style}'>{view_mode}</div>", unsafe_allow_html=True)
        selected_benchmark = st.selectbox(
            label="Benchmark Dropdown",
            options=benchmark_stocks,
            index=benchmark_stocks.index(st.session_state.selected_benchmark) if st.session_state.selected_benchmark else 0,
            label_visibility="collapsed",
            key="benchmark_dropdown"
        )
        st.session_state.selected_benchmark = selected_benchmark
        st.session_state.selected_speaker = "All"
    
    if st.session_state.uploaded_file is None:
        st.markdown("### **Upload Earnings Call PDF**", unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["pdf"], key="uploader")
        if uploaded is not None:
            st.session_state.uploaded_file = uploaded
            st.rerun()

# --- Main Header ---
st.image("vanguard_logo.png", width=180)  

uploaded_file = st.session_state.uploaded_file
selected_speaker = st.session_state.selected_speaker
selected_benchmark = st.session_state.selected_benchmark

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

# --- Benchmark Analysis ---
if view_mode == "Benchmark Analysis" and selected_benchmark:
    years = list(range(2015, 2025))
    prices = {
        "MSFT": [40.12, 52.12, 74.22, 101.57, 134.92, 157.70, 222.42, 295.44, 313.85, 375.62],
        "VGT":  [90.23, 108.87, 137.26, 170.03, 209.88, 238.65, 309.11, 358.74, 388.67, 420.95],
        "GOOGL":[30.42, 39.10, 52.88, 74.91, 100.21, 124.34, 151.77, 137.85, 141.12, 160.54],
        "AAPL": [14.25, 18.95, 28.61, 39.25, 55.56, 73.41, 101.67, 126.03, 162.32, 189.21],
        "AMZN": [312.65, 351.22, 389.61, 442.30, 497.22, 535.65, 566.74, 603.18, 625.44, 678.92]
    }

    def compute_yoy_growth(prices):
        return [None] + [round(((curr - prev) / prev) * 100, 2) for prev, curr in zip(prices[:-1], prices[1:])]

    msft_prices = prices["MSFT"]
    selected_prices = prices[selected_benchmark]
    msft_growth = compute_yoy_growth(msft_prices)
    selected_growth = compute_yoy_growth(selected_prices)

    df = pd.DataFrame({
        "Year": years,
        "MSFT Price": msft_prices,
        "MSFT YoY Growth (%)": msft_growth,
        f"{selected_benchmark} Price": selected_prices,
        f"{selected_benchmark} YoY Growth (%)": selected_growth
    })

    df_formatted = df.copy()
    for col in df_formatted.columns:
        if "Growth" in col:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x}%" if pd.notnull(x) else "—")

    styled_html = (
        "<style>"
        "table { width: 100%; border-collapse: collapse; background-color: white; color: black; }"
        "th, td { padding: 8px 12px; font-weight: bold; border: 1px solid #ddd; }"
        "th { background-color: #f0f0f0 !important; text-align: center !important; }"
        "thead th { text-align: center !important; }"
        "td { text-align: center; }"
        "</style>"
        + df_formatted.to_html(index=False, escape=False)
    )

    st.markdown(f"### Annual Price & Growth: {selected_benchmark} vs MSFT (2014–2024)")
    st.markdown(styled_html, unsafe_allow_html=True)

    msft_total = round(((msft_prices[-1] - msft_prices[0]) / msft_prices[0]) * 100, 2)
    selected_total = round(((selected_prices[-1] - selected_prices[0]) / selected_prices[0]) * 100, 2)

    if selected_total > msft_total:
        insight = f"{selected_benchmark} outperformed MSFT from 2014 to 2024 with a total growth of {selected_total}% vs {msft_total}%."
    elif selected_total < msft_total:
        insight = f"MSFT outperformed {selected_benchmark} from 2014 to 2024 with a total growth of {msft_total}% vs {selected_total}%."
    else:
        insight = f"{selected_benchmark} and MSFT had equal growth of {msft_total}% over the 10-year period."

    st.markdown("#### Key Insights")
    st.markdown(f"<div style='color:black; font-size:16px'>{insight}</div>", unsafe_allow_html=True)
    
# --- Risk Analysis ---
if view_mode == "Risk Analysis" and selected_benchmark:
    st.markdown(f"### Risk Assessment: {selected_benchmark}")

    risk_insights = {
        "AMZN": [
            "Regulatory pressure on cloud and AI services.",
            "Dependence on enterprise contracts for revenue stability.",
            "High valuation may limit upside during market corrections."
        ],
        "AAPL": [
            "Heavy reliance on iPhone sales for majority of revenue.",
            "Supply chain concentration in China.",
            "Slower innovation cycles in recent years."
        ],
        "GOOGL": [
            "Ad revenue vulnerability to macroeconomic cycles.",
            "Rising competition in AI and cloud segments.",
            "Ongoing antitrust scrutiny in U.S. and EU."
        ],
        "VGT": [
            "Broad tech exposure provides diversification.",
            "Lower company-specific risk than individual stocks.",
            "Still subject to sector-wide downturns but mitigated by ETF structure."
        ]
    }

    st.markdown("#### Identified Risks")
    for risk in risk_insights[selected_benchmark]:
        st.markdown(f"- {risk}", unsafe_allow_html=True)

    if selected_benchmark == "VGT":
        st.markdown("#### Summary")
        st.markdown(
            "<div style='color:black; font-size:16px'>"
            "VGT offers a more balanced risk profile by diversifying across multiple tech leaders, "
            "reducing exposure to individual company setbacks while still capturing overall sector growth."
            "</div>",
            unsafe_allow_html=True
        )
        
if view_mode == "Recommendations":

    st.markdown("""
    <ul style='font-size:16px; color:black; padding-left:1rem; line-height:1.8'>
        <li><span style='font-weight:bold'>Think long-term</span>: Successful investing requires patience and discipline over decades, not months.</li>
        <li><span style='font-weight:bold'>Stay diversified</span>: Broad diversification reduces risk and helps capture market returns.</li>
        <li><span style='font-weight:bold'>Minimize costs</span>: Lower fees mean you keep more of your investment returns.</li>
        <li><span style='font-weight:bold'>Stay the course</span>: Avoid emotional decisions during market swings—stick to your plan.</li>
        <li><span style='font-weight:bold'>Focus on what you can control</span>: Set clear goals, choose the right asset mix, and rebalance as needed.</li>
    </ul>
    """, unsafe_allow_html=True)





# --- Transcript + Speaker Summary ---
if uploaded_file and not selected_benchmark:
    pdf_bytes = BytesIO(uploaded_file.getvalue())
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        raw_text = "".join([page.get_text() for page in doc])

    if selected_speaker != "All":
        speaker_name_for_matching = selected_speaker.split(" (")[0]
        pattern = re.compile(
            rf"{re.escape(speaker_name_for_matching)}\s*:\s*(.*?)(?=\n[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s*\n|$)",
            re.DOTALL | re.IGNORECASE
        )
        matches = pattern.findall(raw_text)
        raw_text = "\n".join(matches).strip() if matches else ""
        if not matches:
            st.warning(f"No speech found for {selected_speaker}. Displaying empty result.")

    if not raw_text.strip():
        st.warning("No transcript text available for summarization.")
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
                    clean_line = re.sub(r"^[-\u2022\s]+", "", line)
                    bullet_group += f"<li><span style='color:black;'>{clean_line}</span></li>"

            if bullet_group:
                styled_summary += f"<ul>{bullet_group}</ul>"

            st.markdown("### Summary", unsafe_allow_html=True)
            st.markdown(styled_summary, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Vectorstore creation failed: {e}")

        st.markdown("### Ask a Question")
        st.text_input("", key="chat_input", on_change=lambda: handle_question(vectorstore, llm))

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
                followup_questions = [q.strip("-\u2022 ").strip() for q in followup_response.strip().split("\n") if q.strip()]

                for i, question in enumerate(followup_questions):
                    if st.button(question, key=f"followup_q_{i}", use_container_width=True):
                        st.session_state.pending_question = question

            except Exception as e:
                st.warning(f"Could not generate follow-up questions: {e}")
        else:
            st.info("Transcript not available for generating follow-up questions.")

        # Process pending question if exists
        if "pending_question" in st.session_state:
            question = st.session_state.pending_question
            st.session_state.chat_history.append({"role": "user", "content": question})

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                chain_type="stuff"
            )
            answer = qa_chain.run(question)
            st.session_state.chat_history.append({"role": "ai", "content": answer})

            del st.session_state.pending_question

