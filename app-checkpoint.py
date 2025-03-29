{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'fitz'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-deb716cf2109>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mstreamlit\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mst\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[1;32mimport\u001b[0m \u001b[0mfitz\u001b[0m  \u001b[1;31m# PyMuPDF\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mlangchain\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtext_splitter\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mRecursiveCharacterTextSplitter\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mlangchain\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0membeddings\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mOpenAIEmbeddings\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mlangchain\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mvectorstores\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mFAISS\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'fitz'"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import fitz  # PyMuPDF\n",
    "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
    "from langchain.embeddings import OpenAIEmbeddings\n",
    "from langchain.vectorstores import FAISS\n",
    "from langchain.chat_models import ChatOpenAI\n",
    "from langchain.chains import RetrievalQA\n",
    "from langchain.llms import OpenAI\n",
    "\n",
    "import os\n",
    "\n",
    "# Set your OpenAI API key here or use environment variable\n",
    "os.environ[\"OPENAI_API_KEY\"] = \"your-api-key-here\"\n",
    "\n",
    "st.set_page_config(page_title=\"Earnings Call Analyzer\", layout=\"wide\")\n",
    "\n",
    "st.title(\"📄 Earnings Call PDF Analyzer with LLM + RAG\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Upload an earnings call PDF\", type=\"pdf\")\n",
    "\n",
    "if uploaded_file:\n",
    "    # Step 1: Extract text from PDF\n",
    "    with fitz.open(stream=uploaded_file.read(), filetype=\"pdf\") as doc:\n",
    "        text = \"\"\n",
    "        for page in doc:\n",
    "            text += page.get_text()\n",
    "\n",
    "    st.subheader(\"📄 Extracted Transcript Preview\")\n",
    "    with st.expander(\"Show Raw Text\"):\n",
    "        st.text(text[:3000] + \"...\")\n",
    "\n",
    "    # Step 2: Chunk the text\n",
    "    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)\n",
    "    chunks = splitter.create_documents([text])\n",
    "\n",
    "    # Step 3: Embed and create FAISS index\n",
    "    with st.spinner(\"Embedding and indexing transcript...\"):\n",
    "        embeddings = OpenAIEmbeddings()\n",
    "        vectorstore = FAISS.from_documents(chunks, embeddings)\n",
    "\n",
    "    # Step 4: Summarization (use OpenAI)\n",
    "    st.subheader(\"📌 Summary (LLM Generated)\")\n",
    "    llm = ChatOpenAI(temperature=0)\n",
    "    summary_prompt = \"Summarize the key points of the earnings call transcript, including risks, opportunities, and outlook.\"\n",
    "\n",
    "    summary = llm.predict(summary_prompt + \"\\n\\n\" + text[:3000])  # Only partial input to avoid token limit\n",
    "    st.markdown(summary)\n",
    "\n",
    "    # Step 5: Ask a Question\n",
    "    st.subheader(\"💬 Ask a Question about the Transcript\")\n",
    "    query = st.text_input(\"Enter your question (e.g., What did the CEO say about inflation?)\")\n",
    "\n",
    "    if query:\n",
    "        qa_chain = RetrievalQA.from_chain_type(\n",
    "            llm=llm,\n",
    "            retriever=vectorstore.as_retriever(),\n",
    "            chain_type=\"stuff\"\n",
    "        )\n",
    "        with st.spinner(\"Searching...\"):\n",
    "            answer = qa_chain.run(query)\n",
    "        st.success(answer)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
