import streamlit as st
import fitz  # PyMuPDF
import pytesseract
import shutil
from PIL import Image
import io
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import os
import time

# 🛠️ Auto-detect Tesseract Path
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    st.error("❌ Tesseract OCR is not installed. Please install it first from: https://github.com/UB-Mannheim/tesseract/wiki")
    st.stop()

# 🔐 API Key Handling
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ Please provide your OpenAI API key in Streamlit secrets or as an environment variable.")
    st.stop()

# 📘 Smart PDF Text Extraction (Auto OCR)
def extract_text_from_pdf(uploaded_file):
    try:
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        extracted_text = ""
        total_pages = len(pdf)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for page_num, page in enumerate(pdf):
            # Show progress bar
            progress_bar.progress((page_num + 1) / total_pages)
            status_text.text(f"🔍 Processing page {page_num + 1}/{total_pages}...")

            # Try extracting selectable text
            page_text = page.get_text("text")

            if page_text.strip():
                extracted_text += page_text
            else:
                # If no text, use OCR
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img, lang="eng+urd")
                extracted_text += ocr_text

            time.sleep(0.1)  # Smooth progress bar

        progress_bar.empty()
        status_text.text("✅ PDF processing complete!")

        return extracted_text.strip()
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return ""

# 🧠 Function to create knowledge base
def create_knowledge_base(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    if not chunks:
        st.error("⚠️ No readable text found in the document. Please upload a proper PDF.")
        st.stop()

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

# 🎯 Streamlit UI
st.set_page_config(page_title="AI Lawyer", page_icon="⚖️")
st.title("⚖️ AI Lawyer - Urdu & English Legal Assistant")
st.markdown("Upload a legal case file (PDF), then ask a question about it.")

uploaded_file = st.file_uploader("📂 Upload PDF File", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    if not text:
        st.error("⚠️ Could not extract any text from this PDF. Try uploading a clearer document.")
        st.stop()

    st.success("✅ Document uploaded and processed successfully!")
    vectorstore = create_knowledge_base(text)

    query = st.text_input("❓ Ask a question about the case (English or Urdu):")

    if query:
        try:
            docs = vectorstore.similarity_search(query)
            if not docs:
                st.warning("⚠️ No relevant information found for your question.")
            else:
                llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=query)

                st.markdown("### 🧾 Answer:")
                st.write(response)
        except Exception as e:
            st.error(f"❌ Error while processing your query: {e}")
