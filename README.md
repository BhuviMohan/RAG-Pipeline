# RAG Pipeline – AI Document Q&A

This project implements a Retrieval-Augmented Generation (RAG) pipeline using **Streamlit**, **FAISS**, and the **Google Gemini API**.  
It allows users to upload documents, extract and chunk text, create embeddings, and perform question answering over the document content.

---

## Features
- Upload documents in **PDF, DOCX, or TXT** formats.
- Automatic text extraction and sentence-based chunking.
- Embedding generation using **Sentence Transformers**.
- Vector similarity search using **FAISS**.
- Question answering powered by **Google Gemini**.
- Interactive user interface built with **Streamlit**.

---

## Project Structure
RAG-Pipeline/
│── app.py # Main Streamlit application
│── requirements.txt # Dependencies
│── .env # Environment variables (API keys) – not uploaded
│── .gitignore # Ignore sensitive files
│── README.md # Project documentation

---

## Installation


1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/RAG-Pipeline.git
   cd RAG-Pipeline

2. Create and activate a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows

3. Install dependencies:
pip install -r requirements.txt

4. Create a .env file in the project root and add your Google Gemini API key:
GEMINI_API_KEY=your_api_key_here

Usage

Run the Streamlit application:

streamlit run app.py


Upload a document (PDF, DOCX, or TXT).

The system extracts and chunks the text.

Ask questions related to the document content.

Get AI-powered answers with supporting context.

Requirements

Python 3.8 or higher

Streamlit

PyPDF2

python-docx

nltk

sentence-transformers

faiss

google-generativeai

python-dotenv

(These are included in requirements.txt)

License

This project is licensed under the MIT License. You are free to use, modify, and distribute it with proper attribution.

Author

Developed by Bhuvaneshwari Mohan


