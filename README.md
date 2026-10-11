# Nexora AI Knowledge Assistant

An AI-powered assistant that lets companies upload internal documents and lets employees ask questions across the whole knowledge base. Answers come from the company's own documents, with the source document and page number shown.

## Problem
Company knowledge is scattered across many documents such as HR policies, onboarding guides and manuals. Employees waste time searching for answers, and keyword search misses results when the wording differs.

## Features
- Upload PDF and TXT documents
- Semantic search across all uploaded documents
- Short, relevant answers instead of whole pages
- Source document and page number shown with every answer
- Says "not found" when the answer is not in the documents

## Tech Stack
- Streamlit (web interface)
- PyMuPDF (reads PDF text and page numbers)
- Sentence Transformers, all-MiniLM-L6-v2 (text embeddings)
- NumPy (similarity search)
- Python

## How to Run
1. Install the packages: pip install -r requirements.txt
2. Start the app: streamlit run app.py
3. Upload your documents and click *Process Documents*
4. Type a question and click *Ask Question*
