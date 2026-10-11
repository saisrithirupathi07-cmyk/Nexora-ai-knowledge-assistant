import re

import numpy as np
import pymupdf
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Company Knowledge Assistant", page_icon="📚")

st.title("📚 AI Company Knowledge Assistant")
st.write(
    "Upload company documents and ask questions "
    "to find information from your knowledge base."
)


@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


model = load_model()

# Keep the knowledge base between button clicks
for key, default in [("index", None), ("chunks", []), ("sources", [])]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------
# Reading documents
# ---------------------------------------------------------------
def extract_text(file):
    """Return a list of (text, page_number) for a PDF or TXT file."""
    name = file.name.lower()
    pages = []

    if name.endswith(".pdf"):
        document = pymupdf.open(stream=file.read(), filetype="pdf")
        for page_number, page in enumerate(document, start=1):
            text = page.get_text()
            if text.strip():
                pages.append((text, page_number))

    elif name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
        if text.strip():
            pages.append((text, 1))

    return pages


def split_into_chunks(pages, max_chars=400):
    """Cut each page into small passages (about 400 characters)."""
    chunks = []
    for text, page_number in pages:
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > max_chars:
                chunks.append((current.strip(), page_number))
                current = ""
            current += sentence + " "

        if current.strip():
            chunks.append((current.strip(), page_number))

    # drop tiny fragments such as page footers
    return [(t, p) for t, p in chunks if len(t) >= 30]


def build_knowledge_base(files):
    """Read all files, split into passages, and create embeddings."""
    chunks, sources = [], []

    for file in files:
        for text, page in split_into_chunks(extract_text(file)):
            chunks.append(text)
            sources.append({"file": file.name, "page": page})

    if not chunks:
        return None, [], []

    embeddings = model.encode(
        chunks, normalize_embeddings=True, show_progress_bar=False
    )
    return np.array(embeddings, dtype="float32"), chunks, sources


# ---------------------------------------------------------------
# Answering questions
# ---------------------------------------------------------------
def answer_question(question, index, chunks, sources, top_k=3, min_score=0.25):
    """Return a short answer (best passage) and the matching sources."""
    query = model.encode([question], normalize_embeddings=True)[0]
    scores = index @ query  # cosine similarity (vectors are normalized)
    top = np.argsort(scores)[::-1][:top_k]

    best = float(scores[top[0]])
    if best < min_score:
        return "I couldn't find this in the uploaded documents.", []

    answer_parts = [chunks[top[0]]]
    references = []

    for rank, i in enumerate(top):
        score = float(scores[i])
        if score < min_score:
            break
        # add the 2nd passage only if it is almost as relevant as the best
        if rank == 1 and best - score <= 0.05 and chunks[i] not in answer_parts:
            answer_parts.append(chunks[i])
        references.append({"source": sources[i], "score": score})

    return "\n\n".join(answer_parts), references


# ---------------------------------------------------------------
# Step 1: upload documents
# ---------------------------------------------------------------
st.markdown("### 📄 Upload documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if st.button("Process Documents"):
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        with st.spinner("Reading and indexing documents..."):
            index, chunks, sources = build_knowledge_base(uploaded_files)

        if index is None:
            st.error(
                "No readable text found. Scanned PDFs (images) need OCR."
            )
        else:
            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
            st.session_state["sources"] = sources
            st.success(
                f"Done! Indexed {len(chunks)} passages "
                f"from {len(uploaded_files)} document(s)."
            )

# ---------------------------------------------------------------
# Step 2: ask questions
# ---------------------------------------------------------------
st.markdown("### 💬 Ask your company knowledge base")

question = st.text_input("Enter your question")

if st.button("Ask Question"):
    if st.session_state["index"] is None:
        st.warning("Please upload and process documents first.")
    elif not question.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Searching documents..."):
            answer, references = answer_question(
                question,
                st.session_state["index"],
                st.session_state["chunks"],
                st.session_state["sources"],
            )

        st.markdown("### Answer")
        st.write(answer)

        if references:
            st.markdown("### 📎 Sources")
            for reference in references:
                source = reference["source"]
                st.write(
                    f"*Document:* {source['file']}  \n"
                    f"*Page:* {source['page']}  \n"
                    f"*Match:* {reference['score']:.0%}"
                )