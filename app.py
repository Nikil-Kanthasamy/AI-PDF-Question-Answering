import os
import tempfile
from pathlib import Path
import streamlit as st

from src.pdf_processor import extract_pdf_pages
from src.text_chunker import chunk_pages
from src.vector_store import LocalVectorStore
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Local PDF AI", page_icon="📚", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1120px;padding-top:2rem}
.hero{padding:1.5rem 1.7rem;border:1px solid rgba(120,130,150,.22);border-radius:18px;background:linear-gradient(120deg,rgba(99,102,241,.12),rgba(16,185,129,.08))}
.hero h1{margin:0 0 .35rem;letter-spacing:-.04em}
.muted{opacity:.72;font-size:.92rem}
[data-testid="stChatMessage"]{border-radius:14px}
</style>
<div class="hero"><h1>📚 Ask your PDFs</h1>
<div class="muted">Private local document intelligence · RAG · Ollama</div></div>
""", unsafe_allow_html=True)

store = LocalVectorStore(Path(os.getenv("RAG_DATA_DIR", "./data")) / "index")
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Workspace")
    st.caption("Documents and AI inference stay on this computer.")
    model_name = st.text_input("Ollama model", os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"))
    chunk_size = st.slider("Chunk size", 400, 1400, 800, 100)
    overlap = st.slider("Chunk overlap", 0, 300, 120, 20)
    top_k = st.slider("Retrieved passages", 1, 8, 4)
    st.metric("Indexed passages", len(store.records))
    if st.button("Clear local index", use_container_width=True):
        store.clear()
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.subheader("1 · Add PDFs")
    uploads = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Index PDFs", type="primary", use_container_width=True, disabled=not uploads):
        pipeline = RAGPipeline(store, model_name=model_name)
        progress = st.progress(0, text="Preparing documents…")
        total = 0
        try:
            for i, uploaded in enumerate(uploads):
                if uploaded.size > 30 * 1024 * 1024:
                    st.warning(f"Skipped {uploaded.name}: larger than 30 MB.")
                    continue
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded.getvalue())
                        temp_path = tmp.name
                    pages = extract_pdf_pages(temp_path)
                    for page in pages:
                        page["source"] = uploaded.name
                    chunks = chunk_pages(pages, chunk_size, overlap)
                    total += pipeline.index(chunks)
                finally:
                    if temp_path:
                        Path(temp_path).unlink(missing_ok=True)
                progress.progress((i + 1) / len(uploads), text=f"Processed {uploaded.name}")
            st.success(f"Indexed {total} passages.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    st.caption("Scanned/image-only PDFs need OCR; this version extracts selectable text.")

st.markdown("### 2 · Ask a question")
if not store.records:
    st.info("Upload a PDF in the sidebar, then select **Index PDFs**.")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved source passages"):
                for source in message["sources"]:
                    st.markdown(f"**{source['source']} · page {source['page']}** · similarity {source['score']:.3f}")
                    st.write(source["text"])

question = st.chat_input("Ask about your indexed PDFs…", disabled=not store.records)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving passages and generating locally…"):
            try:
                answer, sources = RAGPipeline(store, model_name=model_name).answer(question, top_k)
                st.markdown(answer)
                with st.expander("Retrieved source passages"):
                    for source in sources:
                        st.markdown(f"**{source['source']} · page {source['page']}** · similarity {source['score']:.3f}")
                        st.write(source["text"])
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as exc:
                st.error(str(exc))
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {exc}"})
st.divider()
st.markdown('<div class="muted">Local-first PDF RAG · Verify important answers against original pages.</div>', unsafe_allow_html=True)
