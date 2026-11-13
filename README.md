# AI PDF Question Answering

A local-first Retrieval-Augmented Generation (RAG) app for asking questions about PDF files. The browser UI is powered by Streamlit; PDF extraction, embeddings, retrieval, and answer generation run on your computer.

## Features
- Private local inference with [Ollama](https://ollama.com/download) — no hosted LLM API.
- Page-aware source metadata and expandable retrieved passages.
- Multiple PDFs per indexing session.
- Compact defaults for an 8 GB RAM machine: `qwen2.5:0.5b` + `all-MiniLM-L6-v2`.
- Persistent local NumPy vector index; no external vector database required.

## Architecture
```text
PDF → PyMuPDF → page-aware chunks → local sentence embeddings
    → NumPy cosine-similarity search → Ollama → cited answer
```

## Run locally

**1. Install Ollama** from https://ollama.com/download and download the model once:
```bash
ollama pull qwen2.5:0.5b
```

**2. Create an environment and install packages**

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Start the browser interface**
```bash
streamlit run app.py
``
Open the local address Streamlit prints (normally http://localhost:8501). Upload PDFs in the sidebar, click **Index PDFs**, then ask questions.

## Offline use
The first setup needs internet to install dependencies and download the Ollama and embedding model weights. Once cached, document processing and model inference are local. Start Ollama before launching questions. For an offline machine, transfer/install dependencies and model files ahead of time.

## Hardware and limitations
8 GB RAM is a practical target, not a guarantee; speed depends on CPU/GPU and document size. The app limits individual uploads to 30 MB. Scanned/image-only PDFs need OCR and are not supported by the current text extractor. Always verify important claims against the cited original pages; small models can make mistakes.

## Project layout
```text
app.py                 Streamlit browser UI
src/pdf_processor.py   PDF text extraction
src/text_chunker.py    Overlapping page-aware chunks
src/embeddings.py      Local Sentence Transformers embeddings
src/vector_store.py    Persistent NumPy similarity index
src/rag_pipeline.py    Retrieval + Ollama generation
docs/index.html        Public project landing page
tests/                 Unit tests
```

## Configuration
Optional environment variables: `OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `qwen2.5:0.5b`), `EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`), and `RAG_DATA_DIR` (default `./data`).

Run tests with `pytest -q`.

## Project webpage
The static landing page is in `docs/`. Enable GitHub Pages in **Settings → Pages**, choose **GitHub Actions** as the source, and the included workflow will publish it on pushes to `main`.

## License
MIT (see [LICENSE](LICENSE)).
