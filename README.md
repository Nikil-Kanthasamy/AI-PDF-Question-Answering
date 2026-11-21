# AI PDF Question Answering

A local-first Retrieval-Augmented Generation (RAG) app for asking questions about PDF files. The browser UI is powered by Streamlit; PDF extraction, OCR, embeddings, retrieval, and answer generation run on your computer.

## Features
- Private local inference with [Ollama](https://ollama.com/download) — no hosted LLM API.
- Page-aware source metadata and expandable retrieved passages.
- Multiple PDFs per indexing session.
- OCR support for scanned and image-only PDF pages using Tesseract.
- Compact defaults for an 8 GB RAM machine: `qwen2.5:0.5b` + `all-MiniLM-L6-v2`.
- Persistent local NumPy vector index; no external vector database required.

## Architecture
```text
PDF → PyMuPDF text extraction / Tesseract OCR → page-aware chunks
    → local sentence embeddings → NumPy cosine-similarity search
    → Ollama → cited answer
```

## Run locally

**1. Install Ollama** from https://ollama.com/download and download the model once:
```bash
ollama pull qwen2.5:0.5b
```

**2. Install Tesseract OCR**

OCR requires the Tesseract application in addition to the Python packages.

- Windows: install Tesseract from the UB Mannheim Windows builds and note the executable path. The common default is `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- macOS: install with `brew install tesseract`.
- Ubuntu/Debian: install with `sudo apt install tesseract-ocr`.

Make sure the English language data (`eng`) is installed. If Tesseract is not on PATH or is installed in a nonstandard location, set `TESSERACT_CMD` to the full executable path before starting the app. For example, in Windows PowerShell:
```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**3. Create an environment and install packages**

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**4. Start the browser interface**
```bash
python -m streamlit run app.py
```
Open the local address Streamlit prints (normally http://localhost:8501). Upload PDFs in the sidebar, click **Index PDFs**, then ask questions.

## Offline use

The initial setup requires internet access to install Python packages and download the Ollama and embedding model weights. Install Tesseract and its English language data before going offline as well. Then run the app once while online and index a sample PDF so the configured Sentence Transformers model is cached locally. Confirm the Ollama model is present with `ollama list`.

After setup and caching, PDF text extraction, OCR, embeddings, retrieval, and Ollama inference can run locally without internet. Keep the Ollama service running while using the app. If you change the configured embedding model or Ollama model, download/cache that model before disconnecting. Package installation and model downloads cannot be performed offline unless the required files are already available locally.

## Hardware and limitations

8 GB RAM is a practical target, not a guarantee; speed depends on CPU/GPU and document size. The app limits individual uploads to 30 MB. OCR uses English (`eng`) by default; install the relevant Tesseract language data and update the OCR language setting in code if you need another language. OCR accuracy depends on scan quality, resolution, rotation, and text clarity. Always verify important claims against the cited original pages; small models can make mistakes.

## Project layout
```text
app.py                 Streamlit browser UI
src/pdf_processor.py   PDF text extraction and OCR
src/text_chunker.py    Overlapping page-aware chunks
src/embeddings.py      Local Sentence Transformers embeddings
src/vector_store.py    Persistent NumPy similarity index
src/rag_pipeline.py    Retrieval + Ollama generation
docs/index.html        Public project landing page
tests/                 Unit tests
```

## Configuration

Optional environment variables: `OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `qwen2.5:0.5b`), `EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`), `RAG_DATA_DIR` (default `./data`), and `TESSERACT_CMD` (full path to the Tesseract executable when it is not discoverable automatically).

Run the automated unit tests with:
```bash
python -m pytest -q
```
The suite covers page-aware chunking, PDF extraction and OCR fallback, vector-index persistence and validation, retrieval, and mocked Ollama responses. The OCR unit tests mock the Tesseract call so the test suite does not require the Tesseract application to be installed on CI machines.

## Project webpage

The static landing page is in `docs/`. Enable GitHub Pages in **Settings → Pages**, choose **GitHub Actions** as the source, and the included workflow will publish it on pushes to `main`.

## License

MIT (see [LICENSE](LICENSE)).
