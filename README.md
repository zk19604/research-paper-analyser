# Research Paper Analyser

Upload a research paper (PDF) in the browser → it's sent to a FastAPI backend →
the paper is ingested (extract → chunk → embed → store in ChromaDB) → ask
questions and get answers grounded in the paper (RAG + reranking via Groq).

```
frontend/        # static UI (HTML/CSS/JS) — upload + chat
rag/
  main.py        # FastAPI server (/upload, /ask, /status)
  rag_pipeline.py# extract, chunk, embed, retrieve, generate
  requirements.txt
```

## Prerequisites

- **Ollama** running with the embedding model:
  `ollama serve` and `ollama pull nomic-embed-text`
- **Groq API key** in `rag/.env`: `GROQ_API_KEY=...`
- `tesseract` installed (only used to OCR scanned/image pages)

## Run

**1. Backend** (from the `rag/` folder):

```bash
cd rag
pip install -r requirements.txt
uvicorn main:app --reload          # serves http://localhost:8000
```

**2. Frontend** (from the `frontend/` folder, any static server):

```bash
cd frontend
python3 -m http.server 5500        # open http://localhost:5500
```

Open the frontend, drop in a PDF, click **Upload & Analyse**, then ask away.
Uploading a new PDF replaces the previous one.
