# Research Paper Analyser

Upload a research paper (PDF) in the browser → it's sent to a FastAPI backend →
the paper is ingested (extract → OCR fallback → chunk → embed → store in
ChromaDB) → ask questions and get answers grounded in the paper, plus an
auto-generated knowledge graph of its ideas. Retrieval is two-stage: vector
search for recall, then Groq (Llama 3.3 70B) reranking and answer generation.

```
frontend/            # React + Vite UI — upload, chat, knowledge graph
rag/
  main.py            # FastAPI server
  rag_pipeline.py    # extract, chunk, embed, retrieve, rerank, generate, graph
  requirements.txt
```

## Prerequisites

- **Ollama** running with the embedding model:
  `ollama serve` and `ollama pull nomic-embed-text`
- **Groq API key** in `rag/.env`: `GROQ_API_KEY=...` (used for reranking,
  answers, and graph extraction)
- **tesseract** installed (OCR fallback for scanned/image PDF pages)
- **Node 18+** and **Python 3.10+**

## Run

**1. Backend** (from `rag/`):

```bash
cd rag
pip install -r requirements.txt
uvicorn main:app --reload          # http://localhost:8000
```

**2. Frontend** (from `frontend/`):

```bash
cd frontend
npm install
npm run dev                        # http://localhost:5173
```

The frontend talks to the backend at `http://localhost:8000` (hardcoded in
`src/App.jsx` / `src/Graph.jsx` — change there if you move the port).

Open the app, drop in a PDF, click **Upload & Analyse**, then ask away or view
the knowledge graph. Uploading a new PDF replaces the previous one.

## API

| Method | Path       | Purpose                                          |
|--------|------------|--------------------------------------------------|
| POST   | `/upload`  | Ingest a PDF (extract → chunk → embed → store)   |
| POST   | `/ask`     | Ask a question, get a grounded answer            |
| GET    | `/graph`   | Knowledge graph for the current paper (`?refresh=true` to rebuild) |
| GET    | `/sources` | Retrieved source chunks                          |
| GET    | `/status`  | Which paper is currently indexed                 |
