"""FastAPI server for the research paper analyser.

Endpoints:
  POST /upload  — receive a PDF, (re)ingest it into the vector DB
  POST /ask     — answer a question about the most recently uploaded PDF
  GET  /status  — how many chunks are currently indexed

Run it from the `rag/` folder:
    uvicorn main:app --reload
"""

import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_pipeline

# Anchor uploads next to this file so the path works regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Research Paper Analyser")

# Allow the frontend (served from a different origin/port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.get("/status")
def status():
    return {"chunks": rag_pipeline.collection.count()}


@app.get("/sources")
def sources():
    """Metadata about the currently indexed paper, for the Sources sidebar."""
    return {"source": rag_pipeline.get_meta()}


@app.get("/graph")
def graph(refresh: bool = False):
    """Knowledge graph (Groq-extracted) for the currently indexed paper."""
    if rag_pipeline.collection.count() == 0:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet.")
    try:
        return rag_pipeline.build_knowledge_graph(force=refresh)
    except Exception as exc:  # surface a clean message to the frontend
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    pdf_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(pdf_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # force=True wipes any previously indexed paper so we answer about THIS one.
    rag_pipeline.ingest_pdf(pdf_path, force=True)

    return {
        "filename": file.filename,
        "chunks": rag_pipeline.collection.count(),
        "message": "PDF ingested. You can now ask questions.",
    }


@app.post("/ask")
def ask(payload: Question):
    if rag_pipeline.collection.count() == 0:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet.")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer, chunks = rag_pipeline.ask(question)
    return {"answer": answer, "chunks": chunks}
