"""FastAPI server for the research paper analyser — v2.0.0

Endpoints (all require an X-Session-Id header so users don't share state):

  POST   /upload                    — ingest a PDF (accumulates, does not replace)
  GET    /papers                    — list all papers in this session
  DELETE /papers/{paper_id}         — remove a paper and its chunks
  GET    /meta?paper_id=            — paper metadata (title, abstract, …)
  POST   /ask                       — grounded answer with inline citations + verification
  GET    /graph?paper_id=&refresh=  — knowledge graph for one paper
  GET    /graph/pending?paper_id=   — low-confidence KG nodes awaiting review
  POST   /graph/approve             — approve a pending node into the main graph
  POST   /graph/reject              — discard a pending node
  GET    /sources                   — (compat) metadata for the session
  GET    /status                    — chunk count for the session

Also serves the built frontend from ./static in production.

Run from the `rag/` folder:
    uvicorn main:app --reload
"""

import os
import re
import shutil

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Research Paper Analyser", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── SESSION GUARD ──────────────────────────────────────────────────────────────
def get_sid(x_session_id: str = Header(...)) -> str:
    if not re.fullmatch(r"[A-Za-z0-9-]{8,64}", x_session_id):
        raise HTTPException(status_code=400, detail="Invalid session id.")
    return x_session_id


# ── REQUEST MODELS ─────────────────────────────────────────────────────────────
class Question(BaseModel):
    question: str
    paper_id: str | None = None   # optional: restrict answer to one paper


class ReviewAction(BaseModel):
    paper_id: str
    node_id:  str


# ── STATUS / COMPAT ────────────────────────────────────────────────────────────
@app.get("/status")
def status(sid: str = Depends(get_sid)):
    return {"chunks": rag_pipeline.collection(sid).count()}


@app.get("/sources")
def sources(sid: str = Depends(get_sid)):
    """Backward-compat endpoint: summary metadata for the session."""
    return {"source": rag_pipeline.get_meta(sid)}


# ── PAPERS ────────────────────────────────────────────────────────────────────
@app.get("/papers")
def list_papers(sid: str = Depends(get_sid)):
    """All papers currently ingested in this session."""
    return {"papers": rag_pipeline.list_papers(sid)}


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: str, sid: str = Depends(get_sid)):
    """Remove a paper and all its chunks from the session."""
    try:
        rag_pipeline.remove_paper(sid, paper_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return {"ok": True}


# ── METADATA ──────────────────────────────────────────────────────────────────
@app.get("/meta")
def meta(paper_id: str | None = None, sid: str = Depends(get_sid)):
    """Full paper metadata (title, abstract, pages, …) for one or all papers."""
    data = rag_pipeline.get_meta(sid, paper_id=paper_id)
    if data is None:
        raise HTTPException(status_code=404, detail="No papers ingested yet.")
    return data


# ── UPLOAD ────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(file: UploadFile = File(...), sid: str = Depends(get_sid)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    pdf_path = os.path.join(rag_pipeline.session_dir(sid), f"upload_{file.filename}")
    with open(pdf_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    paper_meta = rag_pipeline.ingest_pdf(sid, pdf_path, filename=file.filename)

    return {
        "paper_id":  paper_meta["paper_id"],
        "filename":  paper_meta["filename"],
        "title":     paper_meta.get("title", ""),
        "abstract":  paper_meta.get("abstract", ""),
        "pages":     paper_meta.get("pages", 0),
        "chunks":    paper_meta["chunks"],
        "from_cache": False,
        "message":   "PDF ingested. You can now ask questions.",
    }


# ── ASK ───────────────────────────────────────────────────────────────────────
@app.post("/ask")
def ask(payload: Question, sid: str = Depends(get_sid)):
    if rag_pipeline.collection(sid).count() == 0:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet.")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = rag_pipeline.ask(sid, question, paper_id=payload.paper_id)
    return result


# ── KNOWLEDGE GRAPH ────────────────────────────────────────────────────────────
@app.get("/graph")
def graph(
    paper_id: str | None = None,
    refresh:  bool       = False,
    merged:   bool       = False,
    sid:      str        = Depends(get_sid),
):
    """Knowledge graph for one paper, or a merged cross-paper graph (merged=true)."""
    if rag_pipeline.collection(sid).count() == 0:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet.")
    try:
        if merged:
            return rag_pipeline.build_merged_graph(sid)
        return rag_pipeline.build_knowledge_graph(sid, paper_id=paper_id, force=refresh)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/pending")
def graph_pending(paper_id: str, sid: str = Depends(get_sid)):
    """Low-confidence KG nodes awaiting human review for a given paper."""
    return {"pending": rag_pipeline.get_pending_review(sid, paper_id)}


@app.post("/graph/approve")
def graph_approve(payload: ReviewAction, sid: str = Depends(get_sid)):
    """Approve a pending node — moves it into the main graph."""
    rag_pipeline.approve_triple(sid, payload.paper_id, payload.node_id)
    return {"ok": True}


@app.post("/graph/reject")
def graph_reject(payload: ReviewAction, sid: str = Depends(get_sid)):
    """Reject (discard) a pending node."""
    rag_pipeline.reject_triple(sid, payload.paper_id, payload.node_id)
    return {"ok": True}


# ── STATIC FRONTEND ────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
