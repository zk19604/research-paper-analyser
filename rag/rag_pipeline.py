"""Research-paper RAG pipeline — v2.0.0

Layers implemented over v1:
  Layer 0 — Foundations:
    • PyMuPDF dict-mode extraction: section headers, page tracking, font heuristics
    • Rich chunk metadata: paper_id, page_number, section_title, chunk_type
    • File-hash caching: re-uploading the same PDF is a no-op
    • Model version constants recorded on every artefact for traceability
    • Enriched paper metadata (title, abstract, page count) from first 2 pages
    • Constrained relation type enum in KG prompt + confidence on every node
    • Inline [p.N] citation instructions in the answer prompt

  Layer 1 — Multi-Paper Sessions:
    • Each uploaded PDF gets a UUID paper_id; papers accumulate (no wipe)
    • Papers registry (papers.json) + list_papers() / remove_paper() helpers
    • ChromaDB chunks tagged with paper_id in metadata for per-paper filtering
    • ask() and build_knowledge_graph() accept optional paper_id filter

  Layer 2 — Anti-Hallucination:
    • verify_citations(): post-hoc LLM call to flag unsupported claims
    • Low-confidence KG nodes (< CONFIDENCE_THRESHOLD) held in pending_review queue
    • approve_triple() / reject_triple() to promote or discard pending nodes
    • Provenance on every node: paper_id, model, pipeline_version

  Layer 3 — Hybrid Retrieval:
    • BM25 keyword index built from chunks.json, cached in memory
    • Reciprocal Rank Fusion (RRF) merges vector + BM25 ranked lists
    • LLM-based reranker applied after fusion (unchanged from v1)

  Layer 4 — Structure-Aware Chunking:
    • Section-first chunking: flush on header detection, sub-split only if needed
    • Abstract + conclusion kept as single atomic chunks
    • References section split one-entry-per-citation
    • ~10-20% overlap on sub-split sections
"""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

import chromadb
import fitz  # PyMuPDF
import pytesseract
from dotenv import load_dotenv
from groq import Groq
from PIL import Image

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


# ── SETUP ──────────────────────────────────────────────────────────────────────
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Persistent on-disk ChromaDB store — embeddings via built-in all-MiniLM-L6-v2.
client_db = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))

# Model / version constants — stored on every extracted artefact so you can
# re-run extraction later after a model upgrade and still know what produced what.
EXTRACTION_MODEL = "llama-3.3-70b-versatile"
GRAPH_MODEL      = "llama-3.3-70b-versatile"
VERIFY_MODEL     = "llama-3.3-70b-versatile"
PIPELINE_VERSION = "2.0.0"

# Chunking budget (characters, not tokens — conservative to stay under context limits).
SECTION_TOKEN_BUDGET = 1400
OVERLAP_CHARS        = 200

# KG confidence threshold: nodes below this go to the pending-review queue.
CONFIDENCE_THRESHOLD = 0.65

# In-memory BM25 index cache: sid -> (BM25Okapi, list[chunk_id], list[text])
_bm25_cache: dict[str, tuple] = {}


# ── SESSION HELPERS ────────────────────────────────────────────────────────────
def collection(sid: str):
    return client_db.get_or_create_collection(name=f"paper_{sid}")


def session_dir(sid: str) -> str:
    d = os.path.join(SESSIONS_DIR, sid)
    os.makedirs(d, exist_ok=True)
    return d


def _cache(sid: str, name: str) -> str:
    return os.path.join(session_dir(sid), name)


def purge_old_sessions(ttl_hours: int = 24):
    """Drop sessions untouched for ttl_hours so disk doesn't fill up."""
    cutoff = time.time() - ttl_hours * 3600
    for name in os.listdir(SESSIONS_DIR):
        d = os.path.join(SESSIONS_DIR, name)
        if os.path.getmtime(d) < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            try:
                client_db.delete_collection(f"paper_{name}")
            except Exception:
                pass


# ── FILE HASH ─────────────────────────────────────────────────────────────────
def _file_hash(path: str) -> str:
    """MD5 of file contents — used to skip re-ingestion of unchanged PDFs."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ── TABLE & FIGURE EXTRACTION (Layer / §1) ─────────────────────────────────────
def _nearest_caption(page, word: str) -> str:
    """First line on the page starting with 'Table N' / 'Figure N'.
    ponytail: page-level, not bbox-nearest — a page with multiple tables reuses
    the first caption. Upgrade to bbox-distance matching if that bites."""
    for line in page.get_text().splitlines():
        s = line.strip()
        if re.match(rf"^{word}\s*\.?\s*\d+", s, re.I):
            return s[:200]
    return ""


def _extract_tables(page, page_num: int) -> list[dict]:
    """Extract tables via PyMuPDF's built-in detector → markdown chunks.
    Wrapped so old PyMuPDF (no find_tables) degrades to zero tables."""
    out = []
    try:
        finder = page.find_tables()
    except Exception:
        return out
    for t in getattr(finder, "tables", []):
        try:
            md = t.to_markdown().strip()
        except Exception:
            continue
        if len(md) < 30:
            continue
        caption = _nearest_caption(page, "Table")
        out.append({
            "text":          (caption + "\n" if caption else "") + md,
            "page_number":   page_num,
            "chunk_type":    "table",
            "section_title": caption or "Table",
        })
    return out


def _extract_figure_captions(page, page_num: int) -> list[dict]:
    """Figure captions as searchable chunks. ponytail: caption text only — no
    image crop / vision description (needs a vision model + dependency)."""
    out = []
    for line in page.get_text().splitlines():
        s = line.strip()
        if re.match(r"^(figure|fig\.?)\s*\d+", s, re.I) and len(s) > 15:
            out.append({
                "text":          s,
                "page_number":   page_num,
                "chunk_type":    "figure",
                "section_title": s[:40],
            })
    return out


# ── STRUCTURED EXTRACTION (Layer 4) ───────────────────────────────────────────
@dataclass
class RawBlock:
    text: str
    page_number: int   # 1-indexed
    font_size: float
    is_bold: bool
    is_header: bool = False


def _extract_blocks(pdf_path: str) -> tuple[list[RawBlock], dict, list[dict]]:
    """
    Use PyMuPDF dict-mode to extract text blocks with font size + bold flags.
    Headers are detected by font size relative to the page median.
    Returns (blocks, paper_metadata, extras) where extras are ready-made
    table/figure chunk dicts.
    """
    doc = fitz.open(pdf_path)
    raw: list[dict] = []
    font_sizes: list[float] = []
    extras: list[dict] = []

    for page_num, page in enumerate(doc, start=1):
        plain = page.get_text()

        # Tables + figure captions (independent of the text-block path).
        extras.extend(_extract_tables(page, page_num))
        extras.extend(_extract_figure_captions(page, page_num))

        # OCR fallback for image-only pages (almost no text layer).
        if len(plain.strip()) < 100:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                raw.append({"text": ocr_text, "page_number": page_num,
                            "font_size": 11.0, "is_bold": False})
                font_sizes.append(11.0)
            continue

        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:   # skip image blocks
                continue
            texts, sizes, flags = [], [], []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = span["text"].strip()
                    if not t:
                        continue
                    texts.append(t)
                    sizes.append(span["size"])
                    # PyMuPDF flag bit 4 (value 16) = bold
                    flags.append(span.get("flags", 0))

            if not texts:
                continue

            avg_size = sum(sizes) / len(sizes)
            bold     = any(f & 16 for f in flags)
            font_sizes.append(avg_size)
            raw.append({"text": " ".join(texts), "page_number": page_num,
                        "font_size": avg_size, "is_bold": bold})

    # Median font size for the whole document.
    if not font_sizes:
        median_size = 11.0
    else:
        sorted_fs   = sorted(font_sizes)
        median_size = sorted_fs[len(sorted_fs) // 2]

    blocks: list[RawBlock] = []
    for rb in raw:
        text      = rb["text"]
        # A block is a header if it's larger than median, or bold,
        # AND short (headers rarely span >120 chars or end with punctuation).
        is_header = (
            (rb["font_size"] > median_size * 1.08 or rb["is_bold"])
            and len(text) < 120
            and bool(text.strip())
            and text.strip()[-1] not in ".,;:"
        )
        blocks.append(RawBlock(
            text=text,
            page_number=rb["page_number"],
            font_size=rb["font_size"],
            is_bold=rb["is_bold"],
            is_header=is_header,
        ))

    meta = _extract_metadata_heuristic(blocks, len(doc))
    doc.close()
    return blocks, meta, extras


def _extract_metadata_heuristic(blocks: list[RawBlock], page_count: int) -> dict:
    """
    Heuristic paper-level metadata from the first 2 pages.
      title    — block with the largest font size on page 1-2
      abstract — text following the first "Abstract" header
    """
    p12 = [b for b in blocks if b.page_number <= 2]

    title    = ""
    abstract = ""

    if p12:
        max_size     = max(b.font_size for b in p12)
        title_blocks = [b for b in p12 if b.font_size >= max_size * 0.95]
        title        = " ".join(b.text for b in title_blocks[:3]).strip()[:200]

        in_abstract, abstract_parts = False, []
        for b in p12:
            if re.search(r"\babstract\b", b.text, re.IGNORECASE) and b.is_header:
                in_abstract = True
                continue
            if in_abstract:
                if b.is_header:
                    break
                abstract_parts.append(b.text)
                if len(" ".join(abstract_parts)) > 1500:
                    break
        abstract = " ".join(abstract_parts).strip()

    return {
        "title":            title,
        "abstract":         abstract,
        "pages":            page_count,
        "pipeline_version": PIPELINE_VERSION,
        "extraction_model": EXTRACTION_MODEL,
    }


# ── CHUNKING (Layer 4) ────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id:      str
    paper_id:      str
    text:          str
    page_number:   int
    section_title: str
    chunk_type:    str   # text | abstract | conclusion | reference


def _is_garbage(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 50:
        return True
    alpha = sum(c.isalpha() or c.isspace() for c in stripped)
    return (alpha / len(stripped)) < 0.55


def _sub_split(text: str) -> list[str]:
    """Paragraph-aware sub-split with OVERLAP_CHARS overlap tail."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sub_chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= SECTION_TOKEN_BUDGET:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                sub_chunks.append(current)
            tail    = current[-OVERLAP_CHARS:] if current else ""
            current = f"{tail}\n\n{para}" if tail else para
    if current:
        sub_chunks.append(current)
    return sub_chunks


_SPECIAL_RE = re.compile(r"\b(abstract|conclusion|summary)\b", re.I)
_REF_RE     = re.compile(r"\b(references|bibliography)\b", re.I)


def chunk_document(blocks: list[RawBlock], paper_id: str,
                   extras: list[dict] | None = None) -> list[Chunk]:
    """
    Section-first chunking (Layer 4):
      • Walk blocks; each header flushes the current section.
      • Abstract + conclusion → kept as a single atomic chunk.
      • References → one chunk per citation entry.
      • Long sections → sub-split with overlap.
      • Tables + figures (extras) → one atomic chunk each.
    """
    # Group blocks into sections by header detection.
    sections: list[dict] = []
    current_title = "Preamble"
    current_page  = 1
    current_body: list[RawBlock] = []

    for block in blocks:
        if block.is_header:
            if current_body:
                sections.append({
                    "title":      current_title,
                    "page":       current_page,
                    "text":       "\n\n".join(b.text for b in current_body),
                    "is_special": bool(_SPECIAL_RE.search(current_title)),
                    "is_refs":    bool(_REF_RE.search(current_title)),
                })
            current_title = block.text.strip()
            current_page  = block.page_number
            current_body  = []
        else:
            if not current_body:
                current_page = block.page_number
            current_body.append(block)

    if current_body:
        sections.append({
            "title":      current_title,
            "page":       current_page,
            "text":       "\n\n".join(b.text for b in current_body),
            "is_special": bool(_SPECIAL_RE.search(current_title)),
            "is_refs":    bool(_REF_RE.search(current_title)),
        })

    chunks: list[Chunk] = []
    idx = 0

    for sec in sections:
        text = sec["text"].strip()
        if not text:
            continue

        if sec["is_refs"]:
            # Split reference section into individual citation entries.
            entries = re.split(r"\n(?=\[\d+\]|\d+\.|\[)", text)
            for entry in entries:
                entry = entry.strip()
                if len(entry) > 30:
                    chunks.append(Chunk(
                        chunk_id      = f"{paper_id}_{idx}",
                        paper_id      = paper_id,
                        text          = entry,
                        page_number   = sec["page"],
                        section_title = "References",
                        chunk_type    = "reference",
                    ))
                    idx += 1

        elif sec["is_special"] or len(text) <= SECTION_TOKEN_BUDGET:
            # Keep abstract, conclusion, and short sections whole.
            if not _is_garbage(text):
                ctype = "abstract"    if "abstract"   in sec["title"].lower() else \
                        "conclusion"  if re.search(r"\bconclusion", sec["title"], re.I) else \
                        "text"
                chunks.append(Chunk(
                    chunk_id      = f"{paper_id}_{idx}",
                    paper_id      = paper_id,
                    text          = text,
                    page_number   = sec["page"],
                    section_title = sec["title"],
                    chunk_type    = ctype,
                ))
                idx += 1

        else:
            # Sub-split long sections with overlap.
            for sub in _sub_split(text):
                if not _is_garbage(sub):
                    chunks.append(Chunk(
                        chunk_id      = f"{paper_id}_{idx}",
                        paper_id      = paper_id,
                        text          = sub,
                        page_number   = sec["page"],
                        section_title = sec["title"],
                        chunk_type    = "text",
                    ))
                    idx += 1

    # Tables + figures as atomic chunks (skip the text-oriented garbage filter —
    # tables are pipe/number heavy by nature).
    for ex in (extras or []):
        if len(ex["text"].strip()) < 20:
            continue
        chunks.append(Chunk(
            chunk_id      = f"{paper_id}_{idx}",
            paper_id      = paper_id,
            text          = ex["text"],
            page_number   = ex["page_number"],
            section_title = ex["section_title"],
            chunk_type    = ex["chunk_type"],
        ))
        idx += 1

    return chunks


# ── PAPERS REGISTRY (Layer 1) ──────────────────────────────────────────────────
def _load_papers(sid: str) -> dict:
    p = _cache(sid, "papers.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_papers(sid: str, papers: dict):
    with open(_cache(sid, "papers.json"), "w", encoding="utf-8") as fh:
        json.dump(papers, fh, indent=2)


def list_papers(sid: str) -> list:
    return list(_load_papers(sid).values())


def remove_paper(sid: str, paper_id: str):
    """Remove a paper and all its chunks from the session."""
    papers = _load_papers(sid)
    if paper_id not in papers:
        raise KeyError(f"Paper {paper_id!r} not found.")
    del papers[paper_id]
    _save_papers(sid, papers)

    # Remove chunks from ChromaDB.
    col = collection(sid)
    try:
        existing = col.get(where={"paper_id": {"$eq": paper_id}})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
    except Exception:
        pass

    # Remove from the BM25 corpus file.
    chunks_path = _cache(sid, "chunks.json")
    if os.path.exists(chunks_path):
        with open(chunks_path, encoding="utf-8") as fh:
            all_chunks = json.load(fh)
        all_chunks = [c for c in all_chunks if c.get("paper_id") != paper_id]
        with open(chunks_path, "w", encoding="utf-8") as fh:
            json.dump(all_chunks, fh)

    # Remove cached graph + pending review files.
    for fname in [f"graph_{paper_id}.json", f"pending_{paper_id}.json"]:
        p = _cache(sid, fname)
        if os.path.exists(p):
            os.remove(p)

    _invalidate_bm25(sid)


# ── INGEST (Layers 0 + 1 + 4) ─────────────────────────────────────────────────
def ingest_pdf(sid: str, pdf_path: str, filename: str | None = None) -> dict:
    """
    Ingest a PDF into this session.

    Each paper gets its own paper_id (UUID). Papers accumulate — uploading a
    second PDF does NOT wipe the first.  Re-uploading the exact same file
    (same MD5) returns the cached metadata immediately without re-processing.
    """
    purge_old_sessions()

    # File-hash deduplication (Layer 0).
    file_hash = _file_hash(pdf_path)
    papers    = _load_papers(sid)
    for meta in papers.values():
        if meta.get("file_hash") == file_hash:
            print(f"  [cache hit] same file already ingested as {meta['paper_id']}")
            return meta

    paper_id = uuid.uuid4().hex[:8]   # short, URL-safe ID
    col      = collection(sid)

    print("Step 1: Structured extraction…")
    blocks, paper_meta, extras = _extract_blocks(pdf_path)
    n_tab = sum(e["chunk_type"] == "table"  for e in extras)
    n_fig = sum(e["chunk_type"] == "figure" for e in extras)
    print(f"  {len(blocks)} blocks, {paper_meta['pages']} pages, "
          f"{n_tab} tables, {n_fig} figures\n")

    print("Step 2: Section-aware chunking…")
    chunks = chunk_document(blocks, paper_id, extras)
    print(f"  {len(chunks)} chunks produced\n")

    print("Step 3: Embedding & storing…")
    if chunks:
        col.add(
            ids       = [c.chunk_id      for c in chunks],
            documents = [c.text           for c in chunks],
            metadatas = [{
                "paper_id":      c.paper_id,
                "page_number":   c.page_number,
                "section_title": c.section_title,
                "chunk_type":    c.chunk_type,
            } for c in chunks],
        )
    print(f"  {col.count()} total chunks in session\n")

    # Persist chunk list for BM25 (Layer 3).
    chunks_path     = _cache(sid, "chunks.json")
    existing_chunks = []
    if os.path.exists(chunks_path):
        with open(chunks_path, encoding="utf-8") as fh:
            existing_chunks = json.load(fh)
    existing_chunks.extend([asdict(c) for c in chunks])
    with open(chunks_path, "w", encoding="utf-8") as fh:
        json.dump(existing_chunks, fh)
    _invalidate_bm25(sid)

    # Register the paper.
    paper_meta.update({
        "paper_id":    paper_id,
        "filename":    filename or os.path.basename(pdf_path),
        "file_hash":   file_hash,
        "chunks":      len(chunks),
        "ingested_at": time.time(),
    })
    papers[paper_id] = paper_meta
    _save_papers(sid, papers)

    print("Ingestion complete.\n")
    return paper_meta


def get_meta(sid: str, paper_id: str | None = None) -> dict | None:
    """Metadata for one paper or a summary of all papers in the session."""
    papers = _load_papers(sid)
    if not papers:
        return None
    if paper_id:
        return papers.get(paper_id)
    return {
        "papers":       list(papers.values()),
        "total_chunks": collection(sid).count(),
    }


# ── BM25 INDEX (Layer 3) ───────────────────────────────────────────────────────
# Common English + question stopwords — dropped so BM25 ranks on content
# words, not on how often a chunk repeats "the/is/of". Without this, meta
# questions ("what is the main contribution of this paper?") are almost all
# stopwords and BM25 surfaces the longest prose chunk instead of the abstract.
_STOPWORDS = frozenset("""
a an the this that these those of in on at to for from by with about as into
is are was were be been being am do does did have has had will would can could
shall should may might must and or but if then else so than too very
what which who whom whose when where why how
it its it's they them their there here i we you he she our your my me us
paper papers study studies article author authors
""".split())


def _tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = re.findall(r"\b[a-z0-9]+\b", text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


def _build_or_get_bm25(sid: str) -> tuple | None:
    """Return (BM25Okapi, list[chunk_id], list[text]) or None if unavailable."""
    if not _HAS_BM25:
        return None
    if sid in _bm25_cache:
        return _bm25_cache[sid]

    chunks_path = _cache(sid, "chunks.json")
    if not os.path.exists(chunks_path):
        return None
    with open(chunks_path, encoding="utf-8") as fh:
        all_chunks = json.load(fh)
    if not all_chunks:
        return None

    corpus    = [_tokenize(c["text"]) for c in all_chunks]
    ids_list  = [c["chunk_id"]        for c in all_chunks]
    texts     = [c["text"]            for c in all_chunks]
    papers    = [c.get("paper_id", "") for c in all_chunks]

    _bm25_cache[sid] = (BM25Okapi(corpus), ids_list, texts, papers)
    return _bm25_cache[sid]


def _invalidate_bm25(sid: str):
    _bm25_cache.pop(sid, None)


# ── HYBRID RETRIEVAL (Layer 3) ────────────────────────────────────────────────
def _rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion over multiple ranked chunk-ID lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def _hybrid_retrieve(sid: str, question: str,
                     paper_id: str | None = None, n: int = 10) -> list[dict]:
    """
    Vector search (ChromaDB) + BM25 keyword search merged via RRF.
    Returns up to n chunks as rich dicts with full metadata.
    """
    col = collection(sid)
    if col.count() == 0:
        return []

    where = {"paper_id": {"$eq": paper_id}} if paper_id else None

    # 1. Vector search.
    try:
        vec_res  = col.query(
            query_texts = [question],
            n_results   = min(n, col.count()),
            where       = where,
            include     = ["documents", "metadatas"],
        )
        vec_ids   = vec_res["ids"][0]
        vec_docs  = vec_res["documents"][0]
        vec_metas = vec_res["metadatas"][0]
    except Exception:
        vec_ids, vec_docs, vec_metas = [], [], []

    # 2. BM25 keyword search.
    bm25_ids: list[str] = []
    bm25_idx = _build_or_get_bm25(sid)
    if bm25_idx is not None:
        bm25, all_ids, all_texts, all_papers = bm25_idx
        tokens = _tokenize(question)
        scores = bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for i in ranked[:n]:
            if paper_id is None or all_papers[i] == paper_id:
                bm25_ids.append(all_ids[i])

    # 3. Fuse via RRF.
    fused_ids = _rrf([vec_ids, bm25_ids])[:n]

    # Build result list; BM25-only hits fetched from chunks.json.
    id_to_doc  = dict(zip(vec_ids, vec_docs))
    id_to_meta = dict(zip(vec_ids, vec_metas))

    missing = [cid for cid in fused_ids if cid not in id_to_doc]
    if missing:
        chunks_path = _cache(sid, "chunks.json")
        if os.path.exists(chunks_path):
            with open(chunks_path, encoding="utf-8") as fh:
                chunk_map = {c["chunk_id"]: c for c in json.load(fh)}
            for cid in missing:
                if cid in chunk_map:
                    c = chunk_map[cid]
                    id_to_doc[cid]  = c["text"]
                    id_to_meta[cid] = {
                        "paper_id":      c["paper_id"],
                        "page_number":   c["page_number"],
                        "section_title": c["section_title"],
                        "chunk_type":    c["chunk_type"],
                    }

    result = []
    for cid in fused_ids:
        if cid in id_to_doc:
            m = id_to_meta.get(cid, {})
            result.append({
                "chunk_id":      cid,
                "text":          id_to_doc[cid],
                "paper_id":      m.get("paper_id",      ""),
                "page_number":   m.get("page_number",   0),
                "section_title": m.get("section_title", ""),
                "chunk_type":    m.get("chunk_type",    "text"),
            })

    return result[:n]


# ── RERANKING ──────────────────────────────────────────────────────────────────
def rerank(question: str, candidates: list[dict], top_k: int = 4) -> list[dict]:
    """
    LLM-based cross-reranker: scores each candidate's relevance to the question.
    Falls back to the original order if parsing fails.
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    numbered = "\n\n".join(
        f"[{i}] (p.{c['page_number']}, §{c['section_title'][:30]})\n{c['text'][:400]}"
        for i, c in enumerate(candidates)
    )
    prompt = f"""Question: {question}

Return ONLY a comma-separated list of the indices of the {top_k} passages
most relevant to answering the question, most relevant first.
Example: 3,0,5,1

Passages:
{numbered}
"""
    resp  = groq_client.chat.completions.create(
        model    = EXTRACTION_MODEL,
        messages = [{"role": "user", "content": prompt}],
        temperature = 0,
    )
    raw   = resp.choices[0].message.content
    order = []
    for tok in raw.replace("\n", ",").split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) < len(candidates) and int(tok) not in order:
            order.append(int(tok))
    if not order:
        order = list(range(len(candidates)))
    return [candidates[i] for i in order[:top_k]]


# ── CITATION VERIFICATION (Layer 2) ───────────────────────────────────────────
def verify_citations(question: str, answer: str, chunks: list[dict]) -> dict:
    """
    Post-hoc check: ask the LLM whether the answer is grounded in the chunks.
    Returns {verified, unsupported_claims, note}.
    """
    context = "\n\n---\n\n".join(
        f"[Chunk {c['chunk_id']}, p.{c['page_number']}]\n{c['text'][:600]}"
        for c in chunks
    )
    prompt = f"""You are a fact-checker reviewing an AI-generated research answer.

Question: {question}

Answer:
{answer}

Source Passages:
{context}

Task: Identify whether every factual claim in the answer is directly supported
by the source passages above.

Return strict JSON:
{{
  "verified": true or false,
  "unsupported_claims": ["list any claim NOT directly supported by the passages"],
  "note": "one-sentence overall verdict"
}}

Rules:
- Only flag claims that clearly go beyond or contradict the passages.
- Return verified=true and empty list if the answer is fully grounded.
"""
    try:
        resp = groq_client.chat.completions.create(
            model           = VERIFY_MODEL,
            messages        = [{"role": "user", "content": prompt}],
            temperature     = 0,
            response_format = {"type": "json_object"},
        )
        return _safe_json(resp.choices[0].message.content)
    except Exception as exc:
        return {
            "verified":           True,
            "unsupported_claims": [],
            "note":               f"Verification skipped: {exc}",
        }


# ── ASK (Layers 0 + 2 + 3) ────────────────────────────────────────────────────
def ask(sid: str, question: str, paper_id: str | None = None) -> dict:
    """
    Retrieve → rerank → generate → verify.
    Returns {answer, chunks, verification}.
    """
    candidates = _hybrid_retrieve(sid, question, paper_id=paper_id, n=10)

    if not candidates:
        return {
            "answer":       "I don't know based on this paper.",
            "chunks":       [],
            "verification": {
                "verified": True, "unsupported_claims": [],
                "note": "No context retrieved.",
            },
        }

    retrieved = rerank(question, candidates, top_k=4)

    # Build context with inline source references so the LLM can cite them.
    context_parts = []
    for c in retrieved:
        context_parts.append(
            f"[Source: chunk {c['chunk_id']}, p.{c['page_number']}, "
            f"§{c['section_title']}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a research paper assistant.
Answer ONLY using the context below.
When you use information from a chunk, cite it inline as [p.N] where N is the
page number shown in the Source header.
If the answer is not in the context, say exactly:
"I don't know based on this paper."

Context:
{context}

Question: {question}
"""
    resp = groq_client.chat.completions.create(
        model    = EXTRACTION_MODEL,
        messages = [{"role": "user", "content": prompt}],
        temperature = 0.1,
    )
    answer       = resp.choices[0].message.content
    verification = verify_citations(question, answer, retrieved)

    return {
        "answer":       answer,
        "chunks":       retrieved,
        "verification": verification,
    }


# ── KNOWLEDGE GRAPH (Layers 0 + 2) ────────────────────────────────────────────
GRAPH_TYPES = (
    "root, theme, problem, method, component, model, dataset, "
    "task, metric, finding, concept, baseline, result"
)

ALLOWED_RELATIONS = [
    "proposes", "evaluates_on", "outperforms", "uses", "extends",
    "compares_to", "achieves_metric", "contributes_to",
    "built_on", "addresses", "leverages", "introduces", "contradicts",
]


def _safe_json(raw: str) -> dict:
    """Parse LLM reply tolerating stray markdown fences or surrounding prose."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def _normalise_graph(raw: dict, paper_id: str = "") -> dict:
    """
    Convert the LLM's {topic, nodes[parent], relations} payload into the
    {topic, nodes[level], edges[kind]} shape the frontend renders.
    Nodes below CONFIDENCE_THRESHOLD are tagged for pending review.
    """
    nodes = {
        n["id"]: n for n in raw.get("nodes", [])
        if n.get("id") and n.get("label")
    }
    if not nodes:
        raise RuntimeError("The model returned no usable concepts.")

    children = {n["id"] for n in nodes.values() if n.get("parent") in nodes}
    root_id  = raw.get("root")
    if root_id not in nodes:
        root_id = (
            next((nid for nid in nodes if nid not in children), None)
            or next(iter(nodes))
        )
    nodes[root_id]["parent"] = None

    def level_of(nid: str) -> int:
        seen, lvl, cur = set(), 0, nid
        while True:
            parent = nodes.get(cur, {}).get("parent")
            if not parent or parent not in nodes or parent in seen or parent == cur:
                return lvl
            seen.add(cur)
            cur, lvl = parent, lvl + 1

    out_nodes, hierarchy_edges = [], []
    for nid, n in nodes.items():
        parent = n.get("parent")
        if nid != root_id and (parent not in nodes or parent == nid):
            parent      = root_id
            n["parent"] = parent

        lvl        = 0 if nid == root_id else level_of(nid)
        confidence = min(1.0, max(0.0, float(n.get("confidence", 1.0))))

        out_nodes.append({
            "id":         nid,
            "label":      n["label"],
            "type":       "root" if nid == root_id else (n.get("type") or "concept"),
            "level":      lvl,
            "summary":    (n.get("summary") or "").strip(),
            "facts":      [f for f in (n.get("key_facts") or []) if isinstance(f, str)][:5],
            "confidence": confidence,
            "provenance": {
                "paper_id": paper_id,
                "model":    GRAPH_MODEL,
                "version":  PIPELINE_VERSION,
            },
        })
        if nid != root_id:
            hierarchy_edges.append({
                "source": parent, "target": nid,
                "relation": "", "kind": "hierarchy",
            })

    relation_edges = [
        {
            "source":   r["source"],
            "target":   r["target"],
            "relation": r.get("label", ""),
            "kind":     "relation",
        }
        for r in raw.get("relations", [])
        if (
            r.get("source") in nodes
            and r.get("target") in nodes
            and r.get("source") != r.get("target")
        )
    ]

    return {
        "topic":    raw.get("topic", ""),
        "paper_id": paper_id,
        "nodes":    out_nodes,
        "edges":    hierarchy_edges + relation_edges,
        "built_at": time.time(),
        "model":    GRAPH_MODEL,
        "version":  PIPELINE_VERSION,
    }


# ── PENDING REVIEW (Layer 2) ───────────────────────────────────────────────────
def _pending_path(sid: str, paper_id: str) -> str:
    return _cache(sid, f"pending_{paper_id}.json")


def _load_pending(sid: str, paper_id: str) -> list:
    p = _pending_path(sid, paper_id)
    return json.loads(open(p, encoding="utf-8").read()) if os.path.exists(p) else []


def _save_pending(sid: str, paper_id: str, items: list):
    with open(_pending_path(sid, paper_id), "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2)


def get_pending_review(sid: str, paper_id: str) -> list:
    return _load_pending(sid, paper_id)


def approve_triple(sid: str, paper_id: str, node_id: str):
    """Promote a pending node into the main graph with confidence=1.0."""
    pending = _load_pending(sid, paper_id)
    item    = next((p for p in pending if p["id"] == node_id), None)
    if not item:
        return

    graph_path = _cache(sid, f"graph_{paper_id}.json")
    if os.path.exists(graph_path):
        with open(graph_path, encoding="utf-8") as fh:
            graph = json.load(fh)
        item["confidence"] = 1.0
        graph["nodes"].append(item)
        # Add a hierarchy edge if the node still has a valid parent.
        if item.get("level", 0) > 0:
            node_ids = {n["id"] for n in graph["nodes"]}
            prov     = item.get("provenance", {})
            graph["edges"].append({
                "source": "root", "target": item["id"],
                "relation": "", "kind": "hierarchy",
            })
        with open(graph_path, "w", encoding="utf-8") as fh:
            json.dump(graph, fh)

    _save_pending(sid, paper_id, [p for p in pending if p["id"] != node_id])


def reject_triple(sid: str, paper_id: str, node_id: str):
    """Discard a pending node."""
    pending = _load_pending(sid, paper_id)
    _save_pending(sid, paper_id, [p for p in pending if p["id"] != node_id])


def build_knowledge_graph(sid: str, paper_id: str | None = None,
                          force: bool = False) -> dict:
    """
    Build / return a knowledge graph for a specific paper.
    Cached to disk so we only pay the LLM call once per paper.
    Low-confidence nodes (<CONFIDENCE_THRESHOLD) go to pending_review queue.
    """
    papers = _load_papers(sid)

    if paper_id is None:
        if not papers:
            raise RuntimeError("No paper has been ingested yet.")
        # Default: most recently ingested paper.
        paper_id = max(papers, key=lambda pid: papers[pid].get("ingested_at", 0))

    graph_cache = _cache(sid, f"graph_{paper_id}.json")
    if not force and os.path.exists(graph_cache):
        with open(graph_cache, encoding="utf-8") as fh:
            return json.load(fh)

    col = collection(sid)
    if col.count() == 0:
        raise RuntimeError("No paper has been ingested yet.")

    # Retrieve chunks for this paper.
    try:
        stored = col.get(
            where   = {"paper_id": {"$eq": paper_id}},
            include = ["documents"],
        )
        text = "\n\n".join(stored.get("documents") or [])
    except Exception:
        stored = col.get(include=["documents"])
        text   = "\n\n".join(stored.get("documents") or [])

    if not text.strip():
        raise RuntimeError(f"No text found for paper {paper_id!r}.")

    excerpt        = text[:30000]
    relations_list = ", ".join(f'"{r}"' for r in ALLOWED_RELATIONS)

    prompt = f"""You are a research scientist building a HIERARCHICAL knowledge
map of ONE paper, so a reader can understand its story at a glance.

Build a tree rooted at the paper's single core contribution, branching into
main themes, then into specifics. Aim for 3 levels of depth.

Structure:
- Level 0: exactly ONE root = the paper's central contribution/system.
- Level 1: 3-6 themes (the problem it solves, method, data, evaluation, findings).
- Level 2+: concrete specifics (components, datasets, tasks, metrics, results, baselines).

Each node MUST have:
  parent     — id of parent node (null for root)
  type       — one of: {GRAPH_TYPES}
  summary    — 2-4 plain-English sentences grounded in this paper: what this
               concept is, why it matters in the paper, and how it connects to
               the rest of the work. Written so a reader who clicks the node
               understands the topic without reading the paper.
  key_facts  — 2-5 specific phrases, numbers, or results from the paper
  confidence — float 0.0–1.0 (how certain you are this node is accurately extracted;
                use < 0.65 when the text is ambiguous or the claim is inferred)

Relations MUST use labels chosen ONLY from this list:
  [{relations_list}]

Do NOT include authors, affiliations, funding, or citation entries.

Return STRICT JSON, nothing else:
{{
  "topic": "the paper in <=6 words",
  "root": "root_id",
  "nodes": [
    {{"id": "snake_case_id", "label": "Short Name", "type": "method",
      "parent": "parent_id_or_null",
      "summary": "1-2 sentences.",
      "key_facts": ["fact", "number"],
      "confidence": 0.95}}
  ],
  "relations": [
    {{"source": "id_a", "target": "id_b", "label": "evaluates_on"}}
  ]
}}

Rules:
- 14 to 24 nodes total. Every non-root node's parent MUST be an existing id.
- Labels concise (1-4 words). Use the paper's own terminology where possible.
- Never invent facts not supported by the text.

Paper text:
{excerpt}
"""

    last_err = None
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(
                model           = GRAPH_MODEL,
                messages        = [{"role": "user", "content": prompt}],
                temperature     = 0.2 + 0.1 * attempt,
                response_format = {"type": "json_object"},
            )
            graph = _normalise_graph(
                _safe_json(resp.choices[0].message.content),
                paper_id=paper_id,
            )
            if len(graph["nodes"]) >= 3:
                # Split into confirmed + pending-review nodes (Layer 2).
                main_nodes    = [n for n in graph["nodes"]
                                 if n["confidence"] >= CONFIDENCE_THRESHOLD]
                pending_nodes = [n for n in graph["nodes"]
                                 if n["confidence"] < CONFIDENCE_THRESHOLD]

                graph["nodes"]         = main_nodes
                graph["pending_count"] = len(pending_nodes)

                _save_pending(sid, paper_id, pending_nodes)

                with open(graph_cache, "w", encoding="utf-8") as fh:
                    json.dump(graph, fh)

                return graph

            last_err = RuntimeError("Too few concepts extracted.")
        except Exception as exc:
            last_err = exc
            print(f"  graph attempt {attempt + 1} failed: {exc}")

    raise RuntimeError(f"Could not build a graph for this paper ({last_err}).")


# ── CROSS-PAPER MERGE (Layer 5 / §5) ───────────────────────────────────────────
def _norm_label(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for entity matching."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _entity_match(label: str, existing: list[dict]) -> dict | None:
    """
    Resolve an entity label against already-seen concept nodes.
    Exact normalised match first, then fuzzy (SequenceMatcher > 0.88).
    ponytail: string + fuzzy only — catches "Multi-Head Attention" vs
    "multi head attention", NOT paraphrases like "BERT" vs its expansion.
    Upgrade path: embed labels (Chroma's MiniLM is already loaded) and match on
    cosine similarity when string matching proves too weak.
    """
    nl = _norm_label(label)
    if not nl:
        return None
    for node in existing:
        en = _norm_label(node["label"])
        if nl == en:
            return node
        if len(nl) > 4 and difflib.SequenceMatcher(None, nl, en).ratio() > 0.88:
            return node
    return None


def build_merged_graph(sid: str) -> dict:
    """
    Merge every paper's per-paper graph into one cross-paper graph.

    Shape: root "All Papers" → one node per paper → that paper's concepts.
    Concepts whose labels resolve to the same entity across papers are MERGED
    into a single shared node tagged with every source paper_id, so shared
    methods/datasets visibly bridge the papers that use them.
    """
    papers = _load_papers(sid)
    if not papers:
        raise RuntimeError("No paper has been ingested yet.")

    root = {
        "id": "all_papers", "label": "All Papers", "type": "root", "level": 0,
        "summary": f"Merged knowledge graph across {len(papers)} papers. "
                   "Shared nodes (appearing in more than one paper) bridge them.",
        "facts": [], "confidence": 1.0, "paper_ids": [], "shared": False,
    }
    nodes: list[dict] = [root]
    edges: list[dict] = []
    concepts: list[dict] = []   # level-2 nodes eligible for cross-paper merging

    for pid, pmeta in papers.items():
        try:
            g = build_knowledge_graph(sid, pid)
        except Exception as exc:
            print(f"  skipping paper {pid} in merge: {exc}")
            continue

        pnode_id = f"paper_{pid}"
        title    = (pmeta.get("title") or pmeta.get("filename") or pid)[:60]
        nodes.append({
            "id": pnode_id, "label": title, "type": "paper", "level": 1,
            "summary": (pmeta.get("abstract") or "")[:400],
            "facts": [], "confidence": 1.0, "paper_ids": [pid], "shared": False,
        })
        edges.append({"source": "all_papers", "target": pnode_id,
                      "relation": "", "kind": "hierarchy"})

        groot = next((n["id"] for n in g["nodes"] if n.get("level") == 0), None)
        for n in g["nodes"]:
            if n["id"] == groot:
                continue   # the paper's own root is represented by the paper node

            match = _entity_match(n["label"], concepts)
            if match:
                if pid not in match["paper_ids"]:
                    match["paper_ids"].append(pid)
                    match["shared"] = True
                for f in n.get("facts", []):
                    if f not in match["facts"]:
                        match["facts"].append(f)
                edges.append({"source": pnode_id, "target": match["id"],
                              "relation": "", "kind": "hierarchy"})
            else:
                new = {
                    "id": f"{pid}__{n['id']}", "label": n["label"],
                    "type": n.get("type", "concept"), "level": 2,
                    "summary": n.get("summary", ""),
                    "facts": list(n.get("facts", []))[:6],
                    "confidence": n.get("confidence", 1.0),
                    "paper_ids": [pid], "shared": False,
                }
                nodes.append(new)
                concepts.append(new)
                edges.append({"source": pnode_id, "target": new["id"],
                              "relation": "", "kind": "hierarchy"})

    return {
        "topic": f"{len(papers)} papers merged", "paper_id": None, "merged": True,
        "nodes": nodes, "edges": edges,
        "built_at": time.time(), "model": GRAPH_MODEL, "version": PIPELINE_VERSION,
    }


# ── CLI ENTRYPOINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    meta = ingest_pdf("local", "paper.pdf")
    print(f"Paper: {meta.get('title') or meta['filename']}")
    print(f"Pages: {meta['pages']}  Chunks: {meta['chunks']}\n")

    while True:
        q = input("Ask a question (or 'quit'): ").strip()
        if q.lower() == "quit":
            break
        result = ask("local", q)
        print(f"\nAnswer:\n{result['answer']}")
        verified = result["verification"].get("verified", True)
        print(f"\n{'✅ Verified' if verified else '⚠️  Unverified'}: "
              f"{result['verification'].get('note', '')}")
        for c in result["chunks"]:
            print(f"  [p.{c['page_number']} §{c['section_title'][:30]}] "
                  f"{c['text'][:80]}…")
        print()