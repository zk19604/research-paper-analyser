---
title: Research Paper Analyser
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Research Paper Analyser

A retrieval-augmented (RAG) reading assistant for academic papers. Drop in a
PDF, ask questions and get answers **grounded in the paper with inline `[p.N]`
citations**, every answer **fact-checked** against its own sources, and an
auto-generated **knowledge graph** of the paper's ideas. Upload several papers
and it builds a **cross-paper graph** where shared methods and datasets bridge
the papers that use them.

![Main screen](images/main-screen.png)

## Notable features

Built against a serious scientific-RAG checklist. Everything below is
implemented in this repo — not aspirational.

**Layout-aware ingestion**
- **Structure-first extraction** — PyMuPDF dict-mode reads text blocks with font
  size + bold flags; section headers are detected by font size vs. the page
  median, so parsing follows the paper's real hierarchy instead of naive
  `pdftotext`.
- **OCR fallback** — scanned / image-only pages are rasterised and run through
  Tesseract.
- **Tables & figures as atomic chunks** — the table detector emits markdown
  tables (caption kept attached); figure captions become searchable chunks.
- **Reference-level chunking** — the references section is split one chunk per
  citation entry, not one blob.
- **Paper-level metadata** — title, abstract, and page count extracted
  heuristically from the first pages.

**Structure-aware chunking**
- Sections flush on header detection; **abstract and conclusion kept whole**
  (dense with claims); long sections sub-split with ~200-char overlap.
- Every chunk carries `paper_id / page_number / section_title / chunk_type`
  metadata — this is what makes citations verifiable.
- **File-hash (MD5) caching** — re-uploading the same PDF is a no-op.

**Hybrid retrieval**
- Dense vector search (ChromaDB, `all-MiniLM-L6-v2`) **+** sparse BM25 keyword
  search (stopword-filtered), fused via **Reciprocal Rank Fusion**.
- **LLM cross-reranker** scores fused candidates and keeps the top 4.
- Metadata (paper_id, page, section) carried end-to-end so answers cite
  precisely.

**Anti-hallucination**
- **Grounded generation** — answers use only retrieved context; explicit refusal
  when the answer isn't present; low temperature for factual QA.
- **Inline `[p.N]` citations** required in every answer.
- **Post-hoc citation verification** — a second LLM pass checks each claim
  against its sources and flags unsupported ones (`verified / unsupported_claims`).
- **Fixed ontology** — entity + relation extraction is constrained to a closed
  set of node types and 13 allowed relations (`proposes`, `outperforms`,
  `evaluates_on`, `contradicts`, …), so the model can't invent relation types.
- **Schema-constrained JSON extraction** for the graph, not free-form summary.
- **Confidence score per node**; nodes below threshold are held back.
- **Human-in-the-loop** — low-confidence nodes go to a pending-review queue you
  approve or reject before they enter the graph.
- **Provenance chain** on every node — `paper_id / model / pipeline_version`.

**Knowledge graph**
- Hierarchical per-paper concept map: single root contribution → themes →
  specifics, each node with a plain-English summary and key facts.
- Click a node for its summary, facts, confidence, and provenance;
  "explain in more detail" expands on demand.

**Multi-paper & cross-paper merge**
- Multiple PDFs per session, each a unique `paper_id`; papers **accumulate**
  (uploading a second doesn't wipe the first).
- **Entity resolution across papers** — exact + fuzzy (SequenceMatcher) label
  matching merges the same concept phrased differently into one **shared node**
  that bridges the papers using it.
- **`contradicts` relation** in the ontology for disagreements between papers.
- Both **per-paper subgraph** and **merged global** views.

**Per-session isolation**
- Each browser gets its own ChromaDB collection via an `X-Session-Id` header —
  no shared state; idle sessions purged after 24h.

### Grounded answer + citation verification
![Chat with grounded citations](images/chat.png)

### Per-paper knowledge graph
![Knowledge graph](images/knowledge.png)

### Cross-paper merged graph (shared nodes bridge papers)
![Merged knowledge graph](images/knowledge-merged.png)

## Retrieval pipeline

Ingestion and query are both multi-stage — the interesting engineering is here:

**Ingest** (`ingest_pdf`)
1. **Structured extraction** — PyMuPDF dict-mode reads text blocks *with font
   size and bold flags*. Headers are detected by font size relative to the page
   median, so chunking follows the paper's real section structure.
2. **OCR fallback** — pages with almost no text layer (scanned/image PDFs) are
   rasterised and run through Tesseract.
3. **Tables & figures** — PyMuPDF's table detector emits markdown table chunks;
   figure captions become searchable chunks.
4. **Structure-aware chunking** — sections flush on header detection; abstract
   and conclusion stay atomic; references split one-entry-per-citation; long
   sections sub-split with ~200-char overlap. Every chunk carries
   `paper_id / page_number / section_title / chunk_type` metadata.
5. **Dedup** — MD5 of file contents; re-uploading the same PDF is a no-op.

**Query** (`ask`)
1. **Hybrid retrieval** — dense vector search (ChromaDB, `all-MiniLM-L6-v2`) +
   sparse **BM25** keyword search (stopword-filtered), merged with **Reciprocal
   Rank Fusion**. Recall from both semantics and exact terms.
2. **LLM reranker** — a cross-reranker (Llama 3.3 70B) scores the fused
   candidates and keeps the top 4.
3. **Grounded generation** — answer constrained to the retrieved context with
   inline `[p.N]` citation instructions.
4. **Verification** — a separate LLM pass checks each claim against the sources
   and returns `{verified, unsupported_claims, note}`.

Embeddings run **in-process** via ChromaDB's built-in model — no external
embedding service.

## Stack

- **Backend** — FastAPI, one container. `rag_pipeline.py` holds the whole
  pipeline; `main.py` is the HTTP layer and also serves the built frontend.
- **Retrieval** — ChromaDB (persistent, built-in MiniLM embeddings), `rank_bm25`.
- **LLM** — Groq (Llama 3.3 70B) for reranking, answers, verification, and graph
  extraction.
- **Extraction** — PyMuPDF (text/tables/figures), Tesseract (OCR fallback).
- **Frontend** — React + Vite, force-directed knowledge-graph canvas.
- **Deploy** — single Dockerfile (builds frontend + runs backend) on Hugging
  Face Spaces; per-session state, purged after 24h idle.


## API

All endpoints require an `X-Session-Id` header (the frontend generates one per
browser and keeps it in localStorage).

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | Ingest a PDF (accumulates — does not replace prior papers) |
| GET | `/papers` | List all papers in the session |
| DELETE | `/papers/{paper_id}` | Remove a paper and its chunks |
| GET | `/meta?paper_id=` | Paper metadata (title, abstract, pages) |
| POST | `/ask` | Grounded answer with inline citations + verification |
| GET | `/graph?paper_id=&merged=&refresh=` | Per-paper or merged knowledge graph |
| GET | `/graph/pending?paper_id=` | Low-confidence nodes awaiting review |
| POST | `/graph/approve` · `/graph/reject` | Promote / discard a pending node |
| GET | `/status` | Chunk count for the session |
