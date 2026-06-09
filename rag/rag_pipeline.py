import os
import json
import fitz
import chromadb
import ollama
from groq import Groq
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import io

# ── SETUP ──────────────────────────────────────────────────
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_CACHE = os.path.join(BASE_DIR, "last_paper.txt")
GRAPH_CACHE = os.path.join(BASE_DIR, "graph.json")
META_CACHE = os.path.join(BASE_DIR, "doc_meta.json")

# Persistent on-disk store so we ingest/embed once, not on every launch.
client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection(name="research_paper")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        # Prefer the embedded text layer — it is clean and ordered.
        text = page.get_text()

        # Only fall back to OCR for pages that have almost no text layer
        # (e.g. fully scanned/image pages). OCR-ing every page just injects
        # garbled figure/axis text that pollutes retrieval.
        if len(text.strip()) < 100:
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR accuracy
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)

        pages.append(text)
        print(f"  processed page {page_num + 1}/{len(doc)}")

    return "\n".join(pages)


# ── 2. CHUNKING ────────────────────────────────────────────
def _looks_like_garbage(chunk):
    """Drop chunks dominated by non-alphabetic noise (OCR junk, equations)."""
    stripped = chunk.strip()
    if len(stripped) < 50:
        return True
    alpha = sum(c.isalpha() or c.isspace() for c in stripped)
    return (alpha / len(stripped)) < 0.6


def chunk_text(text, chunk_size=1200, overlap=200):
    """Paragraph-aware packing: keep paragraphs together, then pack them
    into ~chunk_size windows with a sentence-level overlap tail. This keeps
    related content (e.g. the Text/Audio/Video embedding paragraphs) in the
    same chunk instead of slicing mid-word on a raw character boundary."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # start the next chunk with an overlap tail for continuity
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}" if tail else para

    if current:
        chunks.append(current)

    return [c for c in chunks if not _looks_like_garbage(c)]


# ── 3. EMBEDDING (local via Ollama) ────────────────────────
# nomic-embed-text REQUIRES task prefixes: documents and queries must be
# embedded with different instructions or their vectors won't align.
def embed_text(text, is_query=False):
    prefix = "search_query: " if is_query else "search_document: "
    response = ollama.embed(
        model="nomic-embed-text",
        input=prefix + text
    )
    return response["embeddings"][0]


# ── 4. INGEST PDF ──────────────────────────────────────────
def reset_collection():
    """Drop every stored chunk so a freshly uploaded PDF starts clean."""
    global collection
    client_db.delete_collection(name="research_paper")
    collection = client_db.get_or_create_collection(name="research_paper")


def ingest_pdf(pdf_path, force=False):
    # With a persistent store, skip the whole pipeline if we already ingested.
    # `force=True` (used when a new PDF is uploaded) re-ingests from scratch.
    if collection.count() > 0:
        if not force:
            print(f"Collection already has {collection.count()} chunks — skipping ingest.\n")
            return
        reset_collection()

    print("Step 1: Extracting text...")
    text = extract_text_from_pdf(pdf_path)
    print(f"  extracted {len(text)} characters\n")

    print("Step 2: Chunking...")
    chunks = chunk_text(text)
    print(f"  created {len(chunks)} chunks\n")

    print("Step 3: Embedding and storing...")
    for i, chunk in enumerate(chunks):
        if chunk.strip() == "":
            continue
        vector = embed_text(chunk)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[vector],
            documents=[chunk]
        )
    print(f"  stored {collection.count()} chunks in vector DB\n")

    # Cache the clean full text for graph extraction, record document metadata,
    # and invalidate any previously generated knowledge graph.
    with open(TEXT_CACHE, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(META_CACHE, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "filename": os.path.basename(pdf_path),
                "chunks": collection.count(),
                "characters": len(text),
            },
            fh,
        )
    if os.path.exists(GRAPH_CACHE):
        os.remove(GRAPH_CACHE)

    print("Ingestion complete.\n")


# ── METADATA ───────────────────────────────────────────────
def get_meta():
    """Info about the currently indexed paper, for the Sources sidebar."""
    if collection.count() == 0:
        return None
    if os.path.exists(META_CACHE):
        with open(META_CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    # Pre-caching ingest: we at least know the chunk count.
    return {"filename": "Indexed paper", "chunks": collection.count()}


# ── KNOWLEDGE GRAPH (Groq) ─────────────────────────────────
# Node categories worth surfacing — deliberately excludes authors/affiliations
# so the graph reflects the paper's *ideas*, not its byline.
GRAPH_TYPES = (
    "root, theme, problem, method, component, model, dataset, "
    "task, metric, finding, concept, baseline"
)


def _safe_json(raw):
    """Parse the model's reply, tolerating stray markdown fences or prose by
    falling back to the outermost {...} block."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def _normalise_graph(raw):
    """Turn the model's {topic, nodes[parent], relations} payload into the
    {topic, nodes[level], edges[kind]} shape the frontend renders.

    - Hierarchy comes from each node's `parent`; we derive `level` by walking
      the parent chain (cycle-safe) so the graph can be many levels deep.
    - Cross links (non-tree relationships) are kept as `relation` edges.
    """
    nodes = {n["id"]: n for n in raw.get("nodes", []) if n.get("id") and n.get("label")}
    if not nodes:
        raise RuntimeError("The model returned no usable concepts.")

    # Pick / validate a single root: the declared one, else a node nobody
    # claims as a child, else just the first node.
    children = {n["id"] for n in nodes.values() if n.get("parent") in nodes}
    root_id = raw.get("root")
    if root_id not in nodes:
        root_id = next((nid for nid in nodes if nid not in children), None) or next(iter(nodes))
    nodes[root_id]["parent"] = None

    def level_of(nid):
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
            # Orphan / broken parent → attach to root so nothing floats free
            # and the whole graph stays one connected tree.
            parent = root_id
            n["parent"] = parent

        lvl = 0 if nid == root_id else level_of(nid)
        out_nodes.append({
            "id": nid,
            "label": n["label"],
            "type": "root" if nid == root_id else (n.get("type") or "concept"),
            "level": lvl,
            "summary": (n.get("summary") or "").strip(),
            "facts": [f for f in (n.get("key_facts") or []) if isinstance(f, str)][:5],
        })
        if nid != root_id:
            hierarchy_edges.append({"source": parent, "target": nid, "relation": "", "kind": "hierarchy"})

    relation_edges = [
        {"source": r["source"], "target": r["target"], "relation": r.get("label", ""), "kind": "relation"}
        for r in raw.get("relations", [])
        if r.get("source") in nodes and r.get("target") in nodes and r.get("source") != r.get("target")
    ]

    return {"topic": raw.get("topic", ""), "nodes": out_nodes, "edges": hierarchy_edges + relation_edges}


def build_knowledge_graph(force=False):
    """Extract a HIERARCHICAL knowledge graph from the paper using Groq.

    The graph is a tree rooted at the paper's core contribution, branching into
    themes and then specifics (multiple levels), with extra cross-links for
    relationships that aren't strictly parent/child. Every node carries a short
    grounded summary and key facts so the UI can show detail on click.

    Cached to disk so we only pay the LLM call once per uploaded paper."""
    if not force and os.path.exists(GRAPH_CACHE):
        with open(GRAPH_CACHE, encoding="utf-8") as fh:
            return json.load(fh)

    if os.path.exists(TEXT_CACHE):
        with open(TEXT_CACHE, encoding="utf-8") as fh:
            text = fh.read()
    elif collection.count() > 0:
        # Paper was ingested before text-caching existed — rebuild the text
        # from the stored chunks so the graph still works without re-uploading.
        stored = collection.get(include=["documents"])
        text = "\n\n".join(stored.get("documents") or [])
    else:
        raise RuntimeError("No paper has been ingested yet.")

    # llama-3.3-70b has a large context window, so feed it a generous slice
    # spanning the problem, method, data and results — not just the abstract.
    excerpt = text[:30000]

    prompt = f"""You are a research scientist building a HIERARCHICAL knowledge
map of ONE paper, so a reader can understand its story at a glance.

Build a tree rooted at the paper's single core contribution, then branch into
its main themes, then into specifics. Aim for 3 levels of depth.

Use this structure:
- Level 0: exactly ONE root = the paper's central contribution/system.
- Level 1: 3-6 themes = the pillars of the paper (e.g. the problem it solves,
  its method/approach, the data it uses, how it is evaluated, key findings).
- Level 2+: the concrete specifics under each theme (individual components,
  datasets, tasks, metrics, results, baselines, concepts).

Each node MUST have a `parent` (the id of the node it sits under); the root's
parent is null. `type` is one of: {GRAPH_TYPES}.

For EVERY node write a `summary`: 1-2 plain-English sentences, grounded in THIS
paper, explaining what it is and why it matters. Add 1-4 `key_facts` (short
specific phrases / numbers from the paper) where the text supports them.

Also list `relations`: meaningful NON-hierarchical links between any two nodes
(e.g. "method" -> "evaluated on" -> "dataset", "model" -> "outperforms" ->
"baseline"). Use short verb-phrase labels.

Do NOT include authors, affiliations, funding, or citation entries.

Return STRICT JSON, nothing else:
{{
  "topic": "the paper in <=6 words",
  "root": "root_id",
  "nodes": [
    {{"id": "snake_case_id", "label": "Short Name", "type": "method",
      "parent": "parent_id_or_null",
      "summary": "1-2 sentences grounded in the paper.",
      "key_facts": ["specific phrase", "number or result"]}}
  ],
  "relations": [
    {{"source": "id_a", "target": "id_b", "label": "evaluated on"}}
  ]
}}

Rules:
- 14 to 24 nodes total. Every non-root node's `parent` MUST be an existing id.
- Labels are concise (1-4 words). Prefer the paper's own terminology.
- Be faithful: never invent facts not supported by the text.

Paper text:
{excerpt}
"""

    # Be resilient: any single attempt can hit a transient Groq error, return
    # malformed JSON, or yield too few nodes. Retry a couple of times before
    # giving up so the feature works across arbitrary papers.
    last_err = None
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2 + 0.1 * attempt,  # nudge off a bad local optimum
                response_format={"type": "json_object"},
            )
            graph = _normalise_graph(_safe_json(resp.choices[0].message.content))
            if len(graph["nodes"]) >= 3:  # a root + at least a couple of branches
                with open(GRAPH_CACHE, "w", encoding="utf-8") as fh:
                    json.dump(graph, fh)
                return graph
            last_err = RuntimeError("Too few concepts extracted.")
        except Exception as exc:
            last_err = exc
            print(f"  graph attempt {attempt + 1} failed: {exc}")

    raise RuntimeError(f"Could not build a graph for this paper ({last_err}).")


# ── 5a. RERANKING ──────────────────────────────────────────
def rerank(question, candidates, top_k=4):
    """Second-stage reranking. Vector search is recall-oriented and noisy;
    we ask the LLM to score each candidate's relevance to the question and
    keep only the best ones before generation. (A local cross-encoder like
    cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers is a
    drop-in alternative if you'd rather not spend an LLM call here.)"""
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
    prompt = f"""Question: {question}

Below are candidate passages. Return ONLY a comma-separated list of the
indices of the {top_k} passages most relevant to answering the question,
most relevant first. Example: 3,0,5,1

Passages:
{numbered}
"""
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content

    order = []
    for tok in raw.replace("\n", ",").split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) < len(candidates) and int(tok) not in order:
            order.append(int(tok))

    if not order:  # fall back to original vector order if parsing fails
        order = list(range(len(candidates)))

    return [candidates[i] for i in order[:top_k]]


# ── 5b. RETRIEVAL + GENERATION ─────────────────────────────
def ask(question):
    query_vector = embed_text(question, is_query=True)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=8
    )
    candidates = results["documents"][0]

    # Rerank the 8 recalled chunks down to the 4 most relevant.
    retrieved_chunks = rerank(question, candidates, top_k=4)

    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""You are a research paper assistant.
Use ONLY the context below to answer the question.
If the answer isn't in the context, say "I don't know based on this paper."

Context:
{context}

Question: {question}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, retrieved_chunks


# ── 6. RUN IT ──────────────────────────────────────────────
if __name__ == "__main__":
    ingest_pdf("paper.pdf")

    while True:
        question = input("Ask a question (or 'quit'): ")
        if question.lower() == "quit":
            break

        answer, chunks = ask(question)
        print(f"\nAnswer: {answer}")
        print(f"\nBased on these chunks:")
        for i, c in enumerate(chunks):
            print(f"  [{i+1}] {c[:80]}...")
        print()