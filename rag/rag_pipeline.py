import os
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

client_db = chromadb.Client()
collection = client_db.get_or_create_collection(name="research_paper")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        # Step 1: extract normal text as before
        text = page.get_text()
        full_text += text

        # Step 2: render the page as an image and OCR it
        # this catches text inside figures, diagrams, tables
        mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR accuracy
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_text = pytesseract.image_to_string(img)

        # Step 3: only add OCR text that isn't already in normal extraction
        # avoids pure duplicates
        for line in ocr_text.splitlines():
            line = line.strip()
            if line and line not in full_text:
                full_text += "\n" + line

        print(f"  processed page {page_num + 1}/{len(doc)}")

    return full_text


# ── 2. CHUNKING ────────────────────────────────────────────
def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


# ── 3. EMBEDDING (local via Ollama) ────────────────────────
def embed_text(text):
    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )
    return response["embeddings"][0]


# ── 4. INGEST PDF ──────────────────────────────────────────
def ingest_pdf(pdf_path):
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
    print("Ingestion complete.\n")


# ── 5. RETRIEVAL + GENERATION ──────────────────────────────
def ask(question):
    query_vector = embed_text(question)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=6
    )
    retrieved_chunks = results["documents"][0]

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