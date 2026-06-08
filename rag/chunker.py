def chunk_text(text, chunk_size = 100, overlap = 10):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# Test it with a fake "paper abstract"
sample = """
Large language models have demonstrated remarkable capabilities across diverse tasks.
However, they suffer from hallucination and outdated knowledge. Retrieval-Augmented 
Generation addresses this by grounding responses in external documents. In this paper 
we propose a novel RAG architecture that improves retrieval precision by 34% on 
standard benchmarks while reducing latency by half. Our method uses semantic chunking 
combined with a reranking step to filter irrelevant passages before generation.
"""

chunks = chunk_text(sample, chunk_size=100, overlap=20)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()