from collections import Counter
import math 

def build_vocab (documents) :
    vocab = set ()
    for doc in documents :
        words = doc.lower().split ()
        vocab.update(words)
    return vocab

def embed_bow (text, vocab) :
    words = text.lower().split ()
    word_counts = Counter(words)
    embedding = [word_counts.get(word, 0) for word in vocab]
    return embedding    

def cosine_similarity (vec1, vec2) :
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0 :
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

docs = [
    "neural networks learn from data",
    "deep learning uses neural networks",
    "cats and dogs are animals",
]

vocab = build_vocab(docs)
print("Vocabulary:", vocab)
print()

for doc in docs:
    vec = embed_bow(doc, vocab)
    print(f"Text  : {doc}")
    print(f"Vector: {vec}")
    print()


# Now search!
query = "machines that learn automatically"
query_vec = embed_bow(query, vocab)

print(f"Query: {query}\n")

scores = []
for doc in docs:
    doc_vec = embed_bow(doc, vocab)
    score = cosine_similarity(query_vec, doc_vec)
    scores.append((score, doc))

# rank by similarity
scores.sort(reverse=True)
for score, doc in scores:
    print(f"Score: {round(score, 3)}  →  {doc}")