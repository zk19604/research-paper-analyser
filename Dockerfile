# Stage 1: build the React frontend
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: FastAPI backend + built frontend, one container
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as uid 1000 without root
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user rag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake Chroma's ONNX embedding model into the image so the first upload
# doesn't stall on an ~80MB model download.
RUN python -c "import chromadb; c = chromadb.EphemeralClient().create_collection('warmup'); c.add(ids=['1'], documents=['warm up the embedding model'])"

COPY --chown=user rag/ .
COPY --from=frontend --chown=user /fe/dist ./static

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
