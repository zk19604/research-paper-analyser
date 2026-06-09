#!/usr/bin/env bash
# Start everything: Ollama check, FastAPI backend, and the frontend server.
# Usage: ./start.sh   (Ctrl+C stops both servers)
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT=8000
FRONTEND_PORT=5500

# ── Ollama (needed for embeddings) ───────────────────────────
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "⚠️  Ollama isn't running. Start it in another terminal with:"
  echo "      ollama serve   (and once: ollama pull nomic-embed-text)"
  echo
fi

# ── Backend (FastAPI) ────────────────────────────────────────
echo "▶ Starting backend on http://localhost:$BACKEND_PORT"
cd "$ROOT/rag"
pip install -q -r requirements.txt
uvicorn main:app --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

# ── Frontend (static server) ─────────────────────────────────
echo "▶ Starting frontend on http://localhost:$FRONTEND_PORT"
cd "$ROOT/frontend"
python3 -m http.server "$FRONTEND_PORT" &
FRONTEND_PID=$!

# ── Stop both on Ctrl+C ──────────────────────────────────────
cleanup() {
  echo
  echo "Stopping servers…"
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

echo
echo "✅ Open the app:  http://localhost:$FRONTEND_PORT"
echo "   (Press Ctrl+C to stop)"
wait
