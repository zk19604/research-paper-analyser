#!/usr/bin/env bash
# Start everything: FastAPI backend and the Vite dev frontend.
# Usage: ./start.sh   (Ctrl+C stops both servers)
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT=8000
FRONTEND_PORT=5173

# ── Backend (FastAPI) ────────────────────────────────────────
echo "▶ Starting backend on http://localhost:$BACKEND_PORT"
cd "$ROOT/rag"
python3 -m pip install -q -r requirements.txt --break-system-packages
uvicorn main:app --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

# ── Frontend (Vite dev server) ───────────────────────────────
echo "▶ Starting frontend on http://localhost:$FRONTEND_PORT"
cd "$ROOT/frontend"
npm run dev &
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
