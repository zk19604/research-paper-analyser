// Same-origin in production; dev proxy to localhost:8000.
// Every request carries a per-browser session id so users don't share papers.
const API = import.meta.env.DEV ? 'http://localhost:8000' : ''

const SID = localStorage.getItem('sid') || crypto.randomUUID()
localStorage.setItem('sid', SID)

export function apiFetch(path, opts = {}) {
  return fetch(`${API}${path}`, {
    ...opts,
    headers: { ...opts.headers, 'X-Session-Id': SID },
  })
}

// ── Typed helpers ────────────────────────────────────────────────────────────

export async function apiListPapers() {
  const res = await apiFetch('/papers')
  if (!res.ok) throw new Error('Failed to list papers')
  return (await res.json()).papers
}

export async function apiDeletePaper(paper_id) {
  const res = await apiFetch(`/papers/${paper_id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete paper')
}

export async function apiGetPending(paper_id) {
  const res = await apiFetch(`/graph/pending?paper_id=${paper_id}`)
  if (!res.ok) return []
  return (await res.json()).pending ?? []
}

export async function apiApproveNode(paper_id, node_id) {
  const res = await apiFetch('/graph/approve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paper_id, node_id }),
  })
  if (!res.ok) throw new Error('Failed to approve node')
}

export async function apiRejectNode(paper_id, node_id) {
  const res = await apiFetch('/graph/reject', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paper_id, node_id }),
  })
  if (!res.ok) throw new Error('Failed to reject node')
}
