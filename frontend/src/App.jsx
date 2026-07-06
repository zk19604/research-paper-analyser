import { useEffect, useRef, useState } from 'react'
import Graph from './Graph'
import { apiFetch, apiListPapers, apiDeletePaper } from './api'
import './App.css'

const SUGGESTIONS = [
  'What is the main contribution of this paper?',
  'Summarise the methodology in a few sentences.',
  'What are the key findings or results?',
  'What datasets were used and how was the model evaluated?',
  'What are the limitations of this work?',
]

function VerificationBadge({ verification }) {
  if (!verification) return null
  const ok     = verification.verified !== false
  const claims = verification.unsupported_claims ?? []
  return (
    <div className={`verify-badge ${ok ? 'verified' : 'unverified'}`}>
      <span className="verify-icon">{ok ? '✅' : '⚠️'}</span>
      <span className="verify-text">
        {ok
          ? 'All claims grounded in source'
          : `${claims.length} unverified claim${claims.length !== 1 ? 's' : ''}`}
      </span>
      {!ok && claims.length > 0 && (
        <details className="verify-claims">
          <summary>Show details</summary>
          <ul>{claims.map((c, i) => <li key={i}>{c}</li>)}</ul>
        </details>
      )}
    </div>
  )
}

function ChunkCitation({ chunk, index }) {
  const text = typeof chunk === 'string' ? chunk : chunk.text
  const page = chunk.page_number
  const sec  = chunk.section_title

  return (
    <blockquote className="chunk-cite">
      {page > 0 && (
        <span className="chunk-meta">
          p.{page}{sec ? ` · §${sec}` : ''}
        </span>
      )}
      <span className="chunk-text">{text}</span>
    </blockquote>
  )
}

function PaperItem({ paper, selected, onSelect, onRemove }) {
  return (
    <div
      className={`source-item ${selected ? 'active' : ''}`}
      onClick={() => onSelect(paper.paper_id)}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && onSelect(paper.paper_id)}
    >
      <span className="source-icon">▤</span>
      <div className="source-meta">
        <div className="source-title" title={paper.filename}>
          {paper.title || paper.filename}
        </div>
        <div className="source-sub">
          {paper.pages ? `${paper.pages} pp · ` : ''}
          {paper.chunks} chunks
        </div>
      </div>
      <button
        className="source-remove"
        title="Remove paper"
        onClick={e => { e.stopPropagation(); onRemove(paper.paper_id) }}
        aria-label="Remove paper"
      >
        ×
      </button>
      {selected && <span className="source-check">✓</span>}
    </div>
  )
}

export default function App() {
  const [papers,          setPapers]          = useState([])   // list of paper metadata objects
  const [selectedPaperId, setSelectedPaperId] = useState(null) // null = all papers
  const [uploading,       setUploading]       = useState(false)
  const [messages,        setMessages]        = useState([])   // {role, text, chunks?, verification?}
  const [question,        setQuestion]        = useState('')
  const [asking,          setAsking]          = useState(false)
  const [error,           setError]           = useState('')
  const [view,            setView]            = useState('chat') // 'chat' | 'graph'

  const fileInputRef = useRef(null)
  const scrollRef    = useRef(null)

  useEffect(() => { refreshPapers() }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, asking])

  async function refreshPapers() {
    try {
      const list = await apiListPapers()
      setPapers(list)
      // Auto-select the most recent paper if nothing selected.
      if (list.length > 0 && selectedPaperId === null) {
        setSelectedPaperId(list[list.length - 1].paper_id)
      }
    } catch {
      setError('Cannot reach the backend. Is the API running on :8000?')
    }
  }

  async function handleUpload(f) {
    if (!f) return
    setError('')
    setUploading(true)
    try {
      const body = new FormData()
      body.append('file', f)
      const res  = await apiFetch('/upload', { method: 'POST', body })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      await refreshPapers()
      setSelectedPaperId(data.paper_id)
      setMessages([])
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  async function handleRemovePaper(paper_id) {
    try {
      await apiDeletePaper(paper_id)
      setPapers(prev => prev.filter(p => p.paper_id !== paper_id))
      if (selectedPaperId === paper_id) {
        const remaining = papers.filter(p => p.paper_id !== paper_id)
        setSelectedPaperId(remaining.length > 0 ? remaining[remaining.length - 1].paper_id : null)
        setMessages([])
      }
    } catch (err) {
      setError(err.message)
    }
  }

  async function ask(raw) {
    const q = (raw ?? question).trim()
    if (!q || asking || papers.length === 0) return
    setError('')
    setQuestion('')
    setMessages(m => [...m, { role: 'user', text: q }])
    setAsking(true)
    try {
      const res  = await apiFetch('/ask', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ question: q, paper_id: selectedPaperId }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Request failed')
      setMessages(m => [...m, {
        role:         'assistant',
        text:         data.answer,
        chunks:       data.chunks ?? [],
        verification: data.verification,
      }])
    } catch (err) {
      setMessages(m => [...m, { role: 'assistant', text: `⚠️ ${err.message}`, error: true }])
    } finally {
      setAsking(false)
    }
  }

  const hasPaper    = papers.length > 0
  const activePaper = papers.find(p => p.paper_id === selectedPaperId)
  const paperName   = activePaper
    ? (activePaper.title || activePaper.filename)
    : hasPaper ? `${papers.length} papers` : 'Untitled paper'

  const totalChunks = papers.reduce((sum, p) => sum + (p.chunks || 0), 0)

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="topbar glass">
        <div className="brand">
          <img className="brand-mark" src="/logo.png" alt="Paper Analyser logo" />
          <span className="brand-name">Paper Analyser</span>
        </div>

        <div className="toggle">
          <button
            className={`toggle-btn ${view === 'chat' ? 'active' : ''}`}
            onClick={() => setView('chat')}
          >
            Chat
          </button>
          <button
            className={`toggle-btn ${view === 'graph' ? 'active' : ''}`}
            onClick={() => setView('graph')}
            disabled={!hasPaper}
            title={hasPaper ? '' : 'Upload a paper first'}
          >
            Knowledge Graph
          </button>
        </div>

        <div className="topbar-status">
          <span className={`dot ${hasPaper ? 'live' : ''}`} />
          {hasPaper
            ? `${papers.length} paper${papers.length !== 1 ? 's' : ''} · ${totalChunks} chunks`
            : 'no papers'}
        </div>
      </header>

      <div className="layout">
        {/* ── Sources sidebar ── */}
        <aside className="sources glass">
          <div className="panel-head">Sources</div>

          <button
            className="add-source"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            {uploading ? 'Ingesting…' : '＋  Add source'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={e => e.target.files?.[0] && handleUpload(e.target.files[0])}
            hidden
          />

          <label
            className="dropzone"
            onDragOver={e => e.preventDefault()}
            onDrop={e => {
              e.preventDefault()
              const f = e.dataTransfer.files?.[0]
              if (f) handleUpload(f)
            }}
          >
            Drop a PDF here
          </label>

          <div className="source-list">
            {papers.length === 0 ? (
              <div className="source-empty">No sources yet. Add a PDF to begin.</div>
            ) : (
              <>
                {/* "All papers" toggle when multiple papers exist */}
                {papers.length > 1 && (
                  <div
                    className={`source-item all-papers ${selectedPaperId === null ? 'active' : ''}`}
                    onClick={() => setSelectedPaperId(null)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => e.key === 'Enter' && setSelectedPaperId(null)}
                  >
                    <span className="source-icon">⊞</span>
                    <div className="source-meta">
                      <div className="source-title">All papers</div>
                      <div className="source-sub">{totalChunks} chunks total</div>
                    </div>
                    {selectedPaperId === null && <span className="source-check">✓</span>}
                  </div>
                )}
                {papers.map(p => (
                  <PaperItem
                    key={p.paper_id}
                    paper={p}
                    selected={selectedPaperId === p.paper_id}
                    onSelect={id => { setSelectedPaperId(id); setMessages([]) }}
                    onRemove={handleRemovePaper}
                  />
                ))}
              </>
            )}
          </div>

          {error && <div className="error">{error}</div>}
        </aside>

        {/* ── Main panel ── */}
        <main className="main glass">
          {view === 'graph' ? (
            <Graph
              papers={papers}
              selectedPaperId={selectedPaperId}
              onSelectPaper={setSelectedPaperId}
            />
          ) : (
            <div className="chat">
              <div className="messages" ref={scrollRef}>
                {messages.length === 0 ? (
                  <div className="welcome">
                    <img className="welcome-logo" src="/logo.png" alt="Paper Analyser logo" />
                    <h1 className="welcome-title">{paperName}</h1>
                    <p className="welcome-sub">
                      {hasPaper
                        ? (activePaper?.abstract
                            ? activePaper.abstract.slice(0, 220) + (activePaper.abstract.length > 220 ? '…' : '')
                            : 'Ask anything grounded in this paper, or explore its knowledge graph.')
                        : 'Add a PDF source on the left, then ask questions about its content.'}
                    </p>

                    {hasPaper && (
                      <div className="suggest-grid">
                        {SUGGESTIONS.map(s => (
                          <button key={s} className="suggest" onClick={() => ask(s)}>
                            {s}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  messages.map((m, i) => (
                    <div key={i} className={`msg ${m.role} ${m.error ? 'is-error' : ''}`}>
                      <div className="bubble">{m.text}</div>

                      {/* Verification badge (assistant messages only) */}
                      {m.role === 'assistant' && !m.error && m.verification && (
                        <VerificationBadge verification={m.verification} />
                      )}

                      {/* Source passages */}
                      {m.chunks?.length > 0 && (
                        <details className="sources-cite">
                          <summary>
                            {m.chunks.length} source passage{m.chunks.length > 1 ? 's' : ''}
                          </summary>
                          {m.chunks.map((c, j) => (
                            <ChunkCitation key={j} chunk={c} index={j} />
                          ))}
                        </details>
                      )}
                    </div>
                  ))
                )}

                {asking && (
                  <div className="msg assistant">
                    <div className="bubble typing">
                      <span /><span /><span />
                    </div>
                  </div>
                )}
              </div>

              <form
                className="composer glass"
                onSubmit={e => { e.preventDefault(); ask() }}
              >
                <input
                  type="text"
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  placeholder={hasPaper ? 'Ask a question…' : 'Add a paper first'}
                  disabled={!hasPaper || asking}
                />
                <button type="submit" disabled={!hasPaper || asking || !question.trim()}>
                  →
                </button>
              </form>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
