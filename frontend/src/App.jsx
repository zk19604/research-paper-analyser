import { useEffect, useRef, useState } from 'react'
import Graph from './Graph'
import './App.css'

const API = 'http://localhost:8000'

const SUGGESTIONS = [
  'What is the main contribution of this paper?',
  'Summarise the methodology in a few sentences.',
  'What are the key findings or results?',
  'What are the limitations of this work?',
]

function App() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [source, setSource] = useState(null) // info about the indexed paper
  const [chunks, setChunks] = useState(0)
  const [messages, setMessages] = useState([]) // { role, text, chunks? }
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [error, setError] = useState('')
  const [view, setView] = useState('chat') // 'chat' | 'graph'

  const fileInputRef = useRef(null)
  const scrollRef = useRef(null)

  // On load, find out whether a paper is already indexed.
  useEffect(() => {
    refreshStatus()
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, asking])

  async function refreshStatus() {
    try {
      const res = await fetch(`${API}/status`)
      const data = await res.json()
      setChunks(data.chunks || 0)
      if (data.chunks > 0) {
        const s = await fetch(`${API}/sources`)
        const sj = await s.json()
        setSource(sj.source)
      }
    } catch {
      setError('Cannot reach the backend. Is the API running on :8000?')
    }
  }

  async function handleUpload(selected) {
    const f = selected || file
    if (!f) return
    setFile(f)
    setError('')
    setUploading(true)
    try {
      const body = new FormData()
      body.append('file', f)
      const res = await fetch(`${API}/upload`, { method: 'POST', body })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setChunks(data.chunks)
      setMessages([])
      await refreshStatus()
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  async function ask(raw) {
    const q = (raw ?? question).trim()
    if (!q || asking || !hasPaper) return
    setError('')
    setQuestion('')
    setMessages((m) => [...m, { role: 'user', text: q }])
    setAsking(true)
    try {
      const res = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Request failed')
      setMessages((m) => [...m, { role: 'assistant', text: data.answer, chunks: data.chunks }])
    } catch (err) {
      setMessages((m) => [...m, { role: 'assistant', text: `⚠️ ${err.message}`, error: true }])
    } finally {
      setAsking(false)
    }
  }

  const hasPaper = chunks > 0
  const paperName = source?.filename || source?.title || file?.name || 'Untitled paper'

  return (
    <div className="app">
      {/* Floating glass header */}
      <header className="topbar glass">
        <div className="brand">
          <span className="brand-mark">◆</span>
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
          {hasPaper ? `${chunks} chunks indexed` : 'no paper'}
        </div>
      </header>

      <div className="layout">
        {/* Sources sidebar */}
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
            onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
            hidden
          />

          <label
            className="dropzone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              const f = e.dataTransfer.files?.[0]
              if (f) handleUpload(f)
            }}
          >
            Drop a PDF here
          </label>

          <div className="source-list">
            {hasPaper ? (
              <div className="source-item active">
                <span className="source-icon">▤</span>
                <div className="source-meta">
                  <div className="source-title">{paperName}</div>
                  {source?.pages != null && <div className="source-sub">{source.pages} pages</div>}
                </div>
                <span className="source-check">✓</span>
              </div>
            ) : (
              <div className="source-empty">No sources yet. Add a PDF to begin.</div>
            )}
          </div>

          {error && <div className="error">{error}</div>}
        </aside>

        {/* Main panel */}
        <main className="main glass">
          {view === 'graph' ? (
            <Graph />
          ) : (
            <div className="chat">
              <div className="messages" ref={scrollRef}>
                {messages.length === 0 ? (
                  <div className="welcome">
                    <h1 className="welcome-title">{paperName}</h1>
                    <p className="welcome-sub">
                      {hasPaper
                        ? 'Ask anything grounded in this paper, or explore its knowledge graph.'
                        : 'Add a PDF source on the left, then ask questions about its content.'}
                    </p>

                    {hasPaper && (
                      <div className="suggest-grid">
                        {SUGGESTIONS.map((s) => (
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
                      {m.chunks?.length > 0 && (
                        <details className="sources-cite">
                          <summary>
                            {m.chunks.length} source passage{m.chunks.length > 1 ? 's' : ''}
                          </summary>
                          {m.chunks.map((c, j) => (
                            <blockquote key={j}>{typeof c === 'string' ? c : c.text}</blockquote>
                          ))}
                        </details>
                      )}
                    </div>
                  ))
                )}

                {asking && (
                  <div className="msg assistant">
                    <div className="bubble typing">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                )}
              </div>

              <form
                className="composer glass"
                onSubmit={(e) => {
                  e.preventDefault()
                  ask()
                }}
              >
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
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

export default App
