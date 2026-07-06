import { useEffect, useMemo, useRef, useState } from 'react'
import { apiFetch, apiGetPending, apiApproveNode, apiRejectNode } from './api'

const NODE_W  = 172
const NODE_H  = 54
const COL_GAP = 260
const ROW_GAP = 80

// Pantone palette from the brief — winery / captain's blue / olive.
const INK = '#17130f'
const TYPE_COLOR = {
  root:      '#6b2737', // winery
  theme:     '#6b2737',
  concept:   '#5c7290', // captain's blue
  problem:   '#686f12', // olive
  component: '#5c7290',
  method:    '#6b2737',
  result:    '#686f12',
  finding:   '#686f12',
  dataset:   '#5c7290',
  metric:    '#686f12',
  model:     '#6b2737',
  task:      '#5c7290',
  baseline:  '#5c7290',
  paper:     '#17130f', // ink — a whole paper in the merged view
}

const TYPE_LABEL = {
  root:      'Topic',
  theme:     'Theme',
  concept:   'Concept',
  problem:   'Problem',
  component: 'Component',
  method:    'Method',
  result:    'Result',
  finding:   'Finding',
  dataset:   'Dataset',
  metric:    'Metric',
  model:     'Model',
  task:      'Task',
  baseline:  'Baseline',
  paper:     'Paper',
}

function layout(nodes, edges) {
  const byLevel = new Map()
  for (const n of nodes) {
    const lvl = n.level ?? 0
    if (!byLevel.has(lvl)) byLevel.set(lvl, [])
    byLevel.get(lvl).push(n)
  }
  const pos    = new Map()
  const levels = [...byLevel.keys()].sort((a, b) => a - b)
  let maxRows  = 0
  for (const lvl of levels) maxRows = Math.max(maxRows, byLevel.get(lvl).length)

  const height = maxRows * (NODE_H + ROW_GAP)
  for (const lvl of levels) {
    const col       = byLevel.get(lvl)
    const colHeight = col.length * (NODE_H + ROW_GAP)
    const offset    = (height - colHeight) / 2
    col.forEach((n, i) => {
      pos.set(n.id, { x: 48 + lvl * COL_GAP, y: offset + i * (NODE_H + ROW_GAP) + 24 })
    })
  }
  const width      = 96 + (levels.length ? Math.max(...levels) : 0) * COL_GAP + NODE_W
  const drawnEdges = edges
    .filter(e => pos.has(e.source) && pos.has(e.target))
    .map(e => ({ ...e, a: pos.get(e.source), b: pos.get(e.target) }))

  return { pos, width: Math.max(width, 400), height: Math.max(height + 48, 300), drawnEdges }
}

const MIN_SCALE = 0.25
const MAX_SCALE = 2.5

// ── Pending Review Panel ──────────────────────────────────────────────────────
function PendingPanel({ paperId, onClose, onChanged }) {
  const [items,   setItems]   = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    apiGetPending(paperId).then(setItems).finally(() => setLoading(false))
  }, [paperId])

  async function approve(nodeId) {
    await apiApproveNode(paperId, nodeId)
    setItems(prev => prev.filter(n => n.id !== nodeId))
    onChanged?.()
  }
  async function reject(nodeId) {
    await apiRejectNode(paperId, nodeId)
    setItems(prev => prev.filter(n => n.id !== nodeId))
  }

  return (
    <div className="pending-panel glass">
      <div className="pending-head">
        <span>🔍 Pending Review</span>
        <button className="close" onClick={onClose}>×</button>
      </div>
      {loading ? (
        <p className="pending-empty">Loading…</p>
      ) : items.length === 0 ? (
        <p className="pending-empty">No nodes awaiting review.</p>
      ) : (
        <div className="pending-list">
          {items.map(n => (
            <div key={n.id} className="pending-item">
              <div className="pending-label">{n.label}</div>
              <div className="pending-type">{TYPE_LABEL[n.type] || n.type}</div>
              {n.summary && <p className="pending-summary">{n.summary}</p>}
              <div className="pending-conf">
                Confidence: <b>{Math.round((n.confidence ?? 0) * 100)}%</b>
              </div>
              <div className="pending-actions">
                <button className="btn-approve" onClick={() => approve(n.id)}>✓ Approve</button>
                <button className="btn-reject"  onClick={() => reject(n.id)}>✕ Reject</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main Graph Component ──────────────────────────────────────────────────────
export default function Graph({ papers = [], selectedPaperId, onSelectPaper }) {
  const [data,        setData]        = useState(null)
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState('')
  const [selected,    setSelected]    = useState(null)   // selected node
  const [paperFilter, setPaperFilter] = useState(selectedPaperId || null)
  const [showPending, setShowPending] = useState(false)
  const [pendingCount, setPendingCount] = useState(0)
  const [expand,      setExpand]      = useState(null)   // {loading, text, error} for selected node

  // Clear the on-demand explanation whenever a different node is selected.
  useEffect(() => { setExpand(null) }, [selected?.id])

  // Fetch a deeper, paper-grounded explanation of the selected node via /ask.
  async function explainNode() {
    if (!selected) return
    const pid = selected.paper_ids?.[0] || selected.provenance?.paper_id || paperFilter || null
    setExpand({ loading: true })
    try {
      const res = await apiFetch('/ask', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: `Explain "${selected.label}" in detail based on this paper: `
                  + `what it is, how it works here, and why it matters.`,
          paper_id: pid,
        }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Failed to expand node')
      setExpand({ loading: false, text: data.answer })
    } catch (err) {
      setExpand({ loading: false, error: err.message })
    }
  }

  const [view,   setView]   = useState({ scale: 1, tx: 0, ty: 0 })
  const canvasRef            = useRef(null)
  const pan                  = useRef(null)

  // Sync paper filter with parent selection.
  useEffect(() => {
    setPaperFilter(selectedPaperId)
  }, [selectedPaperId])

  // Load graph whenever the paper filter changes.
  useEffect(() => {
    load()
  }, [paperFilter])

  // Fetch pending count whenever data changes.
  useEffect(() => {
    const pid = data?.paper_id || paperFilter
    if (pid) {
      apiGetPending(pid).then(items => setPendingCount(items.length)).catch(() => {})
    }
  }, [data])

  function clampScale(s) { return Math.min(MAX_SCALE, Math.max(MIN_SCALE, s)) }

  function zoomBy(factor) {
    const rect = canvasRef.current?.getBoundingClientRect()
    const cx   = rect ? rect.width  / 2 : 0
    const cy   = rect ? rect.height / 2 : 0
    setView(v => {
      const scale = clampScale(v.scale * factor)
      const k     = scale / v.scale
      return { scale, tx: cx - (cx - v.tx) * k, ty: cy - (cy - v.ty) * k }
    })
  }

  function resetView() { setView({ scale: 1, tx: 0, ty: 0 }) }

  function onWheel(e) {
    e.preventDefault()
    const rect = canvasRef.current.getBoundingClientRect()
    const px   = e.clientX - rect.left
    const py   = e.clientY - rect.top
    setView(v => {
      const scale = clampScale(v.scale * (e.deltaY < 0 ? 1.12 : 0.89))
      const k     = scale / v.scale
      return { scale, tx: px - (px - v.tx) * k, ty: py - (py - v.ty) * k }
    })
  }

  function onPointerDown(e) {
    if (e.button !== 0) return
    pan.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty }
  }
  function onPointerMove(e) {
    if (!pan.current) return
    setView(v => ({
      ...v,
      tx: pan.current.tx + (e.clientX - pan.current.x),
      ty: pan.current.ty + (e.clientY - pan.current.y),
    }))
  }
  function onPointerUp() { pan.current = null }

  async function load(refresh = false) {
    setLoading(true)
    setError('')
    setSelected(null)
    try {
      const params = new URLSearchParams()
      // No paper selected + multiple papers → merged cross-paper graph.
      const merged = !paperFilter && papers.length > 1
      if (merged)      params.set('merged', 'true')
      if (paperFilter) params.set('paper_id', paperFilter)
      if (refresh)     params.set('refresh', 'true')
      const res  = await apiFetch(`/graph?${params}`)
      const json = await res.json()
      if (!res.ok) throw new Error(json.detail || 'Failed to build graph')
      setData(json)
      resetView()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const { pos, width, height, drawnEdges } = useMemo(
    () => data ? layout(data.nodes, data.edges)
               : { pos: new Map(), width: 0, height: 0, drawnEdges: [] },
    [data]
  )

  // Only hierarchy edges in the main view (keeps it clean).
  const edges = useMemo(
    () => drawnEdges.filter(e => e.kind !== 'relation'),
    [drawnEdges]
  )

  if (loading) {
    return (
      <div className="graph-state">
        <div className="bubble typing"><span /><span /><span /></div>
        <p>Extracting knowledge graph…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="graph-state">
        <p className="graph-error">⚠️ {error}</p>
        <button onClick={() => load()}>Retry</button>
      </div>
    )
  }

  if (!data) return null

  const types = [...new Set(data.nodes.map(n => n.type))]

  return (
    <div className="graph-wrap">
      {/* ── Toolbar ── */}
      <div className="graph-toolbar glass">
        <div className="graph-topic">{data.topic || 'Knowledge graph'}</div>

        {/* Paper selector */}
        {papers.length > 1 && (
          <select
            className="paper-select"
            value={paperFilter || ''}
            onChange={e => {
              const val = e.target.value || null
              setPaperFilter(val)
              onSelectPaper?.(val)
            }}
          >
            <option value="">All papers (merged)</option>
            {papers.map(p => (
              <option key={p.paper_id} value={p.paper_id}>
                {(p.title || p.filename).slice(0, 30)}
              </option>
            ))}
          </select>
        )}

        <div className="graph-legend">
          {types.map(t => (
            <span key={t} className="legend-item">
              <span className="legend-dot" style={{ background: TYPE_COLOR[t] || '#34d399' }} />
              {TYPE_LABEL[t] || t}
            </span>
          ))}
        </div>

        <div className="graph-toolbar-actions">
          {/* Pending review badge */}
          {pendingCount > 0 && (
            <button
              className="pending-btn"
              onClick={() => setShowPending(v => !v)}
              title="Nodes awaiting review"
            >
              🔍 {pendingCount} pending
            </button>
          )}
          <button className="ghost" onClick={() => load(true)}>↻ Rebuild</button>
        </div>
      </div>

      {/* ── Canvas ── */}
      <div
        className="graph-canvas"
        ref={canvasRef}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        <svg width="100%" height="100%">
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
              <path d="M0,0 L7,3 L0,6 Z" fill={INK} />
            </marker>
            {/* Hand-drawn wobble: displace straight edges/boxes by a turbulence
                field. Applied to shapes only, never to text, so labels stay
                crisp and readable. */}
            <filter id="rough" x="-20%" y="-20%" width="140%" height="140%">
              <feTurbulence type="fractalNoise" baseFrequency="0.018" numOctaves="2" seed="7" result="noise" />
              <feDisplacementMap in="SourceGraphic" in2="noise" scale="4" xChannelSelector="R" yChannelSelector="G" />
            </filter>
          </defs>

          <g
            transform={`translate(${view.tx},${view.ty}) scale(${view.scale})`}
            style={{ willChange: 'transform' }}
          >
            {edges.map((e, i) => {
              const x1     = e.a.x + NODE_W
              const y1     = e.a.y + NODE_H / 2
              const x2     = e.b.x
              const y2     = e.b.y + NODE_H / 2
              const mx     = (x1 + x2) / 2
              const active = selected && (selected.id === e.source || selected.id === e.target)
              return (
                <path
                  key={i}
                  d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
                  fill="none"
                  stroke={active ? '#6b2737' : INK}
                  strokeWidth={active ? 3 : 1.8}
                  strokeLinecap="round"
                  markerEnd="url(#arrow)"
                  filter="url(#rough)"
                />
              )
            })}

            {data.nodes.map(n => {
              const p      = pos.get(n.id)
              if (!p) return null
              const color  = TYPE_COLOR[n.type] || TYPE_COLOR.concept
              const active = selected?.id === n.id
              // Confidence opacity: confirmed nodes = 1.0, lower conf = more transparent
              const conf   = n.confidence ?? 1.0
              const opac   = Math.max(0.45, conf)

              return (
                <g
                  key={n.id}
                  transform={`translate(${p.x},${p.y})`}
                  className="graph-node"
                  onClick={() => setSelected(n)}
                  style={{ opacity: opac }}
                >
                  <rect
                    width={NODE_W}
                    height={NODE_H}
                    rx="2"
                    fill="#f3eddf"
                    stroke={INK}
                    strokeWidth={active ? 3.4 : 2.4}
                    filter="url(#rough)"
                  />
                  <rect width="10" height={NODE_H} fill={color} filter="url(#rough)" />
                  {active && (
                    <rect
                      width={NODE_W}
                      height={NODE_H}
                      rx="2"
                      fill="none"
                      stroke={color}
                      strokeWidth="2"
                      strokeDasharray="5 4"
                      filter="url(#rough)"
                    />
                  )}
                  {/* Confidence indicator dot */}
                  <circle
                    cx={NODE_W - 16}
                    cy={16}
                    r="4"
                    fill={conf >= 0.8 ? '#686f12' : conf >= 0.65 ? '#5c7290' : '#6b2737'}
                    stroke={INK}
                    strokeWidth="1.5"
                    title={`Confidence: ${Math.round(conf * 100)}%`}
                  />
                  <text x={NODE_W / 2} y={NODE_H / 2 + 5} textAnchor="middle" className="node-label">
                    {n.label.length > 20 ? n.label.slice(0, 19) + '…' : n.label}
                  </text>
                </g>
              )
            })}
          </g>
        </svg>

        {/* ── Zoom controls ── */}
        <div className="graph-controls glass">
          <button onClick={() => zoomBy(1.2)} title="Zoom in">＋</button>
          <span className="zoom-level">{Math.round(view.scale * 100)}%</span>
          <button onClick={() => zoomBy(0.8)} title="Zoom out">－</button>
          <button onClick={resetView} title="Reset view">⤢</button>
        </div>
      </div>

      {/* ── Node detail panel ── */}
      {selected && (
        <div className="graph-detail glass">
          <button className="close" onClick={() => setSelected(null)}>×</button>
          <span
            className="detail-type"
            style={{ color: TYPE_COLOR[selected.type] || TYPE_COLOR.concept }}
          >
            {TYPE_LABEL[selected.type] || selected.type}
          </span>
          <h3>{selected.label}</h3>
          {selected.paper_ids?.length > 1 && (
            <div className="detail-shared">
              🔗 Shared across {selected.paper_ids.length} papers
            </div>
          )}
          {selected.summary && <p>{selected.summary}</p>}
          {selected.facts?.length > 0 && (
            <ul>{selected.facts.map((f, i) => <li key={i}>{f}</li>)}</ul>
          )}
          <div className="detail-conf">
            Confidence: <b>{Math.round((selected.confidence ?? 1) * 100)}%</b>
          </div>

          {/* On-demand deeper, paper-grounded explanation */}
          {!expand && (
            <button className="detail-expand" onClick={explainNode}>
              ＋ Explain in more detail
            </button>
          )}
          {expand?.loading && <p className="detail-expanding">Reading the paper…</p>}
          {expand?.error && <p className="graph-error">⚠️ {expand.error}</p>}
          {expand?.text && (
            <div className="detail-more">
              <div className="detail-more-head">Detailed explanation</div>
              <p>{expand.text}</p>
            </div>
          )}
          {selected.provenance && (
            <div className="detail-prov">
              Source: paper {selected.provenance.paper_id} ·
              model {selected.provenance.model?.split('-').slice(0, 2).join('-')} ·
              v{selected.provenance.version}
            </div>
          )}
        </div>
      )}

      {/* ── Pending review panel ── */}
      {showPending && (data?.paper_id || paperFilter) && (
        <PendingPanel
          paperId={data?.paper_id || paperFilter}
          onClose={() => setShowPending(false)}
          onChanged={() => {
            setPendingCount(c => Math.max(0, c - 1))
            load(true)
          }}
        />
      )}
    </div>
  )
}
