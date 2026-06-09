import { useEffect, useMemo, useRef, useState } from 'react'

const API = 'http://localhost:8000'

const NODE_W = 168
const NODE_H = 52
const COL_GAP = 250 // horizontal gap between levels
const ROW_GAP = 78 // vertical gap between siblings

// Distinct, solid colours per node type — cyan/green/white family.
const TYPE_COLOR = {
  root: '#ffffff',
  theme: '#22d3ee', // cyan
  concept: '#34d399', // green
  problem: '#5eead4', // teal
  component: '#a7f3d0', // mint
  method: '#22d3ee',
  result: '#34d399',
  dataset: '#5eead4',
}

const TYPE_LABEL = {
  root: 'Topic',
  theme: 'Theme',
  concept: 'Concept',
  problem: 'Entity',
  component: 'Component',
}

// Lay nodes out in columns by level, edges as bezier curves. No deps.
function layout(nodes, edges) {
  const byLevel = new Map()
  for (const n of nodes) {
    const lvl = n.level ?? 0
    if (!byLevel.has(lvl)) byLevel.set(lvl, [])
    byLevel.get(lvl).push(n)
  }

  const pos = new Map()
  const levels = [...byLevel.keys()].sort((a, b) => a - b)
  let maxRows = 0
  for (const lvl of levels) maxRows = Math.max(maxRows, byLevel.get(lvl).length)

  const height = maxRows * (NODE_H + ROW_GAP)
  for (const lvl of levels) {
    const col = byLevel.get(lvl)
    const colHeight = col.length * (NODE_H + ROW_GAP)
    const offset = (height - colHeight) / 2
    col.forEach((n, i) => {
      pos.set(n.id, {
        x: 48 + lvl * COL_GAP,
        y: offset + i * (NODE_H + ROW_GAP) + 24,
      })
    })
  }

  const width = 96 + (levels.length ? Math.max(...levels) : 0) * COL_GAP + NODE_W
  const drawnEdges = edges
    .filter((e) => pos.has(e.source) && pos.has(e.target))
    .map((e) => ({ ...e, a: pos.get(e.source), b: pos.get(e.target) }))

  return { pos, width: Math.max(width, 400), height: Math.max(height + 48, 300), drawnEdges }
}

const MIN_SCALE = 0.3
const MAX_SCALE = 2.5

export default function Graph() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(null)

  // Pan & zoom viewport over the SVG.
  const [view, setView] = useState({ scale: 1, tx: 0, ty: 0 })
  const canvasRef = useRef(null)
  const pan = useRef(null) // { x, y, tx, ty } while dragging

  function clampScale(s) {
    return Math.min(MAX_SCALE, Math.max(MIN_SCALE, s))
  }

  function zoomBy(factor) {
    // Zoom around the centre of the visible canvas.
    const rect = canvasRef.current?.getBoundingClientRect()
    const cx = rect ? rect.width / 2 : 0
    const cy = rect ? rect.height / 2 : 0
    setView((v) => {
      const scale = clampScale(v.scale * factor)
      const k = scale / v.scale
      return { scale, tx: cx - (cx - v.tx) * k, ty: cy - (cy - v.ty) * k }
    })
  }

  function resetView() {
    setView({ scale: 1, tx: 0, ty: 0 })
  }

  function onWheel(e) {
    e.preventDefault()
    const rect = canvasRef.current.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    setView((v) => {
      const scale = clampScale(v.scale * (e.deltaY < 0 ? 1.12 : 0.89))
      const k = scale / v.scale
      return { scale, tx: px - (px - v.tx) * k, ty: py - (py - v.ty) * k }
    })
  }

  function onPointerDown(e) {
    if (e.button !== 0) return
    pan.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty }
  }

  function onPointerMove(e) {
    if (!pan.current) return
    setView((v) => ({
      ...v,
      tx: pan.current.tx + (e.clientX - pan.current.x),
      ty: pan.current.ty + (e.clientY - pan.current.y),
    }))
  }

  function onPointerUp() {
    pan.current = null
  }

  async function load(refresh = false) {
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API}/graph${refresh ? '?refresh=true' : ''}`)
      const json = await res.json()
      if (!res.ok) throw new Error(json.detail || 'Failed to build graph')
      setData(json)
      setSelected(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  // Re-centre the viewport each time a fresh graph arrives.
  useEffect(() => {
    resetView()
  }, [data])

  const { pos, width, height, drawnEdges } = useMemo(
    () =>
      data
        ? layout(data.nodes, data.edges)
        : { pos: new Map(), width: 0, height: 0, drawnEdges: [] },
    [data]
  )

  // Only show hierarchy edges to keep the graph clean (relations duplicate them).
  const edges = useMemo(() => drawnEdges.filter((e) => e.kind !== 'relation'), [drawnEdges])

  if (loading) {
    return (
      <div className="graph-state">
        <div className="bubble typing"><span></span><span></span><span></span></div>
        <p>Extracting the knowledge graph…</p>
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

  const types = [...new Set(data.nodes.map((n) => n.type))]

  return (
    <div className="graph-wrap">
      <div className="graph-toolbar glass">
        <div className="graph-topic">{data.topic || 'Knowledge graph'}</div>
        <div className="graph-legend">
          {types.map((t) => (
            <span key={t} className="legend-item">
              <span className="legend-dot" style={{ background: TYPE_COLOR[t] || '#34d399' }} />
              {TYPE_LABEL[t] || t}
            </span>
          ))}
        </div>
        <button className="ghost" onClick={() => load(true)}>↻ Rebuild</button>
      </div>

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
            <marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
              <path d="M0,0 L7,3 L0,6 Z" fill="#1f6f6a" />
            </marker>
          </defs>

          <g
            transform={`translate(${view.tx},${view.ty}) scale(${view.scale})`}
            style={{ willChange: 'transform' }}
          >
          {edges.map((e, i) => {
            const x1 = e.a.x + NODE_W
            const y1 = e.a.y + NODE_H / 2
            const x2 = e.b.x
            const y2 = e.b.y + NODE_H / 2
            const mx = (x1 + x2) / 2
            const active = selected && (selected.id === e.source || selected.id === e.target)
            return (
              <path
                key={i}
                d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
                fill="none"
                stroke={active ? '#22d3ee' : '#1f6f6a'}
                strokeWidth={active ? 2.2 : 1.4}
                markerEnd="url(#arrow)"
              />
            )
          })}

          {data.nodes.map((n) => {
            const p = pos.get(n.id)
            if (!p) return null
            const color = TYPE_COLOR[n.type] || TYPE_COLOR.concept
            const active = selected?.id === n.id
            return (
              <g
                key={n.id}
                transform={`translate(${p.x},${p.y})`}
                className="graph-node"
                onClick={() => setSelected(n)}
              >
                <rect
                  width={NODE_W}
                  height={NODE_H}
                  rx="14"
                  fill="rgba(10, 20, 22, 0.72)"
                  stroke={active ? '#ffffff' : color}
                  strokeWidth={active ? 2.6 : 1.6}
                />
                <rect width="6" height={NODE_H} rx="3" fill={color} />
                <circle cx={NODE_W - 16} cy={16} r="3.5" fill={color} />
                <text x={NODE_W / 2} y={NODE_H / 2 + 5} textAnchor="middle" className="node-label">
                  {n.label.length > 20 ? n.label.slice(0, 19) + '…' : n.label}
                </text>
              </g>
            )
          })}
          </g>
        </svg>

        <div className="graph-controls glass">
          <button onClick={() => zoomBy(1.2)} title="Zoom in">＋</button>
          <span className="zoom-level">{Math.round(view.scale * 100)}%</span>
          <button onClick={() => zoomBy(0.8)} title="Zoom out">－</button>
          <button onClick={resetView} title="Reset view">⤢</button>
        </div>
      </div>

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
          {selected.summary && <p>{selected.summary}</p>}
          {selected.facts?.length > 0 && (
            <ul>
              {selected.facts.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
