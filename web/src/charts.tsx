import { useEffect, useRef, useState, type ReactNode } from 'react'
import * as d3 from 'd3'
import { fmt } from './data'

export function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null)
  const [w, setW] = useState(480)
  useEffect(() => {
    const ro = new ResizeObserver((e) => setW(Math.max(240, Math.floor(e[0].contentRect.width))))
    if (ref.current) ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, w]
}

export function useTip() {
  const [tip, setTip] = useState<{ x: number; y: number; t: string } | null>(null)
  const node = tip && <div className="tt" style={{ left: tip.x + 12, top: tip.y + 12 }}>{tip.t}</div>
  return { show: (e: React.MouseEvent, t: string) => setTip({ x: e.clientX, y: e.clientY, t }), hide: () => setTip(null), node }
}

const M = { t: 10, r: 12, b: 38, l: 48 }

/** Histograma (opcionalmente com duas séries sobrepostas). */
export function Hist({ series, xLabel, height = 200, bins = 28, fmtX = (v: number) => fmt(v, 1) }: {
  series: { name: string; values: number[]; color: string }[]; xLabel: string; height?: number; bins?: number; fmtX?: (v: number) => string
}) {
  const [ref, w] = useWidth<HTMLDivElement>(); const tip = useTip()
  const all = series.flatMap((s) => s.values)
  if (!all.length) return <div ref={ref} style={{ height }} className="muted">Carregando…</div>
  const x = d3.scaleLinear().domain(d3.extent(all) as [number, number]).nice().range([M.l, w - M.r])
  const th = d3.ticks(x.domain()[0], x.domain()[1], bins)
  const hs = series.map((s) => d3.bin().domain(x.domain() as [number, number]).thresholds(th)(s.values))
  const ymax = d3.max(hs.flat(), (b) => b.length) ?? 1
  const y = d3.scaleLinear().domain([0, ymax]).nice().range([height - M.b, M.t])
  return (
    <div ref={ref} className="chart">
      <svg width={w} height={height} role="img" aria-label={`Histograma de ${xLabel}`}>
        <g className="grid">{y.ticks(4).map((t) => <line key={t} x1={M.l} x2={w - M.r} y1={y(t)} y2={y(t)} />)}</g>
        {hs.map((h, si) => h.map((b, i) => (
          <rect key={`${si}-${i}`} x={x(b.x0!) + 1} width={Math.max(0, x(b.x1!) - x(b.x0!) - 2)} y={y(b.length)} height={y(0) - y(b.length)}
            fill={series[si].color} opacity={series.length > 1 ? 0.6 : 0.9} rx={2}
            onMouseMove={(e) => tip.show(e, `${series[si].name}\n${fmtX(b.x0!)}–${fmtX(b.x1!)}: ${b.length} linhas`)} onMouseLeave={tip.hide} />
        )))}
        {x.ticks(6).map((t) => <text key={t} x={x(t)} y={height - M.b + 14} textAnchor="middle">{fmtX(t)}</text>)}
        {y.ticks(4).map((t) => <text key={t} x={M.l - 6} y={y(t) + 4} textAnchor="end">{t}</text>)}
        <text x={(M.l + w - M.r) / 2} y={height - 6} textAnchor="middle">{xLabel}</text>
      </svg>
      {series.length > 1 && <div className="ctrl" style={{ justifyContent: 'center' }}>{series.map((s) => <span key={s.name}><i style={{ display: 'inline-block', width: 12, height: 12, background: s.color, opacity: .7, borderRadius: 2, marginRight: 5 }} />{s.name}</span>)}</div>}
      {tip.node}
    </div>
  )
}

/** Barras horizontais com rótulo e valor. */
export function HBars({ items, color = 'var(--count)', fmtV = (v: number) => fmt(v, 2), max }: {
  items: { label: string; value: number; color?: string; note?: string; text?: string }[]; color?: string; fmtV?: (v: number) => string; max?: number
}) {
  const m = max ?? Math.max(...items.map((i) => i.value), 1e-9)
  return (
    <div role="table">
      {items.map((it) => (
        <div key={it.label} className="bar-row" style={{ gridTemplateColumns: '120px 1fr 70px' }} title={it.note}>
          <span>{it.label}</span>
          <div style={{ background: 'var(--bg2)', borderRadius: 4 }}><div className="b" style={{ width: `${Math.max(1, (it.value / m) * 100)}%`, background: it.color ?? color }} /></div>
          <b style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{it.text ?? fmtV(it.value)}</b>
        </div>
      ))}
    </div>
  )
}

/** Cascata (waterfall) em escala log: base + contribuições = total. */
export function Waterfall({ steps, base, baseLabel, totalLabel, height = 230 }: {
  steps: { label: string; value: number }[]; base: number; baseLabel: string; totalLabel: string; height?: number
}) {
  const [ref, w] = useWidth<HTMLDivElement>(); const tip = useTip()
  let acc = base
  const rows = [{ label: baseLabel, a: 0, b: base, kind: 'tot' as const }]
  steps.forEach((s) => { rows.push({ label: s.label, a: acc, b: acc + s.value, kind: s.value >= 0 ? 'up' as never : 'down' as never }); acc += s.value })
  rows.push({ label: totalLabel, a: 0, b: acc, kind: 'tot' })
  const lo = Math.min(0, ...rows.map((r) => Math.min(r.a, r.b))), hi = Math.max(...rows.map((r) => Math.max(r.a, r.b))) * 1.05
  const x = d3.scaleBand().domain(rows.map((r) => r.label)).range([M.l, w - M.r]).padding(0.25)
  const y = d3.scaleLinear().domain([lo, hi]).nice().range([height - M.b - 14, M.t])
  const col = (k: string) => (k === 'tot' ? 'var(--accent)' : k === 'up' ? 'var(--up)' : 'var(--down)')
  return (
    <div ref={ref} className="chart">
      <svg width={w} height={height} role="img" aria-label="Contribuição de cada grupo ao valor previsto em escala logarítmica">
        <g className="grid">{y.ticks(4).map((t) => <line key={t} x1={M.l} x2={w - M.r} y1={y(t)} y2={y(t)} />)}</g>
        {rows.map((r, i) => (
          <g key={i}>
            <rect x={x(r.label)!} width={x.bandwidth()} y={y(Math.max(r.a, r.b))} height={Math.max(2, Math.abs(y(r.a) - y(r.b)))} fill={col(r.kind)} rx={3}
              onMouseMove={(e) => tip.show(e, `${r.label}\n${r.kind === 'tot' ? 'log(1+y) = ' + r.b.toFixed(3) : (r.b - r.a >= 0 ? '+' : '') + (r.b - r.a).toFixed(3) + ' em log'}`)} onMouseLeave={tip.hide} />
            <text x={x(r.label)! + x.bandwidth() / 2} y={height - M.b + 2} textAnchor="middle" style={{ fontSize: 10 }}>{r.label.length > 13 ? r.label.slice(0, 12) + '…' : r.label}</text>
          </g>
        ))}
        {y.ticks(4).map((t) => <text key={t} x={M.l - 6} y={y(t) + 4} textAnchor="end">{t}</text>)}
        <text x={4} y={12} style={{ fontSize: 10 }}>log(1+y)</text>
      </svg>
      {tip.node}
    </div>
  )
}

/** Dispersão EB × CAR em log-log. */
export function Scatter({ pts, onPick, height = 380, xLabel, yLabel }: {
  pts: { x: number; y: number; id: string }[]; onPick?: (id: string) => void; height?: number; xLabel: string; yLabel: string
}) {
  const [ref, w] = useWidth<HTMLDivElement>(); const tip = useTip()
  const ext = d3.extent(pts.flatMap((p) => [p.x, p.y])) as [number, number]
  const lo = Math.max(1, ext[0]) * 0.8, hi = ext[1] * 1.2
  const s = Math.min(w, height + 40)
  const x = d3.scaleLog().domain([lo, hi]).range([M.l + 6, s - M.r])
  const y = d3.scaleLog().domain([lo, hi]).range([height - M.b, M.t])
  const ticks = [10, 100, 1000, 10000, 100000].filter((t) => t >= lo && t <= hi)
  return (
    <div ref={ref} className="chart">
      <svg width={s} height={height} role="img" aria-label="Dispersão das estimativas EB contra CAR em escala log-log">
        <g className="grid">{ticks.map((t) => <g key={t}><line x1={M.l + 6} x2={s - M.r} y1={y(t)} y2={y(t)} /><line y1={M.t} y2={height - M.b} x1={x(t)} x2={x(t)} /></g>)}</g>
        <line x1={x(lo)} y1={y(lo)} x2={x(hi)} y2={y(hi)} stroke="var(--ink2)" strokeDasharray="4 4" />
        {pts.map((p) => (
          <circle key={p.id} cx={x(Math.max(p.x, lo))} cy={y(Math.max(p.y, lo))} r={2.6} fill="var(--est)" opacity={0.5} stroke="var(--card)" strokeWidth={0.5}
            style={{ cursor: onPick ? 'pointer' : undefined }} onClick={() => onPick?.(p.id)}
            onMouseMove={(e) => tip.show(e, `${p.id}\nEB ${fmt(p.x)} · CAR ${fmt(p.y)}`)} onMouseLeave={tip.hide} />
        ))}
        {ticks.map((t) => <g key={t}><text x={x(t)} y={height - M.b + 14} textAnchor="middle">{fmt(t)}</text><text x={M.l} y={y(t) + 4} textAnchor="end">{fmt(t)}</text></g>)}
        <text x={(M.l + s) / 2} y={height - 6} textAnchor="middle">{xLabel}</text>
        <text transform={`translate(12 ${height / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
        <text x={x(hi) - 4} y={y(hi) + 12} textAnchor="end" style={{ fontSize: 10 }}>y = x</text>
      </svg>
      {tip.node}
    </div>
  )
}

export function Card({ title, children, note }: { title?: string; children: ReactNode; note?: string }) {
  return <div className="card">{title && <h3>{title}</h3>}{children}{note && <small>{note}</small>}</div>
}
