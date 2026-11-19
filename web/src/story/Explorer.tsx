import { useMemo, useState } from 'react'
import MapView, { VMD, type Mode, type Sent } from '../MapView'
import { fmt, type ClassData, type Store } from '../data'
import { roadCentroid } from './Models'

interface P { store: Store; cd: ClassData | null; selected: string | null; onSelect: (sre: string | null) => void }

const MODES: [Mode, string][] = [['eb', 'EB'], ['car', 'CAR'], ['media', 'Média'], ['diff', 'Diferença']]
const SENTS: [Sent, string][] = [['ambos', 'Ambos'], ['C', 'Crescente'], ['D', 'Decrescente']]

export default function Explorer({ store, cd, selected, onSelect }: P) {
  const [mode, setMode] = useState<Mode>('media')
  const [cls, setCls] = useState(VMD)
  const [sent, setSent] = useState<Sent>('ambos')
  const [go, setGo] = useState('')
  const [q, setQ] = useState('')
  const [basemap, setBasemap] = useState(false)
  const gos = useMemo(() => [...new Set(store.segs.map((s) => s.go))].sort(), [store])
  const filter = useMemo(() => (go ? (_: string, g: string) => g.replace(/\D/g, '') === go.replace(/\D/g, '') : undefined), [go])
  const view = useMemo(() => {
    if (!selected) return null
    const c = roadCentroid(store, selected)
    return c ? { center: c, zoom: 10 } : null
  }, [selected, store])
  const search = (v: string) => {
    setQ(v)
    const s = v.trim().toUpperCase()
    if (store.bySre.has(s)) onSelect(s)
  }
  return (
    <section className="sec wrap" id="explorador">
      <div className="kicker">7 · Explorador</div>
      <h2>Mapa interativo das SREs</h2>
      <p className="lead">Azul: contagem. Vermelho: estimativa. Cinza: fora do escopo dos modelos. Clique em uma linha para ver os dois sentidos, as duas estimativas e a composição por classe.</p>
      <div className="explorer">
        <div className="top ctrl">
          <div className="seg" role="group" aria-label="Modelo">{MODES.map(([k, l]) => <button key={k} className={mode === k ? 'on' : ''} onClick={() => setMode(k)}>{l}</button>)}</div>
          <label>Classe <select value={cls} onChange={(e) => setCls(e.target.value)}><option value={VMD}>VMD total</option>{cd?.order.filter((t) => t !== '3s1').map((t) => <option key={t}>{t}</option>)}</select></label>
          <div className="seg" role="group" aria-label="Sentido">{SENTS.map(([k, l]) => <button key={k} className={sent === k ? 'on' : ''} onClick={() => setSent(k)}>{l}</button>)}</div>
          <label>Rodovia <select value={go} onChange={(e) => setGo(e.target.value)}><option value="">Todas</option>{gos.map((g) => <option key={g}>{g}</option>)}</select></label>
          <input type="search" list="sres" placeholder="Buscar SRE (ex.: 010EGO0010)" value={q} onChange={(e) => search(e.target.value)} aria-label="Buscar SRE" />
          <datalist id="sres">{[...store.bySre.keys()].slice(0, 3000).map((s) => <option key={s} value={s} />)}</datalist>
          <label><input type="checkbox" checked={basemap} onChange={(e) => setBasemap(e.target.checked)} /> Mapa base OSM (online)</label>
        </div>
        <div className="mapcol"><MapView store={store} cd={cd} mode={mode} cls={cls} sent={sent} className="h-explorer" selected={selected} onSelect={onSelect} view={view} basemap={basemap} roadFilter={filter} /></div>
        <aside className="side" aria-live="polite">
          {selected ? <Detail store={store} cd={cd} sre={selected} cls={cls} /> : <p className="muted">Selecione uma SRE no mapa.</p>}
        </aside>
      </div>
    </section>
  )
}

function Detail({ store, cd, sre, cls }: { store: Store; cd: ClassData | null; sre: string; cls: string }) {
  const ent = store.bySre.get(sre)
  const idx = [ent?.C, ent?.D].filter((x): x is number => x != null)
  if (!idx.length) {
    const f = store.roads.features.find((x) => (x.properties as { sre: string }).sre === sre)?.properties as { go: string; sit: string } | undefined
    return <div className="detail"><h3>{sre}</h3><p className="muted">Fora do escopo dos modelos{f ? ` (situação ${f.sit})` : ''}: não há contagem nem estimativa.</p></div>
  }
  const s0 = store.segs[idx[0]]
  const val = (i: number, m: 'eb' | 'car') => (cls === VMD || !cd ? store.segs[i][m] : cd[m][cls]?.[i] ?? 0)
  return (
    <div className="detail">
      <h3>{sre}</h3>
      <p style={{ margin: 0 }}><b>{s0.go}</b> · {s0.tr ?? '—'} · km {fmt(s0.ki, 1)}–{fmt(s0.kf, 1)} ({fmt(s0.ext, 1)} km)</p>
      <p className="muted" style={{ marginTop: 2 }}>{s0.ini ?? ''} → {s0.fim ?? ''}</p>
      {idx.map((i) => {
        const s = store.segs[i]
        const lo = cd && cls !== VMD ? cd.lo[cls]?.[i] : null, hi = cd && cls !== VMD ? cd.hi[cls]?.[i] : null
        const top = cd ? cd.order.map((t) => ({ t, v: s.o === 'c' || true ? (cd.car[t][i] ?? 0) : 0, e: cd.eb[t][i] ?? 0 })).sort((a, b) => b.v - a.v).slice(0, 8) : []
        const mx = Math.max(...top.map((x) => Math.max(x.v, x.e)), 1)
        return (
          <div key={i} className="card" style={{ margin: '10px 0', padding: 12 }}>
            <div><b>{s.s === 'C' ? 'Crescente' : 'Decrescente'}</b> <span className={`pill ${s.o}`}>{s.o === 'c' ? 'contagem' : 'estimativa'}</span></div>
            <table style={{ marginTop: 6 }}><tbody>
              <tr><td>{cls === VMD ? 'VMD total' : cls} · EB</td><td className="num"><b>{fmt(val(i, 'eb'))}</b></td></tr>
              <tr><td>{cls === VMD ? 'VMD total' : cls} · CAR</td><td className="num"><b>{fmt(val(i, 'car'))}</b></td></tr>
              {lo != null && hi != null && <tr><td>CAR IC 90%</td><td className="num">{fmt(lo)} – {fmt(hi)}</td></tr>}
            </tbody></table>
            {s.o === 'e' && <small>Contagens vizinhas e features determinam a estimativa; veja as seções EB e CAR.</small>}
            <h4 style={{ margin: '10px 0 4px', fontSize: '.78rem' }} className="muted">Composição (CAR · barra) e EB (marca)</h4>
            {top.map((x) => (
              <div key={x.t} className={`bar-row ${s.o}`}><span>{x.t}</span>
                <div style={{ position: 'relative', background: 'var(--bg2)', borderRadius: 4, height: 8 }}>
                  <div className="b" style={{ width: `${(x.v / mx) * 100}%` }} />
                  <div style={{ position: 'absolute', top: -2, bottom: -2, left: `${(x.e / mx) * 100}%`, width: 2, background: 'var(--ink)' }} />
                </div><b style={{ textAlign: 'right' }}>{fmt(x.v)}</b></div>))}
          </div>)
      })}
      <h4 style={{ margin: '12px 0 4px' }}>Features do trecho</h4>
      <table><tbody>
        <tr><td>Classe funcional</td><td>{s0.cl}</td></tr><tr><td>Situação</td><td>{s0.sit}</td></tr><tr><td>Perímetro urbano</td><td>{s0.pu ? 'sim' : 'não'}</td></tr>
        <tr><td>Regional</td><td>{s0.reg}</td></tr><tr><td>Empresas ativas</td><td>{fmt(s0.emp)}</td></tr><tr><td>População</td><td>{fmt(s0.pop)}</td></tr>
        <tr><td>PIB</td><td>{fmt(s0.pib)}</td></tr><tr><td>Frota</td><td>{fmt(s0.fro)}</td></tr><tr><td>Distância ao polo (m)</td><td>{fmt(s0.dist)}</td></tr>
      </tbody></table>
    </div>
  )
}
