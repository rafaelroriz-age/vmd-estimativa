import { useEffect, useMemo, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { ClassData, Store } from './data'
import { fmt } from './data'
import { isDark, pal } from './theme'

export type Mode = 'eb' | 'car' | 'media' | 'diff'
export type Sent = 'ambos' | 'C' | 'D'
export const VMD = 'VMD'

export function segValue(store: Store, cd: ClassData | null, i: number, model: 'eb' | 'car', cls: string): number {
  if (cls === VMD || !cd) return store.segs[i][model]
  return cd[model][cls]?.[i] ?? 0
}

interface Props {
  store: Store; cd: ClassData | null; mode: Mode; cls: string; sent: Sent
  overlay?: GeoJSON.FeatureCollection | null
  view?: { center: [number, number]; zoom: number } | null
  selected?: string | null; onSelect?: (sre: string) => void
  basemap?: boolean; className?: string; interactive?: boolean; legend?: boolean
  roadFilter?: (sre: string, go: string) => boolean
}

const GO_BOUNDS: [[number, number], [number, number]] = [[-53.3, -19.5], [-45.9, -12.8]]
const tiles = (_dark: boolean) => ['https://tile.openstreetmap.org/{z}/{x}/{y}.png']

export default function MapView(p: Props) {
  const el = useRef<HTMLDivElement>(null)
  const map = useRef<maplibregl.Map | null>(null)
  const ready = useRef(false)
  const [dark, setDark] = useState(isDark())
  const [tip, setTip] = useState<{ x: number; y: number; html: string } | null>(null)
  const latest = useRef(p); latest.current = p

  useEffect(() => {
    const mq = matchMedia('(prefers-color-scheme: dark)')
    const h = () => setDark(mq.matches)
    mq.addEventListener('change', h)
    return () => mq.removeEventListener('change', h)
  }, [])

  const fc = useMemo(() => {
    const { store, cd, mode, cls, sent } = p
    const feats = store.roads.features.map((f) => {
      const pr = f.properties as { sre: string; go: string }
      const ent = store.bySre.get(pr.sre)
      const idxs = ent ? (sent === 'ambos' ? [ent.C, ent.D] : [ent[sent]]).filter((x): x is number => x != null) : []
      let o = 'n', v = 0, r = 0
      if (idxs.length && !(p.roadFilter && !p.roadFilter(pr.sre, pr.go))) {
        const eb = idxs.reduce((a, i) => a + segValue(store, cd, i, 'eb', cls), 0)
        const car = idxs.reduce((a, i) => a + segValue(store, cd, i, 'car', cls), 0)
        o = idxs.some((i) => store.segs[i].o === 'c') ? 'c' : 'e'
        v = mode === 'eb' ? eb : mode === 'car' ? car : (eb + car) / 2
        r = Math.log2((car + 1) / (eb + 1))
      }
      const w = o === 'n' ? 1 : 1.2 + Math.sqrt(Math.max(v, 0)) / (cls === VMD ? 16 : 6)
      return { ...f, properties: { ...f.properties, o, v: Math.round(v), w: Math.min(w, 9), r } }
    })
    return { type: 'FeatureCollection', features: feats } as GeoJSON.FeatureCollection
  }, [p.store, p.cd, p.mode, p.cls, p.sent, p.roadFilter])

  useEffect(() => {
    const m = new maplibregl.Map({
      container: el.current!, bounds: GO_BOUNDS, fitBoundsOptions: { padding: 16 },
      attributionControl: { compact: true }, interactive: p.interactive !== false,
      style: { version: 8, sources: {}, layers: [{ id: 'bg', type: 'background', paint: { 'background-color': pal(isDark()).surface } }] },
    })
    map.current = m; (window as unknown as { __maps: unknown[] }).__maps = [...((window as unknown as { __maps?: unknown[] }).__maps ?? []), m]; m.on('error', (e) => console.warn('MAPERR', e.error?.message))
    if (p.interactive !== false) m.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')
    m.on('load', () => {
      m.addSource('base', { type: 'raster', tileSize: 256, attribution: '© OpenStreetMap', tiles: tiles(isDark()) })
      m.addLayer({ id: 'base', type: 'raster', source: 'base', paint: { 'raster-opacity': 0.75, 'raster-saturation': -0.6 } })
      m.addSource('roads', { type: 'geojson', data: latest.current.store.roads })
      m.addLayer({ id: 'sel-halo', type: 'line', source: 'roads', filter: ['==', ['get', 'sre'], ''], layout: { 'line-cap': 'round' }, paint: { 'line-color': '#ffd23f', 'line-width': 14, 'line-opacity': 0.7 } })
      m.addLayer({ id: 'roads', type: 'line', source: 'roads', layout: { 'line-cap': 'round', 'line-join': 'round', 'line-sort-key': ['match', ['get', 'o'], 'c', 2, 'e', 1, 0] }, paint: {} })
      m.addSource('overlay', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      m.addLayer({ id: 'ov-line', type: 'line', source: 'overlay', filter: ['==', ['geometry-type'], 'LineString'], paint: { 'line-color': ['get', 'c'], 'line-width': ['get', 'w'], 'line-opacity': 0.85, 'line-dasharray': [2, 2] } })
      m.addLayer({ id: 'ov-pt', type: 'circle', source: 'overlay', filter: ['==', ['geometry-type'], 'Point'], paint: { 'circle-color': ['get', 'c'], 'circle-radius': ['get', 'r'], 'circle-stroke-width': 2, 'circle-stroke-color': pal(isDark()).surface } })
      ready.current = true
    })
    m.on('click', 'roads', (e) => {
      const sre = e.features?.[0]?.properties?.sre
      if (sre) latest.current.onSelect?.(sre)
    })
    m.on('mousemove', 'roads', (e) => {
      m.getCanvas().style.cursor = 'pointer'
      const pr = e.features?.[0]?.properties as { sre: string; v: number; o: string } | undefined
      if (pr) setTip({ x: e.point.x, y: e.point.y, html: `${pr.sre} · ${pr.o === 'c' ? 'contagem' : pr.o === 'e' ? 'estimativa' : 'sem estimativa'}${pr.o !== 'n' ? ` · ${fmt(pr.v)} veíc./dia` : ''}` })
    })
    m.on('mouseleave', 'roads', () => { m.getCanvas().style.cursor = ''; setTip(null) })
    return () => { m.remove(); map.current = null; ready.current = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const whenReady = (fn: (m: maplibregl.Map) => void) => {
    const m = map.current
    if (!m) return
    if (ready.current) fn(m)
    else m.once('load', () => fn(m))
  }

  useEffect(() => whenReady((m) => (m.getSource('roads') as maplibregl.GeoJSONSource).setData(fc)), [fc])
  useEffect(() => whenReady((m) => {
    const c = pal(dark)
    const col = p.mode === 'diff'
      ? ['match', ['get', 'o'], 'c', c.count, 'e', ['interpolate', ['linear'], ['get', 'r'], -1.5, c.down, 0, c.none, 1.5, c.up], c.none]
      : ['match', ['get', 'o'], 'c', c.count, 'e', c.est, c.none]
    m.setPaintProperty('roads', 'line-color', col as never)
    m.setPaintProperty('roads', 'line-opacity', ['match', ['get', 'o'], 'n', 0.55, 1] as never)
    m.setPaintProperty('roads', 'line-width', ['interpolate', ['linear'], ['zoom'], 5, ['*', ['get', 'w'], 0.55], 9, ['*', ['get', 'w'], 1.1], 12, ['*', ['get', 'w'], 1.8]] as never)
    m.setPaintProperty('bg', 'background-color', c.surface)
    m.setPaintProperty('ov-pt', 'circle-stroke-color', c.surface)
  }), [dark, p.mode])
  useEffect(() => whenReady((m) => {
    const src = m.getSource('base') as (maplibregl.RasterTileSource & { setTiles?: (t: string[]) => void }) | undefined
    src?.setTiles?.(tiles(dark))
    m.setLayoutProperty('base', 'visibility', p.basemap !== false ? 'visible' : 'none')
  }), [dark, p.basemap])
  useEffect(() => whenReady((m) => m.setFilter('sel-halo', ['==', ['get', 'sre'], p.selected ?? ''])), [p.selected])
  useEffect(() => whenReady((m) => (m.getSource('overlay') as maplibregl.GeoJSONSource).setData(p.overlay ?? { type: 'FeatureCollection', features: [] })), [p.overlay])
  useEffect(() => {
    if (p.view) whenReady((m) => m.easeTo({ center: p.view!.center, zoom: p.view!.zoom, duration: 900 }))
  }, [p.view])
  useEffect(() => {
    const m = map.current
    const ro = new ResizeObserver(() => m?.resize())
    ro.observe(el.current!)
    return () => ro.disconnect()
  }, [])

  const c = pal(dark)
  return (
    <div className={`mapwrap ${p.className ?? ''}`}>
      <div ref={el} className="map" role="img" aria-label="Mapa das SREs de Goiás: azul contagem, vermelho estimativa" />
      {tip && <div className="maptip" style={{ left: tip.x + 12, top: tip.y + 12 }}>{tip.html}</div>}
      {p.legend !== false && (
        <div className="legend" aria-label="Legenda">
          <span><i style={{ background: c.count }} /> Contagem</span>
          <span><i style={{ background: c.est }} /> Estimativa</span>
          <span><i style={{ background: c.none }} /> Fora do escopo</span>
          <span className="muted">espessura ∝ √VMD</span>
          {p.mode === 'diff' && <span><i style={{ background: `linear-gradient(90deg,${c.down},${c.none},${c.up})`, width: 54 }} /> CAR menor ← → CAR maior</span>}
        </div>
      )}
    </div>
  )
}
