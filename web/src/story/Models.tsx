import { useMemo, useState } from 'react'
import * as d3 from 'd3'
import MapView from '../MapView'
import Scrolly from '../Scrolly'
import { Tex } from '../Tex'
import { Hist, HBars, Waterfall, Card, useWidth } from '../charts'
import { fmt, nf0, type ClassData, type Store, type Walk } from '../data'
import { pal, isDark } from '../theme'

interface P { store: Store; cd: ClassData | null }
type FC = GeoJSON.FeatureCollection

export function roadCentroid(store: Store, sre: string): [number, number] | null {
  const f = store.roads.features.find((x) => (x.properties as { sre: string }).sre === sre)
  if (!f) return null
  const g = f.geometry as GeoJSON.LineString | GeoJSON.MultiLineString
  const pts = g.type === 'LineString' ? g.coordinates : g.coordinates.flat()
  const p = pts[Math.floor(pts.length / 2)]
  return [p[0], p[1]]
}
const pt = (c: [number, number], col: string, r: number, extra = {}): GeoJSON.Feature => ({ type: 'Feature', properties: { c: col, r, ...extra }, geometry: { type: 'Point', coordinates: c } })
const ln = (a: [number, number], b: [number, number], col: string, w: number): GeoJSON.Feature => ({ type: 'Feature', properties: { c: col, w }, geometry: { type: 'LineString', coordinates: [a, b] } })
const PURPLE = (dark: boolean) => (dark ? '#b497f0' : '#7a52d1')
const goLabel = (n: number) => `GO-${String(n).padStart(3, '0')}`

function ExampleSelect({ walk, ex, setEx }: { walk: Walk[]; ex: number; setEx: (i: number) => void }) {
  return (
    <div className="ctrl">
      <span className="muted">Exemplo real:</span>
      <div className="seg" role="tablist">
        {walk.map((w, i) => <button key={w.sre} className={i === ex ? 'on' : ''} onClick={() => setEx(i)} title={w.label}>{w.sre}</button>)}
      </div>
      <small>{walk[ex].label} · {walk[ex].sentido.toLowerCase()} · classe Passeio</small>
    </div>
  )
}

/* ───────────────────────────── EB ───────────────────────────── */
export function EBModel({ store, cd }: P) {
  const [ex, setEx] = useState(0)
  const dark = isDark(); const c = pal(dark)
  const w = store.walk[ex]
  const target: [number, number] = [w.lon, w.lat]
  const overlay = (step: number): FC | null => {
    if (step < 2) return { type: 'FeatureCollection', features: [pt(target, '#ffd23f', 9)] }
    const f: GeoJSON.Feature[] = [pt(target, '#ffd23f', 9)]
    w.road_nb.forEach((n, i) => { const p: [number, number] = [n.lon, n.lat]; f.push(ln(target, p, c.count, 1 + 3 * (w.w_road[i] ?? 0)), pt(p, c.count, 4 + 8 * (w.w_road[i] ?? 0))) })
    w.geo_nb.forEach((n, i) => { const p: [number, number] = [n.lon, n.lat]; f.push(ln(target, p, PURPLE(dark), 1 + 3 * w.w_geo[i]), pt(p, PURPLE(dark), 4 + 8 * w.w_geo[i])) })
    return { type: 'FeatureCollection', features: f }
  }
  const obsP = useMemo(() => (cd ? store.segs.map((s, i) => (s.o === 'c' ? Math.log1p(cd.eb.Passeio[i] ?? 0) : null)).filter((v): v is number => v != null) : []), [store, cd])
  const ct = w.contrib as Record<string, number | Record<string, number>>
  const grp = (k: string) => ct[k] as number
  const fixedEach = Object.entries(ct.fixed_each as Record<string, number>).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 8)
  const viz = w.pred_log - w.ym - (grp('fixed') + grp('classe') + grp('situacao') + grp('go') + grp('reg'))
  const wf = [
    { label: 'Features', value: grp('fixed') }, { label: 'Classe', value: grp('classe') }, { label: 'Situação', value: grp('situacao') },
    { label: 'Rodovia (GO)', value: grp('go') }, { label: 'Regional', value: grp('reg') }, { label: 'Vizinhança', value: viz },
  ]
  const cmpCar = cd ? cd.car.Passeio[store.index.get(`${w.sre}|${w.sentido[0]}`) ?? 0] : null
  const view = (step: number) => ({ center: target, zoom: step < 2 ? 9 : 8.3 }) as { center: [number, number]; zoom: number }

  const steps = [
    { tag: '1 · Alvo', body: <>
      <h3>Trabalhar em log(1 + y)</h3>
      <p>Para cada classe <i>k</i> (aqui, Passeio) o alvo é transformado em escala log. O “+1” mantém os <b>zeros</b> no treino — o que evita o viés para cima em classes esparsas.</p>
      <div className="formula"><Tex block>{String.raw`z_{ik}=\ln(1+y_{ik})`}</Tex></div>
      <small>O histograma mostra as {fmt(obsP.length)} linhas contadas já em escala log: é sobre elas que o modelo aprende.</small></> },
    { tag: '2 · Design', body: <>
      <h3>Monta a matriz de efeitos</h3>
      <p>Cada linha vira um vetor <Tex>{String.raw`\mathbf{x}_i`}</Tex> com 12 numéricas padronizadas + dummies de classe e situação, mais as dummies de rodovia e regional (efeitos aleatórios).</p>
      <div className="formula"><Tex block>{String.raw`z_{ik}=\mathbf{x}_i^{\top}\boldsymbol\beta_k+u^{GO}_{g(i)k}+u^{reg}_{r(i)k}+\gamma_k\,s_{ik}+\varepsilon_{ik}`}</Tex></div>
      <p>Ao lado: quanto cada feature <i>desloca</i> o log do volume do segmento <b>{w.sre}</b> em relação à média das contadas.</p></> },
    { tag: '3 · Vizinhança', body: <>
      <h3>Suaviza as contagens vizinhas</h3>
      <p><span style={{ color: c.count }}>●</span> <b>Ao longo da rodovia</b>, com peso decaindo com a distância em km; <span style={{ color: PURPLE(dark) }}>●</span> <b>geográfico</b>, os 6 vizinhos contados mais próximos. Ambos são encolhidos para a média global.</p>
      <div className="formula"><Tex block>{String.raw`s_{ik}^{rod}=\frac{\sum_j w_{ij}\,z_{jk}+\lambda\,\bar z_k}{\sum_j w_{ij}+\lambda},\quad w_{ij}=e^{-\mathrm{gap}_{ij}/L}`}</Tex></div>
      <small>L = {store.models.eb.params.l_road} km, λ = {store.models.eb.params.shrink}. A própria SRE (os dois sentidos) é excluída: o termo nunca usa o próprio alvo.</small></> },
    { tag: '4 · Ridge', body: <>
      <h3>Ajuste em forma fechada (BLUP)</h3>
      <p>Os coeficientes saem de uma regressão ridge: nada de MCMC. A penalidade equivale a um prior normal nos efeitos, e as dummies de GO e regional são escaladas por 0,5, definindo a variância relativa dos efeitos aleatórios.</p>
      <div className="formula"><Tex block>{String.raw`\hat{\boldsymbol\beta}=\left(X^{\top}X+\alpha I\right)^{-1}X^{\top}z,\qquad \alpha=3`}</Tex></div>
      <p>A cascata mostra a soma, em log, que leva da média das contadas ao valor previsto.</p></> },
    { tag: '5 · Volta', body: <>
      <h3>De volta a veículos por dia</h3>
      <div className="formula"><Tex block>{String.raw`\hat y_{ik}=\max\!\left(0,\;e^{\hat z_{ik}}-1\right)`}</Tex></div>
      <p>Para {w.sre} ({w.sentido.toLowerCase()}): <Tex>{String.raw`\hat z=`}</Tex> {w.pred_log.toFixed(3)} ⇒ <b>{fmt(w.pred)} passeios/dia</b>. Reproduzido do script e idêntico à planilha EB.</p>
      <small>O EB devolve só o valor pontual: não há intervalo de incerteza.</small></> },
  ]
  return (
    <section className="sec wrap" id="eb">
      <div className="kicker">3 · Modelo GLMM-EB</div>
      <h2>Empírico-Bayes: um ridge com vizinhança, em segundos</h2>
      <p className="lead">Cada classe é ajustada separadamente. Tudo o que vem a seguir foi recalculado com as funções reais de <code>modelo_glmm_eb_vmd.py</code> para um segmento sem contagem.</p>
      <Scrolly steps={steps} visual={(i) => (
        <>
          <ExampleSelect walk={store.walk} ex={ex} setEx={setEx} />
          <div style={{ flex: 1, minHeight: 180 }}><MapView store={store} cd={cd} mode="eb" cls="Passeio" sent="ambos" className="h-fill" overlay={overlay(i)} view={view(i)} selected={w.sre} legend={false} /></div>
          <div className="card" style={{ maxHeight: '42%', overflow: 'auto' }}>
            {i === 0 && <Hist series={[{ name: 'Passeio', values: obsP, color: c.count }]} xLabel="ln(1 + Passeio) nas linhas contadas" height={160} bins={22} />}
            {i === 1 && <><h3 style={{ fontSize: '.85rem' }}>Contribuição por feature (Δ em ln)</h3><HBars items={fixedEach.map(([k, v]) => ({ label: k, value: Math.abs(v), color: v >= 0 ? c.up : c.down, text: (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3) }))} /></>}
            {i === 2 && <><h3 style={{ fontSize: '.85rem' }}>Peso de cada vizinho</h3>
              <HBars max={1} items={[...w.w_road.slice(0, w.road_nb.length).map((v, k) => ({ label: `rod. ${w.road_nb[k]?.sre ?? ''}`.slice(0, 18), value: v, color: c.count, note: `gap ${w.gap_km[k].toFixed(1)} km`, text: `${v.toFixed(2)} · ${w.gap_km[k].toFixed(1)} km` })),
                ...w.w_geo.map((v, k) => ({ label: `geo ${w.geo_nb[k].sre}`.slice(0, 18), value: v, color: PURPLE(dark), text: `${v.toFixed(2)} · ${w.dist_km[k].toFixed(0)} km` }))]} />
              <small>s_rodovia = {fmt(Math.expm1(w.s_road))} · s_geo = {fmt(Math.expm1(w.s_geo))} passeios/dia (volta de log)</small></>}
            {i === 3 && <Waterfall base={w.ym} baseLabel="Média contadas" totalLabel="Previsto" steps={wf} />}
            {i === 4 && <div><div className="stats" style={{ marginTop: 0 }}>
              <div className="stat red"><b>{fmt(w.pred)}</b><span>EB · Passeio/dia</span></div>
              <div className="stat"><b>{cmpCar != null ? fmt(cmpCar) : '—'}</b><span>CAR · mesmo segmento</span></div>
              <div className="stat"><b>{w.pred_log.toFixed(2)}</b><span>ẑ em log</span></div></div></div>}
          </div>
        </>)} />
    </section>
  )
}

/* ───────────────────────────── CAR ───────────────────────────── */
function lgamma(z: number): number {
  const g = 7, co = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z)
  z -= 1; let x = co[0]; const t = z + g + 0.5
  for (let i = 1; i < g + 2; i++) x += co[i] / (z + i)
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x)
}
const gammaPdfLog10 = (l10: number, phi: number, mu: number) => {
  const y = 10 ** l10
  return Math.exp(phi * Math.log(phi / mu) + (phi - 1) * Math.log(y) - (phi * y) / mu - lgamma(phi)) * y * Math.LN10
}

function GammaDemo({ phi0 }: { phi0: number }) {
  const [phi, setPhi] = useState(Math.round(phi0 * 10) / 10)
  const [ref, w] = useWidth<HTMLDivElement>(); const c = pal(isDark())
  const mus = [300, 3000, 30000]; const h = 170
  const x = d3.scaleLinear().domain([1, 5.2]).range([36, w - 10]); const xs = d3.range(1, 5.2, 0.03)
  const ymax = d3.max(mus.flatMap((m) => xs.map((v) => gammaPdfLog10(v, phi, m)))) ?? 1
  const y = d3.scaleLinear().domain([0, ymax]).range([h - 26, 6])
  const line = d3.line<number>()
  const cols = [c.count, c.est, c.up]
  return (
    <div ref={ref} className="chart">
      <svg width={w} height={h} role="img" aria-label="Densidades Gama com mesmo formato em três volumes médios">
        {mus.map((m, k) => <path key={m} d={line.x((v) => x(v)).y((v) => y(gammaPdfLog10(v, phi, m)))(xs)!} fill="none" stroke={cols[k]} strokeWidth={2} />)}
        {[1, 2, 3, 4, 5].map((t) => <text key={t} x={x(t)} y={h - 8} textAnchor="middle">{nf0.format(10 ** t)}</text>)}
        {mus.map((m, k) => <text key={m} x={x(Math.log10(m))} y={10} textAnchor="middle" style={{ fill: cols[k] }}>μ = {nf0.format(m)}</text>)}
      </svg>
      <div className="ctrl"><label>φ (forma) = {phi.toFixed(1)} <input type="range" min={0.5} max={8} step={0.1} value={phi} onChange={(e) => setPhi(+e.target.value)} aria-label="Parâmetro de forma φ" /></label>
        <small>CV = 1/√φ = {(100 / Math.sqrt(phi)).toFixed(0)}% — igual em qualquer μ</small></div>
    </div>
  )
}

function ChainDemo() {
  const [a, setA] = useState(0.9)
  const [ref, w] = useWidth<HTMLDivElement>(); const c = pal(isDark())
  const n = 9; const known: Record<number, number> = { 0: 0.7, 1: 0.5, 5: -0.3, 8: -0.6 }
  const u = useMemo(() => {
    const v = Array.from({ length: n }, (_, i) => known[i] ?? 0)
    for (let it = 0; it < 400; it++) for (let i = 0; i < n; i++) if (!(i in known)) {
      const nb = [i - 1, i + 1].filter((j) => j >= 0 && j < n); v[i] = a * d3.mean(nb.map((j) => v[j]))!
    }
    return v
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [a])
  const h = 150, x = d3.scalePoint<number>().domain(d3.range(n)).range([30, w - 30]); const y = d3.scaleLinear().domain([-1, 1]).range([h - 24, 8])
  return (
    <div ref={ref} className="chart">
      <svg width={w} height={h} role="img" aria-label="Cadeia de nove segmentos: efeito espacial conhecido nos contados e propagado nos demais">
        <line x1={x(0)} x2={x(n - 1)} y1={y(0)} y2={y(0)} stroke="var(--line)" />
        <polyline points={d3.range(n).map((i) => `${x(i)},${y(u[i])}`).join(' ')} fill="none" stroke="var(--ink2)" strokeDasharray="3 3" />
        {d3.range(n).map((i) => (
          <g key={i}><circle cx={x(i)} cy={y(u[i])} r={7} fill={i in known ? c.count : c.est} stroke="var(--card)" strokeWidth={2} />
            <text x={x(i)} y={h - 8} textAnchor="middle">{i in known ? 'contado' : 'sem'}</text></g>))}
      </svg>
      <div className="ctrl"><label>α = {a.toFixed(2)} <input type="range" min={0} max={0.99} step={0.01} value={a} onChange={(e) => setA(+e.target.value)} aria-label="Dependência espacial alfa" /></label>
        <small>α→0: sem vizinhança · α→1: herda forte dos vizinhos</small></div>
    </div>
  )
}

export function CARModel({ store, cd }: P) {
  const [ex, setEx] = useState(0)
  const dark = isDark(); const c = pal(dark)
  const w = store.walk[ex]
  const car = store.models.car
  const hyper = car.hyper.Passeio, coef = car.coef.Passeio
  const hv = (name: string) => Number(hyper.find((h) => String(h.hiperparametro).includes(name))?.mean ?? NaN)
  const phi = hv('gamma_shape') || 2
  const chain = useMemo(() => {
    const g = goLabel(w.go)
    const segs = store.segs.map((s, i) => ({ s, i })).filter(({ s }) => s.go === g && s.s === w.sentido[0]).sort((a, b) => a.s.ki - b.s.ki)
    const nodes = segs.map(({ s }) => ({ s, p: roadCentroid(store, s.sre) })).filter((n): n is { s: typeof n.s; p: [number, number] } => n.p != null)
    const f: GeoJSON.Feature[] = []
    nodes.forEach((n, k) => { if (k) f.push(ln(nodes[k - 1].p, n.p, c.none, 2)); })
    nodes.forEach((n) => f.push(pt(n.p, n.s.o === 'c' ? c.count : c.est, n.s.sre === w.sre ? 9 : 5)))
    return { fc: { type: 'FeatureCollection', features: f } as FC, n: nodes.length, nObs: nodes.filter((n) => n.s.o === 'c').length }
  }, [store, w, c.count, c.est, c.none])
  const target: [number, number] = [w.lon, w.lat]
  const rows = store.walk.map((x) => {
    const i = store.index.get(`${x.sre}|${x.sentido[0]}`) ?? 0
    return { sre: x.sre, car: cd?.car.Passeio[i] ?? 0, lo: cd?.lo.Passeio[i] ?? null, hi: cd?.hi.Passeio[i] ?? null, eb: cd?.eb.Passeio[i] ?? 0 }
  })

  const steps = [
    { tag: '1 · Verossimilhança', body: <>
      <h3>Gama em vez de gaussiana</h3>
      <p>O volume é positivo e sua dispersão cresce com a média. A distribuição Gama com ligação log trata o erro como <b>relativo</b>: o coeficiente de variação é o mesmo em 300 e em 30.000 veíc./dia.</p>
      <div className="formula"><Tex block>{String.raw`y_i\sim\mathrm{Gama}\!\left(\phi,\ \tfrac{\phi}{\mu_i}\right),\quad \mathbb{E}[y_i]=\mu_i,\ \ \mathrm{CV}=\tfrac{1}{\sqrt\phi}`}</Tex></div>
      <small>Para Passeio, φ estimado ≈ {phi.toFixed(2)} (CV ≈ {(100 / Math.sqrt(phi)).toFixed(0)}%).</small></> },
    { tag: '2 · Preditor', body: <>
      <h3>Soma de quatro parcelas em log</h3>
      <div className="formula"><Tex block>{String.raw`\ln\mu_i=\beta_0+\mathbf{x}_i^{\top}\boldsymbol\beta+u^{GO}_{g(i)}+u^{reg}_{r(i)}+u^{CAR}_i`}</Tex></div>
      <p>Só duas features numéricas (empresas, distância ao polo) e três categóricas (classe, situação, perímetro). Cada <Tex>{String.raw`e^{\beta}`}</Tex> é um multiplicador direto sobre o volume.</p></> },
    { tag: '3 · Priors e MCMC', body: <>
      <h3>Tudo estimado, com incerteza</h3>
      <p>Priors fracos: <Tex>{String.raw`\beta\sim N(0,1{,}5)`}</Tex>, <Tex>{String.raw`\sigma_{GO},\sigma_{reg}\sim\text{Meia-Normal}(1)`}</Tex>, <Tex>{String.raw`\alpha\sim U(0;0{,}99)`}</Tex>, <Tex>{String.raw`\tau\sim\text{Gama}(2,1)`}</Tex>, <Tex>{String.raw`\phi\sim\text{Gama}(2;0{,}1)`}</Tex>. Amostragem NUTS com 4 cadeias × 1000 amostras (tune 1500); convergência checada por R̂ (meta ≤ 1,01; algumas classes raras não chegam lá — veja Validação). Rodou {(car.seg / 60).toFixed(0)} min para as 35 classes.</p></> },
    { tag: '4 · Efeito CAR', body: <>
      <h3>Grafo em cadeia ao longo da rodovia</h3>
      <p>Segmentos consecutivos (gap ≤ 0,1 km, mesmo sentido) são vizinhos. O efeito latente segue um CAR: cada trecho tende à média dos vizinhos, com força <b>α</b>.</p>
      <div className="formula"><Tex block>{String.raw`\mathbf{u}\sim N\!\left(0,\ [\tau(D-\alpha W)]^{-1}\right),\qquad \mathbb{E}[u_i\mid u_{-i}]=\alpha\,\overline{u}_{N(i)}`}</Tex></div>
      <p>Na rodovia {goLabel(w.go)}, sentido {w.sentido.toLowerCase()}: {chain.nObs} contados em {chain.n} segmentos.</p></> },
    { tag: '5 · Predição', body: <>
      <h3>Segmento sem contagem + IC 90%</h3>
      <p>Fora da amostra, <Tex>{String.raw`\hat u_i`}</Tex> é aproximado por α × média dos 1–2 vizinhos contados mais próximos (ou, sem vizinho na cadeia, dos 2 mais próximos em linha reta).</p>
      <div className="formula"><Tex block>{String.raw`\hat\mu_i=e^{\,\beta_0+\mathbf{x}_i^{\top}\hat\beta+\hat u^{GO}+\hat u^{reg}+\alpha\bar u_{N(i)}},\quad \mathrm{IC}_{90}=\hat\mu_i\,e^{\pm1{,}645\,s}`}</Tex></div>
      <small><Tex>{String.raw`s^2=\sigma_{GO}^2[\text{GO novo}]+\sigma_{reg}^2[\text{regional nova}]+1/\phi`}</Tex>. Não inclui a incerteza dos coeficientes: tende a ser otimista.</small></> },
  ]
  return (
    <section className="sec wrap" id="car">
      <div className="kicker">4 · Modelo GLMM-CAR v4</div>
      <h2>Bayesiano: Gama, efeitos hierárquicos e dependência espacial</h2>
      <p className="lead">O mesmo problema, resolvido como modelo probabilístico completo (PyMC). Mais lento, porém com hiperparâmetros aprendidos e intervalos de credibilidade.</p>
      <Scrolly steps={steps} visual={(i) => (
        <>
          <ExampleSelect walk={store.walk} ex={ex} setEx={setEx} />
          <div style={{ flex: 1, minHeight: 180 }}><MapView store={store} cd={cd} mode="car" cls="Passeio" sent="ambos" className="h-fill" overlay={i >= 3 ? chain.fc : { type: 'FeatureCollection', features: [pt(target, '#ffd23f', 9)] }} view={{ center: target, zoom: i >= 3 ? 8.2 : 9 }} selected={w.sre} legend={false} /></div>
          <div className="card" style={{ maxHeight: '46%', overflow: 'auto' }}>
            {i === 0 && <GammaDemo phi0={phi} />}
            {i === 1 && <><h3 style={{ fontSize: '.85rem' }}>Coeficientes (Passeio) — multiplicador e^β</h3>
              <div className="tbl-scroll" style={{ maxHeight: 190 }}><table><thead><tr><th>Parâmetro</th><th className="num">β</th><th className="num">e^β</th><th className="num">R̂</th></tr></thead>
                <tbody>{coef.filter((r) => String(r.parametro) !== 'intercept' ).map((r) => <tr key={String(r.parametro)}><td>{String(r.parametro)}</td><td className="num">{Number(r.mean).toFixed(2)}</td><td className="num">×{Math.exp(Number(r.mean)).toFixed(2)}</td><td className="num">{Number(r.r_hat).toFixed(2)}</td></tr>)}</tbody></table></div></>}
            {i === 2 && <><h3 style={{ fontSize: '.85rem' }}>Hiperparâmetros aprendidos (Passeio)</h3>
              <table><thead><tr><th>Parâmetro</th><th className="num">Média</th><th className="num">IC 90%</th><th className="num">R̂</th></tr></thead>
                <tbody>{hyper.map((h) => <tr key={String(h.hiperparametro)}><td>{String(h.hiperparametro)}</td><td className="num">{Number(h.mean).toFixed(2)}</td><td className="num">{Number(h.eti90_lb).toFixed(2)} – {Number(h.eti90_ub).toFixed(2)}</td><td className="num">{Number(h.r_hat).toFixed(2)}</td></tr>)}</tbody></table></>}
            {i === 3 && <><h3 style={{ fontSize: '.85rem' }}>Como α propaga o efeito (ilustração)</h3><ChainDemo /></>}
            {i === 4 && <><h3 style={{ fontSize: '.85rem' }}>Passeio/dia · CAR com IC90 × EB</h3><RangeRows rows={rows} /></>}
          </div>
        </>)} />
    </section>
  )
}

function RangeRows({ rows }: { rows: { sre: string; car: number; lo: number | null; hi: number | null; eb: number }[] }) {
  const [ref, w] = useWidth<HTMLDivElement>(); const c = pal(isDark())
  const all = rows.flatMap((r) => [r.car, r.eb, r.lo ?? r.car, r.hi ?? r.car]).filter((v) => v > 0)
  const x = d3.scaleLog().domain([Math.min(...all) * 0.8, Math.max(...all) * 1.2]).range([88, w - 14])
  return (
    <div ref={ref} className="chart">
      <svg width={w} height={rows.length * 38 + 28} role="img" aria-label="Intervalos de 90 por cento do CAR e ponto do EB por segmento">
        {rows.map((r, i) => (
          <g key={r.sre} transform={`translate(0 ${i * 38 + 14})`}>
            <text x={0} y={4}>{r.sre}</text>
            {r.lo != null && r.hi != null && <line x1={x(r.lo)} x2={x(r.hi)} y1={0} y2={0} stroke={c.est} strokeWidth={5} strokeLinecap="round" opacity={0.35} />}
            <circle cx={x(r.car)} cy={0} r={5} fill={c.est} stroke="var(--card)" strokeWidth={2} />
            <rect x={x(r.eb) - 4.5} y={-4.5} width={9} height={9} fill="var(--ink)" transform={`rotate(45 ${x(r.eb)} 0)`} />
            <text x={x(r.car)} y={-9} textAnchor="middle" style={{ fontSize: 10 }}>{fmt(r.car)}</text>
          </g>))}
        {x.ticks(4).map((t) => <text key={t} x={x(t)} y={rows.length * 38 + 22} textAnchor="middle">{fmt(t)}</text>)}
      </svg>
      <small><span style={{ color: c.est }}>●</span> CAR (barra = IC 90%) · ◆ EB (sem intervalo)</small>
    </div>
  )
}
