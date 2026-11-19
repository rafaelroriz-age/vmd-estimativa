import { useMemo } from 'react'
import * as d3 from 'd3'
import MapView from '../MapView'
import { Tex } from '../Tex'
import { Hist, HBars, Scatter, Card } from '../charts'
import { fmt, nf0, type ClassData, type Store } from '../data'
import { pal, isDark } from '../theme'

interface P { store: Store; cd: ClassData | null; onPick: (sre: string) => void }

const ROWS: [string, string, string][] = [
  ['Alvo e erro', 'ln(1+y), erro gaussiano', 'y > 0, Gama com ligação log (erro relativo)'],
  ['Zeros', 'mantidos no treino', 'descartados no treino da classe'],
  ['Features fixas', '12 numéricas + dummies de classe/situação', '2 numéricas + dummies de classe/situação/perímetro'],
  ['Rodovia e regional', 'ridge: variância relativa fixa (0,5)', 'efeitos aleatórios com σ estimado (Meia-Normal)'],
  ['Vizinhança', 'covariável: média suavizada por kernel (rodovia 30 km + 6 vizinhos geo) com coeficiente γ', 'efeito latente CAR na cadeia de segmentos; predição por 1–2 vizinhos × α'],
  ['Hiperparâmetros', 'fixados por validação cruzada', 'estimados com prior (α, τ, σ, φ)'],
  ['Estimação', 'forma fechada (álgebra linear)', 'MCMC NUTS, 4 cadeias'],
  ['Tempo (35 classes)', '≈ 13 s', '≈ 21 min'],
  ['Incerteza', 'só valor pontual', 'IC 90% aproximado por classe'],
  ['Risco principal', 'encolhe para a média; sem intervalo', 'IC otimista; CAR de predição é aproximação'],
]

export function Compare({ store, cd, onPick }: P) {
  const c = pal(isDark())
  const est = useMemo(() => store.segs.filter((s) => s.o === 'e' && s.eb > 0 && s.car > 0), [store])
  const pts = useMemo(() => est.map((s) => ({ x: s.eb, y: s.car, id: `${s.sre}` })), [est])
  const lr = useMemo(() => est.map((s) => Math.log2(s.car / s.eb)), [est])
  const within = (t: number) => (100 * est.filter((s) => Math.abs(s.car / s.eb - 1) <= t).length) / est.length
  const corr = useMemo(() => {
    const a = est.map((s) => Math.log(s.eb)), b = est.map((s) => Math.log(s.car)); const ma = d3.mean(a)!, mb = d3.mean(b)!
    return d3.sum(a.map((v, i) => (v - ma) * (b[i] - mb))) / Math.sqrt(d3.sum(a.map((v) => (v - ma) ** 2)) * d3.sum(b.map((v) => (v - mb) ** 2)))
  }, [est])
  const top = useMemo(() => [...est].sort((a, b) => Math.abs(Math.log(b.car / b.eb)) - Math.abs(Math.log(a.car / a.eb))).slice(0, 10), [est])
  return (
    <section className="sec wrap" id="comparar">
      <div className="kicker">5 · EB × CAR</div>
      <h2>Mesmo dado, duas filosofias</h2>
      <p className="lead">Os dois modelos concordam na ordem de grandeza, mas divergem onde a vizinhança e a regularização pesam de forma distinta.</p>
      <div className="tbl-scroll" style={{ maxHeight: 'none' }}>
        <table><thead><tr><th>Aspecto</th><th style={{ color: c.est }}>GLMM-EB</th><th style={{ color: c.count }}>GLMM-CAR v4</th></tr></thead>
          <tbody>{ROWS.map(([a, b, d]) => <tr key={a}><td><b>{a}</b></td><td>{b}</td><td>{d}</td></tr>)}</tbody></table>
      </div>
      <div className="cols3" style={{ margin: '20px 0' }}>
        <div className="stat"><b>{corr.toFixed(2)}</b><span>correlação (log) entre EB e CAR em {nf0.format(est.length)} linhas estimadas</span></div>
        <div className="stat"><b>{within(0.3).toFixed(0)}%</b><span>das estimativas diferem ≤ 30% entre modelos</span></div>
        <div className="stat"><b>{d3.median(lr)! >= 0 ? '+' : ''}{(100 * (2 ** d3.median(lr)! - 1)).toFixed(0)}%</b><span>mediana de CAR relativo ao EB</span></div>
      </div>
      <div className="cols2">
        <Card title="VMD total estimado: EB × CAR (clique para abrir o segmento)">
          <Scatter pts={pts} onPick={onPick} xLabel="EB (veíc./dia)" yLabel="CAR (veíc./dia)" />
        </Card>
        <Card title="Razão CAR / EB">
          <Hist series={[{ name: 'log₂(CAR/EB)', values: lr.filter((v) => Math.abs(v) < 4), color: c.est }]} xLabel="log₂(CAR / EB)  ·  0 = iguais, +1 = CAR dobra" height={190} bins={36} fmtX={(v) => (v > 0 ? '+' : '') + v.toFixed(1)} />
          <h3 style={{ fontSize: '.85rem', marginTop: 12 }}>Maiores divergências (clique)</h3>
          <div className="tbl-scroll" style={{ maxHeight: 190 }}><table><thead><tr><th>SRE</th><th>Sent.</th><th className="num">EB</th><th className="num">CAR</th><th className="num">CAR/EB</th></tr></thead>
            <tbody>{top.map((s) => <tr key={s.sre + s.s} style={{ cursor: 'pointer' }} onClick={() => onPick(s.sre)}><td className="mono">{s.sre}</td><td>{s.s}</td><td className="num">{fmt(s.eb)}</td><td className="num">{fmt(s.car)}</td><td className="num">{(s.car / s.eb).toFixed(2)}×</td></tr>)}</tbody></table></div>
        </Card>
      </div>
      <h3 style={{ marginTop: 28 }}>Onde divergem no mapa</h3>
      <p>Cada linha estimada é colorida pela razão CAR/EB: <span style={{ color: c.up }}>laranja</span> onde o CAR estima mais, <span style={{ color: c.down }}>verde-azulado</span> onde estima menos, cinza onde concordam. Trechos contados continuam azuis.</p>
      <MapView store={store} cd={cd} mode="diff" cls="VMD" sent="ambos" className="h-hero" onSelect={onPick} />
      <div className="callout warn" style={{ marginTop: 16 }}><b>Ler o contraste:</b> divergências grandes tendem a aparecer em rodovias sem nenhuma contagem (o EB recorre à vizinhança geográfica e à média; o CAR, ao efeito de rodovia/regional e ao vizinho mais próximo) e em trechos urbanos. Na dúvida, use a faixa de 90% do CAR como indicador de confiança.</div>
    </section>
  )
}

export function Validation({ store }: P) {
  const c = pal(isDark())
  const cv = store.models.eb.cv
  const met = store.models.car.metrics
  const byR2 = [...met].filter((m) => typeof m.R2_orig === 'number').sort((a, b) => Number(b.R2_orig) - Number(a.R2_orig))
  const maxRhat = Math.max(...met.map((m) => Number(m.max_rhat)))
  const minEss = Math.min(...met.map((m) => Number(m.min_ESS)))
  const bad = met.filter((m) => Number(m.max_rhat) > 1.01).map((m) => String(m.target))
  const div = met.reduce((a, m) => a + Number(m.divergencias), 0)
  return (
    <section className="sec wrap" id="validacao">
      <div className="kicker">6 · Validação e limitações</div>
      <h2>Quanto confiar em cada estimativa</h2>
      <p className="lead">A validação relevante é fora da amostra: SREs contadas são separadas em grupos, e cada grupo é estimado sem ter sido visto (os dois sentidos de uma SRE ficam juntos).</p>
      <div className="cols2">
        <Card title="EB · validação cruzada agrupada por SRE (5 folds × 3)">
          <div className="stats" style={{ marginTop: 0 }}>
            <div className="stat"><b>{cv.vmd_r2.toFixed(2)}</b><span>R² do VMD total</span></div>
            <div className="stat"><b>{cv.vmd_mdape_pct.toFixed(0)}%</b><span>erro mediano (MdAPE)</span></div>
            <div className="stat"><b>{cv.vmd_mape_pct.toFixed(0)}%</b><span>MAPE (puxado por baixos volumes)</span></div>
          </div>
          <h3 style={{ fontSize: '.85rem', marginTop: 14 }}>Parcela das SREs com erro dentro de…</h3>
          <HBars max={100} items={[{ label: '±30%', value: cv.vmd_dentro_30pct, color: c.est }, { label: '±50%', value: cv.vmd_dentro_50pct, color: c.est }]} fmtV={(v) => v.toFixed(0) + '%'} />
          <small>Para classes: WAPE {cv.classes_wape_pct.toFixed(0)}%; RMSLE macro {cv.classes_rmsle_macro.toFixed(2)} (Passeio {cv.rmsle_Passeio.toFixed(2)}, Moto {cv.rmsle_Moto.toFixed(2)}, 3S3 {cv.rmsle_3s3.toFixed(2)}, 3T6 {cv.rmsle_3t6.toFixed(2)}).</small>
        </Card>
        <Card title="CAR · diagnóstico do ajuste (34 classes)">
          <div className="stats" style={{ marginTop: 0 }}>
            <div className="stat"><b>{maxRhat.toFixed(3)}</b><span>maior R̂ (alvo ≤ 1,01)</span></div>
            <div className="stat"><b>{nf0.format(minEss)}</b><span>menor ESS</span></div>
            <div className="stat"><b>{nf0.format(div)}</b><span>divergências</span></div>
          </div>
          <small>{bad.length ? `${bad.length} de ${met.length} classes não atingem R̂ ≤ 1,01: ${bad.join(', ')}. Para elas, trate a estimativa CAR com cautela.` : 'Todas as classes convergiram (R̂ ≤ 1,01).'}</small>
          <h3 style={{ fontSize: '.85rem', marginTop: 14 }}>R² no ajuste, por classe (10 melhores e 5 piores)</h3>
          <HBars max={1} items={[...byR2.slice(0, 10), ...byR2.slice(-5)].map((m) => ({ label: String(m.target), value: Math.max(0, Number(m.R2_orig)), color: c.count }))} fmtV={(v) => v.toFixed(2)} />
        </Card>
      </div>
      <div className="callout warn"><b>Não compare os números lado a lado.</b> O R² do CAR acima é <i>in-sample</i> (nos segmentos contados) e fica próximo de 1,00 em classes com poucas contagens — sobreajuste, não precisão. Na rodada anterior (ago/2026), a validação cruzada do CAR para o VMD total deu R² entre 0,41 e 0,65 e MAPE entre 58% e 93% — mesma ordem do EB fora da amostra.</div>
      <h3>Limitações comuns</h3>
      <ul>
        <li>Cada classe é ajustada separadamente: a soma das 35 classes não é forçada a bater com um VMD total independente.</li>
        <li>Para um trecho nunca contado, o modelo acerta ordem de grandeza e hierarquia entre segmentos, mas o erro individual pode ser grande — sobretudo em baixos volumes.</li>
        <li>Cerca de 40% das SREs contadas não têm outra contagem no mesmo trecho contínuo: nesses casos o termo de vizinhança contribui pouco.</li>
        <li>Classes raras (poucas contagens) têm estimativas pouco informativas; 3S1 é zero em todo o dataset e não é modelada.</li>
        <li>A faixa de 90% do CAR não inclui a incerteza dos coeficientes e tende a ser levemente otimista. O EB não tem faixa.</li>
        <li>Taxa de crescimento única (3% a.a.) para todas as classes e regiões, aplicada só quando o ano da contagem difere do ano de referência.</li>
      </ul>
    </section>
  )
}
