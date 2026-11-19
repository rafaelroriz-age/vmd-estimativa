import { useMemo, useState } from 'react'
import * as d3 from 'd3'
import MapView from '../MapView'
import { Tex } from '../Tex'
import { Hist, Card } from '../charts'
import { fmt, nf0, type ClassData, type Store } from '../data'
import { pal, isDark } from '../theme'

interface P { store: Store; cd: ClassData | null }

export function Hero({ store, cd }: P) {
  const m = store.models
  const nEst = m.n_rows - m.n_obs
  return (
    <section className="hero wrap" id="topo">
      <div className="grid">
        <div>
          <div className="kicker">Estimativa de VMD · Goiás · rodada SET/2026</div>
          <h1>Como estimamos o tráfego onde ninguém contou</h1>
          <p className="lead">
            Só {nf0.format(m.n_sre_obs)} das {nf0.format(m.n_sre)} SREs pavimentadas têm contagem. Dois modelos
            estatísticos — <b>GLMM-EB</b> (rápido, determinístico) e <b>GLMM-CAR</b> (bayesiano, com incerteza) — preenchem o resto.
            Este guia mostra, passo a passo, o que acontece com o dado em cada um.
          </p>
          <div className="stats">
            <div className="stat"><b>{nf0.format(m.n_sre)}</b><span>SREs no escopo (PAV/DUP/EOD)</span></div>
            <div className="stat blue"><b>{nf0.format(m.n_obs)}</b><span>linhas com contagem ({m.n_sre_obs} SREs × sentido)</span></div>
            <div className="stat red"><b>{nf0.format(nEst)}</b><span>linhas estimadas</span></div>
          </div>
          <p style={{ marginTop: 18 }}><small>Role para seguir a história ou vá direto ao <a href="#explorador">explorador do mapa</a>.</small></p>
        </div>
        <MapView store={store} cd={cd} mode="media" cls="VMD" sent="ambos" className="h-hero" />
      </div>
    </section>
  )
}

export function Dado({ store }: P) {
  const m = store.models
  const sample = useMemo(() => {
    const c = store.segs.find((s) => s.o === 'c' && s.sit === 'PAV')!
    const e = store.segs.filter((s) => s.o === 'e' && s.sit === 'PAV' && s.go === c.go)
    return [c, e[0], store.segs.find((s) => s.o === 'e' && s.sit === 'DUP')!].filter(Boolean)
  }, [store])
  const obs = useMemo(() => store.segs.filter((s) => s.o === 'c').map((s) => s.eb), [store])
  const c = pal(isDark())
  const sit = Object.entries(m.situacoes).sort((a, b) => b[1] - a[1])
  const inScope = new Set(['PAV', 'DUP', 'EOD'])
  const cutoff = sit.filter(([k]) => !inScope.has(k)).reduce((a, [, v]) => a + v, 0)
  return (
    <section className="sec wrap" id="dado">
      <div className="kicker">1 · O dado</div>
      <h2>Uma linha por SRE e sentido, 35 classes veiculares</h2>
      <p className="lead">A unidade de análise é o par (SRE, sentido). Cada linha traz atributos do trecho e, quando há contagem, o volume diário médio de cada uma das 35 classes (VMD total = soma das classes).</p>
      <div className="cols2">
        <Card title="Três linhas reais do dataset">
          <div className="tbl-scroll"><table>
            <thead><tr><th>SRE</th><th>Sentido</th><th>Rodovia</th><th>Classe</th><th>Sit.</th><th className="num">Empresas</th><th>Origem</th><th className="num">VMD</th></tr></thead>
            <tbody>{sample.map((s) => (
              <tr key={s.sre + s.s}><td className="mono">{s.sre}</td><td>{s.s === 'C' ? 'Cresc.' : 'Decresc.'}</td><td>{s.go}</td><td>{s.cl}</td><td>{s.sit}</td>
                <td className="num">{fmt(s.emp)}</td>
                <td><span className={`pill ${s.o}`}>{s.o === 'c' ? 'contagem' : 'estimar'}</span></td>
                <td className="num">{s.o === 'c' ? fmt(s.eb) : '?'}</td></tr>))}</tbody>
          </table></div>
          <small>Linhas “estimar” têm todos os atributos, mas as 35 colunas de volume estão vazias — é isso que os modelos preenchem.</small>
        </Card>
        <Card title={`Filtro de escopo: ${nf0.format(cutoff)} de ${nf0.format(m.n_rows_total)} linhas ficam de fora`}>
          <div role="table">{sit.map(([k, v]) => (
            <div key={k} className="bar-row" style={{ gridTemplateColumns: '48px 1fr 56px' }}>
              <span>{k}</span><div style={{ background: 'var(--bg2)', borderRadius: 4 }}><div className="b" style={{ width: `${(v / sit[0][1]) * 100}%`, background: inScope.has(k) ? c.est : c.none }} /></div><b style={{ textAlign: 'right' }}>{nf0.format(v)}</b>
            </div>))}</div>
          <small>Só pista pavimentada (PAV), duplicada (DUP) e em obras de duplicação (EOD) entram nos modelos; as demais aparecem em cinza no mapa.</small>
        </Card>
      </div>
      <div className="cols2" style={{ marginTop: 20 }}>
        <Card title="Atualização para o ano de referência">
          <p>Antes de ajustar, cada contagem é levada ao ano de referência com crescimento composto:</p>
          <Tex block>{String.raw`V_{ref}=V_{cont}\,(1+g)^{\,ano_{ref}-ano_{cont}},\qquad g=3\%\ \text{a.a.}`}</Tex>
          <small>Neste dataset todas as 406 contagens já estão em 2026, logo o fator é 1,00; a regra vale para rodadas com contagens antigas.</small>
        </Card>
        <Card title="Distribuição do VMD contado (406 linhas)">
          <Hist series={[{ name: 'VMD', values: obs.map((v) => Math.log10(v + 1)), color: c.count }]} xLabel="log₁₀(1 + VMD)" bins={22} fmtX={(v) => nf0.format(Math.round(10 ** v - 1))} height={190} />
          <small>Assimétrica e positiva: por isso ambos os modelos trabalham em escala logarítmica, onde o erro é relativo (±200 veíc. pesa pouco em 10 mil e muito em 500).</small>
        </Card>
      </div>
      <div className="callout"><b>Duas formas de lidar com a escala.</b> EB modela <Tex>{String.raw`z=\ln(1+y)`}</Tex> com erro gaussiano e mantém zeros no treino. CAR modela <Tex>{String.raw`y>0`}</Tex> direto com verossimilhança Gama e ligação log — zeros são descartados do treino da classe.</div>
    </section>
  )
}

const FEATS: { g: string; name: string; tr: string; eb: string; car: string; key?: 'emp' | 'dist' | 'pop' | 'pib' | 'fro' | 'ext' }[] = [
  { g: 'Socioeconômico', name: 'Empresas ativas (municípios)', tr: 'log1p → z', eb: '●', car: '●', key: 'emp' },
  { g: 'Socioeconômico', name: 'População residente', tr: 'log1p → z', eb: '●', car: '', key: 'pop' },
  { g: 'Socioeconômico', name: 'PIB', tr: 'log1p → z', eb: '●', car: '', key: 'pib' },
  { g: 'Socioeconômico', name: 'Frota ativa', tr: 'log1p → z', eb: '●', car: '', key: 'fro' },
  { g: 'Geográfico', name: 'Distância à cidade polo', tr: 'log1p → z', eb: '●', car: '●', key: 'dist' },
  { g: 'Geográfico', name: 'Latitude / longitude do trecho', tr: 'z', eb: '●', car: '○ só no fallback do CAR' },
  { g: 'Geográfico', name: 'Liga a rodovia federal (BR)', tr: '0/1', eb: '●', car: '' },
  { g: 'Estrutural', name: 'Extensão do segmento', tr: 'ln(x+0,05) → z', eb: '●', car: '', key: 'ext' },
  { g: 'Estrutural', name: 'Classe funcional', tr: 'dummies', eb: '●', car: '●' },
  { g: 'Estrutural', name: 'Situação da pista (PAV/DUP/EOD)', tr: 'dummies', eb: '●', car: '●' },
  { g: 'Estrutural', name: 'Perímetro urbano', tr: '0/1', eb: '●', car: '●' },
  { g: 'Estrutural', name: 'Coincide com outra rodovia', tr: '0/1', eb: '●', car: '' },
  { g: 'Estrutural', name: 'Sentido decrescente', tr: '0/1', eb: '●', car: '' },
  { g: 'Hierarquia', name: 'Rodovia (GO)', tr: 'efeito aleatório', eb: '● ridge, σ fixo', car: '● σ estimado' },
  { g: 'Hierarquia', name: 'Regional', tr: 'efeito aleatório', eb: '● ridge, σ fixo', car: '● σ estimado' },
  { g: 'Vizinhança', name: 'Contagens vizinhas (km / geo)', tr: 'suavização kernel', eb: '● covariável s_road, s_geo', car: '● efeito latente CAR' },
]

export function Features({ store }: P) {
  const [key, setKey] = useState<'emp' | 'dist' | 'pop' | 'pib' | 'fro' | 'ext'>('emp')
  const c = pal(isDark())
  const vals = useMemo(() => store.segs.map((s) => s[key] ?? 0), [store, key])
  const raw = vals
  const lg = useMemo(() => {
    const l = vals.map((v) => (key === 'ext' ? Math.log(v + 0.05) : Math.log1p(Math.max(v, 0))))
    const mu = d3.mean(l)!, sd = d3.deviation(l) || 1
    return l.map((v) => (v - mu) / sd)
  }, [vals, key])
  const names = { emp: 'Empresas ativas', dist: 'Distância ao polo (m)', pop: 'População', pib: 'PIB', fro: 'Frota', ext: 'Extensão (km)' }
  let last = ''
  return (
    <section className="sec wrap" id="features">
      <div className="kicker">2 · As features</div>
      <h2>O que cada modelo enxerga de um trecho</h2>
      <p className="lead">Os atributos são agrupados em cinco famílias. Numéricos assimétricos passam por log e padronização (z-score); categóricos viram dummies. Os dois modelos usam conjuntos bem diferentes.</p>
      <div className="tbl-scroll" style={{ maxHeight: 'none' }}>
        <table>
          <thead><tr><th>Família</th><th>Feature</th><th>Transformação</th><th>GLMM-EB</th><th>GLMM-CAR v4</th></tr></thead>
          <tbody>{FEATS.map((f) => { const first = f.g !== last; last = f.g; return (
            <tr key={f.name}><td style={{ fontWeight: first ? 600 : 400, color: first ? 'var(--ink)' : 'transparent' }}>{f.g}</td><td>{f.name}</td><td className="muted">{f.tr}</td>
              <td>{f.eb ? <span style={{ color: c.est }}>{f.eb}</span> : <span className="muted">—</span>}</td>
              <td>{f.car ? <span style={{ color: c.count }}>{f.car}</span> : <span className="muted">—</span>}</td></tr>) })}</tbody>
        </table>
      </div>
      <small>● usada · ○ uso indireto. O CAR v4 usa poucas features fixas e deixa a estrutura espacial/hierárquica absorver o restante; o EB usa muitas, regularizadas por ridge.</small>
      <div className="cols2" style={{ marginTop: 24 }}>
        <Card title="Antes e depois da transformação">
          <div className="ctrl" style={{ marginBottom: 8 }}>
            <label>Feature <select value={key} onChange={(e) => setKey(e.target.value as typeof key)}>{Object.entries(names).map(([k, v]) => <option key={k} value={k}>{v}</option>)}</select></label>
          </div>
          <h3 style={{ fontSize: '.85rem' }}>Bruta</h3>
          <Hist series={[{ name: names[key], values: raw, color: c.none }]} xLabel={names[key]} height={150} bins={30} fmtX={(v) => (Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(0) + 'M' : Math.abs(v) >= 1e3 ? (v / 1e3).toFixed(0) + 'k' : String(Math.round(v * 10) / 10))} />
          <h3 style={{ fontSize: '.85rem' }}>log → z-score</h3>
          <Hist series={[{ name: 'z', values: lg, color: c.count }]} xLabel="z-score do log" height={150} bins={30} />
        </Card>
        <Card title="Por que transformar?">
          <p>Volumes e variáveis econômicas variam por ordens de grandeza. No log, efeitos viram <b>multiplicativos</b>: um coeficiente <Tex>{String.raw`\beta`}</Tex> corresponde a um fator <Tex>{String.raw`e^{\beta}`}</Tex> no volume.</p>
          <Tex block>{String.raw`z_j=\frac{\ln(1+x_j)-\mu_j}{\sigma_j}`}</Tex>
          <p>A padronização coloca todas as features na mesma escala, o que torna a penalidade ridge do EB e os priors <Tex>{String.raw`N(0,1{,}5)`}</Tex> do CAR comparáveis entre variáveis.</p>
          <div className="callout"><b>Vazamento de alvo:</b> nenhum dos modelos usa o volume da própria SRE como feature. A vizinhança usa só contagens <i>de outras SREs</i>.</div>
        </Card>
      </div>
    </section>
  )
}
