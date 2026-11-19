import { useEffect, useState } from 'react'
import { StaticSource, type ClassData, type Store } from './data'
import { Hero, Dado, Features } from './story/Intro'
import { EBModel, CARModel } from './story/Models'
import { Compare, Validation } from './story/Compare'
import Explorer from './story/Explorer'

const NAV: [string, string][] = [['dado', 'Dado'], ['features', 'Features'], ['eb', 'EB'], ['car', 'CAR'], ['comparar', 'EB × CAR'], ['validacao', 'Validação'], ['explorador', 'Mapa']]

export default function App() {
  const [store, setStore] = useState<Store | null>(null)
  const [cd, setCd] = useState<ClassData | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [sel, setSel] = useState<string | null>(null)
  const [on, setOn] = useState('')

  useEffect(() => {
    StaticSource.load().then((s) => { setStore(s); s.loadClasses().then(setCd) }).catch((e) => setErr(String(e)))
  }, [])
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setOn(e.target.id)), { rootMargin: '-40% 0px -55% 0px' })
    NAV.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el) })
    return () => io.disconnect()
  }, [store])

  if (err) return <div className="wrap" style={{ padding: 40 }}><h2>Falha ao carregar os dados</h2><pre>{err}</pre></div>
  if (!store) return <div className="wrap" style={{ padding: 40 }}><p className="muted">Carregando dados…</p></div>

  const pick = (sre: string | null) => { setSel(sre); if (sre) document.getElementById('explorador')?.scrollIntoView({ behavior: 'smooth' }) }
  return (
    <>
      <nav className="nav" aria-label="Seções"><div className="wrap">
        <b><a href="#topo" style={{ color: 'inherit', padding: 0 }}>VMD · SRE GO</a></b>
        {NAV.map(([id, l]) => <a key={id} href={`#${id}`} className={on === id ? 'on' : ''}>{l}</a>)}
      </div></nav>
      <main>
        <Hero store={store} cd={cd} />
        <Dado store={store} cd={cd} />
        <Features store={store} cd={cd} />
        <EBModel store={store} cd={cd} />
        <CARModel store={store} cd={cd} />
        <Compare store={store} cd={cd} onPick={pick} />
        <Validation store={store} cd={cd} onPick={pick} />
        <Explorer store={store} cd={cd} selected={sel} onSelect={(s) => setSel(s)} />
      </main>
      <footer><div className="wrap">
        <p><b>Fonte dos dados:</b> <code>vmd_SRE_SET-26_features_2026.xlsx</code>, estimativas <code>modelo_volume_glmm_eb_20261118_184542.xlsx</code> e <code>modelo_volume_glmm_car_20261118_195410.xlsx</code>, malha <code>SRE-GO_2026_SET_SGP.shp</code> (geometria simplificada). Estimativas EB e CAR são as saídas oficiais da rodada; o app não reexecuta os modelos.</p>
        <p>A camada de dados é isolada em <code>src/data.ts</code> (interface <code>DataSource</code>) para migrar para o Supabase sem alterar a interface. Mapa base © OpenStreetMap (opcional, online).</p>
      </div></footer>
    </>
  )
}
