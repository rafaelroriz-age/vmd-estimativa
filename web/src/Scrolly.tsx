import { useEffect, useRef, useState, type ReactNode } from 'react'

export interface Step { tag: string; body: ReactNode }

/** Scrollytelling: texto à direita rola, visual fixo à esquerda reage ao passo ativo. */
export default function Scrolly({ steps, visual, onStep }: { steps: Step[]; visual: (i: number) => ReactNode; onStep?: (i: number) => void }) {
  const [active, setActive] = useState(0)
  const refs = useRef<(HTMLDivElement | null)[]>([])
  useEffect(() => {
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting) { const i = Number((e.target as HTMLElement).dataset.i); setActive(i); onStep?.(i) } })
    }, { rootMargin: '-45% 0px -45% 0px' })
    refs.current.forEach((r) => r && io.observe(r))
    return () => io.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [steps.length])
  return (
    <div className="scrolly">
      <div className="visual">{visual(active)}</div>
      <div className="steps">
        {steps.map((s, i) => (
          <div key={i} data-i={i} ref={(el) => { refs.current[i] = el }} className={`step ${i === active ? 'on' : ''}`}>
            <div><span className="n">{s.tag}</span>{s.body}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
