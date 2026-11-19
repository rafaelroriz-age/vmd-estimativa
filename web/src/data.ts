// Camada de dados. Hoje: JSON estático em /data. Amanhã: Supabase (mesma interface DataSource).
export interface Seg {
  sre: string; s: 'C' | 'D'; o: 'c' | 'e'; eb: number; car: number; go: string; reg: number
  cl: string; sit: string; pu: boolean; ext: number; ki: number; kf: number
  emp: number | null; pop: number | null; pib: number | null; fro: number | null; dist: number | null
  ini: string | null; fim: string | null; tr: string | null
}
export interface ClassData {
  order: string[]; eb: Record<string, (number | null)[]>; car: Record<string, (number | null)[]>
  lo: Record<string, (number | null)[]>; hi: Record<string, (number | null)[]>
}
export interface Walk {
  label: string; sre: string; sentido: string; go: number; lat: number; lon: number; classe: string; situacao: string
  target: string; ym: number; pred_log: number; pred: number; s_road: number; s_geo: number
  w_road: number[]; gap_km: number[]; w_geo: number[]; dist_km: number[]
  road_nb: Nb[]; geo_nb: Nb[]; coef_road: number; coef_geo: number
  contrib: Record<string, number | Record<string, number>>
}
export interface Nb { sre: string; go: string; lat: number; lon: number; vmd: number; passeio: number }
export interface Models {
  eb: { cv: Record<string, number>; seg: number; params: Record<string, number> }
  car: { seg: number; metrics: Record<string, number | string>[]; coef: Record<string, Record<string, number | string>[]>; hyper: Record<string, Record<string, number | string>[]> }
  n_rows: number; n_obs: number; n_sre: number; n_sre_obs: number; n_rows_total: number; situacoes: Record<string, number>
}
export interface Store {
  segs: Seg[]; roads: GeoJSON.FeatureCollection; models: Models; walk: Walk[]
  index: Map<string, number>; bySre: Map<string, { C?: number; D?: number }>
  loadClasses(): Promise<ClassData>
}
export interface DataSource { load(): Promise<Store> }

const j = async <T,>(p: string): Promise<T> => (await fetch(`${import.meta.env.BASE_URL}data/${p}`)).json()

export const StaticSource: DataSource = {
  async load() {
    const [segs, roads, models, walk] = await Promise.all([
      j<Seg[]>('segments.json'), j<GeoJSON.FeatureCollection>('roads.geojson'),
      j<Models>('models.json'), j<Walk[]>('walkthrough.json'),
    ])
    const index = new Map<string, number>(); const bySre = new Map<string, { C?: number; D?: number }>()
    segs.forEach((s, i) => {
      index.set(`${s.sre}|${s.s}`, i)
      const e = bySre.get(s.sre) ?? {}; e[s.s] = i; bySre.set(s.sre, e)
    })
    let cache: Promise<ClassData> | null = null
    return { segs, roads, models, walk, index, bySre, loadClasses: () => (cache ??= j<ClassData>('classes.json')) }
  },
}

export const nf0 = new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 0 })
export const nf1 = new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 1 })
export const nf2 = new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 2 })
export const fmt = (v: number | null | undefined, d = 0) => (v == null ? '—' : d === 0 ? nf0.format(v) : d === 1 ? nf1.format(v) : nf2.format(v))
