# VMD por SRE — PWA de storytelling

Guia interativo (pt-BR) de como `modelo_glmm_eb_vmd.py` e `modelo_glmm_car_vmd_v4.py` estimam o VMD, com mapa das SREs
(azul = contagem, vermelho = estimativa, cinza = fora do escopo).

```powershell
cd web
npm install
npm run data      # regenera public/data/* a partir de ../data e ../mapa (Python com pandas, pyshp, shapely)
npm run dev       # desenvolvimento
npm run build; npm run preview   # PWA (service worker só no build)
```

- Dados: `public/data/` (`segments.json`, `classes.json`, `roads.geojson`, `models.json`, `walkthrough.json`).
- `walkthrough.json` é recalculado com as funções reais do modelo EB e conferido contra a planilha EB (< 1e-9).
- Camada de dados trocável: `src/data.ts` (`DataSource`). Rascunho do esquema Supabase em `../docs/supabase_schema.sql`.
- Mapa base OSM é opcional (checkbox no explorador); sem ele o app funciona 100% offline.
