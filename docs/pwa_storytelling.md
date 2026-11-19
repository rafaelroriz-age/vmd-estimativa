# PWA de storytelling: GLMM-EB e GLMM-CAR (VMD por SRE)

Documentação do app em `web/`. Público: quem for manter o app ou revisar a metodologia.

## 1. Objetivo

Explicar, passo a passo e em pt-BR, como os dois modelos de estimativa de VMD produzem
volumes para segmentos sem contagem, e mostrar o resultado num mapa das SREs de Goiás:

- **azul** = trecho com contagem;
- **vermelho** = trecho estimado;
- **cinza** = fora do escopo dos modelos (situação diferente de PAV/DUP/EOD).

## 2. Insumos

| Arquivo | Conteúdo | Uso no app |
|---|---|---|
| `data/vmd_SRE_SET-26_features_2026.xlsx` | 4.864 linhas = 2.432 SREs × 2 sentidos; 406 linhas com contagem | features e contagens |
| `data/modelo_volume_glmm_eb_20261118_184542.xlsx` | estimativas EB (35 classes) | saída oficial EB |
| `data/modelo_volume_glmm_car_20261118_195410.xlsx` | estimativas CAR (35 classes) | saída oficial CAR |
| `data/glmm_car_multitarget_resultados_20261118_195410.xlsx` | coeficientes, hiperparâmetros, IC90, métricas do CAR | seção CAR e validação |
| `data/glmm_car_multitarget_resumo_20261118_195410.json` | tempo de execução do CAR | comparação de custo |
| `data/glmm_eb_resumo_20261118_184542.json` | CV agrupada por SRE do EB | seção validação |
| `mapa/SRE-GO_2026_SET_SGP.shp` | geometria das SREs (SIRGAS 2000, 2.689 feições) | mapa |
| `modelos/modelo_glmm_eb_vmd.py` | funções do EB | walkthrough recalculado |

As planilhas de estimativa já existiam. **O app não reexecuta os modelos.** O CAR leva cerca de 21 min
(MCMC) e não foi rodado de novo.

## 3. Pipeline de dados

`web/scripts/build_data.py` (Python; `npm run data` a partir de `web/`) gera `web/public/data/`:

- `segments.json`: uma entrada por (SRE, sentido): origem (`c` contagem / `e` estimativa), VMD total EB e CAR,
  e features (classe, situação, perímetro urbano, extensão, km, empresas, população, PIB, frota, distância ao polo).
- `classes.json`: arrays por classe (EB, CAR e IC90 do CAR), carregados sob demanda.
- `roads.geojson`: geometria simplificada (Douglas-Peucker, tolerância 2e-4°), 5 casas decimais.
- `models.json`: CV do EB, métricas e coeficientes do CAR, hiperparâmetros do EB.
- `walkthrough.json`: três exemplos de SRE com os intermediários do EB.

Checagens feitas pelo script:

- 3.854 linhas em ambas as planilhas de estimativa, 406 contagens iguais nas duas.
- O walkthrough do EB reproduz a planilha EB com diferença máxima de 3,6e-12 veíc./dia.
- Todas as 1.927 SREs do modelo têm geometria no shapefile.

Pontos conhecidos:

- Classe **3S1** é zero em todo o dataset: não tem modelo e não aparece no CAR.
- A coluna `Vmd` do dataset **não** é a soma das 35 classes; o app usa a soma.
- Os JSON não podem conter `NaN` ou `Infinity` (quebrava o `JSON.parse`); o script converte para `null`.

## 4. Arquitetura do app

```
web/
  scripts/build_data.py   geração dos JSON
  public/data/            dados estáticos
  public/icon*.{svg,png}  ícones do PWA
  src/data.ts             DataSource (StaticSource hoje; Supabase depois)
  src/MapView.tsx         MapLibre GL: camadas, cores, tooltip, overlays
  src/charts.tsx          histograma, barras, cascata, dispersão (D3)
  src/Scrolly.tsx         scrollytelling (texto rola, visual fixo)
  src/Tex.tsx             fórmulas (KaTeX)
  src/story/Intro.tsx     topo, dado, features
  src/story/Models.tsx    passo a passo EB e CAR
  src/story/Compare.tsx   EB × CAR e validação
  src/story/Explorer.tsx  mapa com filtros e painel de detalhe
  vite.config.ts          vite-plugin-pwa (autoUpdate, precache)
```

Versões relevantes: React 19, Vite 8, MapLibre GL **4.7.1** (a 6.x não carregava o worker com o Vite),
D3 7, KaTeX, vite-plugin-pwa 2.

### Mapa

- Motor: MapLibre GL, não Leaflet. Pan, zoom, clique e tooltip se comportam como no Leaflet.
- Fundo: tiles do OpenStreetMap (`tile.openstreetmap.org`), ligado por padrão, com checkbox no explorador.
  Sem rede, o mapa continua funcionando só com as linhas.
- Cores validadas com `dataviz/scripts/validate_palette.js` nos temas claro e escuro
  (ΔE de CVD ≥ 22, contraste ≥ 3:1).
- Espessura ∝ √VMD; estimativas e contagens se distinguem por cor, não só por espessura.

## 5. Metodologia exibida (resumo)

**EB (empírico-Bayes, ridge).** Para cada classe k:

- alvo `z = ln(1 + y)`, com zeros mantidos;
- efeitos fixos padronizados (log de empresas, distância, população, PIB, frota, extensão; dummies de classe,
  situação, perímetro; latitude, longitude, sentido, ligação com BR);
- efeitos aleatórios de rodovia (GO) e regional, penalizados com escala 0,5;
- vizinhança: média suavizada ao longo da rodovia (peso `exp(-gap_km/30)`) e pelos 6 vizinhos geográficos
  (peso `exp(-d_km/30)`), encolhida para a média global (λ = 0,3), excluindo a própria SRE;
- ajuste ridge em forma fechada, `β = (XᵀX + αI)⁻¹ Xᵀ z`, com α = 3; saída `ŷ = max(0, exp(ẑ) − 1)`.

Hiperparâmetros fixados por CV agrupada por SRE (5 folds × 3 repetições). Sem intervalo de incerteza.

**CAR (Gama, bayesiano).** Para cada classe:

- verossimilhança `y ~ Gama(φ, φ/μ)`, com `ln μ = β₀ + xβ + u_GO + u_reg + u_CAR`;
- priors: `β ~ N(0, 1,5)`, desvios de GO e regional ~ Meia-Normal(1), α ~ U(0; 0,99), τ ~ Gama(2, 1), φ ~ Gama(2; 0,1);
- CAR sobre o grafo de segmentos consecutivos (gap ≤ 0,1 km, mesmo sentido);
- NUTS, 4 cadeias, 1.000 amostras (tune 1.500);
- predição fora da amostra com `u_CAR` ≈ α × média dos 1–2 vizinhos contados mais próximos;
- IC90 aproximado por `μ · exp(±1,645 s)`, sem a incerteza dos coeficientes (tende a ser otimista).

**Validação.** EB: R² 0,56, MdAPE 37%, 40% das SREs com erro ≤ 30% e 69% com erro ≤ 50% (VMD total, CV agrupada).
CAR: R² in-sample, pouco informativo (sobreajuste em classes com poucas contagens); R̂ máximo 1,106,
ESS mínimo 29, 5 divergências. Algumas classes raras não atingem R̂ ≤ 1,01; o app lista essas classes.

## 6. Publicação

- Repositório: `rafaelroriz-age/vmd-estimativa`, branch `main`.
- Workflow `.github/workflows/pages.yml`: em push para `main` que altere `web/**` ou o próprio workflow,
  roda `npm ci`, `npm run build` em `web/` e publica `web/dist` no GitHub Pages.
- URL esperada: `https://rafaelroriz-age.github.io/vmd-estimativa/`.
- `vite.config.ts` usa `base: './'`, então os caminhos funcionam em subpasta.
- **Configuração única no GitHub:** Settings → Pages → Source = **GitHub Actions**. Sem isso o deploy não publica.
- O `.gitignore` exclui `node_modules/`, `dist/`, `__pycache__/` e `*.pkl`.
- O repositório inclui o shapefile de 28 MB. Abaixo do limite de 100 MB do GitHub, mas acima de 50 MB o GitHub avisa.

## 7. Como rodar localmente

```powershell
cd web
npm install
npm run data        # opcional: regenera public/data (precisa do Python com pandas, pyshp e shapely)
npm run dev         # desenvolvimento
npm run build       # gera dist/ e o service worker
npm run preview     # testa o PWA
```

O service worker só é gerado no build, então o modo offline só funciona com `preview` ou com a página publicada.

## 8. Limitações conhecidas

- O CAR não foi reexecutado: as estimativas vêm das planilhas de 18/11/2026.
- A faixa de 90% do CAR é aproximada e otimista.
- O mapa usa tiles do OSM, que dependem de rede. Não há cache específico de tiles além do runtime do navegador.
- O app é estático: não há autenticação nem escrita. A migração para Supabase está prevista como `SupabaseSource`
  em `web/src/data.ts`, com esquema rascunhado em `docs/supabase_schema.sql` (ainda não aplicado).
- Não foi feito teste de layout em celular real nem de modo offline completo.

## 9. Histórico de decisões

- Estimativas: usar as planilhas prontas em vez de rodar o MCMC (decisão do usuário).
- Stack: Vite + React + TypeScript + MapLibre + D3 (escolha do usuário sobre HTML puro e Leaflet).
- Mapa: fundo OSM ligado por padrão, a pedido do usuário, que queria comportamento de Leaflet/OSM.
- Repositório: primeiro commit em `main` criado a partir do estado local, pois o remoto estava vazio.
