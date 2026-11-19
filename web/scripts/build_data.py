"""Pre-processa xlsx + shapefile em JSON estaticos para o PWA.

Uso (da raiz do repo):  python web/scripts/build_data.py
Saidas em web/public/data/: roads.geojson, segments.json, classes.json, models.json, walkthrough.json
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import LineString, MultiLineString, mapping

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "web" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("eb", ROOT / "modelos" / "modelo_glmm_eb_vmd.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)
TARGETS = eb.TARGETS

def clean(o):
    """Troca NaN/Inf por None (JSON valido)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


R = lambda x, n=1: None if pd.isna(x) else round(float(x), n)  # noqa: E731

# ── Planilhas de estimativas ────────────────────────────────────────────────
ebx = pd.read_excel(DATA / "modelo_volume_glmm_eb_20261118_184542.xlsx")
carx = pd.read_excel(DATA / "modelo_volume_glmm_car_20261118_195410.xlsx")
for d in (ebx, carx):
    d["origem"] = d["origem"].map(lambda s: "contagem" if str(s).startswith("Contagem") else "estimativa")
key = ["sre", "sentido_trafego"]
assert len(ebx) == len(carx) == 3854
ebx = ebx.set_index(key)
carx = carx.set_index(key).loc[ebx.index]
assert (ebx["origem"] == carx["origem"]).all()
assert int((ebx["origem"] == "contagem").sum()) == 406
obs = ebx["origem"] == "contagem"
assert np.allclose(ebx.loc[obs, TARGETS].to_numpy(float), carx.loc[obs, TARGETS].to_numpy(float))

# IC90 do CAR por classe
ci = {}
res = pd.ExcelFile(DATA / "glmm_car_multitarget_resultados_20261118_195410.xlsx")
for t in TARGETS:
    if f"GLMM_GRAFO_prev_{t}" not in res.sheet_names:  # 3s1: zero em todo o dataset, sem modelo
        ci[t] = pd.DataFrame({"IC90_inferior": [], "IC90_superior": []},
                             index=pd.MultiIndex.from_arrays([[], []], names=["sre", "sentido"]))
        continue
    p = res.parse(f"GLMM_GRAFO_prev_{t}").set_index(["sre", "sentido"])
    ci[t] = p[["IC90_inferior", "IC90_superior"]]

# ── Dataset de features ─────────────────────────────────────────────────────
raw = pd.read_excel(DATA / "vmd_SRE_SET-26_features_2026.xlsx")
raw["sentido_trafego"] = raw["sentido"]
feat = raw.set_index(key)
vmd_chk = feat.loc[obs[obs].index, TARGETS].sum(axis=1)
vmd_is_sum = bool(np.allclose(vmd_chk, feat.loc[obs[obs].index, "Vmd"], rtol=0.02))
print("Vmd == soma das 35 classes:", vmd_is_sum)

idx = ebx.index
sre_l = idx.get_level_values(0)
sent_l = idx.get_level_values(1)
tot_eb = ebx[TARGETS].sum(axis=1)
tot_car = carx[TARGETS].sum(axis=1)

FEATS = ["classe", "situacao", "perim_urb", "regional", "extensao", "km_inicial", "km_final",
         "media_empresas_ativas", "media_populacao_residente", "media_pib", "media_frotas_ativas",
         "distancia_cid_1", "inicio", "fim", "federal", "trecho"]
F = feat.loc[idx, FEATS]
segments = []
for i, (s, d) in enumerate(idx):
    r = F.iloc[i]
    segments.append({
        "sre": s, "s": "C" if d == "CRESCENTE" else "D",
        "o": "c" if obs.iloc[i] else "e",
        "eb": R(tot_eb.iloc[i]), "car": R(tot_car.iloc[i]),
        "go": str(ebx["rodovia"].iloc[i]), "reg": int(r["regional"]),
        "cl": r["classe"], "sit": r["situacao"], "pu": str(r["perim_urb"]).lower() == "s",
        "ext": R(r["extensao"], 2), "ki": R(r["km_inicial"], 3), "kf": R(r["km_final"], 3),
        "emp": R(r["media_empresas_ativas"], 0), "pop": R(r["media_populacao_residente"], 0),
        "pib": R(r["media_pib"], 0), "fro": R(r["media_frotas_ativas"], 0),
        "dist": R(r["distancia_cid_1"], 0),
        "ini": r["inicio"] if isinstance(r["inicio"], str) else None,
        "fim": r["fim"] if isinstance(r["fim"], str) else None,
        "tr": r["trecho"] if isinstance(r["trecho"], str) else None,
    })
(OUT / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

# classes: ordem = segments; arrays por classe
classes = {"order": TARGETS, "eb": {}, "car": {}, "lo": {}, "hi": {}}
for t in TARGETS:
    classes["eb"][t] = [R(v) for v in ebx[t].to_numpy()]
    classes["car"][t] = [R(v) for v in carx[t].to_numpy()]
    c = ci[t].reindex(idx)
    lo = c["IC90_inferior"].to_numpy(float)
    hi = c["IC90_superior"].to_numpy(float)
    classes["lo"][t] = [None if np.isnan(v) or o else R(v) for v, o in zip(lo, obs.to_numpy())]
    classes["hi"][t] = [None if np.isnan(v) or o else R(v) for v, o in zip(hi, obs.to_numpy())]
(OUT / "classes.json").write_text(json.dumps(classes, separators=(",", ":")), encoding="utf-8")

# ── Geometria ───────────────────────────────────────────────────────────────
rd = shapefile.Reader(str(ROOT / "mapa" / "SRE-GO_2026_SET_SGP.shp"), encoding="utf-8")
names = [f[0] for f in rd.fields[1:]]
sre_in_model = set(sre_l)
feats = []
for sr in rd.iterShapeRecords():
    p = dict(zip(names, sr.record))
    parts = [[(x, y) for x, y, *_ in sr.shape.points[a:b]] for a, b in
             zip(list(sr.shape.parts), list(sr.shape.parts[1:]) + [len(sr.shape.points)])]
    geom = MultiLineString([pt for pt in parts if len(pt) > 1]) if len(parts) > 1 else LineString(parts[0])
    geom = geom.simplify(0.0002, preserve_topology=False)
    if geom.is_empty:
        continue
    g = mapping(geom)
    g["coordinates"] = json.loads(json.dumps(g["coordinates"], default=list))

    def rnd(c):
        return [round(c[0], 5), round(c[1], 5)] if isinstance(c[0], (int, float)) else [rnd(x) for x in c]
    g["coordinates"] = rnd(g["coordinates"])
    feats.append({"type": "Feature", "id": len(feats),
                  "properties": {"sre": str(p["sre"]).strip(), "go": str(p["go"]).strip(),
                                 "sit": str(p["situacao"]).strip(), "in": str(p["sre"]).strip() in sre_in_model},
                  "geometry": g})
(OUT / "roads.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                                              separators=(",", ":")), encoding="utf-8")
shp_sres = {f["properties"]["sre"] for f in feats}
print("shp features:", len(feats), "| SRE no modelo com geometria:", len(sre_in_model & shp_sres),
      "de", len(sre_in_model))

# ── Modelos / metricas ──────────────────────────────────────────────────────
car_met = res.parse("GLMM_GRAFO_metricas")
car_sum = json.loads((DATA / "glmm_car_multitarget_resumo_20261118_195410.json").read_text(encoding="utf-8"))
eb_sum = json.loads((DATA / "glmm_eb_resumo_20261118_184542.json").read_text(encoding="utf-8"))
coefs, hypers = {}, {}
for t in ("Passeio", "2c", "3s3"):
    coefs[t] = json.loads(res.parse(f"GLMM_GRAFO_coef_{t}").round(4).to_json(orient="records"))
    hypers[t] = json.loads(res.parse(f"GLMM_GRAFO_hiper_{t}").round(4).to_json(orient="records"))
models = {
    "eb": {"cv": eb_sum["cv_agrupada_por_sre"], "seg": eb_sum["segundos_total"],
           "params": {"ridge": eb.RIDGE_ALPHA, "scale_go": eb.SCALE_GO, "scale_reg": eb.SCALE_REG,
                      "l_road": eb.L_ROAD_KM, "l_geo": eb.L_GEO_KM, "k_geo": eb.K_GEO, "shrink": eb.SHRINK}},
    "car": {"seg": car_sum["tempo_execucao_s"],
            "metrics": json.loads(car_met.round(4).to_json(orient="records")),
            "coef": coefs, "hyper": hypers},
    "n_rows": len(idx), "n_obs": int(obs.sum()), "n_sre": len(sre_in_model),
    "n_sre_obs": int(len({s for s, o in zip(sre_l, obs) if o})),
    "n_rows_total": int(len(raw)), "situacoes": raw["situacao"].value_counts().to_dict(),
}
(OUT / "models.json").write_text(json.dumps(clean(models), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

# ── Walkthrough EB real ─────────────────────────────────────────────────────
data = eb.apply_annual_growth(raw.drop(columns=["sentido_trafego"]), 2026, 0.03)
data = data.loc[data["situacao"].astype(str).str.upper().isin(eb.ALLOWED_SITUATIONS)].reset_index(drop=True)
observed = data[TARGETS].notna().all(axis=1).to_numpy()
obs_i = np.flatnonzero(observed)
D = eb.build_design(data)
YL = np.log1p(data[TARGETS].fillna(0.0).to_numpy(float).clip(min=0))
k = TARGETS.index("Passeio")
# reproduz fit_predict para exportar intermediarios
road_tr, geo_tr, ex_tr = eb.spatial_terms(D, obs_i, obs_i, YL[obs_i])
all_i = np.arange(len(data))
road_all, geo_all, ex_all = eb.spatial_terms(D, all_i, obs_i, YL[obs_i])
base_tr = np.column_stack([D["X"][obs_i], ex_tr, road_tr[:, k], geo_tr[:, k]])
base_all = np.column_stack([D["X"], ex_all, road_all[:, k], geo_all[:, k]])
xm, ym = base_tr.mean(0), YL[obs_i, k].mean()
Xc = base_tr - xm
b = np.linalg.solve(Xc.T @ Xc + eb.RIDGE_ALPHA * np.eye(Xc.shape[1]), Xc.T @ (YL[obs_i, k] - ym))
pred = (base_all - xm) @ b + ym
chk = np.expm1(pred)
ref = ebx.reset_index().set_index(["sre", "sentido_trafego"])
mine = pd.Series(chk, index=pd.MultiIndex.from_arrays([data["sre"], data["sentido"]]))
sel = ~observed
diff = np.abs(mine[sel].to_numpy() - ref.loc[mine[sel].index, "Passeio"].to_numpy()).max()
print("walkthrough EB vs planilha (Passeio), max |diff| veic/dia:", diff)
assert diff < 0.1

lat_m = np.degrees(D["lat"]); lon_m = np.degrees(D["lon"])
nfix = D["X"].shape[1]
fixed_names = (["log empresas", "log dist. polo", "log população", "log PIB", "log frota", "log extensão",
                "perímetro urbano", "conexão c/ outra rodovia", "liga a BR", "sentido decrescente", "latitude", "longitude"])
n_dummy_cls = pd.get_dummies(data["classe"].astype(str), drop_first=True).shape[1]
n_dummy_sit = pd.get_dummies(data["situacao"].astype(str), drop_first=True).shape[1]
groups = {"fixed": (0, 12), "classe": (12, 12 + n_dummy_cls), "situacao": (12 + n_dummy_cls, 12 + n_dummy_cls + n_dummy_sit),
          "go": (nfix - len(np.unique(D["go"])) - data["regional"].nunique(), nfix - data["regional"].nunique()),
          "reg": (nfix - data["regional"].nunique(), nfix)}
rng = np.random.default_rng(3)


def pick(cond):
    c = np.flatnonzero(cond)
    return int(c[len(c) // 2]) if len(c) else None


cands = []
# 1) mesma rodovia com vizinhos contados; 2) isolada; 3) urbana
w_road = road_all[:, k]
has_road = ex_all[:, 0] > np.log1p(0.8)
cands.append(("Com vizinhos contados na mesma rodovia", pick(sel & has_road & (data["perim_urb"].astype(str).str.lower() == "n").to_numpy())))
cands.append(("Sem contagem na rodovia (só vizinhança geográfica)", pick(sel & ~(ex_all[:, 0] > np.log1p(0.05)))))
cands.append(("Trecho em perímetro urbano", pick(sel & (data["perim_urb"].astype(str).str.lower() == "s").to_numpy() & has_road)))
examples = []
for label, i in cands:
    if i is None:
        continue
    row = data.iloc[i]
    # vizinhos usados
    same = D["sre"][obs_i] == D["sre"][i]
    gap = np.maximum(0.0, np.maximum(D["km_lo"][i] - D["km_hi"][obs_i], D["km_lo"][obs_i] - D["km_hi"][i]))
    gap[(D["go"][obs_i] != D["go"][i]) | same] = np.inf
    w = np.exp(-gap / eb.L_ROAD_KM)
    top_r = np.argsort(-w)[:4]
    la, lo = D["lat"], D["lon"]
    a = (np.sin((la[i] - la[obs_i]) / 2) ** 2 + np.cos(la[i]) * np.cos(la[obs_i]) * np.sin((lo[i] - lo[obs_i]) / 2) ** 2)
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1))); dist[same] = np.inf
    nn = np.argsort(dist)[:eb.K_GEO]
    contrib = (base_all[i] - xm) * b
    cg = {n: float(contrib[a:bb].sum()) for n, (a, bb) in groups.items()}
    cg["fixed_each"] = {n: float(contrib[j]) for j, n in enumerate(fixed_names)}
    nb = lambda ii: [{"sre": str(D["sre"][obs_i][j]), "go": str(D["go"][obs_i][j]),  # noqa: E731
                      "lat": float(lat_m[obs_i][j]), "lon": float(lon_m[obs_i][j]),
                      "vmd": float(np.expm1(YL[obs_i][j]).sum()), "passeio": float(np.expm1(YL[obs_i][j, k]))} for j in ii]
    examples.append({
        "label": label, "sre": str(row["sre"]), "sentido": str(row["sentido"]), "go": int(row["go"]),
        "lat": float(lat_m[i]), "lon": float(lon_m[i]), "classe": row["classe"], "situacao": row["situacao"],
        "target": "Passeio", "ym": float(ym), "pred_log": float(pred[i]), "pred": float(chk[i]),
        "s_road": float(road_all[i, k]), "s_geo": float(geo_all[i, k]),
        "w_road": [float(w[j]) for j in top_r], "gap_km": [float(gap[j]) for j in top_r],
        "w_geo": [float(np.exp(-dist[j] / eb.L_GEO_KM)) for j in nn], "dist_km": [float(dist[j]) for j in nn],
        "road_nb": nb(top_r[w[top_r] > 1e-6]), "geo_nb": nb(nn),
        "contrib": cg, "coef_road": float(b[-2]), "coef_geo": float(b[-1]),
    })
(OUT / "walkthrough.json").write_text(json.dumps(clean(examples), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("exemplos:", [(e["sre"], e["sentido"], round(e["pred"])) for e in examples])
for f in sorted(OUT.iterdir()):
    print(f"{f.name:20s} {f.stat().st_size/1e6:6.2f} MB")
