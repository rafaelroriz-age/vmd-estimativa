"""Executa o GLMM Bayesiano Gamma/log com efeito CAR para volumes veiculares.

Este arquivo e um ponto de entrada independente da interface Streamlit.
Autocontido: todo o nucleo matematico do GLMM+CAR/GP (antes vendorizado em
``model_glmm_graph.py``) esta embutido neste arquivo, sem nenhuma dependencia
de outros modulos do projeto. Antes do treino, corrige os targets
observados para o ano de referencia e monta a planilha operacional
internamente (mesma estrutura de colunas de ``02_dados/modelo_volume.xlsx``),
sem depender desse arquivo como entrada.

v4: entrada unica. O dataset ja contem a malha rodoviaria completa (todas as
SREs, por sentido, com ou sem contagem), entao nao ha mais ``--road-network``.
O esqueleto sre+sentido, os atributos estruturais, ``principal`` e
``sre_referencia`` sao derivados do proprio dataset
(``prepare_single_dataset``). Modelo, filtros e saidas sao os mesmos da v2.

Uso:
    python modelo_glmm_car_vmd_v4.py --check-data
    python modelo_glmm_car_vmd_v4.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az

warnings.filterwarnings("ignore")

try:  # pragma: no cover - depende do terminal
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

try:
    import pymc as pm
    import pytensor.tensor as pt

    _HAS_BAYES = True
except ImportError:  # pragma: no cover - depende de ambiente
    _HAS_BAYES = False

try:
    import nutpie  # noqa: F401

    _HAS_NUTPIE = True
except ImportError:  # pragma: no cover - depende de ambiente
    _HAS_NUTPIE = False


DELIVERY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DELIVERY_DIR.parent
DEFAULT_DATASET = (
    PROJECT_ROOT / "02_dados" / "vmd_SRE_SET-26_features_2026.xlsx"
)
DEFAULT_OUTPUT_DIR = DELIVERY_DIR / "outputs"

# Nomes alternativos aceitos no dataset unico -> nome usado pelo modelo.
# So sao aplicados quando o nome canonico nao existe no arquivo.
COLUMN_ALIASES = {
    "lat_inicial": ("lat_inicia", "latitude_inicial"),
    "lon_inicial": ("lon_inicia", "longitude_inicial"),
    "lat_final": ("latitude_final",),
    "lon_final": ("longitude_final",),
    "km_inicial": ("km_inicial_upd",),
    "km_final": ("km_final_upd",),
    "regional_macro": ("regional",),
    "distancia_cid_1_2_3": ("distancia_cid_1",),
}


TARGETS = [
    "Passeio", "Van", "Pickup", "Moto", "2cb", "3cb", "4cb", "2sb1",
    "2ib2", "2c", "3c", "4c", "4cd", "2s1", "2s2", "3s1", "2s3",
    "3s2", "2i2", "2i3", "2j3", "3i2", "3s3", "3i3", "3j3", "4r4",
    "2c2", "2c3", "3c2", "3c3", "3d3", "3d4", "3q4", "3q6", "3t6",
]
TARGET = "vmd"
ALLOWED_SITUATIONS = frozenset({"PAV", "DUP", "EOD"})
MODEL_KIND = "glmm_car"
FIXED_NUMERIC = [
    "log_media_empresas_ativas",
    "log_distancia_cid_1_2_3",
]
FIXED_CATEG = ["classe", "situacao", "perim_urb"]
INCLUDE_REGION_RE = True
GEO_KNN_K = 0
SENTIDOS = ("CRESCENTE", "DECRESCENTE")


# ─────────────────────────────────────────────────────────────────────────────
# Núcleo do modelo GLMM + grafo (CAR/GP) — antes vendorizado em
# model_glmm_graph.py; embutido aqui para que este script seja autocontido.
# ─────────────────────────────────────────────────────────────────────────────

def feature_engineering(
    df: pd.DataFrame,
    targets: list[str],
    group_col: str = "regional_macro",
) -> pd.DataFrame:
    """
    Append engineered features to *df* (original columns unchanged):

    1. **Log-transforms** of skewed numeric columns  → ``log_<col>``
    2. **Socioeconomic ratios** (per-capita, per-km)  → e.g. ``frotas_per_capita``
    3. **Log-transforms of ratio features**           → ``log_<ratio>``
    4. **Target-based location features** — mean and median of each target's
       observed values, grouped by ``go`` and ``group_col``, propagated to
       ALL rows (including unobserved ones).          → ``<target>_media_go``, etc.

    Groups for (4) are built from OBSERVED rows only, so unobserved SREs
    receive only the group-level reference value, not their own target.
    """
    df = df.copy()

    # ── 1. Log-transforms of base numeric features ────────────────────────────
    _log_base = [
        "distancia_cid_1", "distancia_cid_1_2_3",
        "media_pib", "media_populacao_residente",
        "media_frotas_ativas", "media_empresas_ativas",
        "extensao",
    ]
    for col in _log_base:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # ── 2. Socioeconomic ratio features ──────────────────────────────────────
    _pop = df["media_populacao_residente"] if "media_populacao_residente" in df.columns else None
    _ext = df["extensao"]                  if "extensao"                  in df.columns else None
    _pib = df["media_pib"]                 if "media_pib"                 in df.columns else None
    _fro = df["media_frotas_ativas"]       if "media_frotas_ativas"       in df.columns else None
    _emp = df["media_empresas_ativas"]     if "media_empresas_ativas"     in df.columns else None

    if _fro is not None and _pop is not None:
        df["frotas_per_capita"]   = _fro / _pop.replace(0, np.nan)
    if _pib is not None and _pop is not None:
        df["pib_per_capita"]      = _pib / _pop.replace(0, np.nan)
    if _emp is not None and _pop is not None:
        df["empresas_per_capita"] = _emp / _pop.replace(0, np.nan)
    if _fro is not None and _ext is not None:
        df["frotas_per_km"]       = _fro / _ext.replace(0, np.nan)
    if _emp is not None and _ext is not None:
        df["empresas_per_km"]     = _emp / _ext.replace(0, np.nan)
    if _pib is not None and _ext is not None:
        df["pib_per_km"]          = _pib / _ext.replace(0, np.nan)

    # ── 3. Log-transforms of ratio features ──────────────────────────────────
    _ratio_cols = [
        "frotas_per_capita", "pib_per_capita", "empresas_per_capita",
        "frotas_per_km", "empresas_per_km", "pib_per_km",
    ]
    for col in _ratio_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # ── 3b. Excess features (relative to regional average) ───────────────────
    if _fro is not None and _ext is not None and group_col in df.columns:
        reg_avg_frotas_km = df.groupby(group_col)["frotas_per_km"].transform("mean")
        df["excesso_frotas_km"] = df["frotas_per_km"] / reg_avg_frotas_km.clip(lower=1)
        df["log_excesso_frotas_km"] = np.log1p(df["excesso_frotas_km"].clip(lower=0))
    if _pib is not None and _ext is not None and group_col in df.columns:
        reg_avg_pib_km = df.groupby(group_col)["pib_per_km"].transform("mean")
        df["excesso_pib_km"] = df["pib_per_km"] / reg_avg_pib_km.clip(lower=1)
        df["log_excesso_pib_km"] = np.log1p(df["excesso_pib_km"].clip(lower=0))

    # ── 4. Target-based location features ────────────────────────────────────
    _loc_groups: dict[str, str] = {}
    if "go" in df.columns:
        _loc_groups["go"] = "go"
    if group_col in df.columns and group_col not in _loc_groups:
        suffix = "regional" if "regional" in group_col.lower() else group_col
        _loc_groups[group_col] = suffix

    for target in targets:
        if target not in df.columns:
            continue
        obs_mask = df[target].notna()
        if obs_mask.sum() == 0:
            continue

        loc_cols = []
        for gcol, suffix in _loc_groups.items():
            stats = df.loc[obs_mask].groupby(gcol)[target].agg(["mean", "median", "std"])
            mean_col = f"{target}_media_{suffix}"
            med_col  = f"{target}_mediana_{suffix}"
            cv_col   = f"{target}_cv_{suffix}"

            df[mean_col] = df[gcol].map(stats["mean"])
            df[med_col]  = df[gcol].map(stats["median"])
            cv_series = stats["std"] / stats["mean"].clip(lower=1)
            df[cv_col] = df[gcol].map(cv_series)

            loc_cols.extend([mean_col, med_col, cv_col])

        if len(loc_cols) >= 2:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler as LocScaler

            X_loc = df[loc_cols].fillna(df[loc_cols].median()).values.astype(float)
            X_loc_scaled = LocScaler().fit_transform(X_loc)
            pca = PCA(n_components=1)
            df[f"localizacao_{target}"] = pca.fit_transform(X_loc_scaled)[:, 0]

    return df


def normalize_group_label(v) -> str:
    """Normaliza o valor de uma coluna de grupo para string limpa.
    Converte floats inteiros para string sem casas decimais: 1.0 → '1'."""
    s = str(v)
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return s


def build_resultados_wide(
    per_target_rows: dict[str, pd.DataFrame],
    id_cols: tuple[str, ...] = ("sre", "sentido", "go", "regiao"),
) -> pd.DataFrame:
    """Combina os DataFrames por-target numa única planilha "wide" — uma
    linha por segmento, com colunas ``{target}_observado``/``{target}_ajustado``/
    ``{target}_residuo``/``{target}_erro_pct`` por target e uma coluna
    ``origem`` unificada (``contagem``/``estimado``/``misto``)."""
    merged: pd.DataFrame | None = None
    origem_cols: list[str] = []

    for target, fr in per_target_rows.items():
        origem_col = f"_origem_{target}"
        origem_cols.append(origem_col)
        present_id = [c for c in id_cols if c in fr.columns]

        if merged is None:
            merged = fr.copy()
            continue

        merge_on = [c for c in present_id if c in merged.columns]
        if merge_on:
            merged = merged.merge(fr, on=merge_on, how="outer", suffixes=("", "_dup"))
            for c in present_id:
                dup = f"{c}_dup"
                if dup in merged.columns:
                    merged[c] = merged[c].fillna(merged[dup])
                    merged.drop(columns=[dup], inplace=True)
        else:
            merged = merged.join(fr, how="outer")

    if merged is None:
        return pd.DataFrame()

    def _origem_final(row):
        vals = {row[c] for c in origem_cols if c in row.index and pd.notna(row[c])}
        if vals == {"contagem"}:
            return "contagem"
        if vals == {"estimado"}:
            return "estimado"
        if vals:
            return "misto"
        return np.nan

    merged["origem"] = merged.apply(_origem_final, axis=1)
    merged = merged.drop(columns=[c for c in origem_cols if c in merged.columns])

    ordered_id = [c for c in id_cols if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in ordered_id and c != "origem"]
    out = merged[ordered_id + other_cols + ["origem"]]

    if "sre" in out.columns:
        out = out.sort_values("sre", kind="stable")
    return out.reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))

    nz = y_true != 0
    pct_err = np.abs(resid[nz] / y_true[nz]) * 100
    mape = float(np.mean(pct_err)) if nz.any() else float("nan")
    w80 = float(100.0 * np.mean(pct_err <= 80)) if nz.any() else float("nan")
    max_err = float(np.max(pct_err)) if nz.any() else float("nan")
    med_err = float(np.median(pct_err)) if nz.any() else float("nan")

    return dict(r2=r2, rmse=rmse, mae=mae, mape=mape, w80=w80,
                max_err=max_err, med_err=med_err)


def per_region_r2(y_true: np.ndarray, y_pred: np.ndarray, regions: np.ndarray) -> dict:
    out: dict[str, float] = {}
    for reg in pd.unique(regions):
        m = regions == reg
        yt, yp = y_true[m], y_pred[m]
        if len(yt) < 2:
            out[str(reg)] = float("nan")
            continue
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        out[str(reg)] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return out


def convergence_diagnostics(idata) -> dict:
    summ = az.summary(idata)
    max_rhat = float(summ["r_hat"].max()) if "r_hat" in summ else float("nan")
    min_ess = float(summ["ess_bulk"].min()) if "ess_bulk" in summ else float("nan")
    try:
        div = az.extract(idata, group="sample_stats", var_names="diverging")
        divergences = int(np.asarray(div.values).sum())
    except (KeyError, ValueError):
        divergences = 0
    return dict(max_rhat=max_rhat, min_ess=min_ess, divergences=divergences)


GO_COL = "go"
REGION_COL = "regional_macro"
TOLERANCIA_KM = 0.1
MODEL_KINDS = ["glmm", "glmm_car", "glmm_car_gp"]
GP_KERNELS = ["exp", "laplacian", "geo"]
SAMPLE_KWARGS = dict(draws=1000, tune=1500, chains=4, target_accept=0.95, random_seed=42)
DOCS_DATA_DIR = PROJECT_ROOT / "00_docs" / "data"
COORDS_PATH = DOCS_DATA_DIR / "sre-coordenadas.csv"
_GP_JITTER = 1e-6
# Defaults internos do núcleo do modelo, usados apenas quando o chamador não
# passa fixed_numeric/fixed_categ explicitamente; distintos de
# FIXED_NUMERIC/FIXED_CATEG acima (a configuração efetivamente usada por este
# script, sempre passada de forma explícita às funções abaixo).
_FIXED_NUMERIC_DEFAULT = ["log_media_empresas_ativas"]
_FIXED_CATEG_DEFAULT = ["situacao", "perim_urb"]
INCLUDE_REGION_RE_DEFAULT = True


def _haversine_km_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Matriz (n,n) de distâncias geodésicas reais (haversine, km) entre
    todos os pares de segmentos em `lat`/`lon`."""
    r = 6371.0088
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2.0) ** 2)
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * r * np.arcsin(np.sqrt(a))


def _haversine_km_cross(lat_a: np.ndarray, lon_a: np.ndarray,
                        lat_b: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    """Matriz (n_a, n_b) de distâncias haversine reais (km) entre dois
    conjuntos diferentes de pontos."""
    r = 6371.0088
    lat_a_r, lon_a_r = np.radians(lat_a), np.radians(lon_a)
    lat_b_r, lon_b_r = np.radians(lat_b), np.radians(lon_b)
    dlat = lat_a_r[:, None] - lat_b_r[None, :]
    dlon = lon_a_r[:, None] - lon_b_r[None, :]
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat_a_r[:, None]) * np.cos(lat_b_r[None, :]) * np.sin(dlon / 2.0) ** 2)
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * r * np.arcsin(np.sqrt(a))


def assign_components(df: pd.DataFrame, tol: float = TOLERANCIA_KM) -> pd.Series:
    """Atribui um id de componente conexo a cada segmento: dentro de cada
    `go`, ordena pela posição física e quebra em novo componente sempre que
    o gap para o segmento anterior excede `tol`. Agrupa também por "sentido"
    quando a coluna existe (dataset "por sentido")."""
    km_min = np.minimum(df["km_inicial"].astype(float), df["km_final"].astype(float))
    km_max = np.maximum(df["km_inicial"].astype(float), df["km_final"].astype(float))
    group_cols = [GO_COL, "sentido"] if "sentido" in df.columns else GO_COL

    comp = pd.Series(index=df.index, dtype=object)
    for key, idx in df.groupby(group_cols).groups.items():
        order = km_min.loc[idx].sort_values().index
        label_prefix = "_".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
        comp_id = 0
        prev_end = None
        for row_idx in order:
            if prev_end is not None and abs(float(km_min.loc[row_idx]) - prev_end) > tol:
                comp_id += 1
            comp.loc[row_idx] = f"{label_prefix}_{comp_id}"
            prev_end = float(km_max.loc[row_idx])
    return comp


def full_graph_degree(df: pd.DataFrame, tol: float = TOLERANCIA_KM) -> pd.Series:
    """Grau de cada segmento no grafo completo (todos os nós), pela mesma
    regra de adjacência de `assign_components`."""
    km_min = np.minimum(df["km_inicial"].astype(float), df["km_final"].astype(float))
    km_max = np.maximum(df["km_inicial"].astype(float), df["km_final"].astype(float))
    group_cols = [GO_COL, "sentido"] if "sentido" in df.columns else GO_COL

    degree = pd.Series(0, index=df.index, dtype=int)
    for _, idx in df.groupby(group_cols).groups.items():
        order = km_min.loc[idx].sort_values().index.to_numpy()
        for i in range(len(order) - 1):
            a_idx, b_idx = order[i], order[i + 1]
            gap = abs(float(km_min.loc[b_idx]) - float(km_max.loc[a_idx]))
            if gap <= tol:
                degree.loc[a_idx] += 1
                degree.loc[b_idx] += 1
    return degree


def prepare_dataframe(df: pd.DataFrame, target: str = "vmd") -> pd.DataFrame:
    """Aplica feature engineering e anota estrutura de grafo (componente
    conexo, posição km, grau no grafo completo, coordenadas geográficas
    quando disponíveis)."""
    df = df.copy()
    df = feature_engineering(df, [target], REGION_COL)
    df[REGION_COL] = df[REGION_COL].map(normalize_group_label)
    df["_km_mid"] = (df["km_inicial"].astype(float) + df["km_final"].astype(float)) / 2.0
    df["_component_id"] = assign_components(df)
    df["_degree_full"] = full_graph_degree(df)
    # Chave única por linha: no dataset "por sentido", `sre` sozinho se repete
    # (2 linhas por segmento físico) — código que precise indexar por linha
    # deve usar `_node_id`, não `sre` diretamente.
    df["_node_id"] = (
        df["sre"].astype(str) + "__" + df["sentido"].astype(str)
        if "sentido" in df.columns else df["sre"].astype(str)
    )

    if "lat" not in df.columns or "lon" not in df.columns:
        if COORDS_PATH.exists():
            coords = pd.read_csv(COORDS_PATH, usecols=["sre", "lat", "lon"])
            df = df.merge(coords, on="sre", how="left")
        else:
            df["lat"] = np.nan
            df["lon"] = np.nan
    return df


def _build_fixed_effects(df: pd.DataFrame, train_mask: pd.Series,
                         fixed_numeric: list[str] | None = None,
                         fixed_categ: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Padroniza (z-score, estatísticas do subconjunto de treino) os numéricos
    e faz dummy-encoding (drop_first) dos categóricos, sobre todo o dataframe
    (treino + previsão) para garantir colunas idênticas nos dois subconjuntos."""
    fixed_numeric = list(_FIXED_NUMERIC_DEFAULT) if fixed_numeric is None else list(fixed_numeric)
    fixed_categ = list(_FIXED_CATEG_DEFAULT) if fixed_categ is None else list(fixed_categ)

    num_stats: dict[str, tuple[float, float]] = {}
    z_cols = []
    for col in fixed_numeric:
        mu = float(df.loc[train_mask, col].mean())
        sd = float(df.loc[train_mask, col].std(ddof=0)) or 1.0
        df[f"{col}_z"] = ((df[col] - mu) / sd).fillna(0.0)
        num_stats[col] = (mu, sd)
        z_cols.append(f"{col}_z")

    if fixed_categ:
        dummies = pd.get_dummies(df[fixed_categ].astype(str), prefix=fixed_categ, drop_first=True)
        X = pd.concat([df[z_cols], dummies], axis=1).astype(float)
    else:
        X = df[z_cols].astype(float)
    return X, num_stats


def build_chain_graph(component_train: np.ndarray, km_mid_train: np.ndarray,
                      lat_train: np.ndarray | None = None,
                      lon_train: np.ndarray | None = None,
                      geo_knn_k: int = 0) -> dict:
    """Constrói, sobre os n segmentos de treino, a matriz de adjacência CAR
    (conecta vizinhos consecutivos dentro do mesmo componente conexo) e a
    matriz de distância real (km) usada pelo kernel do GP. `geo_knn_k > 0`
    amplia a adjacência com arestas de k-vizinhos-mais-próximos geográficos
    reais (haversine), independente de `go`/componente físico."""
    n = len(component_train)
    W = np.zeros((n, n))
    same_component = np.zeros((n, n), dtype=bool)
    dist = np.zeros((n, n))

    for comp in np.unique(component_train):
        idx = np.where(component_train == comp)[0]
        idx = idx[np.argsort(km_mid_train[idx])]
        for a in idx:
            same_component[a, idx] = True
            dist[a, idx] = np.abs(km_mid_train[a] - km_mid_train[idx])
        for a, b in zip(idx[:-1], idx[1:]):
            W[a, b] = 1.0
            W[b, a] = 1.0

    geo_dist = None
    if lat_train is not None and lon_train is not None:
        lat_arr = np.asarray(lat_train, dtype=float)
        lon_arr = np.asarray(lon_train, dtype=float)
        if not (np.isnan(lat_arr).any() or np.isnan(lon_arr).any()):
            geo_dist = _haversine_km_matrix(lat_arr, lon_arr)

    W_geo_knn = np.zeros((n, n))
    geo_knn_k_used = 0
    if geo_knn_k > 0 and geo_dist is not None and n > 1:
        k_eff = min(geo_knn_k, n - 1)
        order = np.argsort(geo_dist, axis=1)
        for i in range(n):
            neighbors = [j for j in order[i] if j != i][:k_eff]
            for j in neighbors:
                W_geo_knn[i, j] = 1.0
                W_geo_knn[j, i] = 1.0
        W = np.clip(W + W_geo_knn, 0.0, 1.0)
        geo_knn_k_used = geo_knn_k

    degree = W.sum(axis=1)
    isolated = degree == 0
    car_idx = np.where(~isolated)[0]
    W_car = W[np.ix_(car_idx, car_idx)]

    return dict(W=W, W_car=W_car, car_idx=car_idx, dist=dist,
                same_component=same_component, degree=degree, isolated=isolated,
                geo_dist=geo_dist, W_geo_knn=W_geo_knn, geo_knn_k=geo_knn_k_used)


@dataclass
class GraphData:
    df_all: pd.DataFrame
    target: str
    train_df: pd.DataFrame
    pred_df: pd.DataFrame
    X_train: pd.DataFrame
    X_pred: pd.DataFrame
    y_train: np.ndarray
    go_categories: pd.Index
    go_codes_train: np.ndarray
    go_codes_pred: np.ndarray
    graph: dict
    component_train: np.ndarray
    km_mid_train: np.ndarray
    num_stats: dict = field(default_factory=dict)
    fixed_numeric: list = field(default_factory=list)
    fixed_categ: list = field(default_factory=list)
    include_region_re: bool = True
    region_categories: pd.Index | None = None
    region_codes_train: np.ndarray | None = None
    region_codes_pred: np.ndarray | None = None
    lat_train: np.ndarray | None = None
    lon_train: np.ndarray | None = None
    geo_knn_k: int = 0


def prepare_graph_data(df: pd.DataFrame, target: str = "vmd",
                       fixed_numeric: list[str] | None = None,
                       fixed_categ: list[str] | None = None,
                       include_region_re: bool = INCLUDE_REGION_RE_DEFAULT,
                       geo_knn_k: int = 0) -> GraphData:
    """Monta todas as estruturas (efeitos fixos, grupos `go`, grafo CAR/GP)
    necessárias para ajustar qualquer um dos 3 modelos aninhados."""
    fixed_numeric = list(_FIXED_NUMERIC_DEFAULT) if fixed_numeric is None else list(fixed_numeric)
    fixed_categ = list(_FIXED_CATEG_DEFAULT) if fixed_categ is None else list(fixed_categ)

    missing = [c for c in [target, GO_COL, REGION_COL, *fixed_numeric, *fixed_categ,
                            "km_inicial", "km_final"] if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes para o modelo de grafo: {missing}")

    train_mask = df[target].notna() & (df[target] > 0)
    X_all, num_stats = _build_fixed_effects(df, train_mask, fixed_numeric, fixed_categ)

    train_df = df.loc[train_mask].copy()
    pred_df = df.loc[~train_mask].copy()
    X_train = X_all.loc[train_mask].reset_index(drop=True)
    X_pred = X_all.loc[~train_mask].reset_index(drop=True)

    y_train = train_df[target].to_numpy(dtype=float)

    go_categories = pd.Index(sorted(df[GO_COL].map(normalize_group_label).unique()))
    go_map = {g: i for i, g in enumerate(go_categories)}
    go_codes_train = train_df[GO_COL].map(normalize_group_label).map(go_map).to_numpy()
    go_codes_pred = pred_df[GO_COL].map(normalize_group_label).map(go_map).to_numpy()

    region_categories = region_codes_train = region_codes_pred = None
    if include_region_re:
        region_categories = pd.Index(sorted(df[REGION_COL].map(normalize_group_label).unique()))
        region_map = {r: i for i, r in enumerate(region_categories)}
        region_codes_train = train_df[REGION_COL].map(normalize_group_label).map(region_map).to_numpy()
        region_codes_pred = pred_df[REGION_COL].map(normalize_group_label).map(region_map).to_numpy()

    component_train = train_df["_component_id"].to_numpy()
    km_mid_train = train_df["_km_mid"].to_numpy(dtype=float)
    lat_train = train_df["lat"].to_numpy(dtype=float) if "lat" in train_df.columns else None
    lon_train = train_df["lon"].to_numpy(dtype=float) if "lon" in train_df.columns else None
    graph = build_chain_graph(component_train, km_mid_train, lat_train=lat_train, lon_train=lon_train,
                              geo_knn_k=geo_knn_k)

    return GraphData(
        df_all=df, target=target, train_df=train_df.reset_index(drop=True),
        pred_df=pred_df.reset_index(drop=True), X_train=X_train, X_pred=X_pred,
        y_train=y_train, go_categories=go_categories,
        go_codes_train=go_codes_train, go_codes_pred=go_codes_pred,
        graph=graph, component_train=component_train, km_mid_train=km_mid_train,
        num_stats=num_stats, fixed_numeric=fixed_numeric, fixed_categ=fixed_categ,
        include_region_re=include_region_re, region_categories=region_categories,
        region_codes_train=region_codes_train, region_codes_pred=region_codes_pred,
        lat_train=lat_train, lon_train=lon_train, geo_knn_k=graph["geo_knn_k"],
    )


def build_pm_model(gdata: GraphData, model_kind: str, gp_kernel: str = "exp"):
    """Constrói o modelo PyMC para `model_kind` em {'glmm','glmm_car','glmm_car_gp'}.
    Todas as variantes expõem os mesmos deterministics (`u_go`, `u_car`, `r_gp`,
    `mu_train`, `mu_response`), com termos ausentes fixados em zero."""
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind inválido: {model_kind} (opções: {MODEL_KINDS})")
    if gp_kernel not in GP_KERNELS:
        raise ValueError(f"gp_kernel inválido: {gp_kernel} (opções: {GP_KERNELS})")

    X = gdata.X_train.to_numpy(dtype=float)
    y = gdata.y_train
    n, p = X.shape
    n_go = len(gdata.go_categories)
    feature_names = list(gdata.X_train.columns)

    graph = gdata.graph
    car_idx = graph["car_idx"]
    n_car = len(car_idx)
    W_car = graph["W_car"]
    dist = graph["dist"]
    same_component = graph["same_component"].astype(float)
    W_full = graph["W"]

    log_y = np.log(y)
    log_y_mean, log_y_sd = float(log_y.mean()), float(log_y.std(ddof=0)) or 1.0

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=log_y_mean, sigma=2.0 * log_y_sd)
        beta = pm.Normal("beta", mu=0.0, sigma=1.5, shape=p)
        mu_fixed = intercept + pt.dot(X, beta)

        sigma_go = pm.HalfNormal("sigma_go", sigma=1.0)
        z_go = pm.Normal("z_go", mu=0.0, sigma=1.0, shape=n_go)
        u_go = pm.Deterministic("u_go", z_go * sigma_go)
        u_go_train = u_go[gdata.go_codes_train]

        if gdata.include_region_re:
            n_region = len(gdata.region_categories)
            sigma_region = pm.HalfNormal("sigma_region", sigma=1.0)
            z_region = pm.Normal("z_region", mu=0.0, sigma=1.0, shape=n_region)
            u_region = pm.Deterministic("u_region", z_region * sigma_region)
            u_region_train = u_region[gdata.region_codes_train]
        else:
            u_region_train = 0.0

        if model_kind in ("glmm_car", "glmm_car_gp") and n_car > 0:
            alpha_car = pm.Uniform("alpha_car", lower=0.0, upper=0.99)
            tau_car = pm.Gamma("tau_car", alpha=2.0, beta=1.0)
            pm.Deterministic("sigma_car_approx", 1.0 / pt.sqrt(tau_car))
            u_car_sub = pm.CAR("u_car_sub", mu=np.zeros(n_car), W=W_car,
                               alpha=alpha_car, tau=tau_car, shape=n_car)
            u_car_full = pt.zeros(n)
            u_car_full = pt.set_subtensor(u_car_full[car_idx], u_car_sub)
        else:
            u_car_full = pt.zeros(n)
        u_car = pm.Deterministic("u_car", u_car_full)

        if model_kind == "glmm_car_gp":
            sigma_r = pm.HalfNormal("sigma_r", sigma=1.5)
            if gp_kernel == "exp":
                ell = pm.Gamma("ell", alpha=2.0, beta=0.3)
                K = (sigma_r ** 2) * pt.exp(-dist / ell) * same_component
                K = K + np.eye(n) * _GP_JITTER
            elif gp_kernel == "geo":
                geo_dist = graph.get("geo_dist")
                if geo_dist is None:
                    raise ValueError(
                        "gp_kernel='geo' requer coordenadas lat/lon sem NaN em todos os "
                        "segmentos de treino (ver 00_docs/data/sre-coordenadas.csv)."
                    )
                ell = pm.Gamma("ell", alpha=2.0, beta=0.3)
                K = (sigma_r ** 2) * pt.exp(-geo_dist / ell)
                K = K + np.eye(n) * _GP_JITTER
            else:  # laplacian
                alpha_l = pm.Deterministic("alpha_l", pm.HalfNormal("alpha_l_raw", sigma=1.0) + 1e-3)
                D_mat = np.diag(W_full.sum(axis=1))
                L = D_mat - W_full
                K = (sigma_r ** 2) * pt.linalg.inv(L + alpha_l * np.eye(n))
                K = K + np.eye(n) * _GP_JITTER
            L_chol = pt.linalg.cholesky(K)
            z_r = pm.Normal("z_r", mu=0.0, sigma=1.0, shape=n)
            r_gp = pt.dot(L_chol, z_r)
        else:
            r_gp = pt.zeros(n)
        r_gp = pm.Deterministic("r_gp", r_gp)

        mu_train = pm.Deterministic("mu_train", mu_fixed + u_go_train + u_region_train + u_car + r_gp)
        mu_response = pm.Deterministic("mu_response", pt.exp(mu_train))

        gamma_shape = pm.Gamma("gamma_shape", alpha=2.0, beta=0.1)
        pm.Gamma("y_obs", alpha=gamma_shape, beta=gamma_shape / mu_response, observed=y)

    return model, dict(feature_names=feature_names, n=n, p=p, n_go=n_go, n_car=n_car,
                       n_isolated=n - n_car, geo_knn_k=graph.get("geo_knn_k", 0),
                       include_region_re=gdata.include_region_re,
                       n_region=(len(gdata.region_categories) if gdata.include_region_re else 0))


def fit_graph_model(gdata: GraphData, model_kind: str, gp_kernel: str = "exp",
                    sample_kwargs: dict | None = None) -> dict:
    if not _HAS_BAYES:
        raise ImportError(
            "pymc/arviz não instalados. Rode: pip install -r requirements.txt"
        )

    sample_kwargs = {**SAMPLE_KWARGS, **(sample_kwargs or {})}
    nuts_sampler = sample_kwargs.pop("nuts_sampler", "nutpie" if _HAS_NUTPIE else "pymc")

    model, meta = build_pm_model(gdata, model_kind, gp_kernel=gp_kernel)
    with model:
        idata = pm.sample(nuts_sampler=nuts_sampler, idata_kwargs={"log_likelihood": True},
                          **sample_kwargs)

    return dict(model=model, idata=idata, gdata=gdata, meta=meta,
                model_kind=model_kind, gp_kernel=gp_kernel)


def _posterior_mean(idata, var_name: str) -> np.ndarray:
    da = az.extract(idata, group="posterior", var_names=var_name)
    return da.mean("sample").values


def evaluate_graph_fit(fit: dict) -> dict:
    """Métricas in-sample (escala original), diagnóstico MCMC e ELPD LOO."""
    idata, gdata = fit["idata"], fit["gdata"]
    y_pred = _posterior_mean(idata, "mu_response")

    y_true = gdata.train_df[gdata.target].to_numpy(dtype=float)

    metrics = compute_metrics(y_true, y_pred)
    regions = gdata.train_df[REGION_COL].astype(str).to_numpy()
    r2_reg = per_region_r2(y_true, y_pred, regions)
    metrics["r2_min_regiao"] = float(np.nanmin(list(r2_reg.values()))) if r2_reg else float("nan")

    diag = convergence_diagnostics(idata)

    try:
        loo = az.loo(idata)
        elpd_loo = float(getattr(loo, "elpd", getattr(loo, "elpd_loo", float("nan"))))
    except Exception as exc:  # pragma: no cover - diagnóstico
        elpd_loo = float("nan")
        print(f"  [aviso] LOO indisponível: {type(exc).__name__}: {exc}")

    return dict(metrics=metrics, r2_por_regiao=r2_reg, diagnostics=diag,
                elpd_loo=elpd_loo, n_train=len(gdata.train_df),
                model_kind=fit["model_kind"], gp_kernel=fit.get("gp_kernel"))


def coefficient_summary(fit: dict) -> pd.DataFrame:
    """Resumo posterior dos efeitos fixos (intercepto + β)."""
    idata, meta = fit["idata"], fit["meta"]
    summ = az.summary(idata, var_names=["intercept", "beta"], ci_prob=0.9)
    names = ["Intercept"] + meta["feature_names"]
    summ.index = names
    return summ


def hyperparameter_summary(fit: dict) -> pd.DataFrame:
    """Resumo posterior dos hiperparâmetros globais."""
    idata = fit["idata"]
    candidates = ["sigma_go", "sigma_region", "alpha_car", "tau_car", "sigma_car_approx",
                  "sigma_r", "ell", "alpha_l", "gamma_shape"]
    present = [v for v in candidates if v in idata.posterior]
    if not present:
        return pd.DataFrame()
    return az.summary(idata, var_names=present, ci_prob=0.9)


def spatial_effects_table(fit: dict) -> pd.DataFrame:
    """Tabela por segmento de treino: observado, ajustado, efeito CAR (u_i),
    resíduo GP (r_i) e intercepto de rodovia (u_go) — posterior mean + IC90%."""
    idata, gdata = fit["idata"], fit["gdata"]
    tr = gdata.train_df

    def _mean_ci(var):
        da = az.extract(idata, group="posterior", var_names=var)
        mean = da.mean("sample").values
        lo = da.quantile(0.05, dim="sample").values
        hi = da.quantile(0.95, dim="sample").values
        return mean, lo, hi

    u_car_m, u_car_lo, u_car_hi = _mean_ci("u_car")
    r_gp_m, r_gp_lo, r_gp_hi = _mean_ci("r_gp")
    mu_m, mu_lo, mu_hi = _mean_ci("mu_train")
    mu_resp_m, _, _ = _mean_ci("mu_response")

    out = pd.DataFrame({
        "sre": tr["sre"].values if "sre" in tr.columns else tr.index,
        **({"sentido": tr["sentido"].values} if "sentido" in tr.columns else {}),
        "go": tr[GO_COL].values,
        "regional_macro": tr[REGION_COL].values,
        "componente": gdata.component_train,
        "km_mid": gdata.km_mid_train,
        "isolado_no_grafo": gdata.graph["isolated"],
        f"{gdata.target}_observado": tr[gdata.target].values,
        "log_vmd_pred_media": mu_m,
        "log_vmd_pred_ic90_lo": mu_lo,
        "log_vmd_pred_ic90_hi": mu_hi,
        f"{gdata.target}_ajustado": mu_resp_m,
        "u_car_media": u_car_m, "u_car_ic90_lo": u_car_lo, "u_car_ic90_hi": u_car_hi,
        "r_gp_media": r_gp_m, "r_gp_ic90_lo": r_gp_lo, "r_gp_ic90_hi": r_gp_hi,
    })
    return out


def _gp_krige_exp(new_component: np.ndarray, new_km: np.ndarray, gdata: GraphData,
                  sigma_r: float, ell: float, r_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predição condicional exata do GP (kernel exponencial) para novos nós."""
    comp_idx: dict[str, list[int]] = {}
    for i, c in enumerate(gdata.component_train):
        comp_idx.setdefault(c, []).append(i)

    n_new = len(new_component)
    preds = np.zeros(n_new)
    var = np.full(n_new, sigma_r ** 2)

    for k in range(n_new):
        idxs = comp_idx.get(new_component[k], [])
        if not idxs:
            continue
        pos_obs = gdata.km_mid_train[idxs]
        k_vec = sigma_r ** 2 * np.exp(-np.abs(pos_obs - new_km[k]) / ell)
        K_obs = (sigma_r ** 2 * np.exp(-np.abs(pos_obs[:, None] - pos_obs[None, :]) / ell)
                 + np.eye(len(idxs)) * _GP_JITTER)
        try:
            K_inv = np.linalg.inv(K_obs)
        except np.linalg.LinAlgError:
            continue
        r_obs = r_train[idxs]
        preds[k] = float(k_vec @ K_inv @ r_obs)
        var[k] = float(max(sigma_r ** 2 - k_vec @ K_inv @ k_vec, 0.0))
    return preds, var


def _gp_krige_geo(new_lat: np.ndarray, new_lon: np.ndarray, gdata: GraphData,
                  sigma_r: float, ell: float, r_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predição condicional exata do GP com kernel "geo" (haversine, sem
    restrição de componente/GO) para novos nós."""
    lat_train, lon_train = gdata.lat_train, gdata.lon_train
    n_new = len(new_lat)
    if lat_train is None or lon_train is None:
        return np.zeros(n_new), np.full(n_new, sigma_r ** 2)

    n_train = len(lat_train)
    K_obs = (sigma_r ** 2) * np.exp(-_haversine_km_matrix(lat_train, lon_train) / ell)
    K_obs = K_obs + np.eye(n_train) * _GP_JITTER
    try:
        K_inv = np.linalg.inv(K_obs)
    except np.linalg.LinAlgError:
        return np.zeros(n_new), np.full(n_new, sigma_r ** 2)

    valid = ~(np.isnan(new_lat) | np.isnan(new_lon))
    preds = np.zeros(n_new)
    var = np.full(n_new, sigma_r ** 2)
    if not valid.any():
        return preds, var

    cross = _haversine_km_cross(new_lat[valid], new_lon[valid], lat_train, lon_train)
    K_new = (sigma_r ** 2) * np.exp(-cross / ell)
    tmp = K_new @ K_inv
    preds[valid] = tmp @ r_train
    var[valid] = np.clip(sigma_r ** 2 - np.einsum("ij,ij->i", tmp, K_new), 0.0, None)
    return preds, var


def _car_nearest_neighbor(new_component: np.ndarray, new_km: np.ndarray, gdata: GraphData,
                          alpha_car: float, u_car_train: np.ndarray,
                          new_lat: np.ndarray | None = None,
                          new_lon: np.ndarray | None = None) -> np.ndarray:
    """Aproximação prática do CAR condicional: média dos 1-2 vizinhos
    observados mais próximos (mesmo componente), escalada por α. Com fallback
    geográfico (2 segmentos observados mais próximos, haversine, qualquer
    `go`) quando o componente físico não tem nenhum segmento observado."""
    comp_idx: dict[str, list[int]] = {}
    for i, c in enumerate(gdata.component_train):
        comp_idx.setdefault(c, []).append(i)

    has_geo_fallback = (new_lat is not None and new_lon is not None
                        and gdata.lat_train is not None and gdata.lon_train is not None)

    preds = np.zeros(len(new_component))
    for k in range(len(new_component)):
        idxs = comp_idx.get(new_component[k], [])
        if idxs:
            pos = gdata.km_mid_train[np.array(idxs)]
            order = np.argsort(np.abs(pos - new_km[k]))
            nearest = [idxs[i] for i in order[:2]]
            preds[k] = float(np.mean(u_car_train[nearest])) * alpha_car
        elif has_geo_fallback and not (np.isnan(new_lat[k]) or np.isnan(new_lon[k])):
            d = _haversine_km_cross(np.array([new_lat[k]]), np.array([new_lon[k]]),
                                    gdata.lat_train, gdata.lon_train)[0]
            nearest = np.argsort(d)[:2]
            preds[k] = float(np.mean(u_car_train[nearest])) * alpha_car
    return preds


def predict_all_segments(fit: dict) -> pd.DataFrame:
    """Tabela por `sre` para todos os segmentos (medidos e não medidos):
    observado, previsto (média + IC90 aproximado), efeito espacial (u_i),
    resíduo GP (r_i) e features locais de grafo."""
    idata, gdata = fit["idata"], fit["gdata"]
    df_all = gdata.df_all
    target = gdata.target

    beta_m = _posterior_mean(idata, "beta")
    intercept_m = float(_posterior_mean(idata, "intercept"))
    gamma_shape_m = float(_posterior_mean(idata, "gamma_shape"))
    sigma_go = float(_posterior_mean(idata, "sigma_go"))
    u_go_m = _posterior_mean(idata, "u_go")
    u_car_train_m = _posterior_mean(idata, "u_car")
    r_gp_train_m = _posterior_mean(idata, "r_gp")

    has_region_re = "sigma_region" in idata.posterior
    sigma_region = float(_posterior_mean(idata, "sigma_region")) if has_region_re else 0.0
    u_region_m = _posterior_mean(idata, "u_region") if has_region_re else None

    has_car = "alpha_car" in idata.posterior
    has_gp_exp = "sigma_r" in idata.posterior and fit["model_kind"] == "glmm_car_gp" and fit["gp_kernel"] == "exp"
    has_gp_geo = "sigma_r" in idata.posterior and fit["model_kind"] == "glmm_car_gp" and fit["gp_kernel"] == "geo"
    alpha_car_m = float(_posterior_mean(idata, "alpha_car")) if has_car else 0.0
    sigma_r_m = float(_posterior_mean(idata, "sigma_r")) if (has_gp_exp or has_gp_geo) else 0.0
    ell_m = float(_posterior_mean(idata, "ell")) if (has_gp_exp or has_gp_geo) else 1.0

    train_tab = spatial_effects_table(fit)
    train_tab["fonte"] = "observado"

    pred_df = gdata.pred_df
    X_pred = gdata.X_pred.to_numpy(dtype=float)
    mu_fixed_pred = intercept_m + X_pred @ beta_m

    go_codes_pred = gdata.go_codes_pred
    u_go_pred = np.where(go_codes_pred >= 0, u_go_m[np.clip(go_codes_pred, 0, None)], 0.0)
    go_unseen = ~np.isin(go_codes_pred, np.arange(len(gdata.go_categories)))

    if has_region_re:
        region_codes_pred = gdata.region_codes_pred
        u_region_pred = np.where(region_codes_pred >= 0,
                                 u_region_m[np.clip(region_codes_pred, 0, None)], 0.0)
        region_unseen = ~np.isin(region_codes_pred, np.arange(len(gdata.region_categories)))
    else:
        u_region_pred = np.zeros(len(pred_df))
        region_unseen = np.zeros(len(pred_df), dtype=bool)

    new_component = pred_df["_component_id"].to_numpy()
    new_km = pred_df["_km_mid"].to_numpy(dtype=float)
    new_lat = pred_df["lat"].to_numpy(dtype=float) if "lat" in pred_df.columns else None
    new_lon = pred_df["lon"].to_numpy(dtype=float) if "lon" in pred_df.columns else None

    if has_gp_exp:
        r_pred, r_var = _gp_krige_exp(new_component, new_km, gdata, sigma_r_m, ell_m, r_gp_train_m)
    elif has_gp_geo and new_lat is not None and new_lon is not None:
        r_pred, r_var = _gp_krige_geo(new_lat, new_lon, gdata, sigma_r_m, ell_m, r_gp_train_m)
    else:
        r_pred = np.zeros(len(pred_df))
        r_var = np.zeros(len(pred_df))

    if has_car:
        u_car_pred = _car_nearest_neighbor(new_component, new_km, gdata, alpha_car_m, u_car_train_m,
                                           new_lat=new_lat, new_lon=new_lon)
    else:
        u_car_pred = np.zeros(len(pred_df))

    mu_pred = mu_fixed_pred + u_go_pred + u_region_pred + u_car_pred + r_pred
    mu_pred_response = np.exp(mu_pred)

    var_go = np.where(go_unseen, sigma_go ** 2, 0.0)
    var_region = np.where(region_unseen, sigma_region ** 2, 0.0)
    var_eta = var_go + var_region + r_var + (1.0 / gamma_shape_m)
    sd_pred_eta = np.sqrt(var_eta)
    sd_pred_response = mu_pred_response * sd_pred_eta

    z90 = 1.645
    log_lo = mu_pred - z90 * sd_pred_eta
    log_hi = mu_pred + z90 * sd_pred_eta

    pred_tab = pd.DataFrame({
        "sre": pred_df["sre"].values if "sre" in pred_df.columns else pred_df.index,
        **({"sentido": pred_df["sentido"].values} if "sentido" in pred_df.columns else {}),
        "go": pred_df[GO_COL].values,
        "regional_macro": pred_df[REGION_COL].values,
        "componente": new_component,
        "km_mid": new_km,
        "isolado_no_grafo": np.array([gdata.component_train == c for c in new_component]).sum(axis=1) == 0
                             if len(new_component) else np.array([], dtype=bool),
        f"{target}_observado": np.nan,
        "log_vmd_pred_media": mu_pred,
        "log_vmd_pred_ic90_lo": log_lo,
        "log_vmd_pred_ic90_hi": log_hi,
        f"{target}_ajustado": mu_pred_response,
        "u_car_media": u_car_pred, "u_car_ic90_lo": np.nan, "u_car_ic90_hi": np.nan,
        "r_gp_media": r_pred, "r_gp_ic90_lo": np.nan, "r_gp_ic90_hi": np.nan,
        "fonte": "previsto",
    })
    pred_tab[f"{target}_previsto_baixo"] = np.clip(mu_pred_response - z90 * sd_pred_response, 1e-6, None)
    pred_tab[f"{target}_previsto_alto"] = mu_pred_response + z90 * sd_pred_response

    full = pd.concat([train_tab, pred_tab], ignore_index=True, sort=False)

    if "sentido" in full.columns:
        full["_node_id"] = full["sre"].astype(str) + "__" + full["sentido"].astype(str)
    else:
        full["_node_id"] = full["sre"].astype(str)

    comp_counts = df_all["_component_id"].value_counts()
    lookup = (
        df_all[["_node_id", "_degree_full", "_component_id"]]
        .drop_duplicates("_node_id")
        .assign(tamanho_componente=lambda d: d["_component_id"].map(comp_counts))
        .rename(columns={"_degree_full": "grau_grafo_completo"})
    )
    full = full.merge(
        lookup[["_node_id", "grau_grafo_completo", "tamanho_componente"]],
        on="_node_id", how="left",
    )

    top_row = df_all.loc[df_all[target].notna()].nlargest(1, target)
    if len(top_row):
        top_comp = top_row["_component_id"].iloc[0]
        top_km = top_row["_km_mid"].iloc[0]
        same_comp_mask = full["componente"] == top_comp
        full["dist_km_ao_top_vmd"] = np.where(
            same_comp_mask, np.abs(full["km_mid"] - top_km), np.nan
        )

    return full.drop(columns=["_node_id"])


def build_equation_text(coef_df: pd.DataFrame, model_kind: str, target: str,
                        include_region_re: bool = True) -> str:
    if coef_df is None or coef_df.empty or "mean" not in coef_df.columns:
        return ""
    intercept = float(coef_df.loc["Intercept", "mean"])
    padding = " " * (len(target) + 9)
    parts = [f"log(ŷ({target})) = {intercept:+.4f}"]
    for name, row in coef_df.drop(index="Intercept").iterrows():
        coef = float(row["mean"])
        sign = "+" if coef >= 0 else "-"
        parts.append(f"{padding}  {sign} {abs(coef):.4f} × {name}")
    parts.append(f"{padding}  + u_go[go]  (intercepto aleatório por rodovia)")
    if include_region_re:
        parts.append(f"{padding}  + u_regiao[regional_macro]  (intercepto aleatório por região)")
    if model_kind in ("glmm_car", "glmm_car_gp"):
        parts.append(f"{padding}  + u_i      (efeito espacial CAR sobre o grafo)")
    if model_kind == "glmm_car_gp":
        parts.append(f"{padding}  + r_i      (resíduo GP sobre o grafo)")
    parts.append(f"\nŷ({target}) = exp(log(ŷ({target})))   [média Gamma via link log — sem correção lognormal]")
    return "\n".join(parts)


def export_frames(glmm_graph_results: dict) -> dict[str, pd.DataFrame]:
    """Constrói as planilhas do GLMM+Grafo, uma por target. Devolve
    ``{nome_planilha: DataFrame}`` (ajustados x observados, previsões,
    coeficientes, hiperparâmetros, fórmula, métricas e uma aba combinada
    com todos os targets/segmentos)."""
    sheets: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict] = []
    resultados_rows: dict[str, pd.DataFrame] = {}

    for target, r in glmm_graph_results.items():
        ev = r["ev"]
        m, d = ev["metrics"], ev["diagnostics"]
        kind = r["fit"]["model_kind"]
        pred_table = r["pred_table"]

        aj_src = pred_table[pred_table["fonte"] == "observado"]
        aj = pd.DataFrame()
        for col, label in [("sre", "sre"), ("sentido", "sentido"),
                   ("go", "go"), (REGION_COL, "regiao")]:
            if col in aj_src.columns:
                aj[label] = aj_src[col].values
        obs = aj_src[f"{target}_observado"].to_numpy(dtype=float)
        fitv = aj_src[f"{target}_ajustado"].to_numpy(dtype=float)
        aj[f"{target}_observado"] = np.round(obs, 2)
        aj[f"{target}_ajustado"] = np.round(fitv, 2)
        aj[f"{target}_residuo"] = np.round(obs - fitv, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            erro = np.where(obs != 0, (obs - fitv) / obs * 100.0, np.nan)
        aj[f"{target}_erro_pct"] = np.round(erro, 2)
        sheets[f"GLMM_GRAFO_ajust_{target}"[:31]] = aj.reset_index(drop=True)

        pv_src = pred_table[pred_table["fonte"] == "previsto"]
        pv = pd.DataFrame()
        for col, label in [("sre", "sre"), ("sentido", "sentido"),
                   ("go", "go"), (REGION_COL, "regiao")]:
            if col in pv_src.columns:
                pv[label] = pv_src[col].values
        pv[f"{target}_previsto"] = np.round(pv_src[f"{target}_ajustado"].to_numpy(dtype=float), 2)
        pv["IC90_inferior"] = np.round(pv_src[f"{target}_previsto_baixo"].to_numpy(dtype=float), 2)
        pv["IC90_superior"] = np.round(pv_src[f"{target}_previsto_alto"].to_numpy(dtype=float), 2)
        sheets[f"GLMM_GRAFO_prev_{target}"[:31]] = pv.reset_index(drop=True)

        parte = pd.DataFrame()
        for col, label in [("sre", "sre"), ("sentido", "sentido"), ("go", "go"), (REGION_COL, "regiao")]:
            if col in pred_table.columns:
                parte[label] = pred_table[col].values
        obs_all = pred_table[f"{target}_observado"].to_numpy(dtype=float)
        fit_all = pred_table[f"{target}_ajustado"].to_numpy(dtype=float)
        is_obs = (pred_table["fonte"] == "observado").to_numpy()
        parte[f"{target}_observado"] = np.round(obs_all, 2)
        parte[f"{target}_ajustado"] = np.round(fit_all, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            residuo_all = np.where(is_obs, obs_all - fit_all, np.nan)
            erro_all = np.where(is_obs & (obs_all != 0), (obs_all - fit_all) / obs_all * 100.0, np.nan)
        parte[f"{target}_residuo"] = np.round(residuo_all, 2)
        parte[f"{target}_erro_pct"] = np.round(erro_all, 2)
        parte[f"_origem_{target}"] = pred_table["fonte"].map(
            {"observado": "contagem", "previsto": "estimado"}
        )
        resultados_rows[target] = parte.reset_index(drop=True)

        coef = r["coef"].reset_index().rename(columns={"index": "parametro"})
        sheets[f"GLMM_GRAFO_coef_{target}"[:31]] = coef

        hyper = r.get("hyper")
        if hyper is not None and not hyper.empty:
            hyper_out = hyper.reset_index().rename(columns={"index": "hiperparametro"})
            sheets[f"GLMM_GRAFO_hiper_{target}"[:31]] = hyper_out

        formula_text = r.get("formula") or build_equation_text(r["coef"], kind, target)
        if formula_text:
            sheets[f"GLMM_GRAFO_formula_{target}"[:31]] = pd.DataFrame(
                {"formula": formula_text.split("\n")}
            )

        metric_rows.append({
            "target": target,
            "modelo": f"glmm_grafo_{kind}",
            "gp_kernel": r["fit"].get("gp_kernel") if kind == "glmm_car_gp" else "-",
            "MAPE_%": round(m["mape"], 1),
            "w80_%": round(m["w80"], 1),
            "R2_orig": round(m["r2"], 3),
            "R2_min_regiao": round(m.get("r2_min_regiao", float("nan")), 3),
            "RMSE": round(m["rmse"], 1),
            "MAE": round(m["mae"], 1),
            "max_rhat": round(d["max_rhat"], 3),
            "min_ESS": round(d["min_ess"]),
            "divergencias": d["divergences"],
            "ELPD_LOO": round(ev["elpd_loo"], 1),
            "n_treino": ev["n_train"],
        })

    if metric_rows:
        sheets["GLMM_GRAFO_metricas"] = pd.DataFrame(metric_rows)
    if resultados_rows:
        sheets["GLMM_GRAFO_resultados"] = build_resultados_wide(resultados_rows)
    return sheets


def glmm_graph_only_excel(frames: dict[str, pd.DataFrame]) -> bytes:
    """Serializa as planilhas do GLMM+Grafo num workbook Excel."""
    import io

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if "GLMM_GRAFO_metricas" in frames:
            frames["GLMM_GRAFO_metricas"].to_excel(writer, sheet_name="GLMM_GRAFO_metricas", index=False)
        for name, frame in frames.items():
            if name == "GLMM_GRAFO_metricas":
                continue
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


def apply_annual_growth(
    df: pd.DataFrame,
    targets: list[str],
    current_year: int,
    growth_rate: float = 0.03,
) -> tuple[pd.DataFrame, dict]:
    """Atualiza targets observados para o ano de referência antes do treino."""
    if not 0 <= growth_rate < 1:
        raise ValueError("A taxa de crescimento deve estar no intervalo [0, 1).")
    missing_targets = [target for target in targets if target not in df.columns]
    if missing_targets:
        raise KeyError(f"Targets ausentes no dataset: {missing_targets}")

    result = df.copy()
    years = (
        pd.to_numeric(result["ano"], errors="coerce")
        if "ano" in result.columns
        else pd.Series(np.nan, index=result.index, dtype=float)
    )
    observed_any = result[targets].notna().any(axis=1)
    missing_year = observed_any & years.isna()
    if missing_year.any():
        raise ValueError(
            f"{int(missing_year.sum())} linhas observadas nao possuem ano de contagem."
        )
    future_year = observed_any & (years > current_year)
    if future_year.any():
        raise ValueError(
            f"{int(future_year.sum())} linhas observadas possuem ano posterior a {current_year}."
        )

    factors = (1.0 + growth_rate) ** (current_year - years)
    adjusted_counts: dict[str, int] = {}
    for target in targets:
        observed = result[target].notna()
        numeric = pd.to_numeric(result.loc[observed, target], errors="raise")
        result.loc[observed, target] = numeric * factors.loc[observed]
        adjusted_counts[target] = int(observed.sum())

    result["ano"] = current_year
    audit = {
        "ano_referencia": current_year,
        "taxa_crescimento": growth_rate,
        "linhas_observadas": int(observed_any.sum()),
        "ajustes_por_target": adjusted_counts,
    }
    return result, audit


def filter_allowed_situations(
    model_data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Mantem somente SREs com situacao operacional permitida."""
    normalized = model_data["situacao"].astype("string").str.strip().str.upper()
    keep = normalized.isin(ALLOWED_SITUATIONS)
    filtered = model_data.loc[keep].reset_index(drop=True)
    return filtered, {
        "situacoes_permitidas": sorted(ALLOWED_SITUATIONS),
        "linhas_antes_filtro_situacao": int(len(model_data)),
        "linhas_removidas_filtro_situacao": int((~keep).sum()),
        "linhas_apos_filtro_situacao": int(len(filtered)),
        "sres_removidas_filtro_situacao": int(model_data.loc[~keep, "sre"].nunique()),
    }


def _format_road(value) -> str | None:
    if pd.isna(value):
        return None
    label = str(value)
    if label.upper().startswith("GO-"):
        return label.upper()
    try:
        return f"GO-{int(float(label)):03d}"
    except ValueError:
        return label


def build_operational_output(
    adjusted_data: pd.DataFrame,
    prediction_tables: dict[str, pd.DataFrame],
    output_columns: list[str],
    targets: list[str],
    current_year: int,
    growth_rate: float,
) -> pd.DataFrame:
    """Monta a planilha final com observados corrigidos e faltantes previstos."""
    required = {"sre", "sentido", "go", "trecho", *targets}
    missing = sorted(required - set(adjusted_data.columns))
    if missing:
        raise KeyError(f"Colunas ausentes para a saida operacional: {missing}")

    output = pd.DataFrame(
        {
            "sre": adjusted_data["sre"].values,
            "rodovia": adjusted_data["go"].map(_format_road).values,
            "trecho": adjusted_data["trecho"].values,
            "sentido_trafego": adjusted_data["sentido"].values,
            "ano": current_year,
            "taxa_crescimento": growth_rate,
        }
    )
    key_columns = ["sre", "sentido"]
    for target in targets:
        if target not in prediction_tables:
            raise KeyError(f"Previsoes ausentes para o target: {target}")
        prediction_column = f"{target}_ajustado"
        predictions = prediction_tables[target][key_columns + [prediction_column]]
        if predictions.duplicated(key_columns).any():
            raise ValueError(f"Previsoes duplicadas para o target: {target}")
        aligned = adjusted_data[key_columns].merge(
            predictions,
            on=key_columns,
            how="left",
            validate="one_to_one",
        )[prediction_column]
        observed = adjusted_data[target].notna()
        output[target] = adjusted_data[target].where(observed, aligned).values

    observed_counts = adjusted_data[targets].notna().sum(axis=1)
    mixed = observed_counts.between(1, len(targets) - 1)
    if mixed.any():
        raise ValueError(
            "A coluna origem nao representa linhas com targets de origens diferentes."
        )
    output["origem"] = np.where(
        observed_counts.eq(len(targets)),
        "Contagem de Tráfego",
        "Estimativa Técnica",
    )

    missing_output_columns = [column for column in output_columns if column not in output]
    extra_output_columns = [column for column in output if column not in output_columns]
    if missing_output_columns or extra_output_columns:
        raise ValueError(
            "Cabecalhos divergentes do modelo: "
            f"ausentes={missing_output_columns}, extras={extra_output_columns}"
        )
    return output[output_columns]


def build_operational_columns(targets: list[str]) -> list[str]:
    """Cabecalhos da planilha operacional, no mesmo layout do modelo de
    referencia (``02_dados/modelo_volume.xlsx``), montados sem ler o arquivo."""
    return [
        "sre", "rodovia", "trecho", "sentido_trafego", "ano",
        "taxa_crescimento", *targets, "origem",
    ]


def apply_coincident_estimates(
    output: pd.DataFrame,
    model_data: pd.DataFrame,
    targets: list[str],
) -> pd.DataFrame:
    """Sobrescreve os alvos das SREs nao-principais (``principal`` = 'n') com
    os valores da mesma linha de sentido da SRE coincidente (``coin_1``/
    ``coin_2``) de menor numero de rodovia. O modelo so gera estimativas
    proprias para SREs com ``principal`` = 's'; as demais espelham a SRE
    de referencia."""
    if "principal" not in model_data.columns:
        return output

    non_principal = model_data.loc[
        model_data["principal"].eq("n") & model_data["sre_referencia"].notna(),
        ["sre", "sentido", "sre_referencia"],
    ]
    if non_principal.empty:
        return output

    mirrored_columns = [*targets, "origem"] if "origem" in output.columns else list(targets)
    reference_values = output.rename(columns={"sentido_trafego": "sentido"})[
        ["sre", "sentido", *mirrored_columns]
    ]
    lookup = non_principal.merge(
        reference_values.rename(columns={"sre": "sre_referencia"}),
        on=["sre_referencia", "sentido"],
        how="left",
    )

    result = output.copy()
    for _, row in lookup.iterrows():
        mask = result["sre"].eq(row["sre"]) & result["sentido_trafego"].eq(row["sentido"])
        for column in mirrored_columns:
            if pd.notna(row[column]):
                result.loc[mask, column] = row[column]
    return result


def build_road_skeleton(road_network: pd.DataFrame) -> pd.DataFrame:
    """Gera as linhas sre+sentido (CRESCENTE/DECRESCENTE) de toda a malha
    rodoviaria, substituindo a leitura de um arquivo de modelo externo."""
    if road_network.duplicated("sre").any():
        raise ValueError("A malha rodoviaria possui SRE duplicada.")
    go_numeric = pd.to_numeric(
        road_network["go"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    blocks = [
        pd.DataFrame(
            {
                "sre": road_network["sre"].values,
                "sentido": sentido,
                "go": go_numeric.values,
                "trecho": road_network["trecho"].values,
            }
        )
        for sentido in SENTIDOS
    ]
    skeleton = pd.concat(blocks, ignore_index=True)
    if skeleton.duplicated(["sre", "sentido"]).any():
        raise ValueError("O esqueleto gerado possui chaves sre+sentido duplicadas.")
    return skeleton


def _route_prefix_number(sre) -> int:
    """Numero de rodovia (primeiros 3 digitos) do codigo de uma SRE."""
    return int(str(sre)[:3])


def _resolve_coincident_reference(network: pd.DataFrame) -> pd.Series:
    """Para cada sre com principal='n', resolve a sre coincidente (coin_1 ou
    coin_2) de menor numero de rodovia, usada para espelhar a estimativa
    final do segmento nao-principal."""
    if "principal" not in network.columns:
        return pd.Series(dtype=object)
    non_principal = network.loc[network["principal"].astype(str).eq("n")]
    reference: dict[str, str] = {}
    for _, row in non_principal.iterrows():
        options = [
            value
            for value in (row.get("coin_1"), row.get("coin_2"))
            if pd.notna(value)
        ]
        if options:
            reference[row["sre"]] = min(options, key=_route_prefix_number)
    return pd.Series(reference, dtype=object)


def normalize_dataset_columns(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Padroniza o dataset unico: aplica ``COLUMN_ALIASES`` (copiando a coluna
    alternativa para o nome canonico, sem apagar a original) e normaliza
    ``sentido``. Retorna o frame e o mapa de aliases aplicados (auditoria)."""
    df = raw.copy()
    applied: dict[str, str] = {}
    for canonical, alternatives in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alternative in alternatives:
            if alternative in df.columns:
                df[canonical] = df[alternative]
                applied[canonical] = alternative
                break
    required = ["sre", "sentido", "go", "trecho", "classe", "km_inicial", "km_final",
                "situacao", "perim_urb"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Colunas obrigatorias ausentes no dataset: {missing}")
    df["sentido"] = df["sentido"].astype(str).str.strip().str.upper()
    invalid = ~df["sentido"].isin(SENTIDOS)
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} linhas com sentido fora de {SENTIDOS}: "
            f"{sorted(df.loc[invalid, 'sentido'].unique())[:10]}"
        )
    if df.duplicated(["sre", "sentido"]).any():
        raise ValueError("O dataset possui chaves sre+sentido duplicadas.")
    return df, applied


def prepare_single_dataset(
    adjusted_data: pd.DataFrame,
    targets: list[str],
    current_year: int,
) -> tuple[pd.DataFrame, dict]:
    """Equivalente da ``expand_with_road_network`` (v2) usando apenas o dataset.

    O proprio dataset e a malha completa: a "malha" (1 linha por SRE) e
    extraida dele, o esqueleto sre x {CRESCENTE, DECRESCENTE} e gerado e
    cruzado de volta com o dataset. Se algum sentido estiver ausente para
    uma SRE, a linha e criada com os atributos estruturais do outro sentido
    (mesma regra da v2 para SREs ausentes)."""
    # Malha = atributos estruturais por SRE, preferindo a linha CRESCENTE.
    network = (
        adjusted_data.assign(_ordem=adjusted_data["sentido"].ne("CRESCENTE"))
        .sort_values(["sre", "_ordem"], kind="stable")
        .drop_duplicates("sre")
        .drop(columns="_ordem")
        .reset_index(drop=True)
    )
    # Convencao de km da v2: CRESCENTE com km_inicial<=km_final e
    # DECRESCENTE invertido. A malha guarda o par (menor, maior).
    km_low = network[["km_inicial", "km_final"]].astype(float).min(axis=1)
    km_high = network[["km_inicial", "km_final"]].astype(float).max(axis=1)
    network["km_inicial_upd"] = km_low
    network["km_final_upd"] = km_high

    skeleton = build_road_skeleton(network)
    principal_by_sre = (
        network.set_index("sre")["principal"].astype(str).str.strip().str.lower()
        if "principal" in network.columns
        else pd.Series("s", index=network["sre"])
    )
    reference_by_sre = _resolve_coincident_reference(
        network.assign(principal=network["sre"].map(principal_by_sre))
    )
    network = network.set_index("sre")
    base = skeleton.merge(
        adjusted_data,
        on=["sre", "sentido"],
        how="left",
        validate="one_to_one",
        suffixes=("_malha", ""),
        indicator=True,
    )
    absent = base["_merge"].eq("left_only")
    base["go"] = base["go_malha"]
    base.loc[absent, "trecho"] = base.loc[absent, "trecho_malha"]

    structural_columns = [
        "classe", "extensao", "situacao", "perim_urb",
        "lat_inicial", "lon_inicial", "lat_final", "lon_final",
        "regional_macro", "media_empresas_ativas", "distancia_cid_1_2_3",
    ]
    for column in structural_columns:
        if column not in network.columns:
            continue
        mapped = base["sre"].map(network[column])
        if column not in base.columns:
            base[column] = mapped
        else:
            base.loc[absent, column] = mapped.loc[absent]

    # km por sentido (todas as linhas, como nas linhas vindas da malha na v2).
    km_start = base["sre"].map(network["km_inicial_upd"])
    km_end = base["sre"].map(network["km_final_upd"])
    crescente = base["sentido"].eq("CRESCENTE")
    base["km_inicial"] = np.where(crescente, km_start, km_end)
    base["km_final"] = np.where(crescente, km_end, km_start)

    base["ano"] = current_year
    for column in [
        "regional_macro",
        "media_empresas_ativas",
        "distancia_cid_1_2_3",
    ]:
        if column not in base.columns:
            base[column] = np.nan

    base["principal"] = base["sre"].map(principal_by_sre).fillna("s")
    base["sre_referencia"] = base["sre"].map(reference_by_sre)

    missing_network = base.loc[
        absent,
        ["classe", "km_inicial", "km_final", "situacao", "perim_urb"],
    ].isna().any(axis=1)
    if missing_network.any():
        raise ValueError(
            f"{int(missing_network.sum())} linhas do modelo nao possuem features estruturais."
        )

    keep_columns = [
        column
        for column in adjusted_data.columns
        if column in base.columns
    ]
    for column in [
        "sre", "sentido", "go", "classe", "km_inicial", "km_final", "extensao",
        "situacao", "trecho", "perim_urb", "regional_macro",
        "media_empresas_ativas", "distancia_cid_1_2_3", "ano",
        "lat_inicial", "lon_inicial", "lat_final", "lon_final",
        "principal", "sre_referencia", *targets,
    ]:
        if column not in keep_columns:
            keep_columns.append(column)
    result = base[keep_columns].copy()
    non_principal = result["principal"].eq("n")
    audit = {
        "linhas_modelo_saida": int(len(result)),
        "linhas_presentes_dataset": int((~absent).sum()),
        "sres_no_dataset": int(network.index.nunique()),
        "linhas_sentido_completadas": int(absent.sum()),
        "sres_com_sentido_completado": int(base.loc[absent, "sre"].nunique()),
        "linhas_com_contagem": int(result[targets].notna().any(axis=1).sum()),
        "linhas_nao_principais": int(non_principal.sum()),
        "sres_nao_principais_sem_referencia": int(
            result.loc[non_principal, "sre_referencia"].isna().sum()
        ),
    }
    return result, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLMM+CAR Gamma/log multi-target para volumes por classe de veiculo."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--current-year", type=int, default=datetime.now().year)
    parser.add_argument("--growth-rate", type=float, default=0.03)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help="Executa CV de 3 folds agrupada por SRE apos o ajuste principal.",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-draws", type=int, default=400)
    parser.add_argument("--cv-tune", type=int, default=400)
    parser.add_argument("--cv-chains", type=int, default=2)
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Valida dataset, features e grafo sem executar a amostragem MCMC.",
    )
    return parser.parse_args()


def prepare_target_data(raw: pd.DataFrame, target: str) -> GraphData:
    prepared = prepare_dataframe(raw, target=target)
    return prepare_graph_data(
        prepared,
        target=target,
        fixed_numeric=FIXED_NUMERIC,
        fixed_categ=FIXED_CATEG,
        include_region_re=INCLUDE_REGION_RE,
        geo_knn_k=GEO_KNN_K,
    )


def build_target_check(
    dataset: Path,
    raw: pd.DataFrame,
    graph_data: GraphData,
    target: str,
) -> dict:
    positive = raw[target].notna() & (raw[target] > 0)
    observed_zero = raw[target].notna() & (raw[target] == 0)
    missing = raw[target].isna()
    regional = pd.DataFrame(
        {
            "regional_macro": raw[REGION_COL].map(str),
            "n_tr": positive.astype(int),
            "n_zero_observado": observed_zero.astype(int),
            "n_pr": missing.astype(int),
        }
    ).groupby("regional_macro", dropna=False).sum()
    return {
        "dataset": str(dataset.resolve()),
        "target": target,
        "modelo": MODEL_KIND,
        "n_total": int(len(raw)),
        "n_tr": int(positive.sum()),
        "n_zero_observado": int(observed_zero.sum()),
        "n_pr": int(missing.sum()),
        "amostra_pequena": bool(positive.sum() < 30),
        "n_tr_por_sentido": {
            str(key): int(value)
            for key, value in raw.loc[positive].groupby("sentido").size().items()
        },
        "n_pr_por_sentido": {
            str(key): int(value)
            for key, value in raw.loc[missing].groupby("sentido").size().items()
        },
        "amostra_por_regiao": regional.reset_index().to_dict(orient="records"),
        "n_car": int(len(graph_data.graph["car_idx"])),
        "n_isolados_car": int(graph_data.graph["isolated"].sum()),
        "n_arestas_car": int(graph_data.graph["W"].sum() // 2),
        "colunas_design_fixo": list(graph_data.X_train.columns),
    }


def write_operational_workbook(output: pd.DataFrame, path: Path) -> None:
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        output.to_excel(writer, sheet_name="Planilha1", index=False)
        worksheet = writer.book["Planilha1"]
        worksheet.freeze_panes = "G2"
        header_fill = PatternFill("solid", fgColor="1F4E78")
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = Font(color="FFFFFF", bold=True)
        worksheet.column_dimensions["A"].width = 16
        worksheet.column_dimensions["B"].width = 12
        worksheet.column_dimensions["C"].width = 22
        worksheet.column_dimensions["D"].width = 20
        worksheet.column_dimensions["E"].width = 10
        worksheet.column_dimensions["F"].width = 18
        worksheet.column_dimensions[get_column_letter(len(output.columns))].width = 24
        for column_index in range(7, len(output.columns)):
            worksheet.column_dimensions[get_column_letter(column_index)].width = 12
        for cell in worksheet["F"][1:]:
            cell.number_format = "0.00%"
        for row in worksheet.iter_rows(
            min_row=2,
            min_col=7,
            max_col=len(output.columns) - 1,
            max_row=len(output) + 1,
        ):
            for cell in row:
                cell.number_format = "0.00"
        table = Table(
            displayName="ModeloVolumeGLMMCAR",
            ref=f"A1:{get_column_letter(len(output.columns))}{len(output) + 1}",
        )
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        worksheet.add_table(table)


def build_output_paths(
    output_dir: Path,
    generated_at: datetime | None = None,
) -> tuple[Path, Path, Path]:
    timestamp = (generated_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return (
        output_dir / f"modelo_volume_glmm_car_{timestamp}.xlsx",
        output_dir / f"glmm_car_multitarget_resultados_{timestamp}.xlsx",
        output_dir / f"glmm_car_multitarget_resumo_{timestamp}.json",
    )


def prediction_diagnostics_for_missing(
    raw: pd.DataFrame,
    prediction_table: pd.DataFrame,
    target: str,
) -> dict:
    missing_keys = raw.loc[raw[target].isna(), ["sre", "sentido"]]
    predicted = missing_keys.merge(
        prediction_table[["sre", "sentido", f"{target}_ajustado"]],
        on=["sre", "sentido"],
        how="left",
        validate="one_to_one",
    )[f"{target}_ajustado"]
    return {
        "n_pr": int(len(predicted)),
        "n_pr_sem_valor": int(predicted.isna().sum()),
        "unique_preds": int(predicted.nunique()),
        "taxa_unique_preds": float(predicted.nunique() / len(predicted)) if len(predicted) else None,
        "pred_min": float(predicted.min()) if predicted.notna().any() else None,
        "pred_mediana": float(predicted.median()) if predicted.notna().any() else None,
        "pred_max": float(predicted.max()) if predicted.notna().any() else None,
    }


def build_result_entry(fit: dict, target: str) -> tuple[dict, dict]:
    evaluation = evaluate_graph_fit(fit)
    coefficients = coefficient_summary(fit)
    hyperparameters = hyperparameter_summary(fit)
    spatial = spatial_effects_table(fit)
    predictions = predict_all_segments(fit)
    formula = build_equation_text(
        coefficients,
        MODEL_KIND,
        target,
        include_region_re=INCLUDE_REGION_RE,
    )
    return {
        "fit": fit,
        "ev": evaluation,
        "coef": coefficients,
        "hyper": hyperparameters,
        "spatial": spatial,
        "pred_table": predictions,
        "formula": formula,
        "gp_kernel": "exp",
    }, prediction_diagnostics_for_missing(fit["gdata"].df_all, predictions, target)


def load_model_data(dataset: Path) -> tuple[pd.DataFrame, GraphData]:
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {dataset}")

    raw = pd.read_excel(dataset)
    prepared = prepare_dataframe(raw, target=TARGET)
    graph_data = prepare_graph_data(
        prepared,
        target=TARGET,
        fixed_numeric=FIXED_NUMERIC,
        fixed_categ=FIXED_CATEG,
        include_region_re=INCLUDE_REGION_RE,
        geo_knn_k=GEO_KNN_K,
    )
    return raw, graph_data


def data_check_summary(dataset: Path, raw: pd.DataFrame, graph_data: GraphData) -> dict:
    graph = graph_data.graph
    train_mask = raw[TARGET].notna() & (raw[TARGET] > 0)
    key_columns = ["sre", "sentido"] if "sentido" in raw.columns else ["sre"]
    return {
        "dataset": str(dataset.resolve()),
        "target": TARGET,
        "modelo": MODEL_KIND,
        "features_originais": [
            "media_empresas_ativas",
            "distancia_cid_1_2_3",
            *FIXED_CATEG,
        ],
        "features_numericas_modeladas": FIXED_NUMERIC,
        "features_categoricas": FIXED_CATEG,
        "colunas_design_fixo": list(graph_data.X_train.columns),
        "n_total": int(len(raw)),
        "n_tr": int(len(graph_data.train_df)),
        "n_pr": int(len(graph_data.pred_df)),
        "n_sre": int(raw["sre"].nunique()),
        "n_sre_treino": int(raw.loc[train_mask, "sre"].nunique()),
        "n_sre_previsao": int(raw.loc[~train_mask, "sre"].nunique()),
        "chave_linha": "+".join(key_columns),
        "duplicatas_chave_linha": int(raw.duplicated(key_columns).sum()),
        "sentidos": sorted(raw["sentido"].dropna().astype(str).unique().tolist())
        if "sentido" in raw.columns
        else [],
        "n_tr_por_sentido": {
            str(key): int(value)
            for key, value in raw.loc[train_mask].groupby("sentido").size().items()
        }
        if "sentido" in raw.columns
        else {},
        "n_pr_por_sentido": {
            str(key): int(value)
            for key, value in raw.loc[~train_mask].groupby("sentido").size().items()
        }
        if "sentido" in raw.columns
        else {},
        "n_car": int(len(graph["car_idx"])),
        "n_isolados_car": int(graph["isolated"].sum()),
        "n_arestas_car": int(graph["W"].sum() // 2),
        "n_componentes_treino": int(pd.Series(graph_data.component_train).nunique()),
        "geo_knn_k": int(graph_data.geo_knn_k),
    }


def build_regional_diagnostics(
    graph_data: GraphData,
    prediction_table: pd.DataFrame,
    evaluation: dict,
) -> pd.DataFrame:
    rows = []
    r2_by_region = evaluation.get("r2_por_regiao", {})
    for region in graph_data.df_all[REGION_COL].astype(str).sort_values().unique():
        region_rows = prediction_table[
            prediction_table[REGION_COL].astype(str) == str(region)
        ]
        observed = region_rows[region_rows["fonte"] == "observado"]
        predicted = region_rows[region_rows["fonte"] == "previsto"]
        pred_values = predicted[f"{TARGET}_ajustado"].dropna()
        rows.append(
            {
                "regional_macro": region,
                "n_tr": int(len(observed)),
                "n_pr": int(len(predicted)),
                "r2_in_sample": r2_by_region.get(str(region), np.nan),
                "pred_min": float(pred_values.min()) if len(pred_values) else np.nan,
                "pred_mediana": float(pred_values.median()) if len(pred_values) else np.nan,
                "pred_max": float(pred_values.max()) if len(pred_values) else np.nan,
                "unique_preds": int(pred_values.nunique()),
                "taxa_unique_preds": float(pred_values.nunique() / len(pred_values))
                if len(pred_values)
                else np.nan,
                "amostra_pequena": bool(len(observed) < 30),
            }
        )
    return pd.DataFrame(rows)


def build_prediction_diagnostics(
    graph_data: GraphData,
    prediction_table: pd.DataFrame,
) -> dict:
    pred_values = prediction_table.loc[
        prediction_table["fonte"] == "previsto", f"{TARGET}_ajustado"
    ].dropna()
    feature_columns = [
        "media_empresas_ativas",
        "distancia_cid_1_2_3",
        *FIXED_CATEG,
    ]
    pred_features = graph_data.pred_df[feature_columns]
    unique_combinations = int(pred_features.drop_duplicates().shape[0])
    feature_direction_columns = [*feature_columns, "sentido"]
    unique_direction_combinations = (
        int(graph_data.pred_df[feature_direction_columns].drop_duplicates().shape[0])
        if "sentido" in graph_data.pred_df.columns
        else unique_combinations
    )
    n_predictions = int(len(pred_values))
    unique_predictions = int(pred_values.nunique())
    diagnostics = {
        "n_pr": n_predictions,
        "unique_feat_combos": unique_combinations,
        "taxa_unique_feat_combos": unique_combinations / n_predictions if n_predictions else None,
        "unique_feat_sentido_combos": unique_direction_combinations,
        "taxa_unique_feat_sentido_combos": unique_direction_combinations / n_predictions
        if n_predictions
        else None,
        "unique_preds": unique_predictions,
        "taxa_unique_preds": unique_predictions / n_predictions if n_predictions else None,
        "pred_min": float(pred_values.min()) if n_predictions else None,
        "pred_mediana": float(pred_values.median()) if n_predictions else None,
        "pred_max": float(pred_values.max()) if n_predictions else None,
    }
    if "sentido" in prediction_table.columns:
        predicted = prediction_table[prediction_table["fonte"] == "previsto"]
        pairs = predicted.pivot(index="sre", columns="sentido", values=f"{TARGET}_ajustado")
        if {"CRESCENTE", "DECRESCENTE"}.issubset(pairs.columns):
            pairs = pairs.dropna(subset=["CRESCENTE", "DECRESCENTE"])
            absolute_difference = (pairs["CRESCENTE"] - pairs["DECRESCENTE"]).abs()
            pair_mean = pairs[["CRESCENTE", "DECRESCENTE"]].mean(axis=1).clip(lower=1e-9)
            relative_difference = absolute_difference / pair_mean
            identical = np.isclose(
                pairs["CRESCENTE"], pairs["DECRESCENTE"], rtol=1e-9, atol=1e-6
            )
            diagnostics.update(
                {
                    "n_pares_sentido_previstos": int(len(pairs)),
                    "n_pares_predicao_igual": int(identical.sum()),
                    "taxa_pares_predicao_igual": float(identical.mean()) if len(pairs) else None,
                    "diferenca_abs_mediana_sentidos": float(absolute_difference.median()),
                    "diferenca_abs_p95_sentidos": float(absolute_difference.quantile(0.95)),
                    "diferenca_rel_mediana_sentidos": float(relative_difference.median()),
                    "diferenca_rel_p95_sentidos": float(relative_difference.quantile(0.95)),
                }
            )
    return diagnostics


def build_direction_diagnostics(prediction_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction in sorted(prediction_table["sentido"].dropna().astype(str).unique()):
        direction_rows = prediction_table[prediction_table["sentido"].astype(str) == direction]
        observed = direction_rows.loc[
            direction_rows["fonte"] == "observado", f"{TARGET}_observado"
        ].dropna()
        predicted = direction_rows.loc[
            direction_rows["fonte"] == "previsto", f"{TARGET}_ajustado"
        ].dropna()
        rows.append(
            {
                "sentido": direction,
                "n_tr": int(len(observed)),
                "n_pr": int(len(predicted)),
                "observado_min": float(observed.min()),
                "observado_mediana": float(observed.median()),
                "observado_max": float(observed.max()),
                "pred_min": float(predicted.min()),
                "pred_mediana": float(predicted.median()),
                "pred_max": float(predicted.max()),
                "unique_preds": int(predicted.nunique()),
                "taxa_unique_preds": float(predicted.nunique() / len(predicted)),
            }
        )
    return pd.DataFrame(rows)


def run_grouped_sre_cv(
    df_prepared: pd.DataFrame,
    output_dir: Path,
    n_splits: int,
    sample_kwargs: dict,
    seed: int,
) -> dict:
    observed_mask = df_prepared[TARGET].notna() & (df_prepared[TARGET] > 0)
    observed = df_prepared.loc[observed_mask].copy()
    observed_sres = observed["sre"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(observed_sres)
    folds = np.array_split(observed_sres, n_splits)

    fold_rows = []
    prediction_rows = []
    started = time.perf_counter()
    for fold_number, held_sres in enumerate(folds, start=1):
        fold_df = df_prepared.copy()
        held_mask = fold_df["sre"].isin(held_sres) & observed_mask
        fold_df.loc[held_mask, TARGET] = np.nan
        fold_data = prepare_graph_data(
            fold_df,
            target=TARGET,
            fixed_numeric=FIXED_NUMERIC,
            fixed_categ=FIXED_CATEG,
            include_region_re=INCLUDE_REGION_RE,
            geo_knn_k=GEO_KNN_K,
        )
        fold_fit = fit_graph_model(
            fold_data,
            MODEL_KIND,
            gp_kernel="exp",
            sample_kwargs=sample_kwargs,
        )
        fold_predictions = predict_all_segments(fold_fit)
        held = observed[observed["sre"].isin(held_sres)][
            ["sre", "sentido", TARGET]
        ].rename(columns={TARGET: "vmd_observado_cv"})
        matched = fold_predictions[fold_predictions["fonte"] == "previsto"].merge(
            held,
            on=["sre", "sentido"],
            how="inner",
        )
        fold_metrics = compute_metrics(
            matched["vmd_observado_cv"].to_numpy(dtype=float),
            matched[f"{TARGET}_ajustado"].to_numpy(dtype=float),
        )
        convergence = convergence_diagnostics(fold_fit["idata"])
        fold_rows.append(
            {
                "fold": fold_number,
                "n_sre_treino": int(fold_data.train_df["sre"].nunique()),
                "n_sre_teste": int(len(held_sres)),
                "n_tr": int(len(fold_data.train_df)),
                "n_teste": int(len(matched)),
                "R2_oos": fold_metrics["r2"],
                "RMSE_oos": fold_metrics["rmse"],
                "MAE_oos": fold_metrics["mae"],
                "MAPE_oos_pct": fold_metrics["mape"],
                "max_rhat": convergence["max_rhat"],
                "min_ESS": convergence["min_ess"],
                "divergencias": convergence["divergences"],
            }
        )
        matched = matched.assign(fold=fold_number)
        prediction_rows.append(
            matched[[
                "fold", "sre", "sentido", "go", "regional_macro",
                "vmd_observado_cv", f"{TARGET}_ajustado",
            ]]
        )

    cv_predictions = pd.concat(prediction_rows, ignore_index=True)
    consolidated = compute_metrics(
        cv_predictions["vmd_observado_cv"].to_numpy(dtype=float),
        cv_predictions[f"{TARGET}_ajustado"].to_numpy(dtype=float),
    )
    cv_metrics = pd.DataFrame(fold_rows)
    cv_metrics.to_csv(
        output_dir / "glmm_car_vmd_cv_metricas_por_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_predictions.to_csv(
        output_dir / "glmm_car_vmd_cv_predicoes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "protocolo": f"{n_splits}-fold agrupado por SRE; ambos os sentidos no mesmo fold",
        "n_oos": int(len(cv_predictions)),
        "n_sre_oos": int(cv_predictions["sre"].nunique()),
        "r2_oos": float(consolidated["r2"]),
        "rmse_oos": float(consolidated["rmse"]),
        "mae_oos": float(consolidated["mae"]),
        "mape_oos_pct": float(consolidated["mape"]),
        "max_rhat_folds": float(cv_metrics["max_rhat"].max()),
        "min_ess_folds": float(cv_metrics["min_ESS"].min()),
        "divergencias_folds": int(cv_metrics["divergencias"].sum()),
        "tempo_s": round(time.perf_counter() - started, 2),
    }


def build_random_effects_table(fit: dict) -> pd.DataFrame:
    idata = fit["idata"]
    graph_data = fit["gdata"]
    rows = []
    for variable, effect_type, labels in [
        ("u_go", "rodovia", graph_data.go_categories),
        ("u_region", "regiao", graph_data.region_categories),
    ]:
        if variable not in idata.posterior or labels is None:
            continue
        samples = az.extract(idata, group="posterior", var_names=variable)
        means = samples.mean("sample").values
        lowers = samples.quantile(0.05, dim="sample").values
        uppers = samples.quantile(0.95, dim="sample").values
        for label, mean, lower, upper in zip(labels, means, lowers, uppers):
            rows.append(
                {
                    "tipo_efeito": effect_type,
                    "categoria": label,
                    "media": float(mean),
                    "ic90_inferior": float(lower),
                    "ic90_superior": float(upper),
                }
            )
    return pd.DataFrame(rows)


def save_observed_vs_fitted(spatial: pd.DataFrame, output_path: Path) -> None:
    observed = spatial[f"{TARGET}_observado"].to_numpy(dtype=float)
    fitted = spatial[f"{TARGET}_ajustado"].to_numpy(dtype=float)
    lower = float(min(observed.min(), fitted.min()))
    upper = float(max(observed.max(), fitted.max()))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(observed, fitted, alpha=0.72, color="#126782", edgecolor="white", linewidth=0.4)
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#b23a48", linewidth=1.5)
    ax.set(title="VMD observado vs. ajustado - GLMM + CAR", xlabel="VMD observado", ylabel="VMD ajustado")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_spatial_profile(spatial: pd.DataFrame, output_path: Path) -> None:
    group_columns = ["go", "sentido"] if "sentido" in spatial.columns else ["go"]
    ordered = spatial.sort_values([*group_columns, "km_mid"])
    fig, ax = plt.subplots(figsize=(11, 6))
    for _, group in ordered.groupby(group_columns):
        ax.plot(group["km_mid"], group["u_car_media"], marker="o", markersize=2.5, linewidth=0.8, alpha=0.55)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set(title="Efeito espacial CAR por rodovia", xlabel="Posicao media do segmento (km)", ylabel="Efeito CAR no log(VMD)")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_results(
    output_dir: Path,
    dataset: Path,
    graph_data: GraphData,
    fit: dict,
    elapsed_seconds: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate_graph_fit(fit)
    coefficients = coefficient_summary(fit)
    hyperparameters = hyperparameter_summary(fit)
    spatial = spatial_effects_table(fit)
    predictions = predict_all_segments(fit)
    formula = build_equation_text(
        coefficients,
        MODEL_KIND,
        TARGET,
        include_region_re=INCLUDE_REGION_RE,
    )
    random_effects = build_random_effects_table(fit)

    result_entry = {
        "fit": fit,
        "ev": evaluation,
        "coef": coefficients,
        "hyper": hyperparameters,
        "spatial": spatial,
        "pred_table": predictions,
        "formula": formula,
        "gp_kernel": "exp",
    }
    frames = export_frames({TARGET: result_entry})
    excel_bytes = glmm_graph_only_excel(frames)

    regional = build_regional_diagnostics(graph_data, predictions, evaluation)
    direction_diagnostics = build_direction_diagnostics(predictions)
    prediction_diagnostics = build_prediction_diagnostics(graph_data, predictions)
    check = data_check_summary(dataset, graph_data.df_all, graph_data)
    check["n_total"] = int(len(graph_data.df_all))

    metrics = frames["GLMM_GRAFO_metricas"]
    metrics.to_csv(output_dir / "glmm_car_vmd_metricas.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(output_dir / "glmm_car_vmd_coeficientes.csv", encoding="utf-8-sig")
    hyperparameters.to_csv(output_dir / "glmm_car_vmd_hiperparametros.csv", encoding="utf-8-sig")
    spatial.to_csv(output_dir / "glmm_car_vmd_ajuste_observados.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_dir / "glmm_car_vmd_segmentos.csv", index=False, encoding="utf-8-sig")
    random_effects.to_csv(output_dir / "glmm_car_vmd_efeitos_aleatorios.csv", index=False, encoding="utf-8-sig")
    regional.to_csv(output_dir / "glmm_car_vmd_diagnostico_regional.csv", index=False, encoding="utf-8-sig")
    direction_diagnostics.to_csv(
        output_dir / "glmm_car_vmd_diagnostico_sentidos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "glmm_car_vmd_resultados.xlsx").write_bytes(excel_bytes)
    (output_dir / "glmm_car_vmd_formula.txt").write_text(formula, encoding="utf-8")

    save_observed_vs_fitted(spatial, output_dir / "glmm_car_vmd_observado_vs_ajustado.png")
    save_spatial_profile(spatial, output_dir / "glmm_car_vmd_perfil_espacial.png")

    summary = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "tempo_execucao_s": round(elapsed_seconds, 2),
        "configuracao": check,
        "padronizacao": {
            name: {"media": float(values[0]), "desvio_padrao": float(values[1])}
            for name, values in graph_data.num_stats.items()
        },
        "metricas": evaluation,
        "diagnostico_predicoes": prediction_diagnostics,
        "formula": formula,
    }
    (output_dir / "glmm_car_vmd_resumo.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    generated_at = datetime.now()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {args.dataset}")

    raw, aliases_applied = normalize_dataset_columns(pd.read_excel(args.dataset))
    if aliases_applied:
        print(
            "Aviso: colunas alternativas usadas no lugar das canonicas: "
            + ", ".join(f"{dst} <- {src}" for dst, src in aliases_applied.items()),
            flush=True,
        )
    targets = list(dict.fromkeys(args.targets))
    adjusted, growth_audit = apply_annual_growth(
        raw,
        targets,
        current_year=args.current_year,
        growth_rate=args.growth_rate,
    )
    model_data, expansion_audit = prepare_single_dataset(
        adjusted,
        targets,
        current_year=args.current_year,
    )
    expansion_audit["aliases_aplicados"] = aliases_applied
    model_data, situation_audit = filter_allowed_situations(model_data)
    # O modelo GLMM+CAR/GP so gera estimativas proprias para SREs principais
    # (principal='s'); as nao-principais espelham a SRE coincidente depois.
    fit_data = model_data.loc[model_data["principal"].eq("s")].reset_index(drop=True)
    graph_data_by_target = {
        target: prepare_target_data(fit_data, target) for target in targets
    }
    checks = {
        target: build_target_check(
            args.dataset,
            fit_data,
            graph_data_by_target[target],
            target,
        )
        for target in targets
    }
    check_payload = {
        "pre_processamento": growth_audit,
        "expansao_modelo_saida": expansion_audit,
        "filtro_situacao": situation_audit,
        "targets": checks,
    }
    print(json.dumps(check_payload, indent=2, ensure_ascii=False))
    if args.check_data:
        return

    sample_kwargs = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "target_accept": args.target_accept,
        "random_seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: dict[str, dict] = {}
    prediction_diagnostics: dict[str, dict] = {}
    target_times: dict[str, float] = {}
    zero_only_tables: dict[str, pd.DataFrame] = {}
    for index, target in enumerate(targets, start=1):
        target_started = time.perf_counter()
        graph_data = graph_data_by_target[target]
        if len(graph_data.y_train) == 0:
            # Sem nenhuma contagem > 0 (ex.: 3s1 so tem zeros): o modelo Gamma
            # nao e identificavel. Estimativa = 0 para os segmentos sem contagem.
            print(
                f"\n[{index}/{len(targets)}] Target {target} sem contagens positivas: "
                "ajuste ignorado, estimativa = 0.",
                flush=True,
            )
            zero_only_tables[target] = fit_data[["sre", "sentido"]].assign(
                **{f"{target}_ajustado": 0.0}
            )
            prediction_diagnostics[target] = {
                "ajuste_ignorado": "sem contagens positivas; estimativa fixada em 0",
                "n_observado": int(fit_data[target].notna().sum()),
            }
            target_times[target] = round(time.perf_counter() - target_started, 2)
            continue
        print(f"\n[{index}/{len(targets)}] Ajustando target {target}...", flush=True)
        fit = fit_graph_model(
            graph_data_by_target[target],
            MODEL_KIND,
            gp_kernel="exp",
            sample_kwargs=sample_kwargs,
        )
        result_entry, diagnostics = build_result_entry(fit, target)
        results[target] = result_entry
        prediction_diagnostics[target] = diagnostics
        target_times[target] = round(time.perf_counter() - target_started, 2)

    elapsed_seconds = time.perf_counter() - started
    frames = export_frames(results)
    operational_path, technical_path, summary_path = build_output_paths(
        args.output_dir,
        generated_at,
    )
    technical_path.write_bytes(glmm_graph_only_excel(frames))

    output_columns = build_operational_columns(targets)
    operational = build_operational_output(
        model_data,
        {
            target: (
                results[target]["pred_table"]
                if target in results
                else zero_only_tables[target]
            )
            for target in targets
        },
        output_columns,
        targets,
        current_year=args.current_year,
        growth_rate=args.growth_rate,
    )
    operational = apply_coincident_estimates(operational, model_data, targets)
    write_operational_workbook(operational, operational_path)

    summary = {
        "gerado_em": generated_at.isoformat(timespec="seconds"),
        "tempo_execucao_s": round(elapsed_seconds, 2),
        "pre_processamento": growth_audit,
        "expansao_modelo_saida": expansion_audit,
        "filtro_situacao": situation_audit,
        "configuracao_por_target": checks,
        "diagnostico_predicoes": prediction_diagnostics,
        "tempo_por_target_s": target_times,
        "arquivo_operacional": str(operational_path.resolve()),
        "arquivo_tecnico": str(technical_path.resolve()),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if args.run_cv:
        raise NotImplementedError(
            "A CV multi-target deve ser executada por target; use --targets com um target por vez."
        )

    print(f"\nConcluido em {elapsed_seconds:.1f}s")
    print(f"Saida operacional: {operational_path.resolve()}")
    print(f"Saida tecnica: {technical_path.resolve()}")
    print(f"Resumo: {summary_path.resolve()}")


if __name__ == "__main__":
    main()