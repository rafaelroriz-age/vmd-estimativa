"""Estimativa rapida do volume por classe veicular (35 classes) em todas as SREs.

GLMM empirico-Bayesiano (estimativa MAP / BLUP) em escala log, ajustado em
segundos, em substituicao ao GLMM Gamma + CAR por MCMC de
``modelo_glmm_car_vmd_v2.py``. Le UM unico dataset (malha completa com as
contagens) e grava a planilha operacional no mesmo layout de 41 colunas.

Modelo, para cada classe k e cada linha i (SRE x sentido):

    log(1 + y_ik) = x_i' beta_k + u_go[i],k + u_reg[i],k
                    + gamma_k * s_ik + e_ik

- x_i: efeitos fixos (empresas, distancia ao polo, populacao, PIB, frota,
  extensao, perimetro urbano, classe funcional, situacao, sentido, conexao
  com rodovia federal, coordenadas);
- u_go, u_reg: interceptos aleatorios por rodovia e por regional, com prior
  normal (implementados como penalidade ridge sobre as dummies; o resultado
  e o BLUP do modelo misto com razao de variancias fixa);
- s_ik: termo espacial, a media suavizada de log(1 + y) das SREs contadas
  vizinhas: (a) ao longo da mesma rodovia, peso exp(-gap_km / L_ROAD); e
  (b) as K mais proximas em linha reta, peso exp(-dist_km / L_GEO). Ambas sao
  encolhidas para a media global. No treino, a propria SRE (os dois
  sentidos) e excluida do calculo, de modo que o termo nunca usa o proprio
  alvo;
- zeros sao mantidos no treino (o v2 os descartava, o que enviesava as
  classes esparsas para cima).

Uso:
    python modelo_glmm_eb_vmd.py              # ajusta e grava a planilha
    python modelo_glmm_eb_vmd.py --cv         # tambem roda CV agrupada por SRE
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:  # pragma: no cover - depende do terminal
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

DELIVERY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DELIVERY_DIR.parent
DEFAULT_DATASET = PROJECT_ROOT / "02_dados" / "vmd_SRE_SET-26_features_2026.xlsx"
DEFAULT_OUTPUT_DIR = DELIVERY_DIR / "outputs"

TARGETS = [
    "Passeio", "Van", "Pickup", "Moto", "2cb", "3cb", "4cb", "2sb1",
    "2ib2", "2c", "3c", "4c", "4cd", "2s1", "2s2", "3s1", "2s3",
    "3s2", "2i2", "2i3", "2j3", "3i2", "3s3", "3i3", "3j3", "4r4",
    "2c2", "2c3", "3c2", "3c3", "3d3", "3d4", "3q4", "3q6", "3t6",
]
ALLOWED_SITUATIONS = frozenset({"PAV", "DUP", "EOD"})

# Hiperparametros, fixados por CV agrupada por SRE (5 folds x 3 repeticoes).
RIDGE_ALPHA = 3.0        # penalidade geral (prior dos efeitos fixos)
SCALE_GO = 0.5           # escala das dummies de rodovia -> var(u_go)
SCALE_REG = 0.5          # escala das dummies de regional -> var(u_reg)
L_ROAD_KM = 30.0         # alcance da suavizacao ao longo da rodovia
L_GEO_KM = 30.0          # alcance da suavizacao geografica
K_GEO = 6                # vizinhos geograficos
SHRINK = 0.3             # peso da media global na suavizacao espacial


# ─────────────────────────────────────────────────────────────────────────────
# Dados
# ─────────────────────────────────────────────────────────────────────────────
def read_dataset(path: Path, cache_dir: Path | None) -> pd.DataFrame:
    """Le o xlsx (lento, ~40s) e guarda um cache .pkl invalidado quando o
    arquivo de origem muda (tamanho ou data de modificacao)."""
    stat = path.stat()
    cache = None
    if cache_dir is not None:
        cache = cache_dir / f".cache_{path.stem}_{stat.st_size}_{int(stat.st_mtime)}.pkl"
        if cache.exists():
            return pd.read_pickle(cache)
    try:  # calamine e ~10x mais rapido que openpyxl, se estiver instalado
        df = pd.read_excel(path, engine="calamine")
    except (ImportError, ValueError):
        df = pd.read_excel(path)
    if cache is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for old in cache_dir.glob(f".cache_{path.stem}_*.pkl"):
            old.unlink()
        df.to_pickle(cache)
    return df


def validate(df: pd.DataFrame) -> None:
    required = {
        "sre", "sentido", "go", "trecho", "classe", "situacao", "perim_urb",
        "regional", "km_inicial", "km_final", "extensao", "lat_inicia",
        "lon_inicia", "lat_final", "lon_final", "media_empresas_ativas",
        "distancia_cid_1", "media_populacao_residente", "media_pib",
        "media_frotas_ativas", "ano", *TARGETS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Colunas ausentes no dataset: {missing}")
    if df.duplicated(["sre", "sentido"]).any():
        raise ValueError("Dataset com (sre, sentido) duplicado.")
    counts = df[TARGETS].notna().sum(axis=1)
    if counts.between(1, len(TARGETS) - 1).any():
        raise ValueError("Ha linhas com apenas parte das 35 classes preenchidas.")


def apply_annual_growth(df: pd.DataFrame, year: int, rate: float) -> pd.DataFrame:
    """Leva as contagens observadas para o ano de referencia."""
    out = df.copy()
    observed = out[TARGETS].notna().all(axis=1)
    years = pd.to_numeric(out["ano"], errors="coerce")
    if (observed & years.isna()).any():
        raise ValueError("Linhas observadas sem ano de contagem.")
    if (observed & (years > year)).any():
        raise ValueError(f"Linhas observadas com ano posterior a {year}.")
    factor = (1.0 + rate) ** (year - years[observed])
    out.loc[observed, TARGETS] = out.loc[observed, TARGETS].mul(factor, axis=0)
    return out


def _z(s: pd.Series) -> np.ndarray:
    s = s.astype(float)
    sd = s.std()
    return ((s - s.mean()) / (sd if sd > 0 else 1.0)).to_numpy()


def _dummies(s: pd.Series, drop_first: bool) -> np.ndarray:
    return pd.get_dummies(s.astype(str), drop_first=drop_first).to_numpy(float)


def build_design(df: pd.DataFrame) -> dict:
    """Matrizes de efeitos fixos/aleatorios e geometria (sem usar os alvos)."""
    as_text = lambda c: df[c].astype(str)  # noqa: E731
    fixed = np.column_stack([
        _z(np.log1p(df["media_empresas_ativas"])),
        _z(np.log1p(df["distancia_cid_1"])),
        _z(np.log1p(df["media_populacao_residente"])),
        _z(np.log1p(df["media_pib"])),
        _z(np.log1p(df["media_frotas_ativas"])),
        _z(np.log(df["extensao"].astype(float) + 0.05)),
        (as_text("perim_urb").str.lower() == "s").astype(float),
        df["coin_f1"].notna().astype(float) if "coin_f1" in df else np.zeros(len(df)),
        (as_text("inicio").str.contains("BR-") | as_text("fim").str.contains("BR-")).astype(float)
        if {"inicio", "fim"} <= set(df.columns) else np.zeros(len(df)),
        (df["sentido"] == "DECRESCENTE").astype(float),
        _z((df["lat_inicia"] + df["lat_final"]) / 2),
        _z((df["lon_inicia"] + df["lon_final"]) / 2),
        _dummies(df["classe"], drop_first=True),
        _dummies(df["situacao"], drop_first=True),
    ])
    random = np.column_stack([
        _dummies(df["go"], drop_first=False) * SCALE_GO,
        _dummies(df["regional"], drop_first=False) * SCALE_REG,
    ])
    km = df[["km_inicial", "km_final"]].astype(float).to_numpy()
    return {
        "X": np.column_stack([fixed, random]),
        "lat": np.radians(((df["lat_inicia"] + df["lat_final"]) / 2).to_numpy(float)),
        "lon": np.radians(((df["lon_inicia"] + df["lon_final"]) / 2).to_numpy(float)),
        "km_lo": km.min(axis=1),
        "km_hi": km.max(axis=1),
        "go": df["go"].astype(str).to_numpy(),
        "sre": df["sre"].astype(str).to_numpy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Modelo
# ─────────────────────────────────────────────────────────────────────────────
def spatial_terms(D: dict, rows: np.ndarray, tr: np.ndarray, YL_tr: np.ndarray):
    """Suavizacao espacial de log(1+y) das SREs de treino para ``rows``,
    excluindo sempre a propria SRE (os dois sentidos)."""
    mu = YL_tr.mean(axis=0)
    same = D["sre"][rows][:, None] == D["sre"][tr][None, :]

    # (a) ao longo da mesma rodovia: gap entre os intervalos de km
    gap = np.maximum(
        0.0,
        np.maximum(D["km_lo"][rows][:, None] - D["km_hi"][tr][None, :],
                   D["km_lo"][tr][None, :] - D["km_hi"][rows][:, None]),
    )
    gap[(D["go"][rows][:, None] != D["go"][tr][None, :]) | same] = np.inf
    w_road = np.exp(-gap / L_ROAD_KM)
    s_road = w_road.sum(axis=1, keepdims=True)
    road = (w_road @ YL_tr + SHRINK * mu) / (s_road + SHRINK)

    # (b) geografica: K vizinhos mais proximos (haversine)
    la, lo = D["lat"], D["lon"]
    a = (np.sin((la[rows][:, None] - la[tr][None, :]) / 2) ** 2
         + np.cos(la[rows])[:, None] * np.cos(la[tr])[None, :]
         * np.sin((lo[rows][:, None] - lo[tr][None, :]) / 2) ** 2)
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    dist[same] = np.inf
    k = min(K_GEO, max(1, len(tr) - 2))
    nn = np.argsort(dist, axis=1)[:, :k]
    dnn = np.take_along_axis(dist, nn, axis=1)
    w_geo = np.exp(-dnn / L_GEO_KM)
    geo = (np.einsum("ik,ikj->ij", w_geo, YL_tr[nn]) + SHRINK * mu) / (
        w_geo.sum(axis=1, keepdims=True) + SHRINK)

    extra = np.column_stack([np.log1p(s_road), np.log1p(dnn.mean(axis=1))])
    return road, geo, extra


def ridge_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge com intercepto nao penalizado (equivalente ao sklearn.Ridge),
    resolvido em forma fechada: (Xc'Xc + alpha I) b = Xc'yc."""
    xm, ym = Xtr.mean(axis=0), ytr.mean()
    Xc = Xtr - xm
    b = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ (ytr - ym))
    return (Xte - xm) @ b + ym


def fit_predict(D: dict, YL: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """Ajusta nas linhas ``tr`` e preve log(1+y) nas linhas ``te``."""
    YL_tr = YL[tr]
    road_tr, geo_tr, ex_tr = spatial_terms(D, tr, tr, YL_tr)
    road_te, geo_te, ex_te = spatial_terms(D, te, tr, YL_tr)
    base_tr = np.column_stack([D["X"][tr], ex_tr])
    base_te = np.column_stack([D["X"][te], ex_te])
    pred = np.empty((len(te), YL.shape[1]))
    for k in range(YL.shape[1]):
        pred[:, k] = ridge_fit_predict(
            np.column_stack([base_tr, road_tr[:, k], geo_tr[:, k]]), YL_tr[:, k],
            np.column_stack([base_te, road_te[:, k], geo_te[:, k]]), RIDGE_ALPHA)
    return pred


def to_volume(pred_log: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(pred_log), 0.0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Validacao cruzada agrupada por SRE
# ─────────────────────────────────────────────────────────────────────────────
def vmd_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ly, lp = np.log1p(y), np.log1p(p)
    ape = np.abs(p - y) / y
    return {
        "vmd_r2": 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        "vmd_r2_log": 1 - ((ly - lp) ** 2).sum() / ((ly - ly.mean()) ** 2).sum(),
        "vmd_mape_pct": 100 * ape.mean(),
        "vmd_mdape_pct": 100 * np.median(ape),
        "vmd_dentro_30pct": 100 * (ape <= 0.30).mean(),
        "vmd_dentro_50pct": 100 * (ape <= 0.50).mean(),
    }


def class_metrics(Y: np.ndarray, P: np.ndarray) -> dict:
    rmsle = np.sqrt(((np.log1p(P) - np.log1p(Y)) ** 2).mean(axis=0))
    out = {
        "classes_wape_pct": 100 * np.abs(P - Y).sum() / Y.sum(),
        "classes_rmsle_macro": rmsle.mean(),
    }
    out.update({f"rmsle_{c}": rmsle[TARGETS.index(c)] for c in ("Passeio", "Moto", "2c", "3c", "3s3", "3t6")})
    return out


def grouped_folds(groups: np.ndarray, folds: int, seed: int):
    """K folds com grupos (SRE) embaralhados: os dois sentidos de uma SRE
    sempre ficam juntos no mesmo fold."""
    uniq = np.random.default_rng(seed).permutation(np.unique(groups))
    for held in np.array_split(uniq, folds):
        te = np.isin(groups, held)
        yield np.flatnonzero(~te), np.flatnonzero(te)


def run_cv(D: dict, YL: np.ndarray, obs: np.ndarray, folds: int, repeats: int) -> dict:
    Y = np.expm1(YL[obs])
    groups = D["sre"][obs]
    runs = []
    for r in range(repeats):
        P = np.empty_like(Y)
        for itr, ite in grouped_folds(groups, folds, seed=r):
            P[ite] = to_volume(fit_predict(D, YL, obs[itr], obs[ite]))
        runs.append({**vmd_metrics(Y.sum(axis=1), P.sum(axis=1)), **class_metrics(Y, P)})
    return {k: round(float(np.mean([m[k] for m in runs])), 4) for k in runs[0]}


# ─────────────────────────────────────────────────────────────────────────────
# Saida operacional
# ─────────────────────────────────────────────────────────────────────────────
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


def build_operational_output(df: pd.DataFrame, volumes: np.ndarray, observed: np.ndarray,
                             year: int, rate: float) -> pd.DataFrame:
    out = pd.DataFrame({
        "sre": df["sre"].to_numpy(),
        "rodovia": df["go"].map(_format_road).to_numpy(),
        "trecho": df["trecho"].to_numpy(),
        "sentido_trafego": df["sentido"].to_numpy(),
        "ano": year,
        "taxa_crescimento": rate,
    })
    vals = np.where(observed[:, None], df[TARGETS].to_numpy(float), volumes)
    out[TARGETS] = vals
    out["origem"] = np.where(observed, "Contagem de Tráfego", "Estimativa Técnica")
    return out


def write_operational_workbook(output: pd.DataFrame, path: Path) -> None:
    try:
        _write_xlsxwriter(output, path)
    except ImportError:
        _write_openpyxl(output, path)


def _write_xlsxwriter(output: pd.DataFrame, path: Path) -> None:
    """Formatacao por coluna (rapida); mesmo visual da versao openpyxl."""
    import xlsxwriter

    path.parent.mkdir(parents=True, exist_ok=True)
    nrow, ncol = output.shape
    with xlsxwriter.Workbook(str(path)) as wb:
        ws = wb.add_worksheet("Planilha1")
        # escrita direta (o to_excel do pandas e ~10x mais lento)
        for r, row in enumerate(output.to_numpy(object).tolist(), start=1):
            ws.write_row(r, 0, row)
        pct = wb.add_format({"num_format": "0.00%"})
        num = wb.add_format({"num_format": "0.00"})
        for i, width in enumerate((16, 12, 22, 20, 10)):
            ws.set_column(i, i, width)
        ws.set_column(5, 5, 18, pct)
        ws.set_column(6, ncol - 2, 12, num)
        ws.set_column(ncol - 1, ncol - 1, 24)
        ws.add_table(0, 0, nrow, ncol - 1, {
            "name": "ModeloVolumeGLMMEB",
            "style": "Table Style Medium 2",
            "columns": [{"header": str(c), "header_format": wb.add_format(
                {"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78"})}
                for c in output.columns],
        })
        ws.freeze_panes(1, 6)


def _write_openpyxl(output: pd.DataFrame, path: Path) -> None:
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    path.parent.mkdir(parents=True, exist_ok=True)
    ncol = len(output.columns)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        output.to_excel(writer, sheet_name="Planilha1", index=False)
        ws = writer.book["Planilha1"]
        ws.freeze_panes = "G2"
        for cell in ws[1]:
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.font = Font(color="FFFFFF", bold=True)
        for col, width in zip("ABCDEF", (16, 12, 22, 20, 10, 18)):
            ws.column_dimensions[col].width = width
        for i in range(7, ncol):
            ws.column_dimensions[get_column_letter(i)].width = 12
        ws.column_dimensions[get_column_letter(ncol)].width = 24
        for cell in ws["F"][1:]:
            cell.number_format = "0.00%"
        for row in ws.iter_rows(min_row=2, max_row=len(output) + 1, min_col=7, max_col=ncol - 1):
            for cell in row:
                cell.number_format = "0.00"
        table = Table(displayName="ModeloVolumeGLMMEB",
                      ref=f"A1:{get_column_letter(ncol)}{len(output) + 1}")
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        ws.add_table(table)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--current-year", type=int, default=datetime.now().year)
    p.add_argument("--growth-rate", type=float, default=0.03)
    p.add_argument("--cv", action="store_true", help="roda CV agrupada por SRE antes do ajuste final")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=3)
    p.add_argument("--no-cache", action="store_true", help="sempre rele o xlsx, sem cache")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    raw = read_dataset(args.dataset, None if args.no_cache else args.output_dir)
    validate(raw)
    t_read = time.perf_counter() - t0
    data = apply_annual_growth(raw, args.current_year, args.growth_rate)
    keep = data["situacao"].astype(str).str.strip().str.upper().isin(ALLOWED_SITUATIONS)
    data = data.loc[keep].reset_index(drop=True)

    observed = data[TARGETS].notna().all(axis=1).to_numpy()
    obs = np.flatnonzero(observed)
    D = build_design(data)
    YL = np.log1p(data[TARGETS].fillna(0.0).to_numpy(float).clip(min=0))
    print(f"Dataset: {len(data)} linhas (SRE x sentido) em PAV/DUP/EOD; "
          f"{len(obs)} com contagem ({len(set(D['sre'][obs]))} SREs). Leitura: {t_read:.1f}s")

    summary = {"dataset": str(args.dataset.resolve()), "linhas": int(len(data)),
               "linhas_observadas": int(len(obs)), "ano": args.current_year,
               "taxa_crescimento": args.growth_rate}
    if args.cv:
        t = time.perf_counter()
        cv = run_cv(D, YL, obs, args.cv_folds, args.cv_repeats)
        summary["cv_agrupada_por_sre"] = cv
        print(f"CV {args.cv_folds}-fold x {args.cv_repeats} (agrupada por SRE, "
              f"{time.perf_counter() - t:.1f}s):")
        for k, v in cv.items():
            print(f"  {k:22s} {v:10.4f}")

    t = time.perf_counter()
    volumes = to_volume(fit_predict(D, YL, obs, np.arange(len(data))))
    summary["segundos_ajuste_previsao"] = round(time.perf_counter() - t, 2)

    output = build_operational_output(data, volumes, observed, args.current_year, args.growth_rate)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx = args.output_dir / f"modelo_volume_glmm_eb_{stamp}.xlsx"
    write_operational_workbook(output, xlsx)
    summary["arquivo_operacional"] = str(xlsx.resolve())
    summary["segundos_total"] = round(time.perf_counter() - t0, 1)
    (args.output_dir / f"glmm_eb_resumo_{stamp}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Ajuste + previsao (35 classes, {len(data)} linhas): {summary['segundos_ajuste_previsao']}s")
    print(f"Saida operacional: {xlsx.resolve()}")
    print(f"Tempo total: {summary['segundos_total']}s")


if __name__ == "__main__":
    main()
