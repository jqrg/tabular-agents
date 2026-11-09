# -*- coding: utf-8 -*-
"""Análisis estadístico y figuras del Capítulo 5 (comando analyze).

Entrada: results/runs.jsonl.
Salida:  results/summary.csv, results/analysis.md y results/figures/*.png.
Incluye tablas por sistema y problema, Wilcoxon pareado con Holm, Fisher
sobre fallos estructurales, calibración del agente, frecuencia del aumento
polinómico y sección de robustez si hay ejecuciones perturbadas.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin display: las figuras se guardan a fichero
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from .config import Config  # noqa: E402
from .persistencia import leer_jsonl  # noqa: E402

ALFA = 0.05


# ------------------------- utilidades de formato ---------------------------
def df_a_markdown(df: pd.DataFrame, decimales: int = 4) -> str:
    """Tabla markdown sin depender de tabulate (no está en las dependencias)."""
    if df.empty:
        return "_(sin datos)_"

    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "-"
        if isinstance(v, float):
            return f"{v:.{decimales}g}"
        return str(v)

    columnas = [str(c) for c in df.columns]
    lineas = ["| " + " | ".join(columnas) + " |",
              "|" + "|".join(["---"] * len(columnas)) + "|"]
    for _, fila in df.iterrows():
        lineas.append("| " + " | ".join(fmt(v) for v in fila) + " |")
    return "\n".join(lineas)


def correccion_holm(pvalores: list[float]) -> list[float]:
    """p ajustados por Holm (step-down), monótonos y acotados a 1."""
    m = len(pvalores)
    orden = sorted(range(m), key=lambda i: pvalores[i])
    ajustados = [0.0] * m
    previo = 0.0
    for k, i in enumerate(orden):
        ajustado = min(1.0, (m - k) * pvalores[i])
        previo = max(previo, ajustado)
        ajustados[i] = previo
    return ajustados


# ------------------------------ carga --------------------------------------
def cargar_runs(ruta_runs: Path) -> pd.DataFrame:
    """Carga runs.jsonl en un DataFrame."""
    registros = leer_jsonl(ruta_runs)
    if not registros:
        return pd.DataFrame()
    df = pd.DataFrame(registros)
    for col in ("mae_run", "err_norm_run", "seconds", "dispersion_norm"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "valido" in df.columns:
        df["valido"] = df["valido"].fillna(False).astype(bool)
    if "perturbacion" not in df.columns:
        df["perturbacion"] = None
    return df


# ---------------------------- tablas resumen -------------------------------
def tabla_sistema_prompt(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por (sistema, prompt): MAE, mediana, std, err_norm, fallos."""
    if df.empty:
        return pd.DataFrame()
    filas = []
    for (sistema, prompt_id), grupo in df.groupby(["sistema", "prompt_id"]):
        validos = grupo[grupo["valido"]]
        filas.append({
            "sistema": sistema,
            "prompt_id": prompt_id,
            "n_runs": len(grupo),
            "n_validos": len(validos),
            "tasa_fallo": 1.0 - len(validos) / len(grupo),
            "mae": validos["mae_run"].mean(),
            "mediana_ae": validos["mae_run"].median(),
            "std_ae": validos["mae_run"].std(ddof=1),
            "err_norm": validos["err_norm_run"].mean(),
            "seconds_media": grupo["seconds"].mean(),
        })
    return pd.DataFrame(filas).sort_values(["sistema", "prompt_id"]).reset_index(drop=True)


def tabla_agregada(resumen: pd.DataFrame) -> pd.DataFrame:
    """Agregado por sistema (media sobre prompts de los estadísticos)."""
    if resumen.empty:
        return pd.DataFrame()
    agg = resumen.groupby("sistema").agg(
        n_prompts=("prompt_id", "nunique"),
        n_runs=("n_runs", "sum"),
        mae_medio=("mae", "mean"),
        err_norm_medio=("err_norm", "mean"),
        tasa_fallo=("tasa_fallo", "mean"),
        seconds_media=("seconds_media", "mean"),
    ).reset_index()
    return agg.sort_values("err_norm_medio").reset_index(drop=True)


def tabla_por_salida(df: pd.DataFrame, nombres_salida: dict[str, list[str]]) -> pd.DataFrame:
    """Desglose del error por salida individual en los problemas multi-salida
    (el MAE promedia todas las salidas y ocultaría aciertos parciales).

    nombres_salida: {prompt_id_base: [nombres de las salidas]} (del ground
    truth); si falta, se etiquetan salida_1, salida_2, ...
    """
    if df.empty or "errores_abs" not in df.columns:
        return pd.DataFrame()
    filas = []
    validos = df[df["valido"] & df["errores_abs"].notna()]
    for (sistema, prompt_id), grupo in validos.groupby(["sistema", "prompt_id"]):
        base = grupo["prompt_id_base"].iloc[0] if "prompt_id_base" in grupo else prompt_id
        errores = [e for e in grupo["errores_abs"] if isinstance(e, list)]
        if not errores:
            continue
        n_out = len(errores[0])
        if n_out < 2:  # solo interesa donde el promedio oculta detalle
            continue
        nombres = nombres_salida.get(base) or [f"salida_{i+1}" for i in range(n_out)]
        for i in range(n_out):
            valores = [e[i] for e in errores if len(e) == n_out]
            if not valores:
                continue
            filas.append({
                "sistema": sistema,
                "prompt_id": prompt_id,
                "salida": nombres[i] if i < len(nombres) else f"salida_{i+1}",
                "mae_salida": sum(valores) / len(valores),
                "n": len(valores),
            })
    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas).sort_values(["prompt_id", "sistema", "salida"]).reset_index(drop=True)


def escribir_summary_csv(cfg: Config, ruta_runs: Path | None = None) -> Path | None:
    """Regenera results/summary.csv a partir de runs.jsonl."""
    ruta_runs = ruta_runs or cfg.ruta_runs
    df = cargar_runs(ruta_runs)
    if df.empty:
        return None
    resumen = tabla_sistema_prompt(df[df["perturbacion"].isna()])
    cfg.dir_resultados.mkdir(parents=True, exist_ok=True)
    ruta = cfg.dir_resultados / "summary.csv"
    resumen.to_csv(ruta, index=False, encoding="utf-8")
    return ruta


# ------------------------- contrastes estadísticos -------------------------
def wilcoxon_pares(resumen: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank pareado por prompt sobre el MAE, entre pares de
    sistemas, con Holm y tamaño de efecto r = |z|/sqrt(n)."""
    if resumen.empty:
        return pd.DataFrame()
    pivote = resumen.pivot_table(index="prompt_id", columns="sistema", values="mae")
    sistemas = sorted(pivote.columns)
    filas = []
    for a, b in itertools.combinations(sistemas, 2):
        pareja = pivote[[a, b]].dropna()
        n = len(pareja)
        fila = {"sistema_A": a, "sistema_B": b, "n_prompts": n,
                "mediana_dif": None, "p": None, "r": None, "nota": ""}
        if n < 3:
            fila["nota"] = "n insuficiente (<3 prompts comunes)"
            filas.append(fila)
            continue
        difs = pareja[a] - pareja[b]
        fila["mediana_dif"] = float(difs.median())
        if (difs == 0).all():
            fila["nota"] = "diferencias todas nulas"
            filas.append(fila)
            continue
        try:
            res = stats.wilcoxon(pareja[a], pareja[b], zero_method="wilcox",
                                 method="approx")
            p = float(res.pvalue)
            z = float(getattr(res, "zstatistic", float("nan")))
            if math.isnan(z):
                z = stats.norm.isf(p / 2)
            # [WILCOXON-EXACTO 07/2026] con n<=10 pares el p exacto es el
            # adecuado; la z de la aproximación se conserva solo para el
            # tamaño de efecto r.
            try:
                res_ex = stats.wilcoxon(pareja[a], pareja[b],
                                        zero_method="wilcox", method="exact")
                p = float(res_ex.pvalue)
            except ValueError:
                fila["nota"] = "p aproximado (exacto no calculable)"
            fila["p"] = p
            fila["r"] = abs(z) / math.sqrt(n)
        except ValueError as e:
            fila["nota"] = f"no calculable: {e}"
        filas.append(fila)

    if not filas:  # menos de 2 sistemas: no hay pares que contrastar
        return pd.DataFrame()
    df = pd.DataFrame(filas)
    con_p = df["p"].notna()
    if con_p.any():
        ajustados = correccion_holm(df.loc[con_p, "p"].tolist())
        df.loc[con_p, "p_holm"] = ajustados
        df.loc[con_p, "significativo"] = [
            "sí" if p <= ALFA else "no" for p in ajustados
        ]
    else:
        df["p_holm"] = None
        df["significativo"] = None
    return df


def comparacion_fallos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Proporciones de fallo estructural por sistema + Fisher exacto pareado."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    tabla = df.groupby("sistema").agg(
        n_runs=("valido", "size"),
        n_fallos=("valido", lambda v: int((~v).sum())),
    ).reset_index()
    tabla["tasa_fallo"] = tabla["n_fallos"] / tabla["n_runs"]

    filas = []
    for a, b in itertools.combinations(sorted(tabla["sistema"]), 2):
        fa = tabla[tabla["sistema"] == a].iloc[0]
        fb = tabla[tabla["sistema"] == b].iloc[0]
        contingencia = [
            [int(fa["n_fallos"]), int(fa["n_runs"] - fa["n_fallos"])],
            [int(fb["n_fallos"]), int(fb["n_runs"] - fb["n_fallos"])],
        ]
        try:
            _, p = stats.fisher_exact(contingencia)
        except ValueError:
            p = float("nan")
        filas.append({"sistema_A": a, "sistema_B": b,
                      "fallos_A": contingencia[0][0], "fallos_B": contingencia[1][0],
                      "p_fisher": p})
    pares = pd.DataFrame(filas)
    if not pares.empty and pares["p_fisher"].notna().any():
        con_p = pares["p_fisher"].notna()
        pares.loc[con_p, "p_holm"] = correccion_holm(pares.loc[con_p, "p_fisher"].tolist())
    return tabla.sort_values("tasa_fallo").reset_index(drop=True), pares


# ------------------------------ calibración --------------------------------
def calibracion_agente(df: pd.DataFrame) -> dict:
    """Calibración de la señal de incertidumbre del agente propio, desglosada
    por sistema (template y libre no se mezclan). Por cada agente_*:
    Spearman entre dispersion_norm y err_norm_run sobre las ejecuciones
    válidas, y comparación del error high/low confidence (Mann-Whitney U).
    """
    resultado: dict = {"n": 0, "por_sistema": {}}
    if df.empty:
        return resultado
    agente = df[df["sistema"].str.startswith("agente_") & df["valido"]].copy()
    if agente.empty:
        return resultado
    agente["confidence"] = agente["extra"].apply(
        lambda e: (e or {}).get("confidence") if isinstance(e, dict) else None
    )

    calib_total = agente.dropna(subset=["dispersion_norm", "err_norm_run"])
    resultado["n"] = len(calib_total)
    resultado["_puntos"] = calib_total[
        ["sistema", "prompt_id", "dispersion_norm", "err_norm_run"]
    ]

    for sistema, grupo in agente.groupby("sistema"):
        c: dict = {}
        calib = grupo.dropna(subset=["dispersion_norm", "err_norm_run"])
        c["n"] = len(calib)
        if len(calib) >= 3:
            rho, p = stats.spearmanr(calib["dispersion_norm"], calib["err_norm_run"])
            c["spearman_rho"] = float(rho)
            c["spearman_p"] = float(p)
        grupos = {}
        for etiqueta in ("high", "low"):
            sel = grupo[grupo["confidence"] == etiqueta]["err_norm_run"].dropna()
            grupos[etiqueta] = sel
            c[f"n_{etiqueta}"] = len(sel)
            c[f"mediana_err_{etiqueta}"] = (
                float(sel.median()) if len(sel) else None
            )
        if len(grupos["high"]) >= 2 and len(grupos["low"]) >= 2:
            try:
                u, p = stats.mannwhitneyu(grupos["high"], grupos["low"],
                                          alternative="two-sided")
                c["mannwhitney_p"] = float(p)
            except ValueError:
                pass
        resultado["por_sistema"][sistema] = c
    return resultado


def frecuencia_augment(df: pd.DataFrame) -> pd.DataFrame:
    """Frecuencia de augment=True por prompt en las ejecuciones del agente."""
    if df.empty:
        return pd.DataFrame()
    agente = df[df["sistema"].str.startswith("agente_")].copy()
    if agente.empty:
        return pd.DataFrame()

    def _augment(extra):
        if not isinstance(extra, dict):
            return None
        rj = extra.get("result_json") or {}
        return rj.get("augment")

    agente["augment"] = agente["extra"].apply(_augment)
    con_dato = agente.dropna(subset=["augment"])
    if con_dato.empty:
        return pd.DataFrame()
    tabla = con_dato.groupby(["sistema", "prompt_id"]).agg(
        n=("augment", "size"),
        frec_augment=("augment", lambda s: float(pd.Series(s).astype(bool).mean())),
    ).reset_index()
    return tabla


# -------------------------------- figuras ----------------------------------
def generar_figuras(resumen: pd.DataFrame, calibracion: dict, dir_figuras: Path) -> list[Path]:
    """Figuras matplotlib con estilo por defecto (sin colores custom)."""
    dir_figuras.mkdir(parents=True, exist_ok=True)
    rutas: list[Path] = []
    if resumen.empty:
        return rutas

    # 1) Barras: error normalizado medio por sistema.
    agregada = resumen.groupby("sistema")["err_norm"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(agregada.index, agregada.values)
    ax.set_ylabel("Error normalizado medio (MAE/rango)")
    ax.set_title("Error normalizado por sistema (media sobre los 10 problemas)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    ruta = dir_figuras / "mae_normalizado_por_sistema.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    rutas.append(ruta)

    # 2) Heatmap sistema x prompt del error normalizado.
    pivote = resumen.pivot_table(index="sistema", columns="prompt_id", values="err_norm")
    fig, ax = plt.subplots(figsize=(1.2 * max(6, len(pivote.columns)), 0.6 * max(4, len(pivote)) + 2))
    im = ax.imshow(pivote.values, aspect="auto")
    ax.set_xticks(range(len(pivote.columns)),
                  [c.replace("prompt_", "") for c in pivote.columns],
                  rotation=45, ha="right")
    ax.set_yticks(range(len(pivote.index)), pivote.index)
    fig.colorbar(im, ax=ax, label="Error normalizado")
    ax.set_title("Error normalizado por sistema y problema")
    fig.tight_layout()
    ruta = dir_figuras / "heatmap_sistema_prompt.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    rutas.append(ruta)

    # 3) Dispersión de calibración del agente.
    puntos = calibracion.get("_puntos")
    if puntos is not None and len(puntos) >= 3:
        fig, ax = plt.subplots(figsize=(6, 5))
        for sistema, grupo in puntos.groupby("sistema"):
            etiqueta = sistema
            rho = calibracion.get("por_sistema", {}).get(sistema, {}).get("spearman_rho")
            if rho is not None:
                etiqueta += f" (rho={rho:.3f})"
            ax.scatter(grupo["dispersion_norm"], grupo["err_norm_run"], label=etiqueta)
        ax.set_xlabel("Dispersión multi-semilla normalizada (pred_std/rango)")
        ax.set_ylabel("Error normalizado real")
        ax.set_title("Calibración del agente (por configuración)")
        ax.legend()
        fig.tight_layout()
        ruta = dir_figuras / "calibracion_dispersion.png"
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
        rutas.append(ruta)
    return rutas


# ------------------------------ informe md ---------------------------------
def analizar(
    cfg: Config,
    ruta_runs: Path | None = None,
    dir_salida: Path | None = None,
) -> Path:
    """Comando analyze: genera summary.csv, analysis.md y las figuras."""
    ruta_runs = Path(ruta_runs) if ruta_runs else cfg.ruta_runs
    dir_salida = Path(dir_salida) if dir_salida else cfg.dir_resultados
    dir_salida.mkdir(parents=True, exist_ok=True)

    df = cargar_runs(ruta_runs)
    principal = df[df["perturbacion"].isna()] if not df.empty else df
    perturbadas = df[df["perturbacion"].notna()] if not df.empty else pd.DataFrame()

    # Nombres de las salidas (del ground truth) para el desglose por dimensión.
    nombres_salida: dict[str, list[str]] = {}
    if cfg.ruta_soluciones.exists():
        from .soluciones import parsear_soluciones
        nombres_salida = {
            pid: b.get("salidas", []) for pid, b in parsear_soluciones(cfg.ruta_soluciones).items()
        }

    resumen = tabla_sistema_prompt(principal)
    agregada = tabla_agregada(resumen)
    por_salida = tabla_por_salida(principal, nombres_salida)
    wilcoxon = wilcoxon_pares(resumen)
    tabla_fallos, pares_fallos = comparacion_fallos(principal)
    calibracion = calibracion_agente(principal)
    augment = frecuencia_augment(principal)
    figuras = generar_figuras(resumen, calibracion, dir_salida / "figures")

    if not resumen.empty:
        resumen.to_csv(dir_salida / "summary.csv", index=False, encoding="utf-8")

    md: list[str] = [
        "# Análisis de la evaluación (Capítulo 5)",
        "",
        f"Fuente: `{ruta_runs.name}` — {len(df)} ejecuciones "
        f"({len(principal)} principales, {len(perturbadas)} perturbadas).",
        "",
        "## Resumen agregado por sistema",
        "",
        df_a_markdown(agregada),
        "",
        "## Resumen por sistema y problema",
        "",
        df_a_markdown(resumen),
        "",
        "## Desglose del error por salida (problemas multi-salida)",
        "",
        "El MAE por problema promedia todas las salidas; aquí se abre por",
        "dimensión para ver si un sistema acierta unas salidas y falla otras.",
        "",
        df_a_markdown(por_salida) if not por_salida.empty
        else "_(no hay problemas multi-salida con ejecuciones válidas)_",
        "",
        "## Contrastes pareados (Wilcoxon signed-rank sobre MAE por problema)",
        "",
        f"p exacto (salvo nota). Corrección de Holm, alfa={ALFA}. "
        "Tamaño de efecto r = |z|/sqrt(n), con z de la aproximación normal.",
        "mediana_dif = mediana de (MAE_A - MAE_B): negativa favorece al sistema A.",
        "",
        df_a_markdown(wilcoxon),
        "",
        "## Fiabilidad estructural (proporción de fallos)",
        "",
        "Fallo estructural: respuesta vacía, menos números finitos extraíbles que",
        "salidas tiene el problema, valores no finitos, timeout o excepción.",
        "",
        df_a_markdown(tabla_fallos),
        "",
        "### Comparación de proporciones (Fisher exacto, Holm)",
        "",
        df_a_markdown(pares_fallos),
        "",
        "## Calibración del agente propio",
        "",
    ]
    if calibracion.get("n", 0) == 0:
        md.append("_(sin ejecuciones del agente con señal de dispersión)_")
    else:
        md.append("Desglosada por configuración del agente (template y libre "
                  "generan el código de forma distinta; no se mezclan).")
        for sistema in sorted(calibracion.get("por_sistema", {})):
            c = calibracion["por_sistema"][sistema]
            md += ["", f"### {sistema}", ""]
            md.append(f"- Ejecuciones válidas con dispersión: {c.get('n', 0)}")
            if "spearman_rho" in c:
                md.append(
                    f"- Spearman dispersión-error: rho={c['spearman_rho']:.4f}, "
                    f"p={c['spearman_p']:.4g}"
                )
            md.append(
                f"- Error normalizado mediano con confidence=high: "
                f"{c.get('mediana_err_high')} (n={c.get('n_high', 0)})"
            )
            md.append(
                f"- Error normalizado mediano con confidence=low: "
                f"{c.get('mediana_err_low')} (n={c.get('n_low', 0)})"
            )
            if "mannwhitney_p" in c:
                md.append(f"- Mann-Whitney U high vs low: p={c['mannwhitney_p']:.4g}")

    md += ["", "## Frecuencia de selección del aumento polinómico (agente)", ""]
    md.append(df_a_markdown(augment) if not augment.empty
              else "_(sin datos de augment en las ejecuciones del agente)_")

    if not perturbadas.empty:
        resumen_pert = tabla_sistema_prompt(perturbadas)
        md += [
            "",
            "## Robustez ante perturbaciones (P1-P6)",
            "",
            "Ejecutadas tras la evaluación principal, con el agente congelado.",
            "El error se normaliza con los rangos del prompt base.",
            "",
            df_a_markdown(resumen_pert),
        ]

    if figuras:
        md += ["", "## Figuras", ""]
        md += [f"- `figures/{f.name}`" for f in figuras]

    ruta_md = dir_salida / "analysis.md"
    ruta_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[analyze] informe en {ruta_md}")
    for f in figuras:
        print(f"[analyze] figura {f}")
    return ruta_md
