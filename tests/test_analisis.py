# -*- coding: utf-8 -*-
"""Tests del análisis sobre el runs.jsonl sintético (fixture, sin red)."""
import pandas as pd
import pytest

from harness.analisis import (
    analizar,
    cargar_runs,
    correccion_holm,
    df_a_markdown,
    tabla_por_salida,
    tabla_sistema_prompt,
    wilcoxon_pares,
)


def test_correccion_holm_valores_conocidos():
    # p ordenados: 0.01 (x3), 0.03 (x2), 0.04 (x1) -> 0.03, 0.06, 0.06
    assert correccion_holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_monotono_y_acotado():
    ajustados = correccion_holm([0.9, 0.8, 0.001])
    assert ajustados[2] == pytest.approx(0.003)
    assert all(p <= 1.0 for p in ajustados)


def test_df_a_markdown():
    df = pd.DataFrame({"a": [1.23456, None], "b": ["x", "y"]})
    md = df_a_markdown(df)
    assert md.startswith("| a | b |")
    assert "| - | y |" in md


def test_carga_fixture(ruta_fixture_runs):
    df = cargar_runs(ruta_fixture_runs)
    assert len(df) == 120
    assert set(df["sistema"]) == {
        "agente_template", "agente_libre", "ollama_qwen3.5:4b", "manual_chatgpt",
    }
    assert df["prompt_id"].nunique() == 10


def test_por_salida_desglosa_multisalida(ruta_fixture_runs):
    df = cargar_runs(ruta_fixture_runs)
    nombres = {"prompt_09_lorenz": ["X siguiente", "Y siguiente", "Z siguiente"]}
    tabla = tabla_por_salida(df, nombres)
    assert not tabla.empty
    # solo problemas multi-salida (nunca los de 1 salida como gas ideal)
    assert "prompt_02_gas_ideal" not in set(tabla["prompt_id"])
    lorenz = tabla[tabla["prompt_id"] == "prompt_09_lorenz"]
    assert set(lorenz["salida"]) == {"X siguiente", "Y siguiente", "Z siguiente"}
    assert (tabla["mae_salida"] >= 0).all()


def test_por_salida_sin_nombres_usa_generico(ruta_fixture_runs):
    df = cargar_runs(ruta_fixture_runs)
    tabla = tabla_por_salida(df, {})  # sin ground truth de nombres
    assert not tabla.empty
    assert tabla["salida"].str.startswith("salida_").any()


def test_wilcoxon_un_solo_sistema_no_peta(ruta_fixture_runs):
    """Regresión: con <2 sistemas wilcoxon_pares devuelve vacío, no KeyError."""
    df = cargar_runs(ruta_fixture_runs)
    uno = df[df["sistema"] == "agente_template"]
    resumen = tabla_sistema_prompt(uno)
    assert wilcoxon_pares(resumen).empty


def test_wilcoxon_detecta_diferencia_grande(ruta_fixture_runs):
    df = cargar_runs(ruta_fixture_runs)
    resumen = tabla_sistema_prompt(df)
    pares = wilcoxon_pares(resumen)
    fila = pares[
        (pares["sistema_A"] == "agente_template")
        & (pares["sistema_B"] == "ollama_qwen3.5:4b")
    ].iloc[0]
    assert fila["n_prompts"] == 10
    assert fila["p"] is not None and fila["p"] < 0.05
    assert fila["mediana_dif"] < 0, "el agente tiene menor MAE que el LLM directo"
    assert 0 <= fila["r"] <= 1.05


def test_calibracion_desglosada_por_sistema(ruta_fixture_runs):
    """template y libre no se mezclan en una sola correlación."""
    from harness.analisis import calibracion_agente
    df = cargar_runs(ruta_fixture_runs)
    calib = calibracion_agente(df)
    assert set(calib["por_sistema"]) == {"agente_template", "agente_libre"}
    for c in calib["por_sistema"].values():
        assert c["n"] >= 3
        assert "spearman_rho" in c
    # en el fixture solo agente_libre tiene runs con confidence=low
    assert calib["por_sistema"]["agente_libre"]["n_low"] == 5
    assert calib["por_sistema"]["agente_template"]["n_low"] == 0


def test_analizar_genera_informe_y_figuras(cfg_tmp, ruta_fixture_runs, tmp_path):
    salida = tmp_path / "salida_analisis"
    ruta_md = analizar(cfg_tmp, ruta_runs=ruta_fixture_runs, dir_salida=salida)

    contenido = ruta_md.read_text(encoding="utf-8")
    assert "Wilcoxon" in contenido
    assert "Calibración" in contenido
    assert "Spearman" in contenido
    assert "fallos" in contenido.lower()
    assert (salida / "summary.csv").exists()
    figuras = list((salida / "figures").glob("*.png"))
    assert len(figuras) >= 3  # barras + heatmap + calibración


def test_analizar_incluye_seccion_robustez_si_hay_perturbadas(
        cfg_tmp, ruta_fixture_runs, tmp_path):
    """Con solo ejecuciones principales no aparece la sección de robustez."""
    salida = tmp_path / "sin_perturbadas"
    ruta_md = analizar(cfg_tmp, ruta_runs=ruta_fixture_runs, dir_salida=salida)
    assert "Robustez ante perturbaciones" not in ruta_md.read_text(encoding="utf-8")
