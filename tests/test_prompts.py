# -*- coding: utf-8 -*-
"""Tests del parser de prompts, sobre los 10 ficheros reales del repositorio."""
import pytest

from harness.prompts import cargar_prompts, normalizar_prompt_id
from harness.soluciones import parsear_soluciones


@pytest.fixture(scope="module")
def prompts(raiz):
    return {p.prompt_id: p for p in cargar_prompts(raiz / "prompts")}


@pytest.fixture(scope="module")
def soluciones(raiz):
    return parsear_soluciones(raiz / "prompts" / "soluciones.txt")


def test_normalizar_prompt_id():
    assert normalizar_prompt_id("prompt_01_triangulacion (1).txt") == "prompt_01_triangulacion"
    assert normalizar_prompt_id("prompt_02_gas_ideal.txt") == "prompt_02_gas_ideal"
    assert normalizar_prompt_id("prompt_09_lorenz") == "prompt_09_lorenz"


def test_carga_los_diez(prompts):
    assert len(prompts) == 10
    assert not any("(" in pid for pid in prompts), "ids sin sufijo ' (n)'"


def test_estructura_coincide_con_ground_truth(prompts, soluciones):
    """n_inputs y n_outputs de cada prompt casan con el ground truth."""
    for pid, gt in soluciones.items():
        p = prompts[pid]
        assert p.n_inputs == len(gt["query"]), pid
        assert p.n_outputs_csv == len(gt["valores"]), pid
        # los valores del query del prompt son los del ground truth
        assert p.valores_query == pytest.approx(list(gt["query"].values())), pid


def test_rangos_salida_positivos(prompts, soluciones):
    for pid, p in prompts.items():
        rangos = p.rangos_salida(len(soluciones[pid]["valores"]))
        assert len(rangos) == len(soluciones[pid]["valores"])
        assert all(r > 0 for r in rangos), pid


def test_csv_cien_filas(prompts):
    for pid, p in prompts.items():
        assert len(p.filas) == 100, pid
        assert all(len(f) == p.n_cols for f in p.filas), pid
