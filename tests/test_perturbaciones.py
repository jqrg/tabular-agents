# -*- coding: utf-8 -*-
"""Tests de las 6 perturbaciones de robustez (deterministas, sobre el
prompt real prompt_02_gas_ideal del repositorio)."""
import random
from collections import Counter

import pytest

from harness.perturbaciones import (
    N_FILAS_DUPLICADAS,
    generar_perturbaciones,
    p1_duplicar_filas,
    p2_barajar_filas,
    p3_ruido_formato,
    p4_columna_ruido,
    p5_reformular_pregunta,
    p6_corromper_fila,
    _partes,
)
from harness.prompts import cargar_prompts, parsear_texto


@pytest.fixture(scope="module")
def texto_base(raiz):
    prompts = cargar_prompts(raiz / "prompts")
    return next(p for p in prompts if p.prompt_id.startswith("prompt_02")).texto


def _rng():
    return random.Random(1234)


def test_p1_duplica_quince_filas(texto_base):
    variante = p1_duplicar_filas(texto_base, _rng())
    _, cab0, filas0 = _partes(texto_base)
    cabeza1, cab1, filas1 = _partes(variante)
    assert cab1 == cab0
    assert len(filas1) == len(filas0) + N_FILAS_DUPLICADAS
    # las filas añadidas son copias exactas de filas originales
    extra = Counter(filas1) - Counter(filas0)
    assert sum(extra.values()) == N_FILAS_DUPLICADAS
    assert all(fila in set(filas0) for fila in extra)


def test_p2_no_cambia_el_multiconjunto(texto_base):
    variante = p2_barajar_filas(texto_base, _rng())
    _, cab0, filas0 = _partes(texto_base)
    _, cab1, filas1 = _partes(variante)
    assert cab1 == cab0
    assert Counter(filas1) == Counter(filas0), "P2 conserva el multiconjunto"
    assert filas1 != filas0, "P2 cambia el orden"


def test_p3_ruido_de_formato_sigue_parseable(texto_base):
    variante = p3_ruido_formato(texto_base, _rng())
    assert ", " in variante
    # una línea en blanco dentro del bloque CSV
    bloque_csv = variante.split("Basándome", 1)[1]
    assert "\n\n" in bloque_csv
    # el parser del harness lo sigue entendiendo igual
    base = parsear_texto(texto_base)
    pert = parsear_texto(variante)
    assert pert.n_cols == base.n_cols
    assert len(pert.filas) == len(base.filas)
    assert pert.filas[0] == base.filas[0]


def test_p4_anade_exactamente_una_columna(texto_base):
    variante = p4_columna_ruido(texto_base, _rng())
    base = parsear_texto(texto_base)
    pert = parsear_texto(variante)
    assert pert.n_cols == base.n_cols + 1
    # "Variable Z" queda ANTES de las salidas (posición n_inputs)
    assert pert.cabecera[base.n_inputs] == "Variable Z"
    # quitando esa columna se recupera el CSV original
    for f_base, f_pert in zip(base.filas, pert.filas):
        sin_z = f_pert[: base.n_inputs] + f_pert[base.n_inputs + 1:]
        assert sin_z == f_base


def test_p5_reformula_manteniendo_datos(texto_base):
    variante = p5_reformular_pregunta(texto_base, _rng())
    primera = variante.split("\n", 1)[0]
    assert primera.startswith("Necesito saber el valor de")
    assert primera.endswith("cuando")
    # mismos datos y mismos valores del query
    base = parsear_texto(texto_base)
    pert = parsear_texto(variante)
    assert pert.valores_query == base.valores_query
    assert pert.cabecera == base.cabecera
    assert pert.filas == base.filas


def test_p6_corrompe_un_unico_valor(texto_base):
    variante = p6_corromper_fila(texto_base, _rng())
    assert variante.count("N/A") == 1
    _, _, filas0 = _partes(texto_base)
    _, _, filas1 = _partes(variante)
    distintas = [i for i, (a, b) in enumerate(zip(filas0, filas1)) if a != b]
    assert len(distintas) == 1
    # el resto de la fila corrupta se conserva
    i = distintas[0]
    campos0, campos1 = filas0[i].split(","), filas1[i].split(",")
    difs = [j for j, (a, b) in enumerate(zip(campos0, campos1)) if a != b]
    assert len(difs) == 1 and campos1[difs[0]] == "N/A"


def test_deterministas(texto_base):
    for fn in (p1_duplicar_filas, p2_barajar_filas, p4_columna_ruido,
               p6_corromper_fila):
        assert fn(texto_base, random.Random(7)) == fn(texto_base, random.Random(7))


def test_generar_las_18_variantes(cfg_tmp):
    escritos = generar_perturbaciones(cfg_tmp)
    assert len(escritos) == 18  # 3 prompts x P1..P6
    for etiqueta in ("P1", "P2", "P3", "P4", "P5", "P6"):
        ficheros = list((cfg_tmp.dir_perturbados / etiqueta).glob("prompt_*.txt"))
        assert len(ficheros) == 3, etiqueta
    # determinismo extremo a extremo: regenerar produce bytes idénticos
    contenido_antes = {p: p.read_text(encoding="utf-8") for p in escritos}
    generar_perturbaciones(cfg_tmp)
    for p, antes in contenido_antes.items():
        assert p.read_text(encoding="utf-8") == antes
