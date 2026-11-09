# -*- coding: utf-8 -*-
"""Tests del parser de soluciones.txt (contra el fichero real del repositorio)."""
from harness.soluciones import parsear_soluciones


def test_diez_bloques(raiz):
    sol = parsear_soluciones(raiz / "prompts" / "soluciones.txt")
    assert len(sol) == 10
    assert all(pid.startswith("prompt_") for pid in sol)


def test_valores_prompt_01(raiz):
    sol = parsear_soluciones(raiz / "prompts" / "soluciones.txt")
    bloque = sol["prompt_01_triangulacion"]
    assert bloque["salidas"] == ["Posicion X", "Posicion Y"]
    assert bloque["valores"] == [468.805, 413.998]


def test_multisalida_lorenz(raiz):
    sol = parsear_soluciones(raiz / "prompts" / "soluciones.txt")
    assert sol["prompt_09_lorenz"]["valores"] == [1.91375, 3.29606, 18.0857]


def test_query_y_nota(raiz):
    sol = parsear_soluciones(raiz / "prompts" / "soluciones.txt")
    assert sol["prompt_02_gas_ideal"]["query"]["Presion"] == 374.972
    for bloque in sol.values():
        assert bloque["valores"], "todo bloque debe tener ground truth"
        assert bloque["nota"], "todo bloque debe tener nota metodológica"
