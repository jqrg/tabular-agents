# -*- coding: utf-8 -*-
"""Tests de persistencia y reanudación de la parrilla."""
from harness.ejecutor import construir_parrilla
from harness.persistencia import anexar_jsonl, claves_completadas, leer_jsonl
from harness.prompts import Prompt


def _prompts_falsos(n=2):
    return [Prompt(prompt_id=f"prompt_{i:02d}_falso", texto="") for i in range(1, n + 1)]


def test_parrilla_completa_sin_hechas():
    tareas, saltadas = construir_parrilla(
        ["sisA", "sisB"], _prompts_falsos(2), n_reps=3, hechas=set()
    )
    assert len(tareas) == 2 * 2 * 3
    assert saltadas == 0


def test_reanudacion_salta_lo_hecho():
    hechas = {
        ("sisA", "prompt_01_falso", 1),
        ("sisA", "prompt_01_falso", 2),
        ("sisB", "prompt_02_falso", 3),
    }
    tareas, saltadas = construir_parrilla(
        ["sisA", "sisB"], _prompts_falsos(2), n_reps=3, hechas=hechas
    )
    assert saltadas == 3
    assert len(tareas) == 12 - 3
    claves = {(t.sistema, t.prompt_id, t.rep) for t in tareas}
    assert not claves & hechas


def test_perturbacion_cambia_la_clave():
    """Una ejecución base hecha NO bloquea la misma celda perturbada."""
    hechas = {("sisA", "prompt_01_falso", 1)}
    tareas, saltadas = construir_parrilla(
        ["sisA"], _prompts_falsos(1), n_reps=1, hechas=hechas, perturbacion="P2"
    )
    assert saltadas == 0
    assert len(tareas) == 1
    assert tareas[0].prompt_id == "prompt_01_falso__P2"


def test_fallo_de_conexion_no_ocupa_la_celda():
    """Un fallo de conexión (VPN caída, servidor apagado) es del entorno, no
    del sistema: la celda debe reintentarse al reanudar. Timeout y
    agente_failed sí ocupan la celda (son comportamiento del sistema)."""
    registros = [
        {"sistema": "s", "prompt_id": "p", "rep": 1,
         "extra": {"error": "conexion"}},
        {"sistema": "s", "prompt_id": "p", "rep": 2,
         "extra": {"error": "timeout"}},
        {"sistema": "s", "prompt_id": "p", "rep": 3,
         "extra": {"error": "agente_failed"}},
        {"sistema": "s", "prompt_id": "p", "rep": 4, "extra": {}},
    ]
    claves = claves_completadas(registros)
    assert ("s", "p", 1) not in claves
    assert claves == {("s", "p", 2), ("s", "p", 3), ("s", "p", 4)}


def test_jsonl_ida_y_vuelta_tolerante(tmp_path):
    ruta = tmp_path / "runs.jsonl"
    anexar_jsonl(ruta, {"sistema": "s", "prompt_id": "p", "rep": 1, "valido": True})
    anexar_jsonl(ruta, {"sistema": "s", "prompt_id": "p", "rep": 2, "valido": False})
    # línea corrupta en medio: se ignora sin romper
    with open(ruta, "a", encoding="utf-8") as f:
        f.write("{esto no es json\n")
    anexar_jsonl(ruta, {"sistema": "s", "prompt_id": "q", "rep": 1})

    registros = leer_jsonl(ruta)
    assert len(registros) == 3
    assert claves_completadas(registros) == {("s", "p", 1), ("s", "p", 2), ("s", "q", 1)}
