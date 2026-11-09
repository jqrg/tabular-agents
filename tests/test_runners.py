# -*- coding: utf-8 -*-
"""Tests de los runners SIN red: manual, agente con script falso y
manejo de Ollama caído (conexión simulada con monkeypatch)."""
import textwrap

import pytest
import requests

import harness.runners.ollama_directo as mod_ollama
from harness.runners.agente import AgentRunner
from harness.runners.manual import ManualImportRunner
from harness.runners.ollama_directo import OllamaDirectRunner


# ------------------------------ manual --------------------------------------
def _csv_manual(tmp_path):
    ruta = tmp_path / "chatgpt.csv"
    ruta.write_text(
        'prompt_id,rep,respuesta_cruda\n'
        'prompt_01_triangulacion,1,"Analizando...\n'
        'X = 470.2, Y = 415.8"\n'
        'prompt_01_triangulacion,2,"X=469.1 e Y=414.3"\n',
        encoding="utf-8",
    )
    return ruta


def test_manual_respuesta_multilinea(tmp_path):
    runner = ManualImportRunner("manual_chatgpt", _csv_manual(tmp_path))
    res = runner.run("", None, {"prompt_id": "prompt_01_triangulacion", "rep": 1})
    assert res.ok_exec
    assert "470.2" in res.raw_text and "\n" in res.raw_text


def test_manual_respuesta_ausente(tmp_path):
    runner = ManualImportRunner("manual_chatgpt", _csv_manual(tmp_path))
    res = runner.run("", None, {"prompt_id": "prompt_99_inexistente", "rep": 1})
    assert not res.ok_exec
    assert res.extra["error"] == "respuesta_manual_ausente"


def test_manual_columnas_incorrectas(tmp_path):
    ruta = tmp_path / "malo.csv"
    ruta.write_text("id,respuesta\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ManualImportRunner("manual_malo", ruta)


# --------------------- agente (script falso, sin Ollama) --------------------
def test_agente_lee_log_jsonl(tmp_path):
    script = tmp_path / "agente_falso.py"
    script.write_text(textwrap.dedent("""
        # -*- coding: utf-8 -*-
        import json, sys, os
        assert os.environ["GENERATION_MODE"] == "template"
        info = {"prompt_id": sys.argv[2], "confidence": "high",
                "result_json": {"pred_mean": [1.5, 2.5], "pred_std": [0.1, 0.2],
                                 "augment": False}}
        with open("log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(info) + "\\n")
        print("resultado del agente falso")
    """), encoding="utf-8")
    runner = AgentRunner("agente_template", script, mode="template",
                         model="m", ollama_url="http://sin-red", timeout_s=60)
    res = runner.run("texto del prompt", None,
                     {"prompt_id": "prompt_x", "rep": 1, "timeout_s": 60})
    assert res.ok_exec
    assert res.pred == [1.5, 2.5]
    assert res.extra["confidence"] == "high"
    assert "agente falso" in res.raw_text


def test_agente_failed_es_fallo_estructural(tmp_path):
    """Si el agente declara confidence=failed sin pred_mean, el runner lo
    etiqueta como fallo: los números de su mensaje de error y del volcado de
    métricas no deben 'rescatarse' como predicción por la regla últimos-n."""
    script = tmp_path / "agente_fallido.py"
    script.write_text(textwrap.dedent("""
        # -*- coding: utf-8 -*-
        import json
        info = {"confidence": "failed", "result_json": None}
        with open("log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(info) + "\\n")
        print("########### Resultado final ###########")
        print("Fallo tras 3 correcciones (0 backtracks). Ultimo error: shape 100x4")
        print("########### Métricas de la ejecución ###########")
        print(json.dumps({"duration_s": 12.5, "backtracks": 0}))
    """), encoding="utf-8")
    runner = AgentRunner("agente_template", script, mode="template",
                         model="m", ollama_url="http://sin-red", timeout_s=60)
    res = runner.run("texto", None, {"prompt_id": "p", "rep": 1, "timeout_s": 60})
    assert not res.ok_exec
    assert res.extra["error"] == "agente_failed"
    assert res.extra["confidence"] == "failed"


def test_agente_raw_recortado_a_resultado_final(tmp_path):
    """La respuesta registrada es la sección 'Resultado final', sin el volcado
    de métricas posterior (sus números contaminarían la validación)."""
    script = tmp_path / "agente_verboso.py"
    script.write_text(textwrap.dedent("""
        # -*- coding: utf-8 -*-
        import json
        info = {"confidence": "low", "result_json": None}
        with open("log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(info) + "\\n")
        print("[Paso 6] logs de entrenamiento 999.9")
        print("########### Resultado final ###########")
        print("RESULT_JSON no disponible; respuesta textual 1.5 2.5")
        print("########### Métricas de la ejecución ###########")
        print(json.dumps({"duration_s": 77.7, "backtracks": 0}))
    """), encoding="utf-8")
    runner = AgentRunner("agente_template", script, mode="template",
                         model="m", ollama_url="http://sin-red", timeout_s=60)
    res = runner.run("texto", None, {"prompt_id": "p", "rep": 1, "timeout_s": 60})
    assert res.ok_exec
    assert res.extra["raw_seccion_resultado"] is True
    assert "respuesta textual" in res.raw_text
    assert "77.7" not in res.raw_text and "999.9" not in res.raw_text


def test_agente_sin_log_es_fallo_etiquetado(tmp_path):
    script = tmp_path / "agente_mudo.py"
    script.write_text("print('no escribo log')", encoding="utf-8")
    runner = AgentRunner("agente_template", script, mode="template",
                         model="m", ollama_url="http://sin-red", timeout_s=60)
    res = runner.run("texto", None, {"prompt_id": "p", "rep": 1, "timeout_s": 60})
    assert not res.ok_exec
    assert res.extra["error"] == "sin_log_jsonl"


# ------------------- Ollama caído: etiqueta "conexion" ----------------------
def test_ollama_caido_no_revienta(monkeypatch):
    def _post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("simulado")

    monkeypatch.setattr(mod_ollama.requests, "post", _post)
    runner = OllamaDirectRunner("ollama_x", "modelo", "http://localhost:11434", 5)
    res = runner.run("prompt", None, {"prompt_id": "p", "rep": 1, "timeout_s": 5})
    assert not res.ok_exec
    assert res.extra["error"] == "conexion"
