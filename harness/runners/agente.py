# -*- coding: utf-8 -*-
"""Runner del agente propio (Agente/agente.py).

Cada ejecución se lanza como subproceso en un directorio temporal propio,
para aislar los artefactos que el agente escribe en su cwd (train.csv,
log.jsonl, test_*.py). El resultado canónico es la última línea de ese
log.jsonl: de ahí salen pred_mean, confidence y el resto de metadatos.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .base import RawResult, Runner, truncar_raw

# Marcadores que imprime el main() del agente alrededor de su respuesta: el
# resto del stdout es telemetría cuyos números confundirían a la regla de
# "últimos n" cuando falta pred_mean.
MARCA_RESULTADO = "########### Resultado final ###########"
MARCA_METRICAS = "########### Métricas de la ejecución ###########"


def _seccion_resultado_final(stdout: str) -> str | None:
    """Recorta la sección 'Resultado final' del stdout del agente, o None."""
    ini = stdout.rfind(MARCA_RESULTADO)
    if ini < 0:
        return None
    fin = stdout.find(MARCA_METRICAS, ini)
    seccion = stdout[ini + len(MARCA_RESULTADO):fin if fin >= 0 else len(stdout)]
    return seccion.strip()


class AgentRunner(Runner):
    """Ejecuta el agente propio con GENERATION_MODE=template|libre."""

    def __init__(
        self,
        nombre: str,
        ruta_agente: Path,
        mode: str,
        model: str,
        ollama_url: str,
        timeout_s: int,
    ):
        self.nombre = nombre
        self.ruta_agente = Path(ruta_agente)
        self.mode = mode
        self.model = model
        self.ollama_url = ollama_url
        self.timeout_s = timeout_s

    def run(self, prompt_text: str, prompt_path: Path | None, meta: dict) -> RawResult:
        t0 = time.monotonic()
        timeout = meta.get("timeout_s", self.timeout_s)
        env = {
            **os.environ,
            "GENERATION_MODE": self.mode,
            "MODEL_NAME": self.model,
            "OLLAMA_URL": self.ollama_url,
            "PYTHONIOENCODING": "utf-8",
            # El workflow del agente debe agotarse antes del kill del harness,
            # para que dé tiempo a escribir log.jsonl.
            "AGENT_TIMEOUT": str(max(60, int(timeout) - 60)),
        }
        with tempfile.TemporaryDirectory(prefix="harness_agente_") as tmp:
            # El agente lee el prompt de un fichero: si viene de memoria
            # (perturbados aún sin guardar), se materializa en el tmp.
            if prompt_path is None:
                prompt_path = Path(tmp) / f"{meta.get('prompt_id', 'prompt')}.txt"
                prompt_path.write_text(prompt_text, encoding="utf-8")
            cmd = [
                sys.executable,
                str(self.ruta_agente),
                str(Path(prompt_path).resolve()),
                str(meta.get("prompt_id", "sin_id")),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                )
                stdout = (proc.stdout or "") + (
                    ("\n[stderr]\n" + proc.stderr) if proc.stderr else ""
                )
            except subprocess.TimeoutExpired as e:
                raw, rec = truncar_raw(
                    (e.stdout or b"").decode("utf-8", "replace")
                    if isinstance(e.stdout, bytes)
                    else (e.stdout or "")
                )
                return RawResult(
                    raw_text=raw,
                    ok_exec=False,
                    seconds=time.monotonic() - t0,
                    extra={"error": "timeout", "raw_recortado": rec, "mode": self.mode},
                )
            except OSError as e:
                return RawResult(
                    ok_exec=False,
                    seconds=time.monotonic() - t0,
                    extra={"error": f"excepcion:{type(e).__name__}", "mode": self.mode},
                )

            info = self._leer_ultima_linea_log(Path(tmp) / "log.jsonl")

        segundos = time.monotonic() - t0
        if info is None:
            raw, recortado = truncar_raw(stdout)
            return RawResult(
                raw_text=raw,
                ok_exec=False,
                seconds=segundos,
                extra={
                    "error": "sin_log_jsonl",
                    "returncode": proc.returncode,
                    "raw_recortado": recortado,
                    "mode": self.mode,
                },
            )

        # La respuesta del sistema es la sección "Resultado final".
        seccion = _seccion_resultado_final(stdout)
        raw, recortado = truncar_raw(seccion if seccion is not None else stdout)

        result_json = info.get("result_json") or {}
        pred = result_json.get("pred_mean")
        extra = {
            "mode": self.mode,
            "model": self.model,
            "confidence": info.get("confidence"),
            "semantic_issues": info.get("semantic_issues"),
            "result_json": result_json,
            "duration_agente_s": info.get("duration_s"),
            "paso7_executions": info.get("paso7_executions"),
            "backtracks": info.get("backtracks"),
            "info_agente": info,
            "returncode": proc.returncode,
            "raw_recortado": recortado,
            "raw_seccion_resultado": seccion is not None,
        }
        # Fallo declarado por el propio agente: su mensaje de error contiene
        # números que la regla de validación tomaría por una predicción.
        if info.get("confidence") == "failed" and not pred:
            extra["error"] = "agente_failed"
            return RawResult(
                raw_text=raw,
                ok_exec=False,
                seconds=segundos,
                extra=extra,
            )
        return RawResult(
            raw_text=raw,
            pred=pred if isinstance(pred, list) else None,
            ok_exec=True,
            seconds=segundos,
            extra=extra,
        )

    @staticmethod
    def _leer_ultima_linea_log(ruta: Path) -> dict | None:
        """Última línea JSON válida del log.jsonl del agente, o None."""
        if not ruta.exists():
            return None
        ultima = None
        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    ultima = json.loads(linea)
                except json.JSONDecodeError:
                    continue
        return ultima
