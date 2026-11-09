# -*- coding: utf-8 -*-
"""Runner de LLM local directo vía Ollama (/api/chat).

Protocolo del estudio: el prompt exacto del fichero como único mensaje de
usuario, sin system prompt ni instrucciones añadidas. El modelo responde
"de cabeza"; la respuesta cruda se conserva íntegra.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from .base import RawResult, Runner, truncar_raw


class OllamaDirectRunner(Runner):
    """POST {ollama_url}/api/chat con stream=False."""

    def __init__(self, nombre: str, model: str, ollama_url: str, timeout_s: int):
        self.nombre = nombre
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout_s = timeout_s

    def run(self, prompt_text: str, prompt_path: Path | None, meta: dict) -> RawResult:
        t0 = time.monotonic()
        timeout = meta.get("timeout_s", self.timeout_s)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "stream": False,
        }
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat", json=payload, timeout=timeout
            )
        except requests.exceptions.ConnectionError:
            # Ollama caído: fallo estructural etiquetado, nunca un crash.
            return RawResult(
                ok_exec=False,
                seconds=time.monotonic() - t0,
                extra={"error": "conexion", "model": self.model},
            )
        except requests.exceptions.Timeout:
            return RawResult(
                ok_exec=False,
                seconds=time.monotonic() - t0,
                extra={"error": "timeout", "model": self.model},
            )
        except requests.exceptions.RequestException as e:
            return RawResult(
                ok_exec=False,
                seconds=time.monotonic() - t0,
                extra={"error": f"excepcion:{type(e).__name__}", "model": self.model},
            )

        segundos = time.monotonic() - t0
        if resp.status_code != 200:
            return RawResult(
                raw_text=truncar_raw(resp.text)[0],
                ok_exec=False,
                seconds=segundos,
                extra={"error": f"http_{resp.status_code}", "model": self.model},
            )
        try:
            contenido = resp.json().get("message", {}).get("content", "")
        except ValueError:
            return RawResult(
                raw_text=truncar_raw(resp.text)[0],
                ok_exec=False,
                seconds=segundos,
                extra={"error": "respuesta_no_json", "model": self.model},
            )
        raw, recortado = truncar_raw(contenido)
        return RawResult(
            raw_text=raw,
            ok_exec=True,
            seconds=segundos,
            extra={"model": self.model, "raw_recortado": recortado},
        )
