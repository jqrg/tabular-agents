# -*- coding: utf-8 -*-
"""Interfaz común de los runners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# Longitud máxima de la respuesta cruda conservada: se guarda el final (donde
# está la respuesta) y el recorte se marca en extra.
MAX_RAW_TEXT = 20000


@dataclass
class RawResult:
    """Resultado crudo de una ejecución, antes del pipeline de validación.

    - raw_text: respuesta cruda del sistema (siempre se conserva).
    - pred: predicción estructurada si el sistema la aporta (p.ej.
      pred_mean del agente); None si hay que extraerla del texto.
    - ok_exec: False si hubo timeout/excepción/conexión caída (fallo
      estructural etiquetado en extra["error"]).
    - seconds: duración de la ejecución.
    - extra: metadatos específicos del sistema (confidence, result_json...).
    """

    raw_text: str = ""
    pred: list | None = None
    ok_exec: bool = False
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def truncar_raw(texto: str) -> tuple[str, bool]:
    """Trunca la respuesta cruda conservando el final; devuelve (texto, recortado)."""
    texto = texto or ""
    if len(texto) <= MAX_RAW_TEXT:
        return texto, False
    return texto[-MAX_RAW_TEXT:], True


class Runner(ABC):
    """Interfaz común: cada sistema evaluado se envuelve en un Runner."""

    #: nombre del sistema tal y como aparece en runs.jsonl y en el análisis
    nombre: str = "runner"

    @abstractmethod
    def run(self, prompt_text: str, prompt_path: Path | None, meta: dict) -> RawResult:
        """Ejecuta el sistema sobre un prompt.

        meta incluye al menos: prompt_id, rep, n_inputs, n_outputs, timeout_s.
        Nunca debe propagar excepciones: cualquier error se devuelve como
        RawResult(ok_exec=False, extra={"error": <etiqueta>}).
        """
