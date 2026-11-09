# -*- coding: utf-8 -*-
"""Validación y extracción de predicciones: pipeline único para todos los
sistemas (regla del Capítulo 4). Una respuesta es válida si contiene al
menos n_outputs números finitos extraíbles; la predicción son los últimos
n_outputs. Para el agente, si existe result_json.pred_mean se usa con la
misma regla; si no basta, se cae al texto crudo por la vía común.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Números en notación decimal o científica con punto decimal (formato del
# dataset y de las respuestas de los sistemas; regla declarada).
RE_NUMERO = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def extraer_numeros(texto: str) -> list[float]:
    """Extrae todos los números finitos de un texto, en orden de aparición."""
    numeros: list[float] = []
    for m in RE_NUMERO.finditer(texto or ""):
        try:
            v = float(m.group())
        except ValueError:
            continue
        if math.isfinite(v):
            numeros.append(v)
    return numeros


@dataclass
class ResultadoValidacion:
    """Resultado del pipeline de validación/extracción."""

    valido: bool
    pred: list[float] | None = None
    motivo: str = ""       # motivo del fallo estructural si valido=False
    fuente: str = ""       # "pred_directa" | "texto"
    numeros_extraidos: int = 0


def _finitos(valores) -> list[float]:
    out: list[float] = []
    for v in valores or []:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def validar_y_extraer(
    raw_text: str,
    n_outputs: int,
    pred_directa: list | None = None,
) -> ResultadoValidacion:
    """Aplica la regla de validación/extracción a una respuesta cruda.

    pred_directa: predicción estructurada del propio sistema (p.ej.
    pred_mean del agente); tiene prioridad si aporta suficientes números.
    """
    if n_outputs <= 0:
        return ResultadoValidacion(False, motivo="n_outputs_invalido")

    if pred_directa is not None:
        finitos = _finitos(pred_directa)
        if len(finitos) >= n_outputs:
            return ResultadoValidacion(
                True,
                pred=finitos[-n_outputs:],
                fuente="pred_directa",
                numeros_extraidos=len(finitos),
            )
        # pred_directa insuficiente: cae a la vía común sobre el texto.

    if not (raw_text or "").strip():
        return ResultadoValidacion(False, motivo="respuesta_vacia")

    numeros = extraer_numeros(raw_text)
    if len(numeros) >= n_outputs:
        return ResultadoValidacion(
            True,
            pred=numeros[-n_outputs:],
            fuente="texto",
            numeros_extraidos=len(numeros),
        )
    return ResultadoValidacion(
        False,
        motivo="numeros_insuficientes",
        numeros_extraidos=len(numeros),
    )
