# -*- coding: utf-8 -*-
"""Parser del ground truth (prompts/soluciones.txt).

Formato de cada bloque:

    prompt_01_triangulacion.txt
      Query   : Distance A=482.928, ...
      Correcto: Posicion X=468.805, Posicion Y=413.998
      Nota    : ...

Este fichero solo lo consume el harness para calcular errores; jamás se
envía a ningún sistema evaluado.
"""
from __future__ import annotations

import re
from pathlib import Path

from .prompts import normalizar_prompt_id

RE_CABECERA_BLOQUE = re.compile(r"^(prompt_\w+)\.txt\s*$")
RE_CAMPO = re.compile(r"^\s*(Query|Correcto|Nota)\s*:\s*(.*)$")


def _parsear_pares(texto: str) -> dict[str, float]:
    """Parsea 'Nombre A=1.2, Nombre B=3.4' -> {'Nombre A': 1.2, ...}."""
    pares: dict[str, float] = {}
    for trozo in texto.split(","):
        if "=" not in trozo:
            continue
        nombre, valor = trozo.split("=", 1)
        try:
            pares[nombre.strip()] = float(valor.strip())
        except ValueError:
            continue
    return pares


def parsear_soluciones(ruta: str | Path) -> dict[str, dict]:
    """Devuelve {prompt_id: {salidas, valores, query, nota}}.

    - salidas: nombres de las variables objetivo, en orden.
    - valores: lista de floats correctos (mismo orden), el ground truth.
    - query:   dict nombre -> valor de las entradas de la consulta.
    - nota:    texto libre de la nota metodológica.
    """
    with open(ruta, encoding="utf-8") as f:
        lineas = f.read().split("\n")

    soluciones: dict[str, dict] = {}
    actual: str | None = None
    for linea in lineas:
        m = RE_CABECERA_BLOQUE.match(linea.strip())
        if m:
            actual = normalizar_prompt_id(m.group(1))
            soluciones[actual] = {"salidas": [], "valores": [], "query": {}, "nota": ""}
            continue
        if actual is None:
            continue
        mc = RE_CAMPO.match(linea)
        if not mc:
            continue
        campo, contenido = mc.group(1), mc.group(2).strip()
        if campo == "Correcto":
            pares = _parsear_pares(contenido)
            soluciones[actual]["salidas"] = list(pares.keys())
            soluciones[actual]["valores"] = list(pares.values())
        elif campo == "Query":
            soluciones[actual]["query"] = _parsear_pares(contenido)
        elif campo == "Nota":
            soluciones[actual]["nota"] = contenido
    return soluciones
