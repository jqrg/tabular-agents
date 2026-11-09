# -*- coding: utf-8 -*-
"""Parser de los prompts de evaluación.

Formato de cada prompt (ver prompts/generar_dataset.py):

    Quiero predecir la <SALIDA>[ y la <SALIDA2>...] para
        <Nombre> = <valor>
        ...
        Basándome en estos datos previos:
    <CSV con cabecera y ~100 filas>

Convenciones del estudio:
  - n_inputs  = nº de líneas "nombre = valor" del bloque de pregunta.
  - n_outputs = nº de columnas del CSV - n_inputs (para el dataset base
    coincide con el nº de valores del ground truth, que es el canónico).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# Línea "nombre = valor" del bloque de pregunta (mismo criterio que el agente).
RE_LINEA_VALOR = re.compile(
    r"^\s*([^=\n]+?)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
# Sufijo de copia de Windows en nombres de fichero: "prompt_01_x (1).txt".
RE_SUFIJO_COPIA = re.compile(r"\s*\(\d+\)$")
RE_BASANDOME = re.compile(r"^\s*Bas[áa]ndome", re.IGNORECASE)


def normalizar_prompt_id(nombre: str | Path) -> str:
    """Devuelve el id canónico: nombre sin extensión ni sufijo ' (n)'.

    'prompt_01_triangulacion (1).txt' -> 'prompt_01_triangulacion'
    """
    stem = Path(nombre).stem if str(nombre).endswith(".txt") else str(nombre)
    return RE_SUFIJO_COPIA.sub("", stem).strip()


@dataclass
class Prompt:
    """Prompt de evaluación parseado."""

    prompt_id: str
    texto: str
    ruta: Path | None = None
    pregunta: str = ""
    nombres_entrada: list[str] = field(default_factory=list)
    valores_query: list[float] = field(default_factory=list)
    cabecera: list[str] = field(default_factory=list)
    filas: list[list[str]] = field(default_factory=list)

    @property
    def n_inputs(self) -> int:
        return len(self.valores_query)

    @property
    def n_cols(self) -> int:
        return len(self.cabecera)

    @property
    def n_outputs_csv(self) -> int:
        """Salidas inferidas del propio prompt (columnas CSV - entradas)."""
        return max(0, self.n_cols - self.n_inputs)

    def rangos_salida(self, n_outputs: int | None = None) -> list[float]:
        """Rango (max-min) de cada columna de salida, calculado del CSV.

        Ignora valores no numéricos o no finitos (p.ej. el 'N/A' de P6);
        rango mínimo 1e-9 para evitar divisiones por cero.
        """
        n_out = n_outputs if n_outputs is not None else self.n_outputs_csv
        rangos: list[float] = []
        for j in range(self.n_inputs, self.n_inputs + n_out):
            vals: list[float] = []
            for fila in self.filas:
                if j >= len(fila):
                    continue
                try:
                    v = float(fila[j])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    vals.append(v)
            rangos.append(max(max(vals) - min(vals), 1e-9) if vals else 1e-9)
        return rangos


def parsear_texto(texto: str, prompt_id: str = "", ruta: Path | None = None) -> Prompt:
    """Parsea el texto completo de un prompt en sus partes."""
    lineas = texto.split("\n")

    idx_bas = next((i for i, l in enumerate(lineas) if RE_BASANDOME.match(l)), None)
    if idx_bas is not None:
        cabeza, resto = lineas[:idx_bas], lineas[idx_bas + 1:]
    else:
        # Tolerancia: sin línea "Basándome", el CSV empieza en la primera
        # línea con al menos dos comas.
        idx_csv = next(
            (i for i, l in enumerate(lineas) if l.count(",") >= 2), len(lineas)
        )
        cabeza, resto = lineas[:idx_csv], lineas[idx_csv:]

    pregunta = cabeza[0].strip() if cabeza else ""
    nombres, valores = [], []
    for linea in cabeza:
        m = RE_LINEA_VALOR.match(linea)
        if m:
            nombres.append(m.group(1).strip())
            valores.append(float(m.group(2)))

    # Bloque CSV: se ignoran líneas en blanco (tolerante a la perturbación P3)
    # y se recortan espacios alrededor de cada campo.
    csv_lineas = [l for l in resto if l.strip()]
    cabecera: list[str] = []
    filas: list[list[str]] = []
    for l in csv_lineas:
        if "," not in l:
            continue
        campos = [c.strip() for c in l.split(",")]
        if not cabecera:
            cabecera = campos
        else:
            filas.append(campos)

    return Prompt(
        prompt_id=prompt_id or "sin_id",
        texto=texto,
        ruta=ruta,
        pregunta=pregunta,
        nombres_entrada=nombres,
        valores_query=valores,
        cabecera=cabecera,
        filas=filas,
    )


def cargar_prompt(ruta: str | Path) -> Prompt:
    """Carga y parsea un fichero de prompt."""
    ruta = Path(ruta)
    with open(ruta, encoding="utf-8") as f:
        texto = f.read()
    return parsear_texto(texto, prompt_id=normalizar_prompt_id(ruta.name), ruta=ruta)


def cargar_prompts(directorio: str | Path) -> list[Prompt]:
    """Carga todos los prompt_*.txt de un directorio, ordenados por id."""
    directorio = Path(directorio)
    prompts = [cargar_prompt(p) for p in sorted(directorio.glob("prompt_*.txt"))]
    return sorted(prompts, key=lambda p: p.prompt_id)
