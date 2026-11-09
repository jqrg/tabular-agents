# -*- coding: utf-8 -*-
"""Perturbaciones de robustez P1-P6 (protocolo del Capítulo 4), aplicadas a
los prompts de `prompts_perturbar` tras la evaluación principal. Todas
deterministas (semilla derivada de seed_perturbaciones + índices fijos).

  P1: duplicar 15 filas aleatorias del CSV (las copias se añaden al final).
  P2: barajar las filas del CSV (semilla fija).
  P3: ruido de formato (espacio tras cada coma + una línea en blanco en
      medio del CSV).
  P4: añadir columna "Variable Z" de ruido uniforme antes de las columnas
      de salida (desplaza las posiciones de las salidas).
  P5: reformular la pregunta manteniendo los mismos datos.
  P6: corromper una fila (un valor -> "N/A").

Los ficheros se escriben en prompts_perturbados/<P>/<id>.txt y se ejecutan
con `python -m harness run --perturbados`. Ojo: el diseño ejecutado se
recortó a solo prompt_02 y prompts_perturbados/ refleja ese recorte; volver
a ejecutar `perturb` regeneraría también las variantes de 04/09.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

from .config import Config
from .prompts import RE_BASANDOME, RE_LINEA_VALOR, cargar_prompts

PERTURBACIONES = ["P1", "P2", "P3", "P4", "P5", "P6"]

N_FILAS_DUPLICADAS = 15          # P1
RANGO_RUIDO_Z = (0.0, 10.0)      # P4: uniforme, formateado a 6 cifras signif.


def _partes(texto: str) -> tuple[list[str], str, list[str]]:
    """Divide el prompt en (bloque_pregunta, línea_cabecera_csv, filas_csv).

    El bloque de pregunta se conserva literalmente, línea a línea, hasta la
    cabecera del CSV (incluida la línea 'Basándome...' y los blancos).
    """
    lineas = texto.split("\n")
    idx_bas = next((i for i, l in enumerate(lineas) if RE_BASANDOME.match(l)), None)
    inicio_busqueda = (idx_bas + 1) if idx_bas is not None else 0
    idx_cab = next(
        (i for i in range(inicio_busqueda, len(lineas)) if lineas[i].count(",") >= 1),
        None,
    )
    if idx_cab is None:
        raise ValueError("el prompt no contiene un bloque CSV reconocible")
    cabeza = lineas[:idx_cab]
    cabecera = lineas[idx_cab]
    filas = [l for l in lineas[idx_cab + 1:] if l.strip()]
    return cabeza, cabecera, filas


def _reconstruir(cabeza: list[str], cabecera: str, filas: list[str]) -> str:
    return "\n".join(cabeza + [cabecera] + filas)


def _n_inputs(cabeza: list[str]) -> int:
    return sum(1 for l in cabeza if RE_LINEA_VALOR.match(l))


def _f6(x: float) -> str:
    """6 cifras significativas, mismo formato que el generador del dataset."""
    return "%.6g" % x


def p1_duplicar_filas(texto: str, rng: random.Random) -> str:
    """P1: duplica N_FILAS_DUPLICADAS filas aleatorias (copias al final)."""
    cabeza, cabecera, filas = _partes(texto)
    n = min(N_FILAS_DUPLICADAS, len(filas))
    duplicadas = [filas[i] for i in sorted(rng.sample(range(len(filas)), n))]
    return _reconstruir(cabeza, cabecera, filas + duplicadas)


def p2_barajar_filas(texto: str, rng: random.Random) -> str:
    """P2: baraja las filas de datos (mismo multiconjunto, otro orden)."""
    cabeza, cabecera, filas = _partes(texto)
    barajadas = list(filas)
    rng.shuffle(barajadas)
    return _reconstruir(cabeza, cabecera, barajadas)


def p3_ruido_formato(texto: str, rng: random.Random) -> str:
    """P3: espacio tras cada coma del CSV + una línea en blanco en medio."""
    cabeza, cabecera, filas = _partes(texto)
    cabecera = cabecera.replace(",", ", ")
    filas = [f.replace(",", ", ") for f in filas]
    filas.insert(len(filas) // 2, "")
    return _reconstruir(cabeza, cabecera, filas)


def p4_columna_ruido(texto: str, rng: random.Random) -> str:
    """P4: inserta 'Variable Z' (ruido uniforme) ANTES de las salidas."""
    cabeza, cabecera, filas = _partes(texto)
    n_in = _n_inputs(cabeza)
    cols = cabecera.split(",")
    cols.insert(n_in, "Variable Z")
    nuevas = []
    for f in filas:
        campos = f.split(",")
        campos.insert(n_in, _f6(rng.uniform(*RANGO_RUIDO_Z)))
        nuevas.append(",".join(campos))
    return _reconstruir(cabeza, ",".join(cols), nuevas)


def p5_reformular_pregunta(texto: str, rng: random.Random) -> str:
    """P5: reformula la primera línea manteniendo datos y valores.

    'Quiero predecir la X y la Y para'
      -> 'Necesito saber el valor de la X y de la Y cuando'
    """
    cabeza, cabecera, filas = _partes(texto)
    if not cabeza:
        raise ValueError("prompt sin bloque de pregunta")
    primera = cabeza[0]
    m = re.match(r"^\s*Quiero predecir\s+(.*?)\s+para\s*$", primera)
    objetivo = m.group(1) if m else primera.strip()
    # "la X, la Y y la Z" -> "la X, de la Y y de la Z"
    objetivo = re.sub(r"\s+y\s+la\s+", " y de la ", objetivo)
    objetivo = re.sub(r",\s*la\s+", ", de la ", objetivo)
    cabeza = [f"Necesito saber el valor de {objetivo} cuando"] + cabeza[1:]
    return _reconstruir(cabeza, cabecera, filas)


def p6_corromper_fila(texto: str, rng: random.Random) -> str:
    """P6: sustituye un valor aleatorio de una fila aleatoria por 'N/A'."""
    cabeza, cabecera, filas = _partes(texto)
    i = rng.randrange(len(filas))
    campos = filas[i].split(",")
    j = rng.randrange(len(campos))
    campos[j] = "N/A"
    filas[i] = ",".join(campos)
    return _reconstruir(cabeza, cabecera, filas)


FUNCIONES = {
    "P1": p1_duplicar_filas,
    "P2": p2_barajar_filas,
    "P3": p3_ruido_formato,
    "P4": p4_columna_ruido,
    "P5": p5_reformular_pregunta,
    "P6": p6_corromper_fila,
}


def _seed_variante(seed_base: int, prompt_id: str, etiqueta: str) -> int:
    """Semilla determinista por (prompt, perturbación), sin hash() (no estable)."""
    num_prompt = int(re.search(r"prompt_(\d+)", prompt_id).group(1))
    num_p = int(etiqueta[1:])
    return seed_base + 100 * num_prompt + num_p


def generar_perturbaciones(cfg: Config) -> list[Path]:
    """Genera las variantes P1-P6 de los prompts configurados.

    Devuelve la lista de ficheros escritos (nº de prompts configurados x 6;
    con el config original, 3 x 6 = 18 variantes).
    """
    prompts = [
        p for p in cargar_prompts(cfg.dir_prompts)
        if any(p.prompt_id.startswith(pref) for pref in cfg.prompts_perturbar)
    ]
    if not prompts:
        print(f"[perturb] ningún prompt coincide con {cfg.prompts_perturbar}")
        return []

    escritos: list[Path] = []
    for etiqueta in PERTURBACIONES:
        destino = cfg.dir_perturbados / etiqueta
        destino.mkdir(parents=True, exist_ok=True)
        for prompt in prompts:
            rng = random.Random(
                _seed_variante(cfg.seed_perturbaciones, prompt.prompt_id, etiqueta)
            )
            variante = FUNCIONES[etiqueta](prompt.texto, rng)
            ruta = destino / f"{prompt.prompt_id}.txt"
            ruta.write_text(variante, encoding="utf-8")
            escritos.append(ruta)
            print(f"[perturb] {etiqueta}/{prompt.prompt_id}.txt")
    print(f"[perturb] {len(escritos)} variantes generadas en {cfg.dir_perturbados}")
    return escritos
