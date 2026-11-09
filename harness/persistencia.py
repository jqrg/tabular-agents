# -*- coding: utf-8 -*-
"""Persistencia y reanudación: results/runs.jsonl (una línea por ejecución).

La clave de reanudación es (sistema, prompt_id, rep): si ya existe en el
fichero, esa ejecución se salta. La respuesta cruda se conserva siempre.
"""
from __future__ import annotations

import json
from pathlib import Path


def anexar_jsonl(ruta: str | Path, registro: dict) -> None:
    """Añade un registro como una línea JSON (crea el directorio si falta)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")


def leer_jsonl(ruta: str | Path) -> list[dict]:
    """Lee un .jsonl tolerando líneas corruptas (se ignoran con aviso)."""
    ruta = Path(ruta)
    if not ruta.exists():
        return []
    registros: list[dict] = []
    with open(ruta, encoding="utf-8") as f:
        for i, linea in enumerate(f, 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                registros.append(json.loads(linea))
            except json.JSONDecodeError:
                print(f"[persistencia] línea {i} de {ruta.name} corrupta; se ignora.")
    return registros


def claves_completadas(registros: list[dict]) -> set[tuple[str, str, int]]:
    """Conjunto de claves (sistema, prompt_id, rep) ya ejecutadas.

    Los fallos de conexión no cuentan como hechos (son del entorno y la
    celda se reintenta al reanudar); timeout y agente_failed sí ocupan celda.
    """
    claves = set()
    for r in registros:
        extra = r.get("extra")
        if isinstance(extra, dict) and extra.get("error") == "conexion":
            continue
        try:
            claves.add((str(r["sistema"]), str(r["prompt_id"]), int(r["rep"])))
        except (KeyError, TypeError, ValueError):
            continue
    return claves
