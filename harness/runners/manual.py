# -*- coding: utf-8 -*-
"""Runner de respuestas manuales (modelos comerciales): lee
manual_responses/<sistema>.csv (columnas prompt_id,rep,respuesta_cruda) y
las procesa por el mismo pipeline de validación que el resto de sistemas.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .base import RawResult, Runner, truncar_raw


class ManualImportRunner(Runner):
    """Sirve respuestas pre-pegadas indexadas por (prompt_id, rep)."""

    def __init__(self, nombre: str, ruta_csv: Path):
        self.nombre = nombre
        self.ruta_csv = Path(ruta_csv)
        self._respuestas: dict[tuple[str, int], str] = {}
        self._cargar()

    def _cargar(self) -> None:
        # utf-8-sig: tolera el BOM que añaden Excel y el Bloc de notas.
        with open(self.ruta_csv, encoding="utf-8-sig", newline="") as f:
            lector = csv.DictReader(f)
            columnas = {c.strip().lower() for c in (lector.fieldnames or [])}
            requeridas = {"prompt_id", "rep", "respuesta_cruda"}
            if not requeridas.issubset(columnas):
                raise ValueError(
                    f"{self.ruta_csv.name}: se esperan columnas {sorted(requeridas)}, "
                    f"hay {sorted(columnas)}"
                )
            for fila in lector:
                fila = {k.strip().lower(): v for k, v in fila.items() if k}
                try:
                    clave = (str(fila["prompt_id"]).strip(), int(fila["rep"]))
                except (KeyError, TypeError, ValueError):
                    continue
                self._respuestas[clave] = fila.get("respuesta_cruda") or ""

    def run(self, prompt_text: str, prompt_path: Path | None, meta: dict) -> RawResult:
        clave = (str(meta.get("prompt_id")), int(meta.get("rep", 0)))
        if clave not in self._respuestas:
            return RawResult(
                ok_exec=False,
                extra={"error": "respuesta_manual_ausente", "clave": list(clave)},
            )
        raw, recortado = truncar_raw(self._respuestas[clave])
        return RawResult(
            raw_text=raw,
            ok_exec=True,
            extra={"origen": self.ruta_csv.name, "raw_recortado": recortado},
        )
