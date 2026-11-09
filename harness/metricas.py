# -*- coding: utf-8 -*-
"""Métricas por ejecución (definiciones del Capítulo 4): error absoluto por
salida, mae_run y err_norm_run (error normalizado por el rango max-min de
cada columna objetivo, calculado del CSV del propio prompt). Los agregados
por sistema y prompt los calcula analisis.py sobre estos valores.
"""
from __future__ import annotations


def errores_absolutos(pred: list[float], verdad: list[float]) -> list[float]:
    """Error absoluto por salida. Exige misma longitud."""
    if len(pred) != len(verdad):
        raise ValueError(
            f"pred tiene {len(pred)} valores y el ground truth {len(verdad)}"
        )
    return [abs(p - v) for p, v in zip(pred, verdad)]


def error_normalizado(errores: list[float], rangos: list[float]) -> float:
    """Media de los errores normalizados por el rango de cada salida."""
    if len(errores) != len(rangos):
        raise ValueError("errores y rangos deben tener la misma longitud")
    return sum(e / max(r, 1e-9) for e, r in zip(errores, rangos)) / len(errores)
