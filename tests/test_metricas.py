# -*- coding: utf-8 -*-
"""Tests de las métricas con valores conocidos."""
import pytest

from harness.metricas import error_normalizado, errores_absolutos


def test_errores_absolutos():
    assert errores_absolutos([1.0, -2.0], [1.5, -1.0]) == [0.5, 1.0]


def test_errores_longitud_distinta():
    with pytest.raises(ValueError):
        errores_absolutos([1.0], [1.0, 2.0])


def test_error_normalizado_multisalida():
    # (0.5/10 + 1.0/2) / 2 = (0.05 + 0.5) / 2 = 0.275
    assert error_normalizado([0.5, 1.0], [10.0, 2.0]) == pytest.approx(0.275)


def test_error_normalizado_rango_cero_no_divide_por_cero():
    assert error_normalizado([1.0], [0.0]) == pytest.approx(1.0 / 1e-9)
