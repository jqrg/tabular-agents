# -*- coding: utf-8 -*-
"""Tests del pipeline único de validación/extracción (casos borde)."""
import pytest

from harness.validacion import extraer_numeros, validar_y_extraer


def test_extraccion_basica():
    assert extraer_numeros("x = 1.5 y -2 y 3e-2") == [1.5, -2.0, 0.03]


def test_razonamiento_con_numeros_intermedios():
    """Los últimos n_outputs números son la predicción (regla declarada)."""
    texto = (
        "Primero calculo la media: (2+4)/2 = 3. Después ajusto por el factor "
        "1.8 del problema 2. Por tanto, la respuesta final es 42.53"
    )
    res = validar_y_extraer(texto, n_outputs=1)
    assert res.valido
    assert res.pred == [42.53]
    assert res.fuente == "texto"


def test_dos_salidas_toma_las_dos_ultimas():
    res = validar_y_extraer("Posicion X = 470.2 y Posicion Y = 415.8", 2)
    assert res.pred == [470.2, 415.8]


def test_respuesta_vacia():
    res = validar_y_extraer("", 1)
    assert not res.valido
    assert res.motivo == "respuesta_vacia"


def test_numeros_insuficientes():
    res = validar_y_extraer("La respuesta es 5.0", 2)
    assert not res.valido
    assert res.motivo == "numeros_insuficientes"
    assert res.numeros_extraidos == 1


def test_texto_sin_numeros_finitos():
    """'NaN' no es un número extraíble: la respuesta no es válida."""
    res = validar_y_extraer("El resultado es NaN, no puedo calcularlo", 1)
    assert not res.valido


def test_pred_directa_tiene_prioridad():
    res = validar_y_extraer("por el camino salieron 1, 2 y 3", 2,
                            pred_directa=[7.5, 8.5])
    assert res.valido
    assert res.pred == [7.5, 8.5]
    assert res.fuente == "pred_directa"


def test_pred_directa_no_finita_cae_al_texto():
    res = validar_y_extraer("resultado: 9.25", 1,
                            pred_directa=[float("nan")])
    assert res.valido
    assert res.pred == [9.25]
    assert res.fuente == "texto"


def test_pred_directa_insuficiente_y_texto_insuficiente():
    res = validar_y_extraer("sin números", 2, pred_directa=[1.0])
    assert not res.valido


def test_n_outputs_invalido():
    assert not validar_y_extraer("1 2 3", 0).valido


@pytest.mark.parametrize("texto,esperado", [
    ("1e309 y 2.5", [2.5]),          # overflow a inf: se descarta
    ("-.5", [-0.5]),
    ("valor=+3.25e2", [325.0]),
])
def test_finitos_y_formatos(texto, esperado):
    assert extraer_numeros(texto) == esperado
