# -*- coding: utf-8 -*-
"""Fixtures comunes de los tests del harness (todos SIN red ni Ollama)."""
import dataclasses
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from harness.config import cargar_config  # noqa: E402


@pytest.fixture(scope="session")
def raiz() -> Path:
    return RAIZ


@pytest.fixture()
def cfg_tmp(tmp_path):
    """Config real (prompts y soluciones del repo) con salidas en tmp_path."""
    cfg = cargar_config()
    return dataclasses.replace(
        cfg,
        dir_resultados=tmp_path / "results",
        dir_perturbados=tmp_path / "prompts_perturbados",
        dir_manual=tmp_path / "manual_responses",
    )


@pytest.fixture(scope="session")
def ruta_fixture_runs(raiz) -> Path:
    return raiz / "tests" / "fixtures" / "runs_sintetico.jsonl"
