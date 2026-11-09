# -*- coding: utf-8 -*-
"""Runners: adaptadores entre el harness y cada tipo de sistema evaluado.

Todos implementan la interfaz común de base.Runner:
    run(prompt_text, prompt_path, meta) -> RawResult
"""
from .base import RawResult, Runner  # noqa: F401
