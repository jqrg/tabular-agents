# -*- coding: utf-8 -*-
"""Carga de la configuración del harness (harness/config.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

RUTA_CONFIG_DEFECTO = Path(__file__).with_name("config.yaml")


@dataclass
class Config:
    """Configuración resuelta del harness (rutas absolutas)."""

    raiz: Path
    n_reps: int
    timeout_s: int
    ollama_url: str
    ruta_agente: Path
    dir_prompts: Path
    ruta_soluciones: Path
    dir_resultados: Path
    dir_manual: Path
    dir_perturbados: Path
    modelos_ollama: list[str]
    agente_model: str
    agente_configs: list[dict]
    prompts_perturbar: list[str]
    seed_perturbaciones: int
    bruto: dict = field(default_factory=dict)

    @property
    def ruta_runs(self) -> Path:
        return self.dir_resultados / "runs.jsonl"


def cargar_config(ruta: str | Path | None = None) -> Config:
    """Lee el YAML y resuelve las rutas contra la raíz del repositorio.

    La raíz es, por defecto, el directorio padre del paquete harness/;
    puede sobrescribirse con la clave opcional `raiz` del YAML.
    """
    ruta = Path(ruta) if ruta else RUTA_CONFIG_DEFECTO
    with open(ruta, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raiz = Path(data.get("raiz", Path(__file__).resolve().parent.parent)).resolve()
    rutas = data.get("rutas", {})

    def _abs(clave: str, defecto: str) -> Path:
        return (raiz / rutas.get(clave, defecto)).resolve()

    agente = data.get("agente", {})
    return Config(
        raiz=raiz,
        n_reps=int(data.get("n_reps", 5)),
        timeout_s=int(data.get("timeout_s", 1800)),
        ollama_url=str(data.get("ollama_url", "http://localhost:11434")),
        ruta_agente=_abs("agente", "Agente/agente.py"),
        dir_prompts=_abs("prompts", "prompts"),
        ruta_soluciones=_abs("soluciones", "prompts/soluciones.txt"),
        dir_resultados=_abs("resultados", "results"),
        dir_manual=_abs("manual_responses", "manual_responses"),
        dir_perturbados=_abs("perturbados", "prompts_perturbados"),
        modelos_ollama=list(data.get("modelos_ollama", []) or []),
        agente_model=str(agente.get("model", "qwen3.5:4b")),
        agente_configs=list(agente.get("configs", []) or []),
        prompts_perturbar=list(data.get("prompts_perturbar", []) or []),
        seed_perturbaciones=int(data.get("seed_perturbaciones", 20260705)),
        bruto=data,
    )
