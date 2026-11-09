# -*- coding: utf-8 -*-
"""CLI del harness: python -m harness <subcomando>.

Subcomandos: run | analyze | perturb
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import cargar_config


def _cmd_run(args) -> int:
    from .ejecutor import ejecutar

    cfg = cargar_config(args.config)
    ejecutar(
        cfg,
        dry_run=args.dry_run,
        perturbados=args.perturbados,
        sistemas_filtro=args.sistema or None,
    )
    return 0


def _cmd_analyze(args) -> int:
    from .analisis import analizar

    cfg = cargar_config(args.config)
    analizar(
        cfg,
        ruta_runs=Path(args.runs) if args.runs else None,
        dir_salida=Path(args.salida) if args.salida else None,
    )
    return 0


def _cmd_perturb(args) -> int:
    from .perturbaciones import generar_perturbaciones

    cfg = cargar_config(args.config)
    generar_perturbaciones(cfg)
    return 0


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness",
        description="Harness de evaluación del TFM: parrilla, métricas, "
                    "perturbaciones y análisis.",
    )
    parser.add_argument("--config", default=None,
                        help="ruta a un config.yaml alternativo")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_run = sub.add_parser("run", help="ejecuta la parrilla pendiente (reanudable)")
    p_run.add_argument("--dry-run", action="store_true",
                       help="lista lo que se ejecutaría, sin ejecutar nada")
    p_run.add_argument("--perturbados", action="store_true",
                       help="ejecuta sobre prompts_perturbados/P1..P6")
    p_run.add_argument("--sistema", action="append",
                       help="limita a un sistema (repetible), p.ej. agente_template")
    p_run.set_defaults(func=_cmd_run)

    p_an = sub.add_parser("analyze", help="genera summary.csv, analysis.md y figuras")
    p_an.add_argument("--runs", default=None,
                      help="ruta a un runs.jsonl alternativo")
    p_an.add_argument("--salida", default=None,
                      help="directorio de salida alternativo")
    p_an.set_defaults(func=_cmd_analyze)

    p_pert = sub.add_parser("perturb",
                            help="genera las perturbaciones P1-P6 (deterministas)")
    p_pert.set_defaults(func=_cmd_perturb)
    return parser


def main(argv=None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)
    return args.func(args)
