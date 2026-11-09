# -*- coding: utf-8 -*-
"""Orquestación de la parrilla experimental: sistemas x prompts x repeticiones.

Reanudable: si (sistema, prompt_id, rep) ya figura en results/runs.jsonl se
salta. Cada ejecución produce una línea JSON con la respuesta cruda, la
validación, la predicción extraída y los errores frente al ground truth.
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass

from .config import Config
from .metricas import error_normalizado, errores_absolutos
from .persistencia import anexar_jsonl, claves_completadas, leer_jsonl
from .prompts import Prompt, cargar_prompts
from .runners.agente import AgentRunner
from .runners.base import Runner
from .runners.manual import ManualImportRunner
from .runners.ollama_directo import OllamaDirectRunner
from .soluciones import parsear_soluciones
from .validacion import validar_y_extraer

RE_DIR_PERTURBACION = re.compile(r"^P\d+$")


@dataclass
class Tarea:
    """Una celda de la parrilla: (sistema, prompt, repetición)."""

    sistema: str
    prompt: Prompt
    rep: int
    perturbacion: str | None = None

    @property
    def prompt_id(self) -> str:
        """Id registrado: el base más el sufijo __P<n> si está perturbado."""
        if self.perturbacion:
            return f"{self.prompt.prompt_id}__{self.perturbacion}"
        return self.prompt.prompt_id

    @property
    def prompt_id_base(self) -> str:
        return self.prompt.prompt_id


def construir_sistemas(cfg: Config) -> dict[str, Runner]:
    """Instancia todos los runners de la evaluación, por nombre de sistema.

    - agente_<mode> por cada configuración del agente propio.
    - ollama_<model> por cada modelo directo activo.
    - manual_<fichero> por cada CSV presente en manual_responses/.
    """
    sistemas: dict[str, Runner] = {}

    for conf in cfg.agente_configs:
        mode = str(conf.get("mode", "template"))
        model = str(conf.get("model", cfg.agente_model))
        nombre = f"agente_{mode}"
        sistemas[nombre] = AgentRunner(
            nombre=nombre,
            ruta_agente=cfg.ruta_agente,
            mode=mode,
            model=model,
            ollama_url=cfg.ollama_url,
            timeout_s=cfg.timeout_s,
        )

    for model in cfg.modelos_ollama:
        nombre = f"ollama_{model}"
        sistemas[nombre] = OllamaDirectRunner(
            nombre=nombre,
            model=model,
            ollama_url=cfg.ollama_url,
            timeout_s=cfg.timeout_s,
        )

    if cfg.dir_manual.exists():
        for ruta_csv in sorted(cfg.dir_manual.glob("*.csv")):
            nombre = f"manual_{ruta_csv.stem}"
            try:
                sistemas[nombre] = ManualImportRunner(nombre=nombre, ruta_csv=ruta_csv)
            except (ValueError, OSError) as e:
                print(f"[sistemas] {ruta_csv.name} ignorado: {e}")

    return sistemas


def construir_parrilla(
    nombres_sistemas: list[str],
    prompts: list[Prompt],
    n_reps: int,
    hechas: set[tuple[str, str, int]],
    perturbacion: str | None = None,
) -> tuple[list[Tarea], int]:
    """Parrilla pendiente (excluye lo ya presente en runs.jsonl).

    Devuelve (tareas_pendientes, n_saltadas).
    """
    pendientes: list[Tarea] = []
    saltadas = 0
    for sistema in nombres_sistemas:
        for prompt in prompts:
            for rep in range(1, n_reps + 1):
                tarea = Tarea(sistema, prompt, rep, perturbacion)
                if (sistema, tarea.prompt_id, rep) in hechas:
                    saltadas += 1
                else:
                    pendientes.append(tarea)
    return pendientes, saltadas


def _directorios_prompts(cfg: Config, perturbados: bool):
    """Lista de (directorio, etiqueta_perturbacion) a ejecutar."""
    if not perturbados:
        return [(cfg.dir_prompts, None)]
    if not cfg.dir_perturbados.exists():
        raise FileNotFoundError(
            f"{cfg.dir_perturbados} no existe; genera antes con `python -m harness perturb`"
        )
    subdirs = sorted(
        d for d in cfg.dir_perturbados.iterdir()
        if d.is_dir() and RE_DIR_PERTURBACION.match(d.name)
    )
    return [(d, d.name) for d in subdirs]


def _evaluar_y_registrar(cfg: Config, tarea: Tarea, runner: Runner,
                         soluciones: dict, prompts_base: dict[str, Prompt]) -> dict:
    """Ejecuta una tarea, valida por el pipeline único y construye el registro."""
    meta = {
        "prompt_id": tarea.prompt_id,
        "rep": tarea.rep,
        "n_inputs": tarea.prompt.n_inputs,
        "n_outputs": tarea.prompt.n_outputs_csv,
        "timeout_s": cfg.timeout_s,
    }
    gt = soluciones.get(tarea.prompt_id_base)
    # n_outputs canónico: el del ground truth (robusto ante P4, que añade una
    # columna espuria al CSV); si no hay ground truth, el inferido del prompt.
    n_outputs = len(gt["valores"]) if gt and gt["valores"] else tarea.prompt.n_outputs_csv
    meta["n_outputs"] = n_outputs

    res = runner.run(tarea.prompt.texto, tarea.prompt.ruta, meta)

    registro: dict = {
        "sistema": tarea.sistema,
        "prompt_id": tarea.prompt_id,
        "prompt_id_base": tarea.prompt_id_base,
        "perturbacion": tarea.perturbacion,
        "rep": tarea.rep,
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": "v2",
        "n_outputs": n_outputs,
        "ok_exec": res.ok_exec,
        "seconds": round(res.seconds, 3),
        "raw_text": res.raw_text,
        "extra": res.extra,
        "valido": False,
        "motivo_invalido": None,
        "fuente_pred": None,
        "pred": None,
        "verdad": gt["valores"] if gt else None,
        "errores_abs": None,
        "mae_run": None,
        "err_norm_run": None,
        "dispersion_norm": None,
    }

    if not res.ok_exec:
        registro["motivo_invalido"] = res.extra.get("error", "ejecucion_fallida")
        return registro

    val = validar_y_extraer(res.raw_text, n_outputs, pred_directa=res.pred)
    registro["valido"] = val.valido
    registro["motivo_invalido"] = None if val.valido else val.motivo
    registro["fuente_pred"] = val.fuente or None
    registro["pred"] = val.pred

    # Rangos siempre del prompt base (decisión declarada en el README).
    prompt_rangos = prompts_base.get(tarea.prompt_id_base, tarea.prompt)

    if val.valido and gt and gt["valores"]:
        rangos = prompt_rangos.rangos_salida(n_outputs)
        errores = errores_absolutos(val.pred, gt["valores"])
        registro["errores_abs"] = errores
        registro["mae_run"] = sum(errores) / len(errores)
        registro["err_norm_run"] = error_normalizado(errores, rangos)

        # Señal de calibración del agente: pred_std/rango, media sobre salidas.
        pred_std = (res.extra.get("result_json") or {}).get("pred_std")
        if isinstance(pred_std, list) and len(pred_std) == len(rangos):
            try:
                registro["dispersion_norm"] = sum(
                    abs(float(s)) / max(r, 1e-9) for s, r in zip(pred_std, rangos)
                ) / len(rangos)
            except (TypeError, ValueError):
                pass
    return registro


def ejecutar(
    cfg: Config,
    dry_run: bool = False,
    perturbados: bool = False,
    sistemas_filtro: list[str] | None = None,
) -> list[dict]:
    """Comando run: ejecuta (o lista, con dry_run) la parrilla pendiente."""
    sistemas = construir_sistemas(cfg)
    if sistemas_filtro:
        desconocidos = [s for s in sistemas_filtro if s not in sistemas]
        if desconocidos:
            print(f"[run] sistemas desconocidos: {desconocidos}. "
                  f"Disponibles: {sorted(sistemas)}")
        sistemas = {k: v for k, v in sistemas.items() if k in sistemas_filtro}
    if not sistemas:
        print("[run] no hay sistemas que ejecutar.")
        return []

    soluciones = (
        parsear_soluciones(cfg.ruta_soluciones) if cfg.ruta_soluciones.exists() else {}
    )
    if not soluciones:
        print(f"[run] AVISO: sin ground truth ({cfg.ruta_soluciones}); "
              "se registran respuestas pero no errores.")

    prompts_base = {p.prompt_id: p for p in cargar_prompts(cfg.dir_prompts)}
    hechas = claves_completadas(leer_jsonl(cfg.ruta_runs))

    todas: list[Tarea] = []
    total_saltadas = 0
    for directorio, etiqueta in _directorios_prompts(cfg, perturbados):
        prompts = cargar_prompts(directorio)
        if not prompts:
            print(f"[run] AVISO: sin prompts en {directorio}")
            continue
        pendientes, saltadas = construir_parrilla(
            sorted(sistemas), prompts, cfg.n_reps, hechas, etiqueta
        )
        todas.extend(pendientes)
        total_saltadas += saltadas

    print(f"[run] parrilla: {len(todas)} ejecuciones pendientes, "
          f"{total_saltadas} ya hechas (se saltan).")
    if dry_run:
        for t in todas:
            print(f"  - {t.sistema} | {t.prompt_id} | rep {t.rep}")
        return []

    registros: list[dict] = []
    for i, tarea in enumerate(todas, 1):
        print(f"[run {i}/{len(todas)}] {tarea.sistema} | {tarea.prompt_id} | rep {tarea.rep}")
        registro = _evaluar_y_registrar(
            cfg, tarea, sistemas[tarea.sistema], soluciones, prompts_base
        )
        anexar_jsonl(cfg.ruta_runs, registro)
        registros.append(registro)
        estado = "OK" if registro["valido"] else f"FALLO({registro['motivo_invalido']})"
        print(f"        -> {estado} mae={registro['mae_run']} t={registro['seconds']}s")

    # summary.csv actualizado al terminar (también lo regenera analyze).
    try:
        from .analisis import escribir_summary_csv
        escribir_summary_csv(cfg)
    except Exception as e:  # noqa: BLE001 — el summary nunca invalida la tanda
        print(f"[run] no se pudo escribir summary.csv: {e}")
    return registros
