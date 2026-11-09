# -*- coding: utf-8 -*-
# Dependencias con versiones fijadas (validadas; instalar exactamente estas):
#   pip install llama-index-llms-ollama==0.10.1 llama-index-core==0.14.21
"""
Agente — TFM Iván Vega Prieto
=============================
Agente que, a partir de un enunciado en lenguaje natural con datos tabulares
embebidos, infiere la estructura del problema, escribe (o instancia) el código
que entrena una red neuronal a medida, lo ejecuta y devuelve la predicción.
Evolución del prototipo del grupo de investigación: agenteJordan.py (incluido
en prototipo/ como referencia de comparación), que a su vez extendía
agenteJordanprim.py (no incluido).
Misma estructura: Workflow LlamaIndex por pasos + CodeActAgent.

QUÉ HACE, DE UN VISTAZO
=======================
  enunciado.txt
      -> pasos 1/2   nº de entradas y de salidas (LLM + self-consistency)
      -> paso  2b    verificación determinista de esa estructura
      -> paso  3     aislamiento del bloque CSV
      -> paso  4     pregunta + valores del query (determinista, LLM de apoyo)
      -> paso  5     volcado a train.csv
      -> paso  6     código de entrenamiento (plantilla | generación libre),
                     ejecución y puerta semántica sobre la predicción
      -> paso  7     corrección de errores de ejecución con CodeActAgent,
                     con retroceso al paso 1 solo si el error es estructural
      -> stdout      "RESULT_JSON={...}" + log.csv + log.jsonl

MEJORAS RESPECTO AL PROTOTIPO (numeración M#; cada una está anclada con su
etiqueta en el punto exacto del código donde vive)
==========================================================================
  [M1]   GENERATION_MODE = "template" | "libre". Mismo flujo en ambos modos;
         solo cambia quién escribe el código de entrenamiento (Tool determinista
         vs generación libre del LLM). Es la ablación central del estudio:
         una sola variable cambia entre las dos configuraciones.
  [M2]   Dos LLMs separados: llm_extract (restrictivo, para contar y extraer)
         y llm_coder (system prompt de codificación) para el CodeActAgent. El
         prototipo compartía el system prompt restrictivo con el agente de
         código; tenía esta separación apuntada en un comentario con GoogleGenAI
         y quedó sin portar a Ollama.
  [M3]   Self-consistency en los pasos 1/2: SC_N llamadas al LLM y voto por
         mayoría; el empate se trata como "sin respuesta". SC_N=1 => modo 100 %
         determinista.
  [M4]   Verificador determinista de estructura (paso 2b): n_inputs = nº de
         líneas "NOMBRE = VALOR" del bloque de pregunta; n_outputs = n_cols -
         n_inputs. Contrasta al LLM (que se sigue midiendo, es el objeto del
         estudio), le devuelve feedback en los reintentos y, si estos se agotan,
         actúa como fallback en lugar de continuar con valores malos.
  [M5]   Valores del query extraídos determinísticamente (lado derecho de "=").
         Fin del bug de dígitos-en-nombres del prototipo, que los sacaba con una
         expresión regular sobre todo el texto. El relleno con 0.0 queda como
         último recurso y deja de ser silencioso: activa values_padded, degrada
         confidence a "low" y se registra como issue en el log.
  [M6]   Prompts internos sin sesgo del problema de triangulación (sin
         "Distance A, B, C" ni default "Position X y Position Y"): el agente
         debe servir a los diez problemas por igual.
  [M7]   Extractor genérico del bloque CSV: el del prototipo estaba acoplado a
         los nombres de columna de la triangulación y fallaba con cualquier
         otro enunciado.
  [M8]   Expansión polinómica de grado 2 opcional; el grid-search decide con o
         sin ella por validación (PolynomialFeatures, a ciegas en todos los
         problemas; guard: se desactiva si n_in > 6 para no explotar el nº de
         características).
  [M9]   Refit dinámico: las épocas del reentrenamiento final se derivan del
         early stopping observado en el grid-search, no son 300 fijas.
  [M10]  Puerta semántica sobre la predicción (paso 6b): NaN/Inf, predicción
         fuera del rango observado de y, val_mae_norm absurdo -> 1 reintento
         con presupuesto ampliado -> flag de confianza. (El paso 7 queda solo
         para errores de ejecución; son problemas distintos.)
  [M11]  Incertidumbre multi-semilla usada como señal: dispersión entre semillas
         relativa al rango de la variable objetivo (robusta a predicciones
         cercanas a 0) -> reintento ampliado / confidence="low". Se registra
         para el análisis de calibración del Cap. 5.
  [M12]  Bucle de corrección acotado y backtracking selectivo: el prototipo
         corregía sin límite y nunca podía volver a los pasos iniciales. Aquí
         hay tope de iteraciones y se retrocede al paso 1 solo ante errores
         estructurales (shape/index/columns), no ante errores de entorno.
  [M13]  Tools aritméticas de ejemplo sustituidas por una tool útil para este
         dominio (csv_info: cabecera, filas y columnas de train.csv).
  [M14]  Logging estructurado: log.csv (con columnas nuevas; si existe un
         log.csv del prototipo conviene renombrarlo antes) + log.jsonl con
         métricas por ejecución (para el harness y el Cap. 5).

CORRECCIONES POSTERIORES AL CONGELADO
=====================================
  [FIX-EXEC 07/2026] SimpleCodeExecutor ejecuta en un namespace único y
         persistente (tools de locals_dict visibles en el código generado y
         estado conservado entre bloques). Defecto de implementación corregido
         antes de la primera tanda computable del modo libre; el modo template
         no usa este ejecutor, así que sus resultados no cambian.
  [FIX-CTX 07/2026] context_window=8192 en los dos clientes Ollama. La
         versión moderna de llama-index-llms-ollama (default -1) pedía el
         contexto máximo del modelo (131k) en cada llamada -> ~11 GB de caché
         KV -> modelo fuera de GPU -> llamadas lentísimas en todos los
         entornos. Solo dimensiona la reserva de memoria: con prompts de
         ~3.1k tokens, lo generado es idéntico. No cambia el objeto medido.

CONFIG por variables de entorno (todas opcionales)
==================================================
  OLLAMA_URL   (default http://localhost:11434)
  MODEL_NAME   (default qwen3.5:4b — ojo: ese tag no existe en Ollama; el
               harness siempre pasa el modelo real de su config, pero al
               ejecutar el agente a mano hay que exportar un tag existente,
               p.ej. llama3.2:3b)
  GENERATION_MODE  template|libre   (default template)
  SC_N         nº llamadas self-consistency pasos 1/2 (default 3; 1=determinista)
  SC_TEMP      temperatura de extracción si SC_N>1 (default 0.7)
  AUGMENT      auto|off (default auto)
  AGENT_TIMEOUT  segundos del workflow (default 3600)

SALIDAS (en el directorio de trabajo actual)
============================================
  stdout       traza por pasos + "########### Resultado final ###########"
               (el harness recorta exactamente esa sección) + métricas JSON
  log.csv      una fila por ejecución, delimitador ';'            [M14]
  log.jsonl    resultado canónico: un objeto JSON por ejecución   [M14]
  train.csv    datos de entrenamiento aislados del enunciado (paso 5)
  test_<ms>.py código de entrenamiento efectivamente ejecutado (artefacto)

USO
===
  python agente.py ruta/al/prompt.txt [prompt_id]
  (sin argumentos ejecuta un mini-problema de humo)
"""
import re
import os
import sys
import json
import time
import asyncio
import io
import contextlib
import ast
import csv
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step, Context
from llama_index.core.agent.workflow import (
    CodeActAgent,
    ToolCall,
    ToolCallResult,
    AgentStream,
)

load_dotenv()

# ============================== CONFIG =====================================
# Toda la configuración entra por variables de entorno: el harness lanza este
# fichero como subproceso y solo puede hablarle por el entorno y por argv.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5:4b")
# [M1] La ablación central del estudio vive en esta única variable.
GENERATION_MODE = os.getenv("GENERATION_MODE", "template").strip().lower()  # template | libre
if GENERATION_MODE not in ("template", "libre"):
    # Un modo mal escrito degrada a 'template' en vez de abortar: una celda de
    # la parrilla nunca debe perderse por una errata en el entorno.
    print(f"[config] GENERATION_MODE='{GENERATION_MODE}' no reconocido; se usa 'template'.")
    GENERATION_MODE = "template"
SC_N = max(1, int(os.getenv("SC_N", "3")))      # [M3] votos de self-consistency; max(1,·) evita SC_N=0
SC_TEMP = float(os.getenv("SC_TEMP", "0.7"))    # [M3] temperatura solo si hay más de un voto
AUGMENT_MODE = os.getenv("AUGMENT", "auto").strip().lower()                 # [M8] auto | off
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "3600"))  # techo del Workflow; el harness pasa el suyo menos 60 s

# --- Presupuesto de entrenamiento de la plantilla --------------------------
MAX_EPOCHS_BASE = 1000  # techo de épocas por modelo del grid (el early stopping suele cortar antes)
PATIENCE_BASE = 50      # épocas sin mejora de validación antes de parar
# Rejilla de arquitecturas (lista de tamaños de capa oculta). ARCHS_BASE es el
# intento normal; ARCHS_EXT añade modelos mayores y solo se usa en el reintento
# con "presupuesto ampliado" que dispara la puerta semántica [M10].
ARCHS_BASE = [[16], [32], [16, 8], [32, 16], [64, 32],
              [32, 16, 8], [64, 32, 16], [128, 64], [16, 16, 16, 16]]
ARCHS_EXT = ARCHS_BASE + [[128, 64, 32], [256, 128], [64, 64, 64]]

# --- Topes de los bucles de recuperación (el prototipo no tenía ninguno) ----
MAX_PASO7_ITER = 3        # [M12] iteraciones de corrección antes de plantear backtrack
MAX_BACKTRACKS = 2        # [M12] regresos al paso 1 permitidos en toda la ejecución
MAX_2B_RETRIES = 2        # [M4] reintentos de la inferencia de estructura con feedback
MAX_SEMANTIC_RETRIES = 1  # [M10] reintentos con presupuesto ampliado tras fallar el gate

# Última porción de código que el CodeActAgent pidió ejecutar. Es global porque
# se rellena dentro del bucle de eventos del agente (run_agent_and_get_result) y
# se consume fuera, en los pasos 6 y 7, para volver a ejecutarla de forma
# controlada y capturar su stdout.
last_code = ""

# ========================= [M2] DOS LLMs SEPARADOS =========================
# Extracción: system prompt restrictivo (solo el número pedido) y temperatura 0
# si SC_N == 1, lo que hace la extracción 100 % determinista.
llm_extract = Ollama(
    base_url=OLLAMA_URL,
    model=MODEL_NAME,
    request_timeout=600.0,
    temperature=(SC_TEMP if SC_N > 1 else 0),
    # [FIX-CTX 07/2026] sin esto, la librería pide num_ctx = contexto máximo
    # del modelo (131k en llama3.2): ~11 GB solo de caché KV -> el modelo no
    # cabe en GPU y la generación se arrastra. Los prompts del estudio miden
    # ~3.1k tokens: 8192 sobra y no cambia ni un token de lo generado.
    context_window=8192,
    system_prompt="Responde siempre en español. Responde solo con el número solicitado, sin explicación.",
)
# Codificación: el CodeActAgent necesita escribir código, y con el system prompt
# restrictivo de la extracción ("responde solo con el número") era incapaz de
# hacerlo. El prototipo heredaba ese prompt; aquí se separan los dos LLM, como
# el propio prototipo dejaba apuntado en un comentario. Temperatura 0.2: algo de
# margen para reformular código sin volverse errático. request_timeout alto
# porque una respuesta de código completa es mucho más larga que un entero.
llm_coder = Ollama(
    base_url=OLLAMA_URL,
    model=MODEL_NAME,
    request_timeout=6000.0,
    temperature=0.2,
    context_window=8192,   # [FIX-CTX 07/2026] ver nota en llm_extract
    system_prompt=("Eres un programador experto en Python y PyTorch. "
                   "Escribes y ejecutas código Python completo y correcto cuando hace falta. "
                   "Responde en español."),
)

# ============================== EVENTOS ====================================
# En un Workflow de LlamaIndex NO hay un bucle principal que llame a los pasos:
# el enrutado lo decide el TIPO del evento. Cada método @step declara qué evento
# consume (parámetro `ev`) y qué evento(s) puede devolver (anotación de retorno),
# y el motor encadena el que corresponda. Por eso los eventos de abajo arrastran
# el estado de un paso al siguiente: son el único canal entre pasos, y por eso
# también un mismo evento (Step1Event) puede nacer del camino normal o del
# camino de reintento sin que el paso siguiente tenga que distinguirlos.
#
# Campos que se repiten en casi todos los eventos:
#   user_message      enunciado completo, tal cual llegó (nunca se recorta en
#                     el evento; los recortes son locales a cada prompt)
#   n_inputs/n_outputs estructura inferida del problema
#   dataset           bloque CSV ya aislado del enunciado
#   question_summary  qué hay que predecir, en una línea
#   input_values_str  valores del query, pendientes de normalizar a CSV
#   training_data     ruta del CSV volcado a disco ("train.csv")

class NeuralNetStartEvent(StartEvent):
    """Evento de arranque del Workflow: transporta el enunciado del problema.

    Es un StartEvent (no un Event normal) porque LlamaIndex lo trata como
    entrada del flujo; se construye en main() con el contenido del .txt.
    """
    user_message: str = ""


class Step1Event(Event):
    """Salida del paso 1: nº de entradas inferido por el LLM.

    `feedback` y `retry_count` viajan vacíos/0 en el camino normal y llegan
    rellenos cuando el evento lo emite paso1_retry, de modo que el paso 2
    hereda el mismo feedback y el contador de reintentos sigue creciendo.
    """
    user_message: str = ""
    n_inputs: int = 0
    feedback: str = ""
    retry_count: int = 0


class Step2Event(Event):
    """Salida del paso 2: estructura completa (n_inputs, n_outputs) del LLM,
    todavía SIN verificar. La verificación es responsabilidad del paso 2b."""
    user_message: str = ""
    n_inputs: int = 0
    n_outputs: int = 0
    feedback: str = ""
    retry_count: int = 0


class Step2bRetryEvent(Event):
    """[M4] Petición de reintento de la inferencia de estructura.

    Cierra el ciclo 2b -> 1 -> 2 -> 2b llevando el `feedback` que explica al LLM
    por qué su recuento no cuadra. Es también el evento con el que aterriza el
    backtracking [M12] desde el paso 7, así que el reintento por estructura
    incoherente y el reintento por fallo de ejecución comparten camino.
    """
    user_message: str = ""
    feedback: str = ""
    retry_count: int = 0


class Step3Event(Event):
    """Estructura ya validada (o resuelta por fallback) + bloque CSV aislado."""
    user_message: str = ""
    n_inputs: int = 0
    n_outputs: int = 0
    dataset: str = ""


class Step4Event(Event):
    """Entrada del paso 4. Nace con question_summary/input_values_str vacíos:
    los rellena el propio paso 4 al construir el Step5Event."""
    user_message: str = ""
    n_inputs: int = 0
    n_outputs: int = 0
    dataset: str = ""
    question_summary: str = ""
    input_values_str: str = ""


class Step5Event(Event):
    """Todo lo necesario para entrenar, aún en memoria: falta volcar el CSV."""
    user_message: str = ""
    n_inputs: int = 0
    n_outputs: int = 0
    dataset: str = ""
    question_summary: str = ""
    input_values_str: str = ""
    training_data: str = ""


class Step6StartEvent(Event):
    """Disparador del paso 6 con train.csv ya en disco. Mismo contenido que
    Step5Event: se separan los tipos porque el enrutado del Workflow es por
    tipo de evento, no por contenido."""
    user_message: str = ""
    n_inputs: int = 0
    n_outputs: int = 0
    dataset: str = ""
    question_summary: str = ""
    input_values_str: str = ""
    training_data: str = ""


class Step6Event(Event):
    """Error de ejecución en los pasos 6/7: código + error para corrección (paso 7).

    Es el único evento que viaja "hacia atrás" dentro de la fase de código: lo
    emite el paso 6 cuando la ejecución falla y lo vuelve a emitir el paso 7
    tras cada corrección fallida, incrementando `iteration` hasta MAX_PASO7_ITER.

    Campos propios:
      code_to_execute  código que falló (lo que el corrector debe arreglar)
      iteration        nº de intento de corrección, 1-indexado
      error_type       nombre de la excepción (p.ej. "ValueError"); decide si el
                       fallo es estructural y merece backtracking [M12]
      error_msg        "Tipo: mensaje", para el prompt de corrección
      error_traceback  traza completa; al prompt solo van sus primeros 500 chars
    """
    user_message: str = ""
    n_inputs: int = 0
    n_outputs: int = 0
    dataset: str = ""
    question_summary: str = ""
    input_values_str: str = ""
    training_data: str = ""
    code_to_execute: str = ""
    iteration: int = 0
    error_msg: str = ""
    error_type: str = ""
    error_traceback: str = ""


class BacktrackEvent(Event):
    """[M12] Retroceso al inicio del flujo tras agotar las correcciones ante un
    error estructural: la hipótesis es que (n_inputs, n_outputs) estaba mal, así
    que se reabre el paso 1 con el feedback del fallo en vez de seguir parcheando
    código sobre una estructura equivocada."""
    user_message: str = ""
    feedback: str = ""


# ================== EJECUTOR + TOOL ÚTIL [M13] =============================
class SimpleCodeExecutor:
    """Backend de ejecución del CodeActAgent: recibe el código que el LLM quiere
    correr, lo ejecuta y devuelve su stdout como texto.

    Ojo con el reparto de papeles, que es fácil de confundir:
      - esta clase ejecuta el código DENTRO del diálogo del agente, para que el
        LLM vea la salida y pueda iterar;
      - la ejecución que cuenta para el resultado del estudio es la de
        run_code_captured(), fuera del agente y en un namespace limpio.
    """

    def __init__(self, locals_dict: Dict[str, Any], globals_dict: Dict[str, Any]):
        # [FIX-EXEC 07/2026] Namespace único y persistente para exec(): antes
        # exec(code) corría sin estos dicts, así que las tools de locals_dict
        # (csv_info) daban NameError en el código del CodeActAgent y las
        # variables no persistían entre bloques de una misma sesión. Corregido
        # antes de la primera tanda computable del modo libre; el modo
        # template no pasa por este ejecutor (sin efecto sobre sus resultados).
        # Un solo dict para globals y locals: al ser el MISMO objeto, lo que el
        # código define (funciones, variables) queda visible para el código
        # siguiente. Con dos dicts distintos, exec() trata el bloque como si
        # fuera un cuerpo de función y las definiciones no se ven entre sí.
        self.globals = {**globals_dict, **locals_dict}
        self.locals = self.globals

    def execute(self, code: str) -> str:
        """Ejecuta `code` capturando su stdout; devuelve la salida o el error.

        Nunca lanza excepción: el CodeActAgent necesita LEER el error como texto
        para poder corregirlo en la siguiente vuelta. Si el último nodo del
        código es una expresión suelta, se reescribe como asignación a
        __result__ para que evaluarla no se pierda; si esa reescritura falla
        (p.ej. la última línea es la continuación de una expresión multilínea),
        se reintenta el código tal cual.
        """
        try:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                try:
                    tree = ast.parse(code)
                    last_node = tree.body[-1] if tree.body else None
                    if isinstance(last_node, ast.Expr):
                        # Recorta la última línea física y la reasigna: es una
                        # heurística de texto, de ahí el try/except de abajo.
                        last_line = code.rstrip().split("\n")[-1]
                        exec_code = code[: -len(last_line)] + "\n__result__ = " + last_line
                        exec(exec_code, self.globals)
                    else:
                        exec(code, self.globals)
                except Exception:
                    exec(code, self.globals)   # camino de seguridad: el código original
            out = f.getvalue()
        except Exception as e:
            # El error se DEVUELVE como texto (no se propaga) para que el LLM lo vea.
            out = f"Error: {type(e).__name__}: {str(e)}\n" + traceback.format_exc()
        return out


def csv_info(path: str = "train.csv") -> str:
    """[M13] Tool del CodeActAgent: cabecera, nº de filas y nº de columnas del CSV.

    Sustituye a las tools aritméticas de ejemplo del prototipo (sumar, restar),
    que no servían para nada en este dominio. Le permite al LLM comprobar la
    forma real de los datos antes de escribir el código, en lugar de suponerla.
    Devuelve el error como texto, nunca lanza: es una tool, no debe tumbar al agente.
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f.read().split("\n") if l.strip()]
        header = lines[0]
        return f"cabecera: {header} | filas_de_datos: {len(lines)-1} | columnas: {len(header.split(','))}"
    except Exception as e:
        return f"error leyendo {path}: {e}"


# Importados con __import__ y guardados con alias privados por dos motivos: se
# usan aquí dentro (semantic_gate usa _np) y, sobre todo, se inyectan ya
# resueltos en el namespace del ejecutor, de modo que el código generado pueda
# usar np/pd/torch aunque olvide importarlos.
_np = __import__("numpy")
_pd = __import__("pandas")
_torch = __import__("torch")

code_executor = SimpleCodeExecutor(
    locals_dict={"csv_info": csv_info},
    globals_dict={
        "__builtins__": __builtins__,
        "np": _np,
        "pd": _pd,
        "pandas": _pd,
        "torch": _torch,
    },
)

# Agente de código de LlamaIndex: recibe una instrucción en lenguaje natural,
# escribe Python y lo ejecuta llamando a code_execute_fn. Se instancia una sola
# vez a nivel de módulo, pero cada uso crea su propio Context (memoria de
# conversación) para que un intento no arrastre el historial del anterior.
agent = CodeActAgent(
    code_execute_fn=code_executor.execute,
    llm=llm_coder,          # [M2] el agente de código usa el LLM de codificación
    tools=[csv_info],       # [M13] tool útil en lugar de aritméticas de ejemplo
)


# ============= PLANTILLA DE ENTRENAMIENTO (Tool determinista) ==============
# Esta cadena es el corazón del modo "template" [M1]: el programa completo de
# entrenamiento, escrito a mano y validado, del que el agente solo rellena los
# huecos (render_template). El LLM no interviene en su contenido, así que la
# diferencia de resultados entre los dos modos es atribuible a quién escribe el
# código y no a otra cosa.
#
# Los huecos son tokens __XXX__ en lugar de llaves {} porque el cuerpo lleva
# llaves de Python (dicts, format) y un str.format() se atragantaría con ellas.
#
# Qué hace el programa, en orden:
#   1. lee train.csv y parte las columnas en X (primeras N_IN) e y (N_OUT
#      siguientes) — el orden de columnas es el contrato con el paso 5;
#   2. separa 80/20 con random_state=42 y normaliza con z-score ajustado SOLO
#      con el tramo de entrenamiento;
#   3. grid-search sobre (aumento polinómico [M8] x arquitectura), cada modelo
#      con early stopping por validación, y se queda con el de menor MAE
#      normalizado de validación;
#   4. reentrena el ganador con TODOS los datos, tantas épocas como sugiera el
#      early stopping observado [M9], repitiendo con 5 semillas fijas;
#   5. promedia las 5 predicciones y publica media, desviación entre semillas y
#      el rango de y en una línea RESULT_JSON, que es lo que leen la puerta
#      semántica [M10/M11] y el harness.
#
# Determinismo: seeds fijas (42 en el grid; 0-4 en el refit), split fijo y cero
# llamadas al LLM => misma entrada, misma predicción bit a bit. Esa es la razón
# de que el modo template dé predicciones idénticas entre modelos distintos.
#
# NOTA de mantenimiento: el tag [M3] que aparece dentro de la plantilla
# corresponde a [M8] en esta numeración. No se reetiqueta a propósito: la cadena
# se escribe tal cual a los ficheros test_*.py de cada ejecución y se prefiere
# conservarla byte a byte idéntica a la de las tandas ya ejecutadas.
# ===TEMPLATE_BEGIN===
PYTORCH_TEMPLATE_V2 = r"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import copy
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

N_IN = __N_IN__
N_OUT = __N_OUT__
INPUT_VALUES = np.array([[__INPUT_VALUES__]], dtype=np.float64)
ARCHITECTURES = __ARCHS__
MAX_EPOCHS = __MAX_EPOCHS__
PATIENCE = __PATIENCE__
AUGMENT_MODE = "__AUGMENT_MODE__"   # auto | off
LR = 1e-3
WD = 1e-4
SEEDS = [0, 1, 2, 3, 4]

df = pd.read_csv("train.csv")
data = df.values.astype(np.float64)
X_all = data[:, :N_IN]
y_all = data[:, N_IN:N_IN + N_OUT]

# [M3] aumento polinómico de grado 2 (a ciegas; la validación decide)
def make_features(X, augment):
    if not augment:
        return X.astype(np.float32)
    pf = PolynomialFeatures(degree=2, include_bias=False)
    return pf.fit_transform(X).astype(np.float32)

def build_model(arch, n_in, n_out):
    layers = []
    prev = n_in
    for h in arch:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.SiLU())
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)

def train_model(model, Xtr, ytr, Xval, yval, max_epochs, patience):
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    best_epoch = max(50, max_epochs // 5)
    bad = 0
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xval), yval).item()
        sched.step(vloss)
        if vloss < best_val - 1e-7:
            best_val = vloss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vmae = torch.mean(torch.abs(model(Xval) - yval)).item()
    return model, vmae, best_epoch

X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

aug_options = [False, True] if (AUGMENT_MODE == "auto" and N_IN <= 6) else [False]

print("Grid-search (arquitectura x aumento):")
results = []
torch.manual_seed(42)
np.random.seed(42)
for aug in aug_options:
    Xtr_f = make_features(X_tr, aug)
    Xval_f = make_features(X_val, aug)
    sx = StandardScaler().fit(Xtr_f)
    sy = StandardScaler().fit(y_tr)
    Xtr_t = torch.from_numpy(sx.transform(Xtr_f).astype(np.float32))
    Xval_t = torch.from_numpy(sx.transform(Xval_f).astype(np.float32))
    ytr_t = torch.from_numpy(sy.transform(y_tr).astype(np.float32))
    yval_t = torch.from_numpy(sy.transform(y_val).astype(np.float32))
    n_feat = Xtr_t.shape[1]
    for arch in ARCHITECTURES:
        model = build_model(arch, n_feat, N_OUT)
        model, vmae, best_ep = train_model(model, Xtr_t, ytr_t, Xval_t, yval_t, MAX_EPOCHS, PATIENCE)
        print("aug={} arch={} val_mae_norm={:.6f} best_epoch={}".format(aug, arch, vmae, best_ep))
        results.append((aug, arch, vmae, best_ep))

results.sort(key=lambda r: r[2])
best_aug, best_arch, best_vmae, best_epoch = results[0]
print("Mejor: aug={} arch={} val_mae_norm={:.6f}".format(best_aug, best_arch, best_vmae))

# Refit con TODOS los datos, épocas dinámicas según early stopping del grid
Xfull_f = make_features(X_all, best_aug)
sx_full = StandardScaler().fit(Xfull_f)
sy_full = StandardScaler().fit(y_all)
Xfull_t = torch.from_numpy(sx_full.transform(Xfull_f).astype(np.float32))
yfull_t = torch.from_numpy(sy_full.transform(y_all).astype(np.float32))

inp_f = make_features(np.vstack([X_all, INPUT_VALUES]), best_aug)[-1:].astype(np.float64)
inp_t = torch.from_numpy(sx_full.transform(inp_f).astype(np.float32))

REFIT_EPOCHS = int(np.clip(best_epoch * 1.25, 100, 1500))
predictions = []
for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(best_arch, Xfull_t.shape[1], N_OUT)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(REFIT_EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(Xfull_t), yfull_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pn = model(inp_t).numpy()
    predictions.append(sy_full.inverse_transform(pn)[0])

predictions = np.array(predictions)
pred_mean = predictions.mean(axis=0)
pred_std = predictions.std(axis=0)

print("Prediccion media:", pred_mean)
print("Desviacion std:", pred_std)

result = {
    "pred_mean": [float(v) for v in pred_mean],
    "pred_std": [float(v) for v in pred_std],
    "best_arch": best_arch,
    "augment": bool(best_aug),
    "val_mae_norm": float(best_vmae),
    "best_epochs": int(best_epoch),
    "refit_epochs": int(REFIT_EPOCHS),
    "y_min": [float(v) for v in y_all.min(axis=0)],
    "y_max": [float(v) for v in y_all.max(axis=0)],
}
print("RESULT_JSON=" + json.dumps(result))
"""
# ===TEMPLATE_END===


def render_template(n_in: int, n_out: int, input_values_csv: str,
                    archs: List[List[int]], max_epochs: int, patience: int,
                    augment_mode: str) -> str:
    """Instancia la plantilla sustituyendo sus tokens __XXX__ por valores reales.

    Es la "Tool determinista" del modo template [M1]: dada la misma estructura y
    los mismos valores de entrada, devuelve exactamente el mismo programa.
    `archs`, `max_epochs` y `patience` llegan ampliados cuando se trata del
    reintento con presupuesto extra que concede la puerta semántica [M10].
    Los reemplazos son textuales y sin escapado: todos los valores son numéricos
    o listas de enteros serializadas con json.dumps, nunca texto del usuario.
    """
    t = PYTORCH_TEMPLATE_V2
    t = t.replace("__N_IN__", str(n_in))
    t = t.replace("__N_OUT__", str(n_out))
    t = t.replace("__INPUT_VALUES__", input_values_csv)
    t = t.replace("__ARCHS__", json.dumps(archs))
    t = t.replace("__MAX_EPOCHS__", str(max_epochs))
    t = t.replace("__PATIENCE__", str(patience))
    t = t.replace("__AUGMENT_MODE__", augment_mode)
    return t


# ============================== HELPERS ====================================
# Funciones deterministas y sin estado. Concentran todo lo que NO debe depender
# del LLM: lo que aquí se resuelve por reglas deja de ser una fuente de error
# ajena al objeto de estudio.

def _extract_csv_block(text: str) -> str:
    """[M7] Aísla el bloque CSV del enunciado y devuelve desde la cabecera hasta
    el final del texto.

    El extractor del prototipo buscaba nombres de columna concretos (los de la
    triangulación) y fallaba con cualquier otro enunciado. Aquí el criterio es
    genérico: la cabecera es la primera línea que "parece" una fila CSV.
    """
    lines = text.strip().split("\n")
    start = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        # Las líneas de redacción del enunciado también pueden llevar comas
        # ("Quiero predecir A, B y C"): se descartan por su comienzo.
        if low.startswith("quiero") or low.startswith("basandome") or low.startswith("basándome"):
            continue
        if "," not in s:
            continue
        toks = [t.strip() for t in s.split(",")]
        if len(toks) < 2:
            continue
        # Todos los campos deben ser "limpios" (palabras, espacios, puntos o
        # guiones): descarta frases con signos de puntuación, ':' o paréntesis y
        # deja pasar tanto la cabecera de nombres como las filas numéricas.
        if not all(t and re.match(r"^[\w\s\.\-]+$", t) for t in toks):
            continue
        start = i
        break
    # Desde la cabecera hasta el final: en estos enunciados el CSV siempre cierra
    # el mensaje, así que no hace falta detectar dónde termina.
    return "" if start < 0 else "\n".join(lines[start:])


# Línea "NOMBRE = VALOR" completa. Notas del patrón:
#   ([^=\n]+?)  nombre no ávido y sin '=' -> admite espacios y dígitos en el
#               nombre ("Sensor 3"), que es justo lo que rompía al prototipo [M5]
#   valor       entero, decimal o notación científica, con signo opcional
#   ^...$       ancla ambos extremos: una frase que mencione un '=' no cuela
_QLINE = re.compile(r"^\s*([^=\n]+?)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")


def parse_query_deterministic(msg: str) -> Tuple[List[str], List[float]]:
    """[M4/M5] Líneas 'NOMBRE = VALOR' del bloque de pregunta (antes de 'Basándome').

    Devuelve (nombres, valores) en el orden en que aparecen, que es el orden de
    columnas del CSV. Es la pieza sobre la que se apoyan dos mejoras: el nº de
    valores es el n_inputs de referencia del verificador [M4] y los valores son
    los que se inyectan en el código de entrenamiento [M5].

    El corte por 'Basándome' es esencial: sin él, las filas del CSV y cualquier
    '=' posterior contaminarían el recuento.
    """
    head = re.split(r"Bas[áa]ndome", msg, maxsplit=1)[0]   # con y sin tilde
    names, vals = [], []
    for line in head.split("\n"):
        m = _QLINE.match(line)
        if m:
            names.append(m.group(1).strip())
            try:
                vals.append(float(m.group(2)))
            except ValueError:
                # Inalcanzable con el patrón actual; se deja por seguridad para
                # que un valor raro no tumbe la extracción entera.
                pass
    return names, vals


def count_csv_cols(dataset: str) -> int:
    """Nº de columnas del CSV, leído de la primera línea con al menos 2 campos.

    Se usa como segunda referencia de estructura en el paso 2b: n_cols debe ser
    igual a n_inputs + n_outputs. Los campos vacíos no cuentan, así que una coma
    final sobrante no infla el recuento. Devuelve 0 si no hay nada parseable, y
    ese 0 desactiva la comprobación de la suma en lugar de dar un falso error.
    """
    for line in dataset.strip().split("\n"):
        s = line.strip()
        if not s:
            continue
        cols = [c for c in s.split(",") if c.strip()]
        if len(cols) >= 2:
            return len(cols)
    return 0


def sc_complete_int(llm, prompt: str, n_calls: int) -> Tuple[Optional[int], List[int]]:
    """[M3] Self-consistency: n_calls llamadas al LLM y voto por mayoría.

    Devuelve (ganador, votos). El ganador es None si no hubo ningún voto válido
    o si hay empate en la cabeza: un empate no es información, y es preferible
    delegar en el verificador determinista del paso 2b antes que quedarse con un
    valor arbitrario. La lista completa de votos se guarda en info: mide la
    (in)estabilidad del LLM y alimenta el Cap. 5.

    Con n_calls=1 el comportamiento es el del LLM sin más, pero a temperatura 0.
    """
    votes: List[int] = []
    for _ in range(n_calls):
        try:
            r = llm.complete(prompt)
            m = re.search(r"\d+", str(r).strip())   # primer entero de la respuesta
            if m:
                votes.append(int(m.group()))
        except Exception:
            # Una llamada caída no invalida la votación: se sigue con el resto.
            continue
    if not votes:
        return None, votes
    top = Counter(votes).most_common()          # [(valor, frecuencia), ...] desc.
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None, votes                      # empate -> sin decisión
    return top[0][0], votes


def _extract_numbers_from_str(text: str, n: int) -> Tuple[List[float], bool]:
    """Extractor bruto de n números (último recurso). Devuelve (valores, padded).

    Solo entra en juego cuando la extracción determinista no cuadró y hay que
    interpretar texto libre del LLM. Si faltan números, rellena con 0.0 y avisa
    con padded=True: ese flag es el que degrada la confianza a "low" [M5], para
    que un relleno inventado nunca pase por una predicción fiable.
    """
    if not text:
        return [0.0] * n, True
    # Dos alternativas: primero la forma con decimales (para no cortar "3.14" en
    # "3"), después la entera; ambas con exponente y signo opcionales.
    matches = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", text)
    nums: List[float] = []
    for m in matches:
        try:
            nums.append(float(m))
        except ValueError:
            continue
    if len(nums) >= n:
        return nums[:n], False        # sobran: se toman los n primeros, en orden
    return nums + [0.0] * (n - len(nums)), True


def parse_result_json(output: str) -> Optional[dict]:
    """Rescata el dict de la línea RESULT_JSON={...} del stdout del entrenamiento.

    Es el contrato de salida común a los dos modos [M1]: la plantilla la imprime
    siempre y al modo libre se le exige explícitamente en el prompt. Si no
    aparece o no es JSON válido, devuelve None y quien llama lo trata como fallo
    estructural (no hay resultado que medir).

    El '.*' es ávido a propósito: se queda con el cierre más lejano de la línea,
    de modo que un dict con dicts anidados no se corte por la primera llave.
    """
    m = re.search(r"RESULT_JSON=(\{.*\})", output)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def semantic_gate(res: dict) -> Tuple[bool, List[str]]:
    """[M10+M11] Puerta de calidad sobre RESULT_JSON: ¿la predicción es creíble?

    Que un programa termine sin excepción no significa que su resultado valga:
    puede devolver NaN, un número absurdo o una predicción que ni siquiera se
    parece a los datos. Esta función revisa el resultado con criterios que no
    necesitan conocer la solución (el ground truth jamás llega al agente) y
    devuelve (todo_ok, lista_de_problemas). Las etiquetas de la lista son las
    que acaban en el log y sostienen el análisis de calibración del Cap. 5.

    Comprobaciones, en orden:
      result_json_incompleto     falta alguna clave o no es convertible a float
      nan_o_inf                  la red divergió
      prediccion_fuera_de_rango  fuera de [ymin, ymax] ampliado media anchura a
                                 cada lado; el margen del 50 % es deliberado:
                                 los queries son de interpolación, así que una
                                 predicción muy lejos del rango observado
                                 delata un fallo, no una extrapolación legítima
      val_mae_norm_mayor_que_1   el error de validación normalizado supera 1,
                                 es decir, la red predice peor que la media
      dispersion_seeds_alta      [M11] las 5 semillas no coinciden
    """
    issues: List[str] = []
    try:
        pred = _np.array(res["pred_mean"], dtype=float)
        std = _np.array(res["pred_std"], dtype=float)
        ymin = _np.array(res["y_min"], dtype=float)
        ymax = _np.array(res["y_max"], dtype=float)
    except Exception:
        # Sin las cuatro claves no hay nada que juzgar: se rechaza de entrada.
        return False, ["result_json_incompleto"]
    if not _np.all(_np.isfinite(pred)) or not _np.all(_np.isfinite(std)):
        issues.append("nan_o_inf")
    # Rango observado de cada salida, con un suelo positivo para no dividir
    # entre 0 cuando una salida es constante (ymax == ymin).
    rng = _np.maximum(ymax - ymin, 1e-9 + 1e-6 * _np.abs(ymax))
    lo, hi = ymin - 0.5 * rng, ymax + 0.5 * rng
    if _np.any(pred < lo) or _np.any(pred > hi):
        issues.append("prediccion_fuera_de_rango")
    if float(res.get("val_mae_norm", 0.0)) > 1.0:
        issues.append("val_mae_norm_mayor_que_1")
    # [M11] dispersión entre semillas relativa al rango de y (no a |pred|):
    # con |pred| cercana a 0 saltaban falsas alarmas aunque la predicción fuera buena.
    # Se toma el máximo sobre las salidas: basta con que UNA sea inestable.
    rel = float(_np.max(std / rng))
    if rel > 0.25:      # umbral: >25 % del rango de y entre semillas es inaceptable
        issues.append("dispersion_seeds_alta")
    return (len(issues) == 0), issues


# Excepciones que casi siempre significan "el nº de columnas no era el que creía".
STRUCTURAL_ERR_TYPES = {"IndexError", "KeyError"}


def is_structural_error(err_type: str, err_msg: str) -> bool:
    """[M12] ¿El error sugiere que la estructura inferida está mal?

    Decide si merece la pena retroceder al paso 1. El criterio separa dos
    familias que el prototipo confundía:
      - estructurales (columna inexistente, tensores que no encajan): volver a
        contar entradas y salidas puede arreglarlos;
      - de entorno (falta un paquete, no existe un fichero): retroceder no
        arregla nada y solo quema presupuesto de tiempo.

    ValueError y RuntimeError son demasiado genéricos para clasificarlos por el
    tipo, así que se miran palabras clave del mensaje (shape, dimension, column,
    broadcast, size mismatch) en minúsculas.
    """
    if err_type in STRUCTURAL_ERR_TYPES:
        return True
    low = (err_msg or "").lower()
    if err_type == "ValueError" and any(k in low for k in ("shape", "dimension", "column", "broadcast")):
        return True
    if err_type == "RuntimeError" and ("size mismatch" in low or "shapes" in low):
        return True
    return False


def run_code_captured(code: str) -> Tuple[bool, str, str, str, str]:
    """Ejecuta el código de entrenamiento y captura su salida.

    Devuelve (ok, stdout, err_type, err_msg, err_trace). ESTA es la ejecución
    que cuenta para el estudio, la que produce el RESULT_JSON que se mide, y es
    idéntica en los dos modos [M1]: el código de la plantilla y el escrito por
    el LLM pasan exactamente por la misma puerta.

    Detalles que importan:
      - namespace nuevo en cada llamada, con __name__ == "__main__" para que se
        ejecuten los bloques `if __name__ == "__main__":` que el LLM suele
        escribir. Al no reutilizar el namespace del CodeActAgent, un intento no
        puede aprovecharse de variables que dejó otro;
      - el stdout se redirige a memoria: el print de RESULT_JSON no debe
        mezclarse con la traza del agente;
      - si hay excepción, stdout se descarta y se devuelven tipo, mensaje y
        traza, que son los tres ingredientes del prompt de corrección del paso 7.
    """
    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            exec(code, {"__name__": "__main__"})
        return True, f.getvalue(), "", "", ""
    except Exception as e:
        et = type(e).__name__
        tr = traceback.format_exc()
        return False, "", et, f"{et}: {e}", tr


async def run_agent_and_get_result(agent, ctx, query: str, log_code: bool = False) -> str:
    """Lanza el CodeActAgent sobre `query` y devuelve su respuesta como texto.

    Además del texto, tiene un efecto lateral que es el motivo real de existir:
    deja en la global `last_code` el ÚLTIMO código que el agente pidió ejecutar.
    Los pasos 6 y 7 lo recogen de ahí para volver a ejecutarlo con
    run_code_captured() y quedarse con su stdout.

    Consume el stream de eventos del agente:
      ToolCall        el LLM pide ejecutar código -> se guarda en last_code
      ToolCallResult  salida de esa ejecución    -> se acumula en la respuesta
      AgentStream     texto que va generando el LLM (deltas)

    max_iterations=5 acota el diálogo interno: sin tope, un modelo pequeño puede
    quedarse dando vueltas hasta agotar el timeout de la celda.

    Tolerancia a fallos: si el agente revienta a media conversación se devuelve
    `snapshot`, el texto acumulado ANTES del último evento procesado. Es una
    respuesta parcial, pero suele bastar para que el paso 7 encuentre un bloque
    ```python y siga trabajando en lugar de perder la ejecución entera.
    """
    print("[CodeAct] Iniciando ejecución del agente...")
    global last_code
    result_parts = []
    snapshot = ""
    it = 0
    try:
        handler = agent.run(query, ctx=ctx, max_iterations=5)
        async for event in handler.stream_events():
            snapshot = "".join(str(p) for p in result_parts)   # copia previa por si el stream muere
            if isinstance(event, ToolCallResult):
                result_parts.append(event.tool_output)
            elif isinstance(event, ToolCall):
                code = event.tool_kwargs.get("code", "")       # vacío si la tool no es la de código
                if code:
                    last_code = code                            # <- el efecto lateral que se consume fuera
                    it += 1
                    if log_code:
                        print(f"\n[CodeAct] Código generado (it {it}):\n{code}\n")
                    # No re-ejecutar aquí: el CodeActAgent ya ejecuta cada tool
                    # call vía code_execute_fn y su salida llega en ToolCallResult.
                    # (Re-ejecutar duplicaba entrenamientos y efectos secundarios.)
            elif isinstance(event, AgentStream):
                if event.delta:
                    result_parts.append(event.delta)
        response = await handler        # espera al final y recoge la respuesta completa
        return str(response)
    except Exception as e:
        print(f"[CodeAct] Excepción en iteración {it}: {type(e).__name__}")
        if snapshot:
            return snapshot
        return "".join(str(p) for p in result_parts) if result_parts else f"Error: {e}"


# ============================== WORKFLOW ===================================
class NeuralNetAgentV2(Workflow):
    """El agente: siete pasos encadenados por eventos.

    Los pasos no se llaman entre sí; cada @step consume un tipo de evento y
    emite otro, y el motor de LlamaIndex hace el enrutado. El camino feliz es
    1 -> 2 -> 2b -> 3 -> 4 -> 5 -> 6 -> fin, con dos ciclos de recuperación:

      * estructura incoherente: 2b -> Step2bRetryEvent -> 1 (hasta MAX_2B_RETRIES,
        luego fallback determinista) [M4]
      * el código falla al ejecutarse: 6 -> 7 -> 7 ... (hasta MAX_PASO7_ITER) y,
        solo si el error es estructural, 7 -> BacktrackEvent -> 1 [M12]

    Reparto de papeles: el LLM decide la ESTRUCTURA (pasos 1/2) y, en modo libre
    o al corregir, escribe CÓDIGO; todo lo demás (aislar el CSV, extraer valores,
    entrenar, juzgar el resultado) es determinista. Esa frontera es lo que el
    Cap. 5 mide.

    Estado de la instancia (fuera de los eventos, porque debe sobrevivir a los
    ciclos): dos contadores de recuperación y el diccionario `info`, que es el
    registro de la ejecución y se vuelca tal cual a log.jsonl [M14].
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paso7_executions = 0   # veces que se ha entrado a corregir código
        self.backtrack_count = 0    # regresos al paso 1 ya consumidos (tope MAX_BACKTRACKS)
        # Telemetría de la ejecución. Cada clave responde a una pregunta del
        # análisis, así que se rellena aunque el camino termine en fallo:
        self.info: Dict[str, Any] = {
            # --- configuración con la que se ejecutó (para reproducir la celda)
            "mode": GENERATION_MODE, "model": MODEL_NAME, "sc_n": SC_N,
            "augment_mode": AUGMENT_MODE,
            # --- qué dijo el LLM al contar la estructura: una lista de votos por
            #     intento (crece con cada reintento/backtrack) [M3]
            "llm_n_in_votes": [], "llm_n_out_votes": [],
            # --- qué decía el verificador determinista y el CSV [M4]
            "det_n_in": None, "n_cols": None,
            # --- estructura con la que finalmente se entrenó
            "n_in_final": None, "n_out_final": None,
            # --- ¿hubo que rescatar al LLM? extraction_llm_ok guarda el acierto
            #     del PRIMER intento (no se sobrescribe después): es la métrica
            #     limpia de fiabilidad del modelo contando variables
            "extraction_fallback_used": False, "extraction_llm_ok": None,
            # --- de dónde salieron los valores del query: deterministic | llm |
            #     raw_head; values_padded avisa de relleno con ceros [M5]
            "input_source": None, "values_padded": False,
            # --- esfuerzo de recuperación y veredicto de la puerta semántica
            "retries_2b": 0, "semantic_retries": 0, "semantic_issues": [],
            # --- confianza declarada (high | low | failed) y resultado crudo.
            #     Arranca en "high" y solo puede degradarse; el harness conserva
            #     estas etiquetas para el análisis de calibración [M11]
            "confidence": "high", "result_json": None,
        }

    # ------------- pasos 1 y 2: el LLM medido, con self-consistency --------
    # Los prompts se construyen en métodos aparte para que el camino normal y el
    # de reintento usen literalmente el mismo texto, con la única diferencia del
    # bloque de feedback. Si divergieran, el reintento dejaría de ser comparable.
    def _prompt_n_inputs(self, msg: str, feedback: str = "") -> str:
        """Prompt del paso 1: cuántas columnas son entradas.

        Describe la regla ("los inputs son los que aparecen como 'nombre =
        valor'") sin nombrar ninguna variable de ningún problema concreto [M6],
        y exige responder solo con un entero para que la respuesta sea parseable
        sin ambigüedad.
        """
        fb = f"\n\nFEEDBACK DEL INTENTO ANTERIOR: {feedback}\n" if feedback else ""
        # [M6] sin ejemplos de un problema concreto
        return (
            "En el siguiente mensaje del usuario hay una pregunta sobre predecir algo usando datos en formato CSV. "
            "La primera línea de datos es la cabecera con nombres de columnas. "
            "Las columnas de ENTRADA (inputs) son aquellas cuyos valores aparecen listados en la pregunta, "
            "una por línea con el formato 'nombre = valor'."
            + fb +
            "\nResponde SOLO con un número entero: cuántas columnas son entradas (inputs) de la red.\n\nMensaje:\n" + msg
        )

    def _prompt_n_outputs(self, msg: str, feedback: str = "") -> str:
        """Prompt del paso 2: cuántas columnas son salidas. Mismo formato de
        respuesta que el del paso 1; la pista es la frase 'Quiero predecir' del
        enunciado, común a los diez problemas."""
        fb = f"\n\nFEEDBACK DEL INTENTO ANTERIOR: {feedback}\n" if feedback else ""
        return (
            "En el siguiente mensaje hay datos CSV. Las columnas de SALIDA (objetivo) son las que el usuario "
            "quiere predecir, mencionadas al inicio tras 'Quiero predecir'."
            + fb +
            "\nResponde SOLO con un número entero: cuántas columnas son salidas (outputs) de la red.\n\nMensaje:\n" + msg
        )

    @step
    async def paso1_num_entradas(self, ev: NeuralNetStartEvent) -> Step1Event:
        """Paso 1: nº de entradas vía LLM. El LLM se mantiene aquí a propósito,
        porque es parte del objeto medido: los modelos pequeños fallan con
        frecuencia ya en el recuento de variables (observación temprana de la
        línea de investigación). Su fiabilidad se contrasta en el paso 2b con
        el verificador determinista. [M3]

        El enunciado se recorta a 3000 caracteres: la cabecera y las primeras
        filas bastan para contar columnas, y así el prompt no se dispara con los
        problemas de CSV largo.
        """
        print(f"[Paso 1] n_inputs vía LLM (self-consistency x{SC_N})...")
        msg = (ev.user_message or "").strip()[:3000]
        n, votes = sc_complete_int(llm_extract, self._prompt_n_inputs(msg), SC_N)
        self.info["llm_n_in_votes"].append(votes)   # se registra el intento aunque no haya ganador
        if n is None:
            n = 0  # se resolverá en 2b con el verificador determinista
        print(f"[Paso 1] votos={votes} -> n_inputs={n}")
        return Step1Event(user_message=ev.user_message, n_inputs=n)

    @step
    async def paso1_retry(self, ev: Step2bRetryEvent) -> Step1Event:
        """Paso 1 bis: mismo trabajo que paso1_num_entradas, pero con el feedback
        del intento fallido dentro del prompt.

        Existe como paso separado porque el enrutado es por tipo de evento: este
        consume Step2bRetryEvent y aquel NeuralNetStartEvent. Propaga feedback y
        retry_count al Step1Event para que el contador de reintentos no se
        reinicie al pasar por los pasos 2 y 2b.
        """
        print(f"[Paso 1 · retry {ev.retry_count}] feedback: {ev.feedback}")
        msg = (ev.user_message or "").strip()[:3000]
        n, votes = sc_complete_int(llm_extract, self._prompt_n_inputs(msg, ev.feedback), SC_N)
        self.info["llm_n_in_votes"].append(votes)
        if n is None:
            n = 0
        return Step1Event(user_message=ev.user_message, n_inputs=n,
                          feedback=ev.feedback, retry_count=ev.retry_count)

    @step
    async def paso_backtrack(self, ev: BacktrackEvent) -> Step2bRetryEvent:
        """[M12] Backtracking: el prototipo nunca podía volver a los pasos
        iniciales (limitación señalada por el tutor); aquí se reabre la
        inferencia de estructura aportando el feedback del fallo.

        Traduce el BacktrackEvent en un Step2bRetryEvent con retry_count=0: el
        presupuesto de reintentos de estructura se concede entero de nuevo, ya
        que la evidencia que motiva el regreso (el código falló por formas) es
        distinta de la que agotó los reintentos anteriores. El tope global lo
        pone self.backtrack_count contra MAX_BACKTRACKS, comprobado en el paso 7.
        """
        print(f"[Backtrack {self.backtrack_count}] hacia paso 1. Feedback: {ev.feedback}")
        return Step2bRetryEvent(user_message=ev.user_message, feedback=ev.feedback, retry_count=0)

    @step
    async def paso2_num_salidas(self, ev: Step1Event) -> Step2Event:
        """Paso 2: nº de salidas vía LLM, con la misma mecánica de votación que
        el paso 1 [M3]. Se pregunta por separado (y no "dame ambos números") para
        que cada recuento tenga su propia votación y su propio registro de votos.
        """
        print(f"[Paso 2] n_outputs vía LLM (self-consistency x{SC_N})...")
        msg = (ev.user_message or "").strip()[:3000]
        n, votes = sc_complete_int(llm_extract, self._prompt_n_outputs(msg, ev.feedback), SC_N)
        self.info["llm_n_out_votes"].append(votes)
        if n is None:
            n = 0
        print(f"[Paso 2] votos={votes} -> n_outputs={n}")
        return Step2Event(user_message=ev.user_message, n_inputs=ev.n_inputs, n_outputs=n,
                          feedback=ev.feedback, retry_count=ev.retry_count)

    # ---------------- paso 2b: validación + verificador determinista -------
    @step
    async def paso2b_validar_consistencia(self, ev: Step2Event) -> Step3Event | Step2bRetryEvent:
        """Paso 2b: doble verificación de estructura. La suma n_in+n_out==n_cols
        sola dejaba pasar fallos silenciosos (p.ej. defaults 3+2 en un problema
        4+1); el verificador determinista (nº de líneas 'nombre = valor' del
        enunciado) cierra ese hueco. Retry con feedback -> fallback determinista. [M4]

        Es el paso que sostiene la validez del estudio: sin él, un recuento malo
        no da error, da una predicción plausible y equivocada. Tres desenlaces:
          - coherente            -> Step3Event y a seguir
          - incoherente y quedan reintentos -> Step2bRetryEvent con feedback
          - incoherente y agotados -> fallback determinista (o, si ni eso, los
            números del LLM saneados a >= 1, dejando constancia)
        """
        print("[Paso 2b] Validando estructura (suma + verificador determinista)...")
        # Si no se reconoce el bloque CSV se usa el mensaje entero: peor recuento,
        # pero nunca cadena vacía (que sí dejaría el resto del flujo sin datos).
        dataset = _extract_csv_block(ev.user_message or "") or (ev.user_message or "")
        n_cols = count_csv_cols(dataset)
        det_names, det_vals = parse_query_deterministic(ev.user_message or "")
        det_n_in = len(det_vals) if det_vals else None
        # El verificador solo es utilizable si encontró valores Y son menos que
        # las columnas del CSV (si no, no quedaría ninguna columna de salida).
        det_ok_possible = det_n_in is not None and n_cols > det_n_in > 0

        self.info["n_cols"] = n_cols
        self.info["det_n_in"] = det_n_in

        # Comprobación 1: las partes suman el total y ninguna es 0.
        sum_ok = n_cols > 0 and (ev.n_inputs + ev.n_outputs == n_cols) and ev.n_inputs > 0 and ev.n_outputs > 0
        # Comprobación 2: el LLM coincide con el verificador. Si no hay
        # verificador utilizable se da por buena en vez de bloquear el flujo.
        det_match = (not det_ok_possible) or (ev.n_inputs == det_n_in)
        llm_ok = sum_ok and det_match
        # Solo el PRIMER veredicto se guarda: mide si el LLM acertó sin ayuda.
        # Los reintentos ya llevan feedback, así que no serían comparables.
        if self.info["extraction_llm_ok"] is None:
            self.info["extraction_llm_ok"] = bool(llm_ok)

        print(f"[Paso 2b] llm=({ev.n_inputs},{ev.n_outputs}) n_cols={n_cols} det_n_in={det_n_in} -> ok={llm_ok}")

        if llm_ok:
            return Step3Event(user_message=ev.user_message, n_inputs=ev.n_inputs,
                              n_outputs=ev.n_outputs, dataset=dataset)

        if ev.retry_count < MAX_2B_RETRIES:
            self.info["retries_2b"] += 1
            # El feedback es concreto y verificable (columnas reales, valores
            # realmente listados): a un modelo pequeño, "vuelve a intentarlo" no
            # le sirve de nada, pero un conteo explícito sí suele reconducirlo.
            fb = (f"Tu respuesta dio n_inputs={ev.n_inputs} y n_outputs={ev.n_outputs}, pero el CSV tiene "
                  f"{n_cols} columnas")
            if det_ok_possible:
                fb += (f" y en la pregunta hay exactamente {det_n_in} valores de entrada listados "
                       f"('nombre = valor'), por lo que n_inputs debe ser {det_n_in} y "
                       f"n_outputs {n_cols - det_n_in}")
            fb += ". Vuelve a contar con cuidado."
            return Step2bRetryEvent(user_message=ev.user_message, feedback=fb,
                                    retry_count=ev.retry_count + 1)

        # [M4] fallback determinista tras agotar reintentos (antes: continuar con
        # valores malos). El LLM ya ha fallado con feedback: se prefiere la
        # estructura que dictan el enunciado y el CSV, y queda registrado en
        # extraction_fallback_used para poder contar cuántas veces hizo falta.
        if det_ok_possible:
            self.info["extraction_fallback_used"] = True
            n_in, n_out = det_n_in, n_cols - det_n_in
            print(f"[Paso 2b] Fallback determinista -> ({n_in},{n_out})")
            return Step3Event(user_message=ev.user_message, n_inputs=n_in,
                              n_outputs=n_out, dataset=dataset)
        # Caso extremo: ni verificador ni suma coherente (enunciado con formato
        # inesperado). Se sigue con lo del LLM, saneado a >= 1 porque un 0 haría
        # imposible construir la red; si está mal, el error saldrá en el paso 6/7.
        print("[Paso 2b] Sin verificador disponible; continúo con valores del LLM (flag).")
        return Step3Event(user_message=ev.user_message, n_inputs=max(1, ev.n_inputs),
                          n_outputs=max(1, ev.n_outputs), dataset=dataset)

    # ------------------------- pasos 3, 4 y 5 ------------------------------
    @step
    async def paso3_aislar_dataset(self, ev: Step3Event) -> Step4Event:
        """Paso 3: fija el bloque CSV que se usará como datos de entrenamiento.

        El paso 2b ya lo aisló para contar columnas; aquí se reutiliza y solo se
        recalcula si viniera vacío (defensa, no camino normal). Es también el
        punto donde la estructura se da por definitiva: se copia a n_in_final /
        n_out_final, que es lo que el análisis compara con la estructura real.
        """
        print("[Paso 3] Aislando bloque CSV...")
        dataset = ev.dataset or _extract_csv_block(ev.user_message or "") or (ev.user_message or "")
        self.info["n_in_final"] = ev.n_inputs
        self.info["n_out_final"] = ev.n_outputs
        return Step4Event(user_message=ev.user_message, n_inputs=ev.n_inputs,
                          n_outputs=ev.n_outputs, dataset=dataset,
                          question_summary="", input_values_str="")

    @step
    async def paso4_aislar_pregunta_y_valores(self, ev: Step4Event) -> Step5Event:
        """Paso 4: qué hay que predecir y con qué valores de entrada.

        Se le pregunta al LLM (sigue siendo objeto de medida) pero su respuesta
        NO es la primera opción para los valores: la prioridad es
        determinista -> llm -> raw_head [M5], y la fuente elegida se anota en
        info["input_source"]. Basta con que el enunciado tenga tantas líneas
        'nombre = valor' como entradas para que el LLM quede fuera del camino
        crítico, que es lo habitual en los diez problemas.
        """
        print("[Paso 4] Pregunta y valores (LLM medido + determinista)...")
        msg = (ev.user_message or "").strip()[:4000]
        prompt = (
            "Del siguiente mensaje extrae:\n"
            "1) La pregunta concreta (qué variable o variables quiere predecir el usuario).\n"
            "2) Los valores de ENTRADA que da el usuario para la predicción (las líneas 'nombre = valor').\n"
            "Responde en dos líneas:\n"
            "PREGUNTA: <qué hay que predecir>\n"
            "VALORES_ENTRADA: <solo los valores numéricos, separados por comas, en el mismo orden>\n\nMensaje:\n" + msg
        )
        question_summary, llm_values = "", ""
        try:
            text = str(llm_extract.complete(prompt)).strip()
            for line in text.split("\n"):
                # Parseo tolerante: se acepta cualquier línea que empiece por
                # PREGUNTA o que mencione VALORES, porque los modelos pequeños
                # rara vez respetan el formato al pie de la letra.
                if line.strip().upper().startswith("PREGUNTA"):
                    question_summary = line.split(":", 1)[-1].strip()
                elif "VALORES" in line.upper():
                    llm_values = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
        except Exception:
            pass   # el LLM es opcional aquí: si falla, manda la vía determinista

        det_names, det_vals = parse_query_deterministic(ev.user_message or "")
        # Preferencia 1: los valores del enunciado, si su número coincide
        # exactamente con las entradas ya validadas. El formato '.10g' conserva
        # las 6 cifras significativas del dataset sin arrastrar ruido binario.
        if len(det_vals) == ev.n_inputs:
            input_values_str = ", ".join(f"{v:.10g}" for v in det_vals)   # [M5]
            self.info["input_source"] = "deterministic"
        # Preferencia 2: lo que dijo el LLM (se normalizará luego con
        # _extract_numbers_from_str, que puede rellenar y marcar values_padded).
        elif llm_values:
            input_values_str = llm_values
            self.info["input_source"] = "llm"
        # Preferencia 3: el texto crudo anterior a 'Basándome', del que después
        # se sacarán los números a la brava. Último recurso.
        else:
            input_values_str = msg.split("Basándome", 1)[0][:500]
            self.info["input_source"] = "raw_head"
        if not question_summary:
            question_summary = "la variable objetivo indicada en la pregunta"  # [M6] sin defaults de un problema concreto

        print(f"[Paso 4] fuente_valores={self.info['input_source']}")
        return Step5Event(user_message=ev.user_message, n_inputs=ev.n_inputs, n_outputs=ev.n_outputs,
                          dataset=ev.dataset, question_summary=question_summary,
                          input_values_str=input_values_str, training_data="")

    @step
    async def paso5_datos_entrenamiento(self, ev: Step5Event) -> Step6StartEvent:
        """Paso 5: vuelca el bloque CSV a train.csv en el directorio de trabajo.

        A partir de aquí los datos viajan por disco y no por el prompt: el código
        de entrenamiento (plantilla o generado) lee siempre "train.csv". Ese
        nombre fijo es el contrato con los dos modos [M1] y con la tool csv_info.
        El harness ejecuta cada celda en su propio directorio, así que no hay
        riesgo de que una ejecución pise el train.csv de otra.
        """
        print("[Paso 5] Guardando train.csv...")
        with open("train.csv", "w", encoding="utf-8") as f:
            f.write(ev.dataset or "")
        return Step6StartEvent(user_message=ev.user_message, n_inputs=ev.n_inputs,
                               n_outputs=ev.n_outputs, dataset=ev.dataset,
                               question_summary=ev.question_summary,
                               input_values_str=ev.input_values_str, training_data="train.csv")

    # --------------- paso 6: template | libre + gate semántico -------------
    def _input_values_csv(self, ev) -> Tuple[str, bool]:
        """Normaliza los valores del query a exactamente n_inputs números en una
        cadena "valor1, valor2, ...", lista para inyectar en el código.

        Cierra el paso 4: venga la cadena de donde venga (determinista, LLM o
        texto crudo), aquí queda con la longitud correcta. El flag padded es
        pegajoso —se acumula con OR en info— porque un relleno con ceros en
        cualquier intento invalida la confianza de toda la ejecución [M5].
        """
        vals, padded = _extract_numbers_from_str(ev.input_values_str, ev.n_inputs)
        self.info["values_padded"] = self.info["values_padded"] or padded
        return ", ".join(f"{v:.10g}" for v in vals), padded

    def _freegen_prompt(self, ev, values_csv: str) -> str:
        """Modo libre: prompt 'espejo' con requisitos de protocolo equivalentes a la
        Tool (normalización, split, selección por validación, multi-semilla,
        RESULT_JSON). Deliberadamente no exige el aumento de características [M8]:
        ese conocimiento embebido es parte de lo que la ablación template-vs-libre mide.

        La comparación es justa porque el prompt pide el MISMO protocolo que
        implementa la plantilla, punto por punto; lo que no se le da es la
        implementación. Dos restricciones son puramente operativas: "sin clases"
        (los modelos pequeños producen clases a medio escribir que ni siquiera
        parsean) y "no inventes datos", contra la tendencia a fabricar un CSV de
        ejemplo en lugar de leer train.csv.
        """
        return (
            "Escribe y ejecuta un programa Python COMPLETO que haga lo siguiente:\n"
            f"1) Lee train.csv (CSV con cabecera). Las primeras {ev.n_inputs} columnas son entradas "
            f"y las {ev.n_outputs} siguientes son salidas.\n"
            "2) Normaliza entradas y salidas con z-score (StandardScaler).\n"
            "3) Separa 80/20 entrenamiento/validación (random_state=42).\n"
            "4) Prueba al menos 3 arquitecturas MLP distintas en PyTorch (nn.Sequential, sin clases) "
            "con early stopping por validación, y elige la de menor error de validación.\n"
            "5) Reentrena la mejor con TODOS los datos con 5 semillas (0,1,2,3,4) y promedia las predicciones.\n"
            f"6) Predice para los valores de entrada: [{values_csv}] (desnormaliza la salida).\n"
            "7) Imprime al final UNA única línea EXACTAMENTE con este formato:\n"
            'RESULT_JSON={"pred_mean": [..], "pred_std": [..], "best_arch": [..], "augment": false, '
            '"val_mae_norm": .., "best_epochs": .., "y_min": [..], "y_max": [..]}\n'
            "No uses clases de Python. No inventes datos: los datos reales están en train.csv."
        )

    async def _obtain_code_and_run(self, ev, budget_ext: bool, issues_fb: str = "") -> Tuple[bool, str, str, str, str, str]:
        """Devuelve (ok, stdout, code, err_type, err_msg, err_trace) según el modo.
        issues_fb: en modo libre, feedback de la puerta semántica para el reintento
        (contrapartida del presupuesto ampliado de la Tool: misma oportunidad de mejora).

        Aquí es donde se bifurca la ablación [M1] y donde vuelve a juntarse: cambia
        el ORIGEN del código (plantilla instanciada vs. LLM) pero no su ejecución
        ni su medición, que son comunes.

        budget_ext=True traduce "reintento con más presupuesto" a cada modo de la
        forma que le corresponde: a la plantilla, más arquitecturas, un 50 % más
        de épocas y 25 de paciencia extra; al modo libre, el feedback de lo que
        salió mal, que es la única palanca que tiene un generador de código.
        """
        global last_code
        values_csv, _ = self._input_values_csv(ev)
        if GENERATION_MODE == "template":
            code = render_template(
                ev.n_inputs, ev.n_outputs, values_csv,
                ARCHS_EXT if budget_ext else ARCHS_BASE,
                int(MAX_EPOCHS_BASE * (1.5 if budget_ext else 1.0)),
                PATIENCE_BASE + (25 if budget_ext else 0),
                "auto" if (AUGMENT_MODE == "auto") else "off",
            )
        else:  # libre
            # Context nuevo por intento: el reintento debe partir del mismo punto
            # que el primero, más el feedback, y no arrastrar la conversación.
            local_ctx = Context(agent)
            fp = self._freegen_prompt(ev, values_csv)
            if issues_fb:
                fp += ("\n\nATENCIÓN: un intento anterior produjo un resultado con problemas: "
                       + issues_fb + ". Corrige el programa para evitarlos.")
            # La respuesta en texto se descarta: lo que interesa es el efecto
            # lateral sobre last_code (el código que el agente pidió ejecutar).
            _ = await run_agent_and_get_result(agent, local_ctx, fp, log_code=True)
            code = last_code or ""
            if not code.strip():
                # El LLM contestó sin escribir código: fallo declarado, no un
                # error de ejecución. Es el desenlace típico del modo libre con
                # modelos pequeños y por eso tiene su propio tipo de error.
                return False, "", "", "NoCode", "El agente no produjo código", ""
        # Artefacto de auditoría: queda en disco el código EXACTO que se ejecutó,
        # con marca de tiempo en milisegundos para no pisar el de otro intento.
        with open(f"test_{int(time.time() * 1000)}.py", "w", encoding="utf-8") as f:
            f.write(code)
        # Ejecución canónica y medida, idéntica para los dos modos.
        ok, out, et, em, tr = run_code_captured(code)
        return ok, out, code, et, em, tr

    @step
    async def paso6_entrenar_y_predecir(self, ev: Step6StartEvent) -> StopEvent | Step6Event:
        """Paso 6: obtiene el código de entrenamiento según GENERATION_MODE
        (Tool determinista o generación libre del LLM; ablación de una sola
        variable [M1]) y lo ejecuta. Integra el paso 6b: puerta semántica
        [M10/M11] con un reintento ampliado. Los errores de ejecución van al
        paso 7; los semánticos se resuelven aquí.

        La separación entre los dos tipos de fallo es deliberada: que el programa
        reviente es un problema de código (lo arregla el paso 7 reescribiéndolo);
        que devuelva un número increíble es un problema de calidad (se ataca
        reentrenando con más presupuesto, no reescribiendo).

        El bucle da como mucho 1 + MAX_SEMANTIC_RETRIES vueltas y sale siempre
        por un `return`. Desenlaces posibles:
          Step6Event -> al paso 7   (excepción, o salida sin RESULT_JSON)
          StopEvent  confidence=high (gate OK)
          StopEvent  confidence=low  (gate OK pero valores rellenados, o issues
                                      que persisten tras el reintento)
        """
        print(f"[Paso 6 · modo={GENERATION_MODE}] Construyendo y ejecutando entrenamiento...")
        semantic_retries = 0
        budget_ext = False      # el primer intento va con presupuesto normal
        issues_fb = ""
        while True:
            ok, out, code, et, em, tr = await self._obtain_code_and_run(ev, budget_ext, issues_fb)
            if not ok:
                # Error de ejecución -> no se reintenta aquí: es trabajo del paso 7.
                print(f"[Paso 6] Error de ejecución: {em}")
                return Step6Event(user_message=ev.user_message, n_inputs=ev.n_inputs,
                                  n_outputs=ev.n_outputs, dataset=ev.dataset,
                                  question_summary=ev.question_summary,
                                  input_values_str=ev.input_values_str, training_data="train.csv",
                                  code_to_execute=code, iteration=1,
                                  error_type=et, error_msg=em, error_traceback=tr)
            res = parse_result_json(out)
            if res is None:
                # El programa terminó bien pero no publicó el resultado en el
                # formato acordado: sin RESULT_JSON no hay nada que medir, así
                # que se trata como fallo y se manda a corregir al paso 7.
                print("[Paso 6] Sin RESULT_JSON parseable -> tratado como fallo estructural.")
                return Step6Event(user_message=ev.user_message, n_inputs=ev.n_inputs,
                                  n_outputs=ev.n_outputs, dataset=ev.dataset,
                                  question_summary=ev.question_summary,
                                  input_values_str=ev.input_values_str, training_data="train.csv",
                                  code_to_execute=code, iteration=1,
                                  error_type="NoResultJSON",
                                  error_msg="La salida no contiene la línea RESULT_JSON={...} requerida",
                                  error_traceback="")
            ok_gate, issues = semantic_gate(res)   # [M10+M11]
            if ok_gate and self.info["values_padded"]:
                # [M5] valores de entrada incompletos rellenados con 0.0: la
                # predicción no es fiable aunque el gate numérico pase. El gate
                # solo mira el resultado; este caso es un defecto de la ENTRADA,
                # invisible desde el RESULT_JSON, y por eso se comprueba aparte.
                self.info["result_json"] = res
                self.info["confidence"] = "low"
                self.info["semantic_issues"].append(["valores_entrada_incompletos"])
                print("[Paso 6b] Gate OK pero valores de entrada incompletos -> confidence=low.")
                # Las marcas CONFIDENCE=/ISSUES= se anexan al stdout devuelto:
                # así viajan en la respuesta que lee el harness, además de en el log.
                return StopEvent(result=out + "\nCONFIDENCE=low ISSUES=valores_entrada_incompletos")
            if ok_gate:
                # Camino feliz: única salida con confidence="high".
                self.info["result_json"] = res
                self.info["confidence"] = "high"
                print("[Paso 6b] Gate semántico OK.")
                return StopEvent(result=out)
            print(f"[Paso 6b] Gate semántico FALLA: {issues}")
            self.info["semantic_issues"].append(issues)   # se apila cada intento, no se sustituye
            if semantic_retries < MAX_SEMANTIC_RETRIES:
                semantic_retries += 1
                self.info["semantic_retries"] = semantic_retries
                budget_ext = True                    # segunda vuelta con más presupuesto
                issues_fb = ",".join(issues)         # y, en modo libre, sabiendo qué falló
                print("[Paso 6b] Reintento con presupuesto ampliado...")
                continue
            # Agotado el reintento: se DEVUELVE el resultado igualmente, marcado.
            # Descartarlo perdería una predicción utilizable y, sobre todo,
            # rompería el análisis de calibración, que necesita justamente los
            # casos de baja confianza para comprobar si el agente se conoce [M11].
            self.info["result_json"] = res
            self.info["confidence"] = "low"        # [M11] se acepta con flag
            print("[Paso 6b] Persisten issues -> confidence=low (se devuelve con flag).")
            return StopEvent(result=out + "\nCONFIDENCE=low ISSUES=" + ",".join(issues))

    # ---------------- paso 7: corrección de errores de ejecución -----------
    @step
    async def paso7_corregir(self, ev: Step6Event) -> StopEvent | Step6Event | BacktrackEvent:
        """Paso 7: corrección de errores con generación libre de código. Aquí la
        adaptabilidad del LLM aporta valor y el determinismo no alcanza, así que
        incluso en modo template el LLM sigue escribiendo código en esta fase
        (la generación libre queda reservada al papel donde es imprescindible).
        Tras MAX_PASO7_ITER fallos, backtracking solo si el error es estructural
        [is_structural_error]: retroceder no arregla errores de entorno y
        quemaría presupuesto. [M12]

        Al corrector se le da el código que falló, el tipo de error, el mensaje y
        una traza recortada, y se le repiten los requisitos de protocolo para que
        "arreglar" no acabe siendo "escribir otro programa más simple" que ya no
        normalice o no publique RESULT_JSON.

        Cuatro desenlaces:
          StopEvent   confidence=failed  el agente no devolvió código
          StopEvent   high/low           ejecutó (con o sin issues del gate)
          Step6Event                     otro intento, iteration+1
          BacktrackEvent                 error estructural y quedan backtracks
        """
        print(f"[Paso 7 · it {ev.iteration}] Corrigiendo con CodeActAgent...")
        self.paso7_executions += 1
        global last_code
        last_code = ""      # se limpia para no confundir código viejo con el nuevo
        values_csv, _ = self._input_values_csv(ev)
        prompt = (
            f"Entrena una red neuronal con {ev.n_inputs} entradas y {ev.n_outputs} salidas leyendo train.csv "
            f"y predice para los valores de entrada: [{values_csv}].\n\n"
            f"El siguiente código falló:\n```python\n{ev.code_to_execute}\n```\n\n"
            f"Error: {ev.error_type}: {ev.error_msg}\nTraza (resumida): {ev.error_traceback[:500]}\n\n"
            "Corrige el código manteniendo: normalización z-score, separación train/validación, "
            "early stopping, promedio multi-semilla, y la línea final "
            'RESULT_JSON={"pred_mean": [..], "pred_std": [..], "best_arch": [..], "augment": .., '
            '"val_mae_norm": .., "best_epochs": .., "y_min": [..], "y_max": [..]}. '
            "No uses clases de Python. Ejecuta el código corregido."
        )
        local_ctx = Context(agent)   # memoria limpia en cada corrección
        result = await run_agent_and_get_result(agent, local_ctx, prompt, log_code=True)

        # Rescate: si el agente no llegó a pedir la ejecución del código (no hubo
        # ToolCall, así que last_code sigue vacío), puede que igualmente lo haya
        # escrito en su respuesta dentro de un bloque markdown. Se intenta sacarlo
        # de ahí antes de dar la ejecución por perdida.
        if not last_code or not last_code.strip():
            m = re.search(r"```python\n(.*?)\n```", result, re.DOTALL)   # DOTALL: el bloque es multilínea
            if m:
                last_code = m.group(1)
            else:
                self.info["confidence"] = "failed"   # sin código: fallo estructural
                return StopEvent(result=result)

        ok, out, et, em, tr = run_code_captured(last_code)
        if ok:
            res = parse_result_json(out)
            if res is not None:
                self.info["result_json"] = res
                gate_ok, issues = semantic_gate(res)
                if self.info["values_padded"]:
                    # Mismo criterio que en el paso 6: entrada rellenada con
                    # ceros invalida la confianza pase lo que pase el gate. [M5]
                    gate_ok = False
                    issues = issues + ["valores_entrada_incompletos"]   # [M5]
                if not gate_ok:
                    self.info["semantic_issues"].append(issues)
                    self.info["confidence"] = "low"
                    out += "\nCONFIDENCE=low ISSUES=" + ",".join(issues)
                # Si el gate pasa, confidence se queda como estaba ("high" salvo
                # que algo la hubiera degradado antes): recuperarse de un error de
                # ejecución no penaliza el resultado, lo que se mide es la predicción.
            else:
                self.info["confidence"] = "low"      # ejecutó pero sin RESULT_JSON parseable
                out += "\nCONFIDENCE=low ISSUES=sin_result_json"
            return StopEvent(result=out)

        print(f"[Paso 7 · it {ev.iteration}] Error: {em}")
        if ev.iteration >= MAX_PASO7_ITER:
            # Agotadas las correcciones. Última carta: si el error huele a
            # estructura mal inferida y aún quedan backtracks, se vuelve al
            # paso 1 en lugar de rendirse.
            if is_structural_error(et, em) and self.backtrack_count < MAX_BACKTRACKS:
                self.backtrack_count += 1
                fb = (f"Se generó código para {ev.n_inputs} entradas y {ev.n_outputs} salidas pero falló "
                      f"{ev.iteration} veces con errores de forma/estructura. Revisa si el número de "
                      f"columnas de entrada/salida es distinto.")
                return BacktrackEvent(user_message=ev.user_message, feedback=fb)
            # Fin del camino: fallo declarado y explícito. El harness lo reconoce
            # como 'agente_failed' y ocupa su celda; no es un error de medida.
            self.info["confidence"] = "failed"
            return StopEvent(result=f"Fallo tras {ev.iteration} correcciones "
                                    f"({self.backtrack_count} backtracks). Último error: {em}")
        # Quedan intentos: se reenvía a este mismo paso con el código y el error
        # nuevos (los del último intento, no los del original) e iteration+1.
        return Step6Event(user_message=ev.user_message, n_inputs=ev.n_inputs,
                          n_outputs=ev.n_outputs, dataset=ev.dataset,
                          question_summary=ev.question_summary,
                          input_values_str=ev.input_values_str, training_data="train.csv",
                          code_to_execute=last_code, iteration=ev.iteration + 1,
                          error_type=et, error_msg=em, error_traceback=tr)


# =============================== MAIN ======================================
# Mini-problema de humo (z ~ 2x + y, 20 filas): sirve para comprobar de un
# vistazo que hay servidor Ollama, que el flujo entero corre y que la predicción
# es razonable, sin tocar los diez problemas de la evaluación. Su formato imita
# el de los prompts reales (pregunta con 'nombre = valor', 'Basándome en estos
# datos previos:' y el CSV al final), que es justo lo que esperan el extractor
# genérico [M7] y el parser determinista del query [M5].
DEMO_MESSAGE = """Quiero predecir la Salida Z para
    Entrada A = 4.5
    Entrada B = 1.5
 
    Basándome en estos datos previos:
 
Entrada A,Entrada B,Salida Z
1,1,3.1
2,1,5.05
3,2,7.9
1.5,2.5,6.4
2.5,0.5,5.6
4,3,11.1
0.5,1,2.05
3.5,1.5,8.6
2,3,8.9
4.5,2,10.9
1,2,5.1
3,0.5,6.4
5,1,11.2
2.2,2.2,7.7
3.8,2.8,10.9
0.8,0.8,2.6
4.2,1.2,9.5
1.7,1.7,5.9
2.9,2.4,9.0
4.8,0.6,10.1
"""


async def main():
    """Punto de entrada: lee el enunciado, ejecuta el Workflow y registra todo.

    Argumentos (ambos opcionales):
      argv[1]  ruta del .txt con el enunciado; si falta, se usa DEMO_MESSAGE
      argv[2]  prompt_id para el log; si falta, el nombre del fichero

    Deja tres rastros de cada ejecución: log.csv (tabla rápida), log.jsonl
    (registro canónico que consume el harness) [M14] y el stdout, del que el
    harness recorta exactamente la sección "Resultado final".

    Ninguna incidencia al escribir los logs debe tumbar la ejecución: el
    resultado ya está calculado y perderlo por un fichero bloqueado sería
    absurdo, de ahí los try/except con aviso por pantalla.
    """
    prompt_path = sys.argv[1] if len(sys.argv) > 1 else ""
    prompt_id = sys.argv[2] if len(sys.argv) > 2 else (os.path.basename(prompt_path) or "demo")
    if prompt_path:
        with open(prompt_path, encoding="utf-8") as f:
            user_message = f.read()
    else:
        user_message = DEMO_MESSAGE
        print("[main] Sin fichero de prompt: usando mini-problema de humo.")

    # timeout del Workflow: si se agota, LlamaIndex aborta el flujo. El harness
    # exporta AGENT_TIMEOUT = su propio timeout menos 60 s, para que el agente
    # muera antes y dé tiempo a escribir los logs de la celda.
    flow = NeuralNetAgentV2(timeout=AGENT_TIMEOUT, verbose=True)
    start_ms = int(time.time() * 1000)
    result = await flow.run(start_event=NeuralNetStartEvent(user_message=user_message))
    end_ms = int(time.time() * 1000)

    # Al terminar, se completa la telemetría con lo que solo se sabe aquí
    # (identificación de la celda, tiempos y esfuerzo de recuperación).
    info = flow.info
    info.update({
        "prompt_id": prompt_id, "start_ms": start_ms, "end_ms": end_ms,
        "duration_s": (end_ms - start_ms) / 1000.0,
        "paso7_executions": flow.paso7_executions,
        "backtracks": flow.backtrack_count,
    })
    # `or {}` porque result_json es None en las ejecuciones fallidas: así las
    # columnas del CSV se rellenan con "" en vez de reventar el volcado.
    rj = info.get("result_json") or {}
    # [M14] log.csv: mismas columnas que el del prototipo (para no romper nada
    # que lo leyera) más las nuevas. Delimitador ';' porque los campos de listas
    # (pred_mean, best_arch) llevan comas dentro. La cabecera se escribe solo si
    # el fichero aún no existe: el modo 'a' permite acumular ejecuciones.
    try:
        exists = os.path.exists("log.csv")
        with open("log.csv", "a", encoding="utf-8", newline="") as lf:
            w = csv.writer(lf, delimiter=";")
            if not exists:
                w.writerow(["start_ms", "end_ms", "diff_seconds", "model_name", "response",
                            "paso7_executions", "best_arch", "prompt_id", "mode", "confidence",
                            "pred_mean", "pred_std", "augment", "val_mae_norm"])
            # La respuesta se recorta a 2000 chars: en el CSV solo es una ayuda
            # visual; la versión íntegra la guarda el harness en runs.jsonl.
            w.writerow([start_ms, end_ms, info["duration_s"], MODEL_NAME, str(result)[:2000],
                        flow.paso7_executions, rj.get("best_arch", ""), prompt_id, GENERATION_MODE,
                        info["confidence"], rj.get("pred_mean", ""), rj.get("pred_std", ""),
                        rj.get("augment", ""), rj.get("val_mae_norm", "")])
    except Exception as e:
        print(f"[log.csv] no se pudo escribir: {e}")
    # [M14] log.jsonl: resultado canónico, una línea JSON por ejecución. Es el
    # fichero que lee el harness, y de él salen las tablas del Cap. 5.
    # ensure_ascii=False conserva las tildes; default=str evita que un objeto no
    # serializable (numpy, etc.) haga perder el registro entero.
    try:
        with open("log.jsonl", "a", encoding="utf-8") as jf:
            jf.write(json.dumps(info, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        print(f"[log.jsonl] no se pudo escribir: {e}")

    # Estos dos marcadores son un contrato con el harness, que se queda SOLO con
    # lo que va entre "Resultado final" y el bloque de métricas: cambiar el texto
    # de las cabeceras rompería la extracción de respuestas.
    print("\n########### Resultado final ###########")
    print(result)
    print("\n########### Métricas de la ejecución ###########")
    # result_json se omite aquí porque ya viaja dentro del stdout del
    # entrenamiento (la línea RESULT_JSON) y duplicarlo solo ensucia la traza.
    print(json.dumps({k: v for k, v in info.items() if k != "result_json"},
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    # asyncio.run porque el Workflow de LlamaIndex es asíncrono de principio a fin.
    asyncio.run(main())
