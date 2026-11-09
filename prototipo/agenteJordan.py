# ---------------------------------------------------------------------------
# Prototipo del grupo de investigación, previo a este TFM. Se incluye tal
# cual como referencia de comparación (Capítulo 4 de la memoria): no forma
# parte de la evaluación y no está pensado para ejecutarse ni mantenerse.
# Únicos cambios respecto al original: la URL del servidor Ollama interno
# se ha anonimizado (sustituida por localhost) y un emoji de aviso en un
# mensaje de consola se ha sustituido por texto. El agente propuesto que
# evoluciona de este prototipo está en Agente/agente.py, cuya cabecera
# detalla las mejoras (M1-M14) respecto a este fichero.
# ---------------------------------------------------------------------------
# pip install llama-index-llms-ollama
"""
Workflow + CodeActAgent: determina estructura de la red, aísla dataset y pregunta,
luego pide al CodeActAgent que construya la red en PyTorch, entrene y prediga.
"""
import code
import re
import asyncio
import io
import contextlib
import ast
import time
from tkinter import N
import traceback
from typing import Any, Dict, Tuple
import os
import csv
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step, Context
from llama_index.core.agent.workflow import (
    CodeActAgent,
    ToolCall,
    ToolCallResult,
    AgentStream,
)
import traceback
import io

last_code = ""
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- LLM para los pasos del workflow ---
# debe dar: 125,     484
# http://localhost:11434/
# http://localhost:11434/api/tags
# qwen3.5:2b o no acaba, o no da solución
# qwen3.5:4b rapido 156 , 376 
# qwen3:8b funciona bien 158, 489
# qwen3.5:9b rapido, bien 142, 476
# gemma3:4b tarda un poco: 156, 376
# gemma3:12b no termina
# gemma3:27b rapido bastante pero predice algo mal 263, 333
# gpt-oss:20b: muy rápido  113, 491
# codegemma:2b no genera nada que funcione
# codegemma:7b no genera nada que funcione
# codellama:7b no genera codigo que funcione
# codellama:13b
# llama3.2:1b
# llama3.2:3b
# llama3.1:8b no determina bien las entradas y salidas
# deepseek-r1:7b
# deepseek-r1:14b
# deepseek-v2:16b

model_name = "qwen3.5:4b"
llm_workflow = Ollama(
    base_url="http://localhost:11434",
    model = model_name,
    request_timeout=6000.0,
    system_prompt="Responde siempre en español. Responde solo con el número solicitado, sin explicación.",
)

#llm_workflow = GoogleGenAI(
#    model=MODEL_NAME,
#    temperature=0.1, # Temperatura baja para que el Paso 1 y 2 sean precisos con los números
#    system_prompt="Responde siempre en español. Responde solo con el número solicitado, sin explicación.",
#    api_key=api_key
#)

# --- Eventos del workflow ---
class NeuralNetStartEvent(StartEvent):
    """Entrada: mensaje del usuario con pregunta y datos CSV."""
    user_message: str = ""


class Step1Event(Event):
    """Tras paso 1: número de entradas de la red."""
    user_message: str = ""
    n_inputs: int = 0


class Step2Event(Event):
    """Tras paso 2: número de salidas de la red."""
    user_message: str = ""
    n_inputs: int = 0
    n_outputs: int = 0


class Step3Event(Event):
    """Tras paso 3: dataset aislado en una variable (texto CSV)."""
    user_message: str = ""
    n_inputs: int = 0
    n_outputs: int = 0
    dataset: str = ""


class Step4Event(Event):
    """Tras paso 4: pregunta concreta y valores de entrada para predecir."""
    user_message: str = ""
    n_inputs: int = 0
    n_outputs: int = 0
    dataset: str = ""
    question_summary: str = ""
    input_values_str: str = ""


class Step5Event(Event):
    """Tras paso 5: datos para entrenamiento (mismo dataset)."""
    user_message: str = ""
    n_inputs: int = 0
    n_outputs: int = 0
    dataset: str = ""
    question_summary: str = ""
    input_values_str: str = ""
    training_data: str = ""

class Step6Event(Event):
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



# --- Code executor y CodeActAgent (como en llama_index_example_IA.py) ---
class SimpleCodeExecutor:
    def __init__(self, locals_dict: Dict[str, Any], globals_dict: Dict[str, Any]):
        self.globals = globals_dict
        self.locals = locals_dict

    def execute(self, code: str) -> str:
        try:
            f = io.StringIO()

            # Redirigimos la salida hacia nuestro objeto 'f'
            with contextlib.redirect_stdout(f):
                try:
                    tree = ast.parse(code)
                    last_node = tree.body[-1] if tree.body else None
                    if isinstance(last_node, ast.Expr):
                        last_line = code.rstrip().split("\n")[-1]
                        exec_code = code[: -len(last_line)] + "\n__result__ = " + last_line
                        exec(exec_code)
                    else:
                        exec(code)
                except Exception:
                    exec(code)

            # Extraemos el contenido acumulado
            out = f.getvalue()
        except Exception as e:
            out = f"Error: {type(e).__name__}: {str(e)}\n" + traceback.format_exc()
        return out


# Herramientas para el agente (el agente usará sobre todo código)
def add(a: int, b: int) -> int:
    return a + b


def subtract(a: int, b: int) -> int:
    return a - b


def multiply(a: int, b: int) -> int:
    return a * b


def divide(a: int, b: int) -> float:
    return a / b


_np = __import__("numpy")
_pd = __import__("pandas")
_torch = __import__("torch")

code_executor = SimpleCodeExecutor(
    locals_dict={"add": add, "subtract": subtract, "multiply": multiply, "divide": divide},
    globals_dict={
        "__builtins__": __builtins__,
        "np": _np,
        "pd": _pd,
        "pandas": _pd,
        "torch": _torch,
    },
)
 

#llm_agent = GoogleGenAI(
#    model=MODEL_NAME,
#    temperature=0.1,
#    system_prompt="Responde siempre en español. Escribes y ejecutas código Python cuando hace falta.",
#    api_key=api_key
#)

agent = CodeActAgent(
    code_execute_fn=code_executor.execute,
    llm=llm_workflow,
    tools=[add, subtract, multiply, divide],
)
ctx = Context(agent)


async def run_agent_and_get_result(
    agent, ctx, query: str, code_executor=None, log_code: bool = False
) -> str:
    """
    Ejecuta el agente. Si llega al límite de iteraciones, 
    retorna el resultado acumulado hasta la última iteración exitosa.
    """
    print("[Run code] Iniciando ejecución del agente...")
    global last_code
    result_parts = []
    snapshot_anterior = "" # Aquí guardaremos el progreso estable
    code_iteration = 0
    
    try:
        # Aumentamos el margen a 50 para dar aire al modelo
        handler = agent.run(query, ctx=ctx, max_iterations=5)
        
        async for event in handler.stream_events():
            # Actualizamos el snapshot con lo que llevamos acumulado
            snapshot_anterior = "".join([str(p) for p in result_parts])

            if isinstance(event, ToolCallResult):
                result_parts.append(event.tool_output)
                if log_code:
                    print(f"[Iteración {code_iteration}] Tool Output recibida.")
            
            elif isinstance(event, ToolCall):
                code = event.tool_kwargs.get("code", "")
                if code:
                    last_code = code
                    code_iteration += 1
                    if log_code:
                        print(f"\n[Run code] Generado (it {code_iteration}):\n{code}\n")
                    
                    if code_executor is not None:
                        exec_out = code_executor.execute(code)
                        result_parts.append(f"\nResultado ejecución:\n{exec_out}")

            elif isinstance(event, AgentStream):
                if event.delta:
                    result_parts.append(event.delta)

        # Si termina normal, devolvemos el resultado final oficial
        response = await handler
        return str(response)

    except Exception as e:
        print(f"\n[aviso] EXCEPCIÓN EN ITERACIÓN {code_iteration}: {type(e).__name__}")
        print(f"Retornando snapshot de la iteración anterior para evitar bloqueo del Workflow...")
        
        # Si el error es por límite de iteraciones, devolvemos lo que teníamos 
        # justo antes de que el agente intentara el paso que rompió el límite.
        if snapshot_anterior:
            return snapshot_anterior
        
        # Si falló muy pronto y no hay snapshot, devolvemos lo acumulado
        return "".join([str(p) for p in result_parts]) if result_parts else f"Error: {e}"


def _extract_csv_block(text: str) -> str:
    """Extrae el bloque CSV del mensaje (línea de cabecera + filas)."""
    lines = text.strip().split("\n")
    start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "," in stripped and not stripped.startswith("Quiero") and not stripped.startswith("Basandome"):
            # Posible cabecera CSV (contiene comas)
            if re.match(r"^[\w\s,\.]+$", stripped) and any(
                x in stripped for x in ["Distance", "Position", "X", "Y", "A", "B", "C"]
            ):
                start = i
                break
    if start < 0:
        return ""
    return "\n".join(lines[start:])


# --- Workflow: NeuralNetWithCodeAct ---
class NeuralNetWithCodeAct(Workflow):
    llm = llm_workflow

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Contador de cuántas veces se ejecuta el paso 7
        self.paso7_executions = 0

    @step
    async def paso1_num_entradas(self, ev: NeuralNetStartEvent) -> Step1Event:
        """Paso 1: Determina cuántas entradas tiene la red neuronal según el mensaje y los datos."""
        print("[Paso 1] Determinando número de entradas (inputs) de la red...")
        msg = (ev.user_message or "").strip()[:3000]
        prompt = (
            "En el siguiente mensaje del usuario hay una pregunta sobre predecir algo usando datos en formato CSV. "
            "La primera línea de datos es la cabecera con nombres de columnas. "
            "Las columnas que son ENTRADA (inputs) de la red neuronal son las que se usan para predecir (por ejemplo Distance A, B, C). "
            "Responde SOLO con un número entero: cuántas columnas son entradas (inputs) de la red.\n\nMensaje:\n" + msg
        )
        resp = self.llm.complete(prompt)
        n = 3
        try:
            n = int(re.search(r"\d+", str(resp).strip()).group())
        except Exception:
            pass
        print(f"[Paso 1] Listo. n_inputs = {n}")
        return Step1Event(user_message=ev.user_message, n_inputs=n)

    @step
    async def paso2_num_salidas(self, ev: Step1Event) -> Step2Event:
        """Paso 2: Determina cuántas salidas tiene la red neuronal."""
        print("[Paso 2] Determinando número de salidas (outputs) de la red...")
        msg = (ev.user_message or "").strip()[:3000]
        prompt = (
            "En el siguiente mensaje hay datos CSV. Las columnas que son SALIDA (objetivo a predecir) son las que el usuario quiere predecir (por ejemplo Position X, Position Y). "
            "Responde SOLO con un número entero: cuántas columnas son salidas (outputs) de la red.\n\nMensaje:\n" + msg
        )
        resp = self.llm.complete(prompt)
        n = 2
        try:
            n = int(re.search(r"\d+", str(resp).strip()).group())
        except Exception:
            pass
        print(f"[Paso 2] Listo. n_outputs = {n}")
        return Step2Event(
            user_message=ev.user_message,
            n_inputs=ev.n_inputs,
            n_outputs=n,
        )

    @step
    async def paso3_aislar_dataset(self, ev: Step2Event) -> Step3Event:
        """Paso 3: Aísla el dataset en una variable (bloque CSV del mensaje)."""
        print("[Paso 3] Aislando bloque CSV del mensaje (dataset)...")
        dataset = _extract_csv_block(ev.user_message or "")
        if not dataset.strip():
            dataset = ev.user_message or ""
        print(f"[Paso 3] Listo. Dataset extraído: {len(dataset)} caracteres")
        return Step3Event(
            user_message=ev.user_message,
            n_inputs=ev.n_inputs,
            n_outputs=ev.n_outputs,
            dataset=dataset,
        )

    @step
    async def paso4_aislar_pregunta_y_valores(self, ev: Step3Event) -> Step4Event:
        """Paso 4: Aísla la pregunta concreta y los valores de entrada para los que hay que predecir."""
        print("[Paso 4] Extrayendo pregunta y valores de entrada para la predicción...")
        msg = (ev.user_message or "").strip()[:4000]
        prompt = (
            "Del siguiente mensaje extrae:\n"
            "1) La pregunta concreta (qué quiere predecir el usuario, ej: 'Position X y Position Y').\n"
            "2) Los valores de ENTRADA que da el usuario para hacer la predicción (ej: Distance A = 499.79..., Distance B = ..., Distance C = ...).\n"
            "Responde en dos líneas:\n"
            "PREGUNTA: <resumen de qué hay que predecir>\n"
            "VALORES_ENTRADA: <los valores numéricos de entrada que menciona, tal cual>\n\nMensaje:\n" + msg
        )
        resp = self.llm.complete(prompt)
        text = str(resp).strip()
        question_summary = ""
        input_values_str = ""
        for line in text.split("\n"):
            if line.strip().upper().startswith("PREGUNTA"):
                question_summary = line.split(":", 1)[-1].strip()
            elif "VALORES" in line.upper() or "ENTRADA" in line.upper():
                input_values_str = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
        if not input_values_str:
            # Fallback: buscar patrones Distance A = número
            for m in re.finditer(r"(Distance\s*[ABC]\s*=\s*[\d.]+)", msg, re.I):
                input_values_str += m.group(1) + " "
        print(f"[Paso 4] Listo. Pregunta: {question_summary[:80]}... | Valores entrada: {input_values_str[:80]}...")
        return Step4Event(
            user_message=ev.user_message,
            n_inputs=ev.n_inputs,
            n_outputs=ev.n_outputs,
            dataset=ev.dataset,
            question_summary=question_summary or "Position X y Position Y",
            input_values_str=input_values_str or msg[:500],
        )

    @step
    async def paso5_aislar_datos_entrenamiento(self, ev: Step4Event) -> Step5Event:
        """Paso 5: Aísla los datos que se usarán para entrenar y los guarda en train.csv."""
        print("[Paso 5] Preparando datos de entrenamiento y guardando en train.csv...")
        training_data = ev.dataset or ""
        with open("train.csv", "w", encoding="utf-8") as f:
            f.write(training_data)
        print("[Paso 5] Listo. Datos guardados en train.csv")
        return Step5Event(
            user_message=ev.user_message,
            n_inputs=ev.n_inputs,
            n_outputs=ev.n_outputs,
            dataset=ev.dataset,
            question_summary=ev.question_summary,
            input_values_str=ev.input_values_str,
            training_data="train.csv",
        )

    @step
    async def paso6_codeact_construir_entrenar_predecir(self, ev: Step5Event) -> Step6Event:
        """Paso 6: CodeActAgent construye la red en PyTorch, entrena con los datos y predice."""
        print(f"[Paso 6] Llamando a CodeActAgent: construir red ({ev.n_inputs} in, {ev.n_outputs} out), entrenar y predecir...")
        global last_code


        prompt = (
            "Crea y ejecuta el siguiente programa para obtener el resultado:\n"
            "1) La red neuronal tiene {} entradas y {} salidas.\n"
            "2) Dataset de entrenamiento se llama train.csv (CSV con cabecera):\n{}\n\n"
            "3) Nunca uses datos de Mock, los datos de entrenamiento reales estan en train.csv\n"
            "4) Nunca uses class de Python\n"
            "5) Usa PyTorch: crea un modelo (nn.Sequential con capas lineales y ReLU), entrénalo con estos datos (columnas de entrada -> columnas de salida)\n"
            "6) No uses nunca class de Python; declara el modelo de forma similar a: model = nn.Sequential(nn.Linear(NUMERO DE ENTRADAS, 6), nn.ReLU(), nn.Linear(6, 3), nn.ReLU(), nn.Linear(3, NUMERO DE SALIDAS))\n"
            "7) Una vez entrenado el modelo, guarda el fichero de python y el modelo entrenado (.pth)"
            "8) Realiza una predicción para los siguientes valores de entrada:\n{}\n\n"
       
        ).format(
            ev.n_inputs,
            ev.n_outputs,
            ev.training_data.strip()[:8000],
            ev.input_values_str.strip()[:500],
        )

        result = await run_agent_and_get_result(agent, ctx, prompt, log_code=True)

        print("[Paso 6] CodeActAgent terminó. Devolviendo resultado final.")
        print("resultado original del agente -------------------------")
        print(result)
        print("-------------------------")

        # Si last_code está vacío, intentamos extraer el código del resultado entre ```python y ```
        if not last_code or not last_code.strip():
            match = re.search(r'```python\n(.*?)\n```', result, re.DOTALL)
            if match:
                last_code = match.group(1)
                print("[Paso 6] last_code estaba vacío, se ha extraído el código desde el bloque ```python``` del resultado.")
            else:
                print("[Paso 6] No se ha encontrado un bloque de código python en la variable result; no se ejecuta ningún código.")

        print("Codigo del agente -------------------------")
        print(last_code)
        print("-------------------------")

        # Aqui esta el código
        timestamp = time.time()
        with open(f"test_{timestamp}.py", 'w', encoding='utf-8') as f:
            f.write(last_code)
            
        try:
            f = io.StringIO()

            # Redirigimos la salida hacia nuestro objeto 'f'
            with contextlib.redirect_stdout(f):
                exec(last_code)

            # Extraemos el contenido acumulado
            result = f.getvalue()
            return StopEvent(result=result)
            
        except Exception as e:
            err_type = type(e).__name__
            err_trace = traceback.format_exc()
            # err_msg pasa a contener el LOG completo de la traza
            err_msg = f"{err_type}: {str(e)}\n{err_trace}"
            print(f"[Paso 6] Error detectado: {err_msg}") 
            return Step6Event(
                user_message=ev.user_message,
                n_inputs=ev.n_inputs,
                n_outputs=ev.n_outputs,
                dataset=ev.dataset,
                question_summary=ev.question_summary,
                input_values_str=ev.input_values_str,
                training_data="train.csv",
                code_to_execute=last_code,
                iteration=1,
                error_type=err_type,
                error_msg=err_msg,
                error_traceback=err_trace
            )
        print("No se ha encontrado un bloque de código python en la variable result")
        return StopEvent(result=result)

    @step
    async def paso7_codeact_execute_and_evaluate(self, ev: Step6Event) -> StopEvent:
        print(f"[Paso 7] Llamando a CodeActAgent: Ejecutar el código del paso 6, para solucionar errores en el código")
        # Incrementamos el contador de ejecuciones del paso 7
        self.paso7_executions += 1
        global last_code
        


        prompt = (
            "Analiza el siguiente código como solución a el problema que se plantea a continuación:\n"
            "1) La red neuronal tiene {} entradas y {} salidas.\n"
            "2) Dataset de entrenamiento (CSV con cabecera):\n{}\n\n"
            "3) Nunca uses datos de Mock, los datos de entrenamiento reales estan en train.csv\n"
            "4) No uses clases de Python; declara el modelo de forma similar a: model = nn.Sequential(nn.Linear(2, 6), nn.ReLU(), nn.Linear(6, 3), nn.ReLU(), nn.Linear(3, 1))\n"
            "5) Usa PyTorch: crea un modelo (nn.Sequential con capas lineales y ReLU), entrénalo con estos datos "
            "(columnas de entrada -> columnas de salida), y haz la predicción para los siguientes valores de entrada:\n{}\n\n"
            "El código generado anteriormente es el siguiente: \n{}\n"
            "Este código se ha ejecutado y ha generado una excepción. Tu misión es corregirla. A cointinuación te aporto información sobre el error generado: \n"
            "- Mensaje de error: {}\n"
            "- Tipo de error: {}\n"
            "- Traza del error: {}\n"
            "Arreglalo y ejecuta el código para obtener el resultado correcto."
        ).format(
            ev.n_inputs,
            ev.n_outputs,
            ev.training_data.strip()[:8000],
            ev.input_values_str.strip()[:500],
            ev.code_to_execute,
            ev.error_msg,
            ev.error_type,
            ev.error_traceback
        )

        print("Código recibido en el paso 7 -------------------------")
        print(ev.code_to_execute)
        print("-------------------------")
        print("Error recibido en el paso 7 -------------------------")
        print(ev.error_msg)
        print(ev.error_type)
        print(ev.error_traceback)
        print("-------------------------")


        result = await run_agent_and_get_result(agent, ctx, prompt, log_code=True)
        print("[Paso 7] CodeActAgent terminó. Devolviendo resultado final.")
        print("resultado original del agente -------------------------")
        print(result)
        print("-------------------------")

        # Si last_code está vacío, intentamos extraer el código del resultado entre ```python y ```
        if not last_code or not last_code.strip():
            match = re.search(r'```python\n(.*?)\n```', result, re.DOTALL)
            if match:
                last_code = match.group(1)
                print("[Paso 7] last_code estaba vacío, se ha extraído el código desde el bloque ```python``` del resultado.")
            else:
                print("[Paso 7] No se ha encontrado un bloque de código python en la variable result; no se ejecuta ningún código.")
                return StopEvent(result=result)

        print("Codigo del agente -------------------------")
        print(last_code)
        print("-------------------------")
        
        
        print(f"[Paso 7] CodeActAgent terminó. Iteración {ev.iteration}.")

        if last_code:
            print(f"[Paso 7 - Iteración {ev.iteration}] CodeActAgent propuso una solución nueva")
            try:
                f = io.StringIO()

                # Redirigimos la salida hacia nuestro objeto 'f'
                with contextlib.redirect_stdout(f):
                    exec(last_code)

                # Extraemos el contenido acumulado
                result = f.getvalue()
                print("Se genero el Stop Event")
                return StopEvent(result=result)

            except Exception as e:
                err_type = type(e).__name__
                err_trace = traceback.format_exc()
                # err_msg pasa a contener el LOG completo de la traza
                err_msg = f"{err_type}: {str(e)}\n{err_trace}"
                print(f"[Paso 7 - Iteración {ev.iteration}] Error detectado: {err_msg}") 

                return Step6Event(
                    user_message=ev.user_message,
                    n_inputs=ev.n_inputs,
                    n_outputs=ev.n_outputs,
                    dataset=ev.dataset,
                    question_summary=ev.question_summary,
                    input_values_str=ev.input_values_str,
                    training_data="train.csv",
                    code_to_execute=last_code,
                    iteration=ev.iteration+1,
                    error_type=err_type,
                    error_msg=err_msg,
                    error_traceback=err_trace

                )
        print(f"[Paso 7 - Iteración {ev.iteration}] CodeActAgent no propuso una solución nueva")

        return StopEvent(result=result)

async def main():
    flow = NeuralNetWithCodeAct(timeout=120000000, verbose=True)

    user_message = """
    Quiero predecir la Posicion X y la Posicion Y para
    Distance A = 689.1390078666366
    Distance B = 300.95425448708454
    Distance C = 639.9909661880821

    Basándome en estos datos previos:

Distance A,Distance B,Distance C,Position X,Position Y
349.0382428454984,527.5630801365121,340.6925340041829,789.4678851359404,340.15416270113695
331.3761553627671,301.35323317680593,380.6114226494857,550.9582979459686,344.3592807557936
607.5490515249687,318.5824067934828,442.6018729445093,315.77671461731137,539.1765584796316
463.42539499102753,101.29186518590416,747.8294956611395,196.03779597793525,159.8042610099495
699.786432642459,319.73732163238117,622.75849505553,136.10858470321318,511.1916135596304
208.44894768905183,611.9892264476865,680.3140587088182,846.9527562844256,5.416352428771476
949.0342581999085,881.3027357854266,291.4873165720313,746.1387307369966,968.0264426565021
607.4540038644383,206.90659525241173,802.6501894563085,68.30726615313642,231.94757319120484
238.21397934402663,231.70485337223548,508.6874316932816,506.73388349486646,222.84525745504757
934.1725814799915,809.4536698360473,308.00654908750107,612.8405256921505,958.8049413736718
806.5052794433345,441.55071970286843,629.4477268946027,108.59879951435315,632.194748183267
479.2534901080185,564.9721102950839,196.33088625671655,776.0069719727604,484.39051531595055
764.2798020668255,719.489155519281,108.34843901953248,725.2100083649141,784.4611201217513
626.9207754959406,236.68602257583802,625.0368944120887,161.74137042397106,431.02342816833885
367.77066621446016,48.40720679060206,624.344851798413,321.6282305056682,210.10352422067234
970.7571633411077,667.3142069292561,585.715857764268,189.0312941782718,884.9603815978332
465.6218792889319,413.5388318958338,237.16758038966896,592.4631451726037,488.25815105199774
623.9102073587067,214.8168468479399,684.0187089283112,122.72781480007633,374.7050604856683
658.6948527819491,689.816023437903,72.39420222934106,807.3454425763967,661.942649571532
192.21756441536957,222.59685858753315,620.5294183152416,470.21441758362295,116.19607327684089
512.5157430980904,254.15939709047487,898.432129018534,126.97968890762468,16.62345309548785
551.0623529100429,677.5290705932401,216.993032784563,884.8771617824623,518.3907669160726
464.21721557900446,70.60959335684282,612.4626385856233,259.77934497366704,292.1538522753017
271.57286262824283,628.2125537504174,580.2163084158834,894.4369556239049,118.40047338403437
498.07298264169106,195.47264067391163,488.5781606464734,325.8314631507612,411.9663879652382
464.3677990774851,195.79160019777484,837.5577117087853,175.98478584378284,54.31866821999842
397.5685419846819,165.29369573937336,788.7811889150605,243.48641795144178,60.95460949902098
807.6907944700347,667.1693380730017,235.8104176024827,555.911104712217,828.3721610480998
230.17635204423374,189.00696116581653,588.8162143618807,453.3086258007283,160.4403636809074
904.1665963726767,562.3917844214874,625.5126035319848,116.92979883338317,762.9227549757173
669.429476502347,664.0359434545048,17.847630532028216,752.3505644958203,684.8471342394772
977.5594147155514,741.0588001334114,475.83344000177584,354.18672914117445,960.0297454871484
930.2001728165839,685.6085236689842,457.803082014145,340.074741651485,905.7267750575403
396.2585477311909,389.87889606717016,285.2806109935363,611.4360222441718,420.27946521244894
723.1473955533116,363.4627681995229,583.6049188866458,162.90021338037397,568.9450981445904
160.12498231869273,467.21203996381297,524.5743445271214,736.8019282711632,152.1252471439606
1034.0826249656704,736.6933613392904,616.3182685254994,186.335011553184,954.5454280000721
931.2489460990774,605.3338023636795,606.7291775061217,145.60899010890526,814.5476902472763
787.6085306428186,798.9024558449338,151.9704823534824,841.9035985628916,786.1482874470188
658.2440073063184,243.5982859709837,764.0444550590871,55.8566322875691,329.5284629354237
677.6223563287571,796.9353259221999,238.15115057403906,967.2845658043669,618.0380245679444
649.5964176931903,795.5833206001214,274.57124361356205,990.1977725217018,571.7604508185091
694.7634259212465,498.7828562625662,291.6856836828331,445.1908124551816,692.0704225234156
885.376280878252,748.0623464726798,281.66317570194633,575.8913050350027,908.1044081368505
411.25789742897143,632.7468594337332,366.33040764403603,895.6015894565152,346.73662402172056
596.9474808907189,294.6022692702971,940.2574544997588,42.72218230198843,42.03708777025306
482.1298937075739,615.6012310298394,238.99332996267395,842.2638412107125,462.3965900341793
763.1079000025055,706.7389960730322,113.70697255645847,703.2495701724537,785.4450527506676
957.5320178537393,782.6715250574949,377.99076295627367,501.15131160868384,972.5061140248462
148.08106950613572,277.64564760118975,579.6559965267759,537.3726226378803,132.3091631501886
273.8623890797798,141.05003789151286,625.0068268680315,396.80945835507555,152.04395731536013
237.62283447556266,178.80899825118775,600.7876858662419,439.97284079629685,154.17239905445845
415.73343770203746,445.9907579984307,247.34115233589895,664.8031221858578,439.9689861539342
469.6773191848327,254.50297243368453,402.36425939733977,413.4722796913629,436.7647346881255
240.44988757641522,440.781923478495,423.9064238217606,714.7797303839991,253.3484088973269
199.95959428818395,216.6045764034652,600.3956221369604,473.6815731831714,136.86941446069824
843.2262194548398,620.8876049916538,374.5708180421455,395.72763949968726,832.2537032247968
223.34428904797807,408.6797577927649,435.941415028247,683.1785500687299,244.02777666467634
837.2603647359683,770.8102297040431,184.39691997255966,710.4593109025888,859.2525070749699
261.0190515534371,659.6614416577189,684.0252175423116,900.148038493783,12.545593049680836
821.52994888899,626.6684374444732,321.49036731620447,451.0481443828231,824.6512617069341
266.7980240025779,317.96142205805506,417.95481471939837,586.6220169602586,286.5310042716999
613.2709291950263,611.5389098538038,45.60730056172042,730.2985257972547,631.5118366446953
888.1745300505339,803.309545597632,239.92077164360944,688.7047462154504,911.8172051852781
724.022897699529,636.0808146448941,126.23595837105428,632.9945577059152,749.0050802502402
854.3889773437912,721.9789289103983,254.13458896666728,580.5329803765313,877.3675131581983
255.60927557962296,487.6929527002061,428.1617957320872,762.0238988998107,249.30086362152548
165.2648514591502,572.476876208059,677.8332387825719,803.1027552300848,2.1491651091259945
970.5170583301444,945.1598057205725,317.0221967081056,851.7602683063982,972.0154321339729
685.3998953981154,292.0456097606374,653.1535241285538,117.23539493986101,468.95799837620325
722.5247876092415,602.8357200962807,173.24556006933972,577.1932075762192,744.8503546305858
534.8466103182404,208.8558273261472,854.6541031675516,109.3230057402369,96.08019672376422
675.7523862094329,269.7100301223933,692.9183401374602,92.2446427978404,421.536890546646
563.5900740660386,681.7685665657151,203.9805326853288,881.972483925667,533.739918969671
963.966978131127,591.4119902093048,738.7308631535175,1.1288002311281575,747.3735452298582
519.4689642551248,146.27370876984165,577.961268944589,248.45798115047867,367.0507948635562
937.2835211198033,964.0293571723853,317.22953549285967,953.4398410822484,908.1280792654725
500.22520786169554,198.35752513319125,845.0816505735711,141.83056761308322,76.22842433523502
856.0561965126557,651.1700752870732,353.677835166648,431.23988975625537,855.3663553320264
874.0352450338061,884.0818263141043,238.51645345887496,883.9733980451625,864.1380222040257
921.1222433093775,653.0308012154616,491.54705165999644,287.1976794579812,876.1281589031772
681.3877870398082,338.0757653260564,538.8328360088354,211.5005216970387,555.263919272263
468.1308115036335,403.82728216703475,244.9279215330203,578.8738815520989,489.20877431931444
1105.9435965525158,777.8681891959749,728.4326636550477,71.45582867309119,973.9681916955766
411.76741022047577,366.90467801287997,292.23112240364134,576.7485641433339,431.9798438066624
275.4096362614421,139.42553775855637,618.7348991092119,398.4918331846728,158.43043238633157
391.7346435108708,41.21463656598554,608.2799139742987,312.37809669234593,240.6381597891213
560.9569716864614,202.70114969172138,553.0184923719579,244.750592981408,423.63766966063633
423.7974789740936,234.3073028127727,413.12589051967905,433.7615846654208,395.55916369381106
583.7755747573747,211.42591521059796,839.4883714521895,72.26815566576938,163.3039061358993
929.2172616304472,820.8146556415988,291.1229398235905,648.4110763436698,954.1845829120062
525.3183546527656,368.9470740200595,293.14145433549874,484.46361059342115,526.9527679512013
623.0939937436478,770.0587923716919,265.5344192466378,971.1494947441405,552.463291293909
466.8431536632046,602.1804180935516,246.49137631382374,832.9267503550891,449.86402252560964
351.6138164651256,716.7899760234264,622.5857064855526,981.9279303384726,104.54219925385789
882.3675444019199,553.9508181349458,584.7070225315333,158.432193796362,764.751705623806
    650.179111192292,251.67760519862713,655.3833204031566,129.79819786708237,428.7576636822368
    """

    # Medimos tiempos de ejecución en milisegundos
    start_ms = int(time.time() * 1000)

    start_ev = NeuralNetStartEvent(user_message=user_message)
    result = await flow.run(start_event=start_ev)

    end_ms = int(time.time() * 1000)
    diff_seconds = (end_ms - start_ms) / 1000.0

    # Registro en log.csv: inicio_ms, fin_ms, diferencia_s, model_name, respuesta, num_paso7
    log_path = "log.csv"
    file_exists = os.path.exists(log_path)
    paso7_count = getattr(flow, "paso7_executions", 0)
    try:
        print(f"Escribiendo en {log_path}: {start_ms}, {end_ms}, {diff_seconds}, {model_name}, {result}, {paso7_count}")
        with open(log_path, mode="a", encoding="utf-8", newline="") as log_file:
            # Usamos ';' como separador en el CSV
            writer = csv.writer(log_file, delimiter=";")
            if not file_exists:
                writer.writerow(["start_ms", "end_ms", "diff_seconds", "model_name", "response", "paso7_executions"])
            writer.writerow([start_ms, end_ms, diff_seconds, model_name, str(result), paso7_count])
    except Exception as e:
        print(f"No se pudo escribir en {log_path}: {e}")

    print("\n########### Resultado final del workflow ###########")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
