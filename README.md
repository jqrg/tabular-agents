# Evaluación de agentes LLM en predicción numérica sobre datos tabulares

Código y datos del Trabajo Fin de Máster (MUIA, UAX) de Iván Vega Prieto:
evaluación comparativa entre un agente LLM que genera y ejecuta código para
entrenar redes neuronales y LLMs que predicen directamente "de cabeza", sobre
10 problemas analíticos con datos tabulares embebidos en el enunciado.

## Contenido del repositorio

- `Agente/agente.py` — el agente propuesto: workflow de LlamaIndex por pasos
  más CodeActAgent, con Ollama como backend de LLM y PyTorch para el
  entrenamiento. Su cabecera documenta el flujo completo, las mejoras respecto
  al prototipo (M1-M14) y su configuración. Está congelado: es el objeto
  medido en la evaluación.
- `prototipo/agenteJordan.py` — prototipo previo del grupo de investigación,
  incluido tal cual como referencia de comparación (solo se ha anonimizado la
  URL de su servidor interno). No forma parte de la evaluación.
- `harness/` — el instrumento de evaluación: paquete Python con CLI que
  ejecuta la parrilla sistemas x problemas x repeticiones, valida todas las
  respuestas con un pipeline único, calcula las métricas y genera el análisis
  estadístico y las figuras.
- `prompts/` — los 10 problemas de evaluación (`prompt_01`...`prompt_10`),
  `soluciones.txt` (ground truth, solo para el harness) y
  `generar_dataset.py` (anexo que documenta cómo se generó el dataset; no
  hace falta ejecutarlo).
- `prompts_perturbados/` — variantes de robustez P1-P6 del prompt_02, tal y
  como se ejecutaron (el diseño de robustez se recortó a ese problema; no
  regenerar con `perturb` si se quiere conservar ese recorte).
- `manual_responses/` — respuestas de modelos comerciales recogidas a mano
  (un CSV por sistema más el texto original de cada conversación), procesadas
  por el mismo pipeline que el resto de sistemas.
- `results*/` — resultados crudos de cada tanda (`runs.jsonl`, una línea JSON
  por ejecución con la respuesta cruda siempre conservada, y `summary.csv`):
  `results/` (llama3.2:3b), `results_qwen25/`, `results_llama31/`,
  `results_gemma3_12b/`, `results_gptoss20b/` (solo sistemas del agente) y
  `results_comerciales/`.
- `tests/` — 58 tests del harness, sin red ni Ollama.

## Instalación

Python 3.10 o superior.

```bash
# Dependencias del harness (tests y análisis funcionan sin nada más)
pip install -r requirements.txt

# Dependencias adicionales para ejecutar el agente (versiones fijadas a
# propósito: otras versiones de llama-index cambian el manejo del contexto)
pip install llama-index-llms-ollama==0.10.1 llama-index-core==0.14.21 python-dotenv torch scikit-learn
```

Para ejecutar el agente o los modelos directos hace falta un servidor Ollama
accesible. La URL de los `config*.yaml` viene como `http://localhost:11434`
(la del servidor original está anonimizada): ajustarla antes de lanzar nada y
verificar que los tags de modelo existen en el servidor (`ollama list`).

## Uso del harness

```bash
python -m pytest tests -q                  # tests (sin red)
python -m harness run --dry-run            # lista la parrilla sin ejecutar
python -m harness run                      # ejecuta lo pendiente (reanudable)
python -m harness run --sistema <s>        # limita a un sistema (repetible)
python -m harness perturb                  # genera variantes P1-P6 (ver aviso)
python -m harness run --perturbados        # ejecuta las variantes de robustez
python -m harness analyze                  # summary.csv + analysis.md + figuras
```

Cada tanda tiene su configuración: `harness/config.yaml` (llama3.2:3b, por
defecto), `config_qwen25.yaml`, `config_llama31.yaml`, `config_gemma3_12b.yaml`,
`config_gptoss20b.yaml` y `config_comerciales.yaml`, cada una con su directorio
de resultados. La opción `--config` va antes del subcomando:

```bash
python -m harness --config harness/config_qwen25.yaml analyze
```

`analyze` regenera en el directorio de resultados el informe `analysis.md`,
las figuras y `summary.csv` a partir de `runs.jsonl`: tablas por sistema y
problema, contrastes Wilcoxon pareados con corrección de Holm, Fisher exacto
sobre fallos estructurales, calibración del agente (por configuración) y
desglose del error por salida en los problemas multi-salida.

La ejecución es reanudable: la clave (sistema, prompt_id, repetición) ya
presente en `runs.jsonl` se salta; los fallos de conexión no ocupan celda y se
reintentan, mientras que timeout y fallo declarado del agente sí cuentan.

Avisos:

- `python -m harness run` con `config.yaml` incluiría también los sistemas
  `manual_*` (se autodescubren de `manual_responses/*.csv`) en la parrilla
  local. La tanda de comerciales va con `config_comerciales.yaml` y sus
  resultados ya están en `results_comerciales/`.
- `perturb` regeneraría las variantes de los prompts 02, 04 y 09; el diseño
  ejecutado se recortó a solo prompt_02 y `prompts_perturbados/` refleja ese
  recorte.
- `n_reps` va a 3 (el protocolo original pedía 5; se redujo por coste, y
  subirlo solo ejecutaría las repeticiones que falten).

## Ejecutar el agente suelto

```bash
python Agente/agente.py prompts/prompt_02_gas_ideal.txt [prompt_id]
# sin argumentos ejecuta un mini-problema de humo
```

Configuración por variables de entorno (todas opcionales): `OLLAMA_URL`,
`MODEL_NAME` (exportar un tag real del servidor, por ejemplo `llama3.2:3b`),
`GENERATION_MODE` (`template` o `libre`, la ablación central del estudio),
`SC_N`, `SC_TEMP`, `AUGMENT` y `AGENT_TIMEOUT`. El detalle completo está en la
cabecera de `Agente/agente.py`. Deja en el directorio de trabajo `log.csv`,
`log.jsonl` (resultado canónico), `train.csv` y los `test_*.py` con el código
ejecutado.

## Sistemas evaluados

| Sistema | Origen |
|---|---|
| `agente_template` / `agente_libre` | `Agente/agente.py` como subproceso aislado (misma tubería; solo cambia quién escribe el código de entrenamiento) |
| `ollama_<modelo>` | POST a Ollama con el prompt exacto como único mensaje, sin system prompt |
| `manual_<nombre>` | CSV pegado a mano en `manual_responses/` |

## Reglas del estudio que el harness garantiza

- Pipeline único de validación para todos los sistemas: una respuesta es
  válida si contiene al menos tantos números finitos extraíbles como salidas
  tiene el problema, y la predicción son los últimos n números (para el
  agente, su `pred_mean` si existe).
- El prompt se envía tal cual, sin instrucciones añadidas.
- `prompts/soluciones.txt` jamás llega a ningún sistema evaluado: solo lo lee
  el harness para calcular errores.
- Todo determinista y reanudable: semillas fijas, ejecuciones hechas se
  saltan, respuesta cruda siempre conservada (hasta 20 000 caracteres, se
  guarda el final y el recorte queda marcado).
- El error se normaliza por el rango de cada columna objetivo calculado del
  CSV del propio prompt; las variantes perturbadas usan los rangos del prompt
  base para mantener la escala comparable.

## Nota sobre el perímetro de la evaluación

El modo directo de `gemma3:12b` y `gpt-oss:20b` quedó fuera del perímetro por
una decisión de validez: el servidor usado cargaba esos modelos con un techo
de contexto de 4096 tokens que truncaba los prompts largos del modo directo
(los prompts internos del agente son cortos y no se ven afectados). Los
directorios de resultados correspondientes contienen por ello solo los
sistemas del agente.
