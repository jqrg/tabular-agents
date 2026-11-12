# Evaluating LLM agents on numerical prediction from tabular data

**English** | [Español](README.es.md)

This repository contains the code and data for Iván Vega Prieto’s master’s thesis (MUIA, UAX). The study compares two ways to predict numerical values: an LLM agent writes and executes code to train a neural network, while other LLMs predict the values directly. Each of the ten benchmark problems includes its data table in the prompt.

## Repository contents

- `Agente/agente.py` — the agent evaluated in the study. It combines a multistep LlamaIndex workflow with CodeActAgent, uses Ollama for inference, and trains networks with PyTorch. The module header describes the workflow, configuration, and changes from the prototype (M1–M14). The agent is frozen for evaluation.
- `prototipo/agenteJordan.py` — the research group’s earlier prototype, included for reference and excluded from the evaluation. Only the internal server URL has been anonymized.
- `harness/` — the Python package and CLI that run each combination of system, problem, and repetition. The harness validates all responses through the same procedure, computes metrics, and produces statistical analyses and figures.
- `prompts/` — the ten benchmark problems (`prompt_01` through `prompt_10`), the reference answers in `soluciones.txt` (for the harness only), and `generar_dataset.py`, which documents how the dataset was generated. You do not need to run this script.
- `prompts_perturbados/` — the P1–P6 robustness variants used in the study. Only problem 02 was included. To preserve that selection, keep the existing files; see the note on `perturb` below.
- `manual_responses/` — responses collected manually from commercial models, with one CSV per system and the original conversation text. The harness processes them in the same way as other responses.
- `results*/` — batch outputs: `runs.jsonl` contains one JSON record per run, including the raw response, and `summary.csv` contains the summary. Batch directories are `results_llama3_23b/`, `results_qwen25/`, `results_llama31/`, `results_gemma3_12b/`, `results_gptoss20b/`, and `results_comerciales/`. The Gemma and GPT-OSS batches contain agent runs only. Historical records and the runs added in November 2026 remain in `results/`; they are not automatically migrated.
- `tests/` — 58 harness tests that run without network access or Ollama.

## Installation

Use Python 3.10 or later.

```bash
# These dependencies are sufficient for harness tests and analysis.
pip install -r requirements.txt

# Additional dependencies for running the agent. Keep the LlamaIndex versions
# shown here: other versions handle context differently.
pip install llama-index-llms-ollama==0.10.1 llama-index-core==0.14.21 python-dotenv torch scikit-learn
```

The agent and direct model runs require access to an Ollama server. The supplied `config*.yaml` files use `http://localhost:11434` in place of the original server address. Before running an evaluation, check the address and verify that the configured model tags are available on the server (`ollama list`).

## Using the harness

Run these commands from the repository root:

```bash
python -m pytest tests -q                         # Run the offline tests
python -m harness run --dry-run                   # List pending runs
python -m harness run                             # Run or resume the evaluation
python -m harness run --sistema agente_template   # Select a system; the flag is repeatable
python -m harness perturb                         # Generate P1–P6 variants (see below)
python -m harness run --perturbados               # Evaluate the robustness variants
python -m harness analyze                         # Generate summary.csv, analysis.md, and figures
```

Each batch has its own configuration and results directory. The default, `harness/config.yaml`, uses `llama3.2:3b`. The alternatives in `harness/` are `config_qwen25.yaml`, `config_llama31.yaml`, `config_gemma3_12b.yaml`, `config_gptoss20b.yaml`, and `config_comerciales.yaml`. Place `--config` before the subcommand:

```bash
python -m harness --config harness/config_qwen25.yaml analyze
```

The `analyze` command reads `runs.jsonl` and regenerates `analysis.md`, the figures, and `summary.csv` in the results directory. It reports results by system and problem, paired Wilcoxon tests with Holm correction, Fisher’s exact test for structural failures, agent calibration by configuration, and errors for each output of problems with multiple outputs.

When resuming, the harness skips combinations of system, prompt ID, and repetition already recorded in `runs.jsonl`. It retries connection failures. Timeouts and failures reported by the agent count as completed runs.

- The default `run` command also discovers `manual_*` systems from `manual_responses/*.csv`. The commercial-model batch uses `config_comerciales.yaml`; its results are already in `results_comerciales/`.
- **Running `perturb` generates variants for problems 02, 04, and 09.** The study used only the variants of problem 02, as reflected in the supplied `prompts_perturbados/` directory.
- The main configuration sets `n_reps` to 3. The original protocol called for 5, but the count was reduced to limit cost. Increasing it runs only the missing repetitions.

## Running the agent on its own

```bash
python Agente/agente.py prompts/prompt_02_gas_ideal.txt [prompt_id]
```

The prompt ID is optional. With no arguments, the agent runs a small smoke test.

Optional environment variables are `OLLAMA_URL`, `MODEL_NAME` (set it to a tag available on your server, such as `llama3.2:3b`), `GENERATION_MODE` (`template` or `libre`, the two modes compared in the ablation study), `SC_N`, `SC_TEMP`, `AUGMENT`, and `AGENT_TIMEOUT`. The header of `Agente/agente.py` documents each setting. The agent writes `log.csv`, `log.jsonl` (the canonical result), `train.csv`, and the executed `test_*.py` files to the working directory.

## Systems evaluated

| System | How it produces a response |
|---|---|
| `agente_template` / `agente_libre` | Runs `Agente/agente.py` in an isolated subprocess. Both modes use the same workflow; only the source of the training code differs. |
| `ollama_<model>` | Sends the original prompt to Ollama in a POST request as the only message, with no system prompt. |
| `manual_<name>` | Reads manually collected responses from a CSV in `manual_responses/`. |

## Evaluation rules

- All systems use the same validation procedure. For a problem with *n* outputs, a text response must contain at least *n* extractable finite numbers. The last *n* numbers form the prediction. For the agent, the harness uses its structured `pred_mean` when available.
- The prompt is sent verbatim, without additional instructions.
- Only the harness reads `prompts/soluciones.txt` to calculate errors. No evaluated system receives the reference answers.
- The harness uses fixed seeds, skips completed runs, and retains raw responses. For responses longer than 20,000 characters, it keeps the last 20,000 and marks the record as truncated.
- Each target column’s error is normalized by its range in the prompt’s CSV data. Perturbed variants use the ranges from the original prompt so their errors remain comparable.

## Scope of the evaluation

The study excluded direct predictions from `gemma3:12b` and `gpt-oss:20b`. The server loaded these models with a 4,096-token context limit, which truncated long prompts in direct mode. The agent’s shorter internal prompts were unaffected. The corresponding results directories therefore contain only the agent runs.
