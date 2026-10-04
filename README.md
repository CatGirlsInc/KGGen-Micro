# KGGen-Micro

Benchmark KGGen with small, locally hosted Ollama models under a consumer-hardware budget of **8 GB VRAM and 16 GB system RAM**. The goal is to establish a reproducible baseline on a fixed 10-record pilot from the published MINE dataset, then recommend whether and how to proceed with a larger benchmark and future MediaWiki extraction.

This is an experimental benchmark, not a production knowledge-graph service. The initial pilot does not include the full MINE dataset, model fine-tuning, or a custom MediaWiki extraction pipeline.

## Local Setup

### Pinned environment

The project uses the CPython 3.13.15 interpreter selected by `.python-version`, plus `kg-gen==0.4.0`, `dspy-ai==3.4.0`, and `datasets==4.4.1`. `uv.lock` pins their transitive dependencies. Let `uv` install the selected interpreter and sync the environment with:

```bash
uv python install
uv sync --locked
uv run scripts/prepare_pilot.py
```

The selector loads the published dataset at immutable revision `aacf51485de5cb6f4a411599f1d0846842c14009`, samples 10 of its 101 training records with seed `20261004`, and saves both source text and IDs to `data/mine-pilot.json`. The selected IDs, in sampling order, are `24, 7, 102, 40, 52, 37, 70, 26, 101, 96`. Re-run the selector only when intentionally regenerating that fixture.

### Ollama and local run

The configured pilot model is `qwen2.5:7b-instruct-q4_K_M` (GGUF `Q4_K_M`, 4-bit), wrapped with a 4096-token context, temperature 0, and seed 42. Pull the source model and create the configured alias:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama create kggen-qwen25-7b:Q4_K_M -f config/Modelfile
uv run scripts/run_pilot.py --limit 1
```

Omit `--limit 1` to attempt all 10 fixed records. Override the model with `--model MODEL_TAG` or `KGGEN_MODEL`; set `OLLAMA_HOST` when the server is not at `http://localhost:11434`. Runs are written under ignored `runs/` with the model digest, settings, record-level status/timing/RSS, sampled GPU memory, graph JSON, KGGen log, and full tracebacks for failures. Failed records remain visible in the summary and do not stop later records. The runner explicitly unloads its selected model in a `finally` cleanup path and records the result in `run.json`; the model files remain installed on disk for subsequent runs. Other Ollama models are left untouched.

With DSPy 3.4.0, its default `lm15` engine returned HTTP 404 for this Ollama endpoint even though both Ollama chat endpoints were healthy. The runner explicitly selects DSPy's `litellm` engine, which completed a direct local chat test and the KGGen run below. The original MINE evaluator is separate: upstream `_1_evaluation.py` configures `openai/gpt-5` and requires `OPENAI_API_KEY`; local graph generation alone does not produce a paper-comparable MINE score. The published dataset includes essays, generated queries, and reference KG/evaluation fields, but this setup does not run or substitute for the hosted evaluator.

### Setup verification

Hardware observed on 2026-10-04: AMD Ryzen 7 3700X, GeForce RTX 2080 with 8 GiB VRAM, and 15 GiB system RAM (about 6.3 GiB available before the run). Ollama client/server reported 0.32.15; the client also displayed a 0.35.0 version warning. The successful local smoke run used model digest `d2d4b544bd31f7e8082779f9e914858a005bf804fe56400338bedf09c05cc24a` on MINE record 24 and produced a graph with extraction and clustering in 263.1 seconds. GPU usage sampled after the run was 5274 MiB of 8192 MiB; this is an observed sample, not a peak measurement. A 3B Llama attempt exposed strict cluster-label parsing failures, so it is not counted as a compatible model. These checks validate the environment and one end-to-end path only; they are not a multi-model benchmark, quality evaluation, or a claim that all 10 records succeed.

## Scope

1. **Set up a reproducible environment.** Install and configure Ollama, KGGen, and dependencies; inspect the original MINE evaluation requirements; pin versions and inference settings; record hardware; and select 10 records with a fixed random seed. Save record identifiers so every model receives identical inputs.
2. **Check model compatibility and resource use.** Test the initial shortlist below through KGGen extraction, clustering, and structured-output stages. Confirm actual memory use and document configuration changes or exclusions before finalizing the shortlist.
3. **Run the pilot.** Process the same 10 records with consistent KGGen settings. Save intermediate extractions, final graphs, execution logs, timings, resource measurements, retries, and failures.
4. **Evaluate and compare.** Reproduce the original MINE evaluation wherever possible, inspect representative outputs, and compare quality, reliability, graph size, speed, and memory use.
5. **Recommend the next phase.** Name a preferred model and configuration, explain the trade-offs and MediaWiki-relevant limitations, and make a reasoned go/no-go decision for the full MINE benchmark.

## Candidate Models

| Model | Pinned Ollama tag | Approx. download size | Role |
| --- | --- | ---: | --- |
| Qwen3.5 4B | `qwen3.5:4b-q4_K_M` | 3.3 GB | Primary small-model candidate |
| Gemma 4 E2B | `gemma4:e2b-it-q4_K_M` | 4.6 GB | Architecture comparison |
| Ministral 3 3B | `ministral-3:3b-instruct-2512-q4_K_M` | 3.0 GB | Compact structured-output candidate |
| Phi-4-mini 3.8B | `phi4-mini:3.8b-q4_K_M` | 2.5 GB | Optional lightweight baseline |
| Qwen3.5 9B | `qwen3.5:9b-q4_K_M` | 6.6 GB | Stretch candidate; test only if it fits the hardware budget |

Download sizes are approximate, not a guarantee of runtime fit. Final selection depends on measured peak VRAM/RAM and compatibility with every required KGGen stage. Use identical generation settings where supported and record any model-specific deviations.

### CAT-79 Compatibility Screening

On 2026-10-04, the pinned tags were screened on MINE record 24 using KGGen 0.4.0, `cluster=True`, chunk size 3000, temperature 0, seed 42, and a 4096-token context. Each completed call exercised extraction, relation generation, clustering, and structured graph output. The resource monitor samples global GPU and host memory once per second from model initialization through unload; process peak RSS is also recorded separately. These are single-record compatibility checks, not the full 10-record pilot or a quality comparison.

| Candidate | Result on record 24 | Elapsed | Peak GPU memory | Peak host memory | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3.5 4B Q4_K_M | Passed | 188.7 s | 4379 / 8192 MiB | 11.4 / 15 GiB | Ollama `think=false` required because reasoning otherwise appeared only in `reasoning_content`, leaving DSPy with an empty response. |
| Gemma 4 E2B Q4_K_M | Passed | 277.5 s | 3654 / 8192 MiB | 11.3 / 15 GiB | No model-specific options. |
| Ministral 3 3B Q4_K_M | Passed | 341.0 s | 4107 / 8192 MiB | 10.9 / 15 GiB | No model-specific options. |
| Phi-4-mini 3.8B Q4_K_M | Passed | 65.8 s | 3694 / 8192 MiB | 11.0 / 15 GiB | No model-specific options. |

All four completed candidates fit within the 8 GiB VRAM and 16 GiB system RAM ceilings in this screening. These results confirm compatibility only on record 24; run the same fixed 10 records per selected model before comparing reliability or quality. The benchmark host reported Ollama server version 0.32.15. New run settings also record the server version. The run-specific settings, model digests, graphs, logs, and resource measurements are in the ignored `runs/` directory on the benchmark machine.

To repeat a one-record check with peak sampling:

```bash
uv run scripts/run_pilot.py --model qwen3.5:4b-q4_K_M --limit 1
uv run scripts/run_pilot.py --model gemma4:e2b-it-q4_K_M --limit 1
uv run scripts/run_pilot.py --model ministral-3:3b-instruct-2512-q4_K_M --limit 1
uv run scripts/run_pilot.py --model phi4-mini:3.8b-q4_K_M --limit 1
```

## Evaluation

Record, at minimum:

| Measure | What it captures |
| --- | --- |
| MINE score | Information retained in the generated graph |
| Extraction success rate | Pipeline reliability across the fixed records |
| Entity and relationship counts | Graph size and output characteristics |
| Clustering reduction | Changes from deduplication and aggregation |
| Time per document | Processing speed |
| Peak VRAM and system RAM | Fit within the hardware budget |
| Retries and failures | Operational reliability and limitations |

If the published evaluation depends on an inaccessible hosted model, keep two result categories distinct: **paper-comparable** results using the original evaluator and settings, and **fully local** results using a fixed Ollama-hosted evaluator. Scores from different evaluators are not directly comparable. Record unavailable dependencies and all deviations from the published method.

Manually inspect representative graphs for incorrect entity merges, missing relationships, and unsupported extractions. Preserve failed outputs and logs; do not silently omit unsuccessful runs from comparisons.

## Reproducibility and Deliverables

The benchmark should leave enough information for another person to repeat and interpret each run:

- Pinned KGGen/dependency versions, Ollama model tags and quantization, inference settings, hardware, context length, and available GPU memory.
- The fixed pilot record identifiers and the seed or selection procedure.
- A configurable runner and documented commands once implementation is in place.
- Intermediate extractions, final graphs, evaluation outputs, resource measurements, and error logs, organized by run/model.
- A comparison report that states the evaluator used, methodology deviations, model exclusions, quality/resource trade-offs, recommendation, and full-benchmark decision.

Acceptance target: evaluate at least three compatible models, or document why fewer qualify; use the same 10 records for each; make successful runs reproducible from pinned configuration; and report quality, runtime, memory, and failures. No benchmark results are claimed here until runs have actually been completed.

## References

- [KGGen implementation](https://github.com/stair-lab/kg-gen)
- [KGGen NeurIPS 2025 paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2b368455e832d2b1a60bcad8c4c6481f-Abstract-Conference.html)
- [MINE evaluation dataset](https://huggingface.co/datasets/josancamon/kg-gen-MINE-evaluation-dataset)
- [Ollama](https://ollama.com/)
- [Project tracking in Linear](https://linear.app/cat-girls-inc/project/kggen-micro-856c7e8aac95)

## Later Work

If pilot quality and resource use justify expansion, run the full MINE dataset with the two strongest models. Then test the selected configuration on a small, manually annotated MediaWiki corpus covering entities, relationships, conditional rules, and mathematical expressions.
