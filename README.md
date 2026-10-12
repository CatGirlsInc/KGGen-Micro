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

The selector loads the published dataset at immutable revision `aacf51485de5cb6f4a411599f1d0846842c14009`, samples 10 of its 101 training records with seed `20261004`, and saves source text and IDs locally to ignored `data/mine-pilot.json`. The selected IDs, in sampling order, are `24, 7, 102, 40, 52, 37, 70, 26, 101, 96`. Regenerate it with `uv run scripts/prepare_pilot.py` when needed; dataset text is not distributed in this repository.

### Ollama and local run

The configured pilot model is `qwen2.5:7b-instruct-q4_K_M` (GGUF `Q4_K_M`, 4-bit), wrapped with a 4096-token context, temperature 0, and seed 42. Pull the source model and create the configured alias:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama create kggen-qwen25-7b:Q4_K_M -f config/Modelfile
uv run scripts/run_pilot.py --limit 1
```

Omit `--limit 1` to attempt all 10 fixed records. Override the model with `--model MODEL_TAG` or `KGGEN_MODEL`; set `OLLAMA_HOST` when the server is not at `http://localhost:11434`. To replay a saved run configuration and its recorded pilot record IDs, pass either its `settings.json` or `run.json`:

```bash
uv run scripts/run_pilot.py --settings runs/RUN_ID/settings.json
```

Use `--limit N` with replay for a partial rerun; `--run-id` can assign a stable, new output directory. Replay verifies that the dataset revision and installed model digest match the saved configuration. Use `--no-cluster` only for extraction diagnostics; the effective setting is recorded, and such a run is not a complete pilot result. Runs are written under ignored `runs/` with the model digest, effective settings, record-level status/timing/RSS, sampled GPU memory, final graph JSON, per-record KGGen intermediate extraction (`intermediate_ID/graph.json`), KGGen log, and full tracebacks for failures. Failed records remain visible in the summary and do not stop later records. The runner explicitly unloads its selected model in a `finally` cleanup path and records the result in `run.json`; the model files remain installed on disk for subsequent runs. Other Ollama models are left untouched.

With DSPy 3.4.0, its default `lm15` engine returned HTTP 404 for this Ollama endpoint even though both Ollama chat endpoints were healthy. The runner explicitly selects DSPy's `litellm` engine, which completed a direct local chat test and the KGGen run below. The original MINE evaluator is separate: upstream `_1_evaluation.py` configures `openai/gpt-5` and requires `OPENAI_API_KEY`; local graph generation alone does not produce a paper-comparable MINE score. The published dataset includes essays, generated queries, and reference KG/evaluation fields, but this setup does not run or substitute for the hosted evaluator.

To configure the original MINE evaluator once on Linux, run `bash scripts/setup_openai_key.sh` in a terminal and enter the key at its hidden prompt. The script saves it outside the repository at `~/.config/openai/openai.env` with owner-only permissions; new interactive Bash shells load that file through `~/.bashrc`. No key belongs in this repository, shell command arguments, or chat. Open a new terminal (or source the env file in an existing one) before running an evaluator process.

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

All four completed candidates fit within the 8 GiB VRAM and 16 GiB system RAM ceilings in this screening. The full fixed 10-record pilot results are summarized below. The benchmark host reported Ollama server version 0.32.15. Run settings record the server version; run-specific settings, model digests, graphs, logs, and resource measurements are in the ignored `runs/` directory on the benchmark machine.

To repeat a one-record check with peak sampling:

```bash
uv run scripts/run_pilot.py --model qwen3.5:4b-q4_K_M --limit 1
uv run scripts/run_pilot.py --model gemma4:e2b-it-q4_K_M --limit 1
uv run scripts/run_pilot.py --model ministral-3:3b-instruct-2512-q4_K_M --limit 1
uv run scripts/run_pilot.py --model phi4-mini:3.8b-q4_K_M --limit 1
```

### CAT-78 Full Pilot Results

On 2026-10-04, all four compatibility-screened models were run on the same 10 records with KGGen 0.4.0, chunk size 3000, `cluster=True`, temperature 0, seed 42, 4096-token context, and DSPy `litellm` engine. Qwen3.5 used the previously established `think=false` option. Each run wrote per-record extraction graphs, final graphs where clustering succeeded, settings, timing/resource manifests, KGGen logs, and full error logs for failures under ignored `runs/`.

| Model | Succeeded | Failed | Mean seconds/record | Total record time | Peak GPU | Peak system memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ministral 3 3B Q4_K_M | 8/10 | 2 | 325.2 s | 54.2 min | 4147 / 8192 MiB | 11.09 / 15 GiB |
| Gemma 4 E2B Q4_K_M | 9/10 | 1 | 294.9 s | 49.1 min | 3670 / 8192 MiB | 11.31 / 15 GiB |
| Phi-4-mini 3.8B Q4_K_M | 10/10 | 0 | 118.8 s | 19.8 min | 3704 / 8192 MiB | 11.48 / 15 GiB |
| Qwen3.5 4B Q4_K_M | 8/10 | 2 | 206.7 s | 34.4 min | 4406 / 8192 MiB | 11.82 / 15 GiB |

Peak GPU and system figures are global samples taken once per second; system-memory values are host-wide, not process-only. The listed total is the sum of per-record elapsed time, excluding setup and cleanup. All four models stayed within the 8 GiB VRAM and 16 GiB system RAM ceilings. Failure records were Ministral 7 and 70, Gemma 40, and Qwen3.5 40 and 37. They remain represented by `record_ID.json`, intermediate extraction graphs, and `error_ID.log`; failures were not retried manually or silently dropped. Phi-4-mini was the only model to complete clustering on all 10 records.

The observed failures were DSPy `AdapterParseError`s from model responses that violated KGGen's strict structured-output literals. Ministral failed during edge clustering on records 7 and 70; Gemma failed during clustering on record 40; Qwen3.5 failed during extraction on records 40 and 37. These are operational success results, not a quality ranking. The original MINE evaluator requires `openai/gpt-5` and `OPENAI_API_KEY`; no MINE scores or paper-comparable quality claims were produced. DSPy 3.4.0's configured budget is three transient request retries, recorded in the runner's settings. Where the inference stack exposes reliable retry telemetry, observed counts should be recorded; otherwise, preserve raw interaction logs and report counts as unavailable rather than inferring them.

During diagnosis, KGGen 0.4.0's clustering loop was found to reset its no-progress counter when a nonempty validated cluster contained only items already removed from the remaining set. A stale model suggestion could therefore prevent the configured no-progress limit from firing. The runner now intersects suggested and validated clusters with remaining items before accepting progress and falls back to KGGen's existing batch path after the limit. A regression test covers the stale-suggestion case. This guard does not suppress model parse failures; it bounds the stale-response loop while preserving unclustered items.

Reproduce a full run from its recorded settings (use the appropriate run directory):

```bash
uv run scripts/run_pilot.py --settings runs/cat78-phi4-full-phi4-mini-3.8b-q4_K_M/settings.json
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

The completed CAT-80 comparison is in [reports/cat80-evaluation.md](reports/cat80-evaluation.md). It reports the upstream MINE retrieval-QA results on the fixed 10-record pilot, separately labeled local judgments, operational and graph-size/clustering measurements, failures, and a bounded recommendation. The CAT-81 decision and confirmation plan are in [reports/cat81-benchmark-plan.md](reports/cat81-benchmark-plan.md): Phi-4-mini is the operational baseline, Qwen3.5 is the quality challenger, and the full 101-record run is gated on a larger paired comparison. CAT-108's 20-record generation results and no-go decision are in [reports/cat108-confirmation-benchmark.md](reports/cat108-confirmation-benchmark.md). Generate the ignored local cohort `data/mine-confirmation-cohort.json` with `uv run scripts/prepare_pilot.py && uv run scripts/prepare_confirmation_cohort.py`, then pass it to the runner with `--cohort data/mine-confirmation-cohort.json`. CAT-114 manifests are similarly regenerated by `uv run scripts/prepare_cat114_cohort.py` and `uv run scripts/prepare_cat114_heldout.py`. These scripts download the pinned dataset to build local inputs; review and follow the dataset provider's terms. Reproduce the original-procedure evaluation against saved graphs with `uv run scripts/evaluate_mine_saved_graphs.py RUN_DIR... --run-id RUN_ID --resume` after providing `OPENAI_API_KEY` to the process. Per-answer judgments and retrieved contexts are preserved under ignored `runs/`; API token/cost telemetry was not captured. Recompute operational summaries with `uv run scripts/summarize_runs.py RUN_DIR...`; reproduce the local judgment with `uv run scripts/evaluate_local.py RUN_DIR... --run-id RUN_ID` after installing the recorded evaluator model in Ollama. Local judgments are not paper-comparable MINE scores.

## Reproducibility and Deliverables

The benchmark should leave enough information for another person to repeat and interpret each run:

- Pinned KGGen/dependency versions, Ollama model tags and quantization, inference settings, hardware, context length, and available GPU memory.
- The fixed pilot record identifiers and the seed or selection procedure.
- A configurable runner and documented commands once implementation is in place.
- Intermediate extractions, final graphs, evaluation outputs, resource measurements, and error logs, organized by run/model.
- A comparison report that states the evaluator used, methodology deviations, model exclusions, quality/resource trade-offs, recommendation, and full-benchmark decision.

The CAT-78 pilot execution, CAT-80 original-procedure MINE retrieval-QA pilot, and separate local-judge evaluation are complete for four compatible models on the same 10 records. The original procedure was applied to saved graphs for the sampled subset only; the full 101-record MINE evaluation remains unrun. The report documents this scope, the local-judge limitations, and the recommendation.

CAT-108 is closed as a documented 4096-token no-go. Its 8192-token diagnostics are separate from the controlled CAT-114 experiment, which uses distinct cohort manifests, settings, run IDs, and per-attempt artifacts. Regenerate the 20-record development manifest with `uv run scripts/prepare_cat114_cohort.py`; its settings are `config/cat114-8k-phi4-settings.json` and `config/cat114-8k-qwen35-settings.json`. The 8K development and non-overlapping held-out generation runs and original-MINE evaluations are complete. The user explicitly adopted the AI-assisted five-record evidence audit as a substitute for a separate human audit; its provenance and findings are recorded in ignored `runs/cat114-human-audit/packet.md`. It is not represented as independent human review. The held-out manifest and settings are generated by `uv run scripts/prepare_cat114_heldout.py`; the paired bootstrap analysis is reproducible with `scripts/analyze_paired_mine.py` against the saved evaluator summaries. CAT-114 is complete with a no-go decision; the full 101-record MINE expansion remains unauthorized because held-out reliability fails the threshold and paired quality remains inconclusive.

The generation commands use unique run IDs and do not overwrite the CAT-108 runs:

```bash
uv run --locked scripts/run_pilot.py --settings config/cat114-8k-phi4-settings.json --run-id cat114-8k-phi4-development
uv run --locked scripts/run_pilot.py --settings config/cat114-8k-qwen35-settings.json --run-id cat114-8k-qwen35-development
uv run --locked scripts/prepare_cat114_heldout.py
uv run --locked scripts/run_pilot.py --settings config/cat114-heldout-phi4-settings.json --run-id cat114-8k-phi4-heldout
uv run --locked scripts/run_pilot.py --settings config/cat114-heldout-qwen35-settings.json --run-id cat114-8k-qwen35-heldout
```

## Results and Conclusion

### Hypothesis and decision

The project tested whether a small, locally hosted, quantized model could generate useful KGGen graphs reliably within the 8 GiB VRAM / 16 GB system RAM budget, with original-procedure MINE retrieval-QA results strong enough to justify expanding to all 101 records. **That hypothesis was not supported for the tested models and settings.** The resource ceiling was met, but neither candidate passed the held-out reliability gate of at least 19/20 successful records, and neither paired MINE comparison established a quality winner. Do not launch the 101-record expansion from these results.

### Main comparison

| Experiment | Model | Graph generation | MINE correct / evaluated answers |
| --- | --- | ---: | ---: |
| CAT-108, 4K baseline, n=20 | Phi-4-mini | 18/20 | 155/270 (57.4%) |
| CAT-108, 4K baseline, n=20 | Qwen3.5 | 13/20 | 139/195 (71.3%) |
| CAT-114, 8K development, n=20 | Phi-4-mini | 14/20 | 121/210 (57.6%) |
| CAT-114, 8K development, n=20 | Qwen3.5 | 19/20 | 198/285 (69.5%) |
| CAT-114, 8K held-out, n=20 | Phi-4-mini | 14/20 | 106/210 (50.5%) |
| CAT-114, 8K held-out, n=20 | Qwen3.5 | 17/20 | 119/195 (61.0%) |

| Paired comparison | Paired records | Qwen3.5 minus Phi-4-mini mean | 95% record-bootstrap CI |
| --- | ---: | ---: | ---: |
| CAT-108, 4K baseline | 12 | +11.7 pp | -6.7 to +28.9 pp |
| CAT-114, 8K development | 13 | +9.2 pp | -7.2 to +26.2 pp |
| CAT-114, 8K held-out | 9 | +11.9 pp | -15.6 to +33.3 pp |

“Paired” results use only records with valid graphs for both models. Failed generations are reported separately and are not assigned zero MINE accuracy. All three 95% intervals include zero, so the observed Qwen advantages do not establish superiority. The 8K development set reuses the frozen CAT-108 cohort after those baseline failures were observed, so it is a development confirmation; the separately frozen, non-overlapping held-out set is a held-out check of the 8K configuration, which was selected after observing CAT-108 results. Qwen reached 19/20 only on development and fell to 17/20 held-out. Phi reached 14/20 in both 8K sets. The resource ceilings passed in all CAT-114 runs, with sampled peaks of 4.27–4.68 GiB VRAM and 11.40–12.73 GiB system RAM.

### Method and relationship to KGGen

The experiment uses KGGen 0.4.0's extraction and entity/relation clustering pipeline [Mo et al., 2025](https://doi.org/10.52202/085713-1010) on the pinned MINE essays, preserving the same input cohort across candidate models within each comparison. Shared generation settings were chunk size 3000, clustering enabled, temperature 0, and seed 42. CAT-108 used 4096-token context/output limits; the separate CAT-114 configuration used 8192-token limits, with up to two recorded whole-record retries. Models were local Ollama Q4_K_M quantizations; Qwen3.5 required `think=false`. Model/provider, quantization, token limits, dataset subset, and hardware-constrained scope differ from the paper's broader benchmark, so this is not a reproduction of its full model comparison.

Quality evaluation follows the upstream MINE retrieval-QA path pinned to KGGen revision `6259b4c71523ceae75dafb4acec2eb833bef8f8e`: use the dataset's 15 generated reference answers per essay, retrieve graph context with `all-MiniLM-L6-v2` at `k=8`, then score answer presence with the upstream DSPy Chain-of-Thought evaluator (`openai/gpt-5`, high reasoning, temperature 1, max tokens 16000). The paired uncertainty intervals resample records (50,000 bootstrap draws, seed 108). This preserves the paper's MINE task and evaluator approach as closely as possible while testing locally hosted graph generators; it does not compare our models against the paper's reported systems on the full dataset. Exact cohorts, run settings, failures, and detailed results are in [reports/cat108-confirmation-benchmark.md](reports/cat108-confirmation-benchmark.md).

The user-approved AI-assisted audit of five selected 8K development records found Qwen more useful on four, but neither model acceptable on record 37. Both showed material graph-correctness problems, especially destructive entity/predicate canonicalization and false or reversed relations. This qualitative review substitutes for a separate human audit only by explicit user authorization; it is not independent human review and is not a statistical quality result. Together with the held-out reliability shortfall and inconclusive paired MINE evidence, the audit supports the no-go decision rather than a model qualification.

## References

- [KGGen implementation](https://github.com/stair-lab/kg-gen)
- Mo, B., Yu, K., Kazdan, J., Mpala, P., Yu, L., Kanatsoulis, C., & Koyejo, S. (2025). KGGen: Extracting Knowledge Graphs from Plain Text with Language Models. *Advances in Neural Information Processing Systems 38*. https://doi.org/10.52202/085713-1010. [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2b368455e832d2b1a60bcad8c4c6481f-Abstract-Conference.html)

```bibtex
@inproceedings{mo2025kggen,
  title = {KGGen: Extracting Knowledge Graphs from Plain Text with Language Models},
  author = {Mo, Belinda and Yu, Kyssen and Kazdan, Joshua and Mpala, Proud and Yu, Lisa and Kanatsoulis, Charilaos and Koyejo, Sanmi},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {38},
  year = {2025},
  doi = {10.52202/085713-1010}
}
```
- MINE evaluation dataset by `josancamon`, pinned revision [`aacf51485de5cb6f4a411599f1d0846842c14009`](https://huggingface.co/datasets/josancamon/kg-gen-MINE-evaluation-dataset/tree/aacf51485de5cb6f4a411599f1d0846842c14009). The upstream dataset card at this revision does not specify a license. Dataset-derived files are downloaded/generated locally and excluded from this source tree; review the provider's terms before use or redistribution.
- [Ollama](https://ollama.com/)

## Public Release Readiness

The paper citation and pinned data provenance are documented, generated run artifacts and dataset-derived cohort JSONs are excluded from the current source tree and reachable Git history, and no credential patterns were found in tracked files or committed history during the release check. The MIT license in [LICENSE](LICENSE) covers original KGGen-Micro source, scripts, configuration, tests, and documentation, excluding third-party materials and dataset-derived JSON fixtures; it does not relicense KGGen or its dependencies. See [NOTICE](NOTICE) for acknowledgments and dataset provenance. The dataset card does not specify a license; users must review the provider's terms before downloading or using its records. Do not include API keys or local run artifacts in a release.

## Later Work

The current experiments do not authorize a full MINE run. Reconsider expansion only after a separately documented intervention improves held-out structured-output reliability and a new paired evaluation supports quality under record-level uncertainty. MediaWiki evaluation remains future work and requires a small, manually annotated corpus covering entities, relationships, conditional rules, and mathematical expressions.
