# CAT-80: Model Evaluation and Comparison

**Evaluation date:** 2026-10-08 (original MINE evaluation added)
**Dataset:** `josancamon/kg-gen-MINE-evaluation-dataset`, revision `aacf51485de5cb6f4a411599f1d0846842c14009`  
**Records:** fixed 10-record pilot: `24, 7, 102, 40, 52, 37, 70, 26, 101, 96`

## Executive Summary

The original MINE retrieval-QA evaluation has now been run for all available saved graphs on the fixed pilot. Among successful graphs, Qwen3.5 4B scored highest at 70.0% (84/120 answers), followed by Ministral 3 3B at 66.7% (80/120), Phi-4-mini 3.8B at 58.0% (87/150), and Gemma 4 E2B at 36.3% (49/135). These are pilot results from the upstream procedure, not full-dataset results. The original evaluator was not run on the five failed generations; generation success rates therefore remain a separate required comparison.

For a separate local judgment, a fixed Ollama `qwen2.5:7b-instruct-q4_K_M` evaluator reviewed every available final graph against its source essay. Qwen3.5 4B had the highest mean across the four judgment axes (3.81/5 across 8 graphs), followed by Ministral 3 3B (3.66/5, 8 graphs), Gemma 4 E2B (3.58/5, 9 graphs), and Phi-4-mini 3.8B (3.23/5, 10 graphs). This remains a provisional local-judge ranking, not a substitute for the original MINE results. Manual cross-checking found the local judge incorrectly called an edge in Qwen3.5's rocket graph unsupported even though that exact edge is in the graph and source.

Phi-4-mini is the strongest operational pilot choice: it completed all 10 records, was fastest (118.8 seconds per attempted record on average), and stayed within the observed hardware limits. Qwen3.5 is the candidate to recheck for quality, not an established quality winner: it had 2 clustering/extraction failures, the highest sampled peak GPU use, and the local judge is demonstrably fallible. Do not start the full-dataset benchmark based on this pilot alone. First use human-reviewed examples to calibrate the local evaluator or obtain access to the original MINE evaluator, then repeat the comparison on a larger sample.

## Method

The four CAT-78 complete pilot runs share the same 10 source records, KGGen 0.4.0, 3000-token chunk size, clustering enabled, temperature 0, seed 42, and 4096-token generation context. Qwen3.5 used `think=false`; this model-specific option was required for nonempty structured responses. Models were run sequentially on the same host. The original run manifests provide model digests, timings, sampled global GPU/system memory, process peak RSS, and per-record outcomes.

The original procedure was taken from `stair-lab/kg-gen` commit `6259b4c71523ceae75dafb4acec2eb833bef8f8e` (`experiments/MINE/_1_evaluation.py`). It retrieves context for each of the dataset's 15 generated reference answers using `all-MiniLM-L6-v2` with the upstream default `k=8`, then uses DSPy `ChainOfThought` and hosted `openai/gpt-5` (reasoning effort `high`, temperature 1.0, max tokens 16000) to classify whether the context contains the answer. The dataset is pinned to revision `aacf51485de5cb6f4a411599f1d0846842c14009`. This pilot adapter evaluates the already-generated CAT-78 graphs rather than invoking upstream's default local graph-generation mode; retrieval and answer scoring follow the original path. It covers only the fixed 10 sampled records, not all 101 essays. No graph-generation or evaluator calls were repeated for records whose saved generation had failed.

Per-query answers and retrieved contexts, per-record accuracy, failures, and settings are preserved on the benchmark host in `runs/cat80-mine-gpt5-openai/`. Run the same saved-graph evaluation with:

```bash
uv run scripts/evaluate_mine_saved_graphs.py \
  runs/cat78-ministral3-full-ministral-3-3b-instruct-2512-q4_K_M \
  runs/cat78-gemma4-full-gemma4-e2b-it-q4_K_M \
  runs/cat78-phi4-full-phi4-mini-3.8b-q4_K_M \
  runs/cat78-qwen35-full-qwen3.5-4b-q4_K_M \
  --run-id cat80-mine-gpt5-openai --resume
```

The adapter preserves the upstream output fields and adds record-level metadata for reproducible pilot aggregation. Per-request latency and OpenAI token/cost usage were not captured during this run, so no cost or evaluator-throughput comparison is claimed.

The separate local evaluation used Ollama `qwen2.5:7b-instruct-q4_K_M`, digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`, Ollama 0.32.15, temperature 0, seed 42, 8192-token context, and a 900-token output limit. It scored fidelity, coverage, relation quality, and entity quality from 1 to 5 against the source essay. It did not compare against the dataset's reference answers and is not an original MINE score.

The full local-judge output and settings are preserved on the benchmark host under `runs/cat80-local-eval-qwen25-7b-q4_K_M/`. Recreate the local judgments with:

```bash
uv run scripts/evaluate_local.py \
  runs/cat78-ministral3-full-ministral-3-3b-instruct-2512-q4_K_M \
  runs/cat78-gemma4-full-gemma4-e2b-it-q4_K_M \
  runs/cat78-phi4-full-phi4-mini-3.8b-q4_K_M \
  runs/cat78-qwen35-full-qwen3.5-4b-q4_K_M \
  --run-id cat80-local-eval-qwen25-7b-q4_K_M
```

To recompute success, timing, memory, graph-size, and clustering summaries from run artifacts:

```bash
uv run scripts/summarize_runs.py \
  runs/cat78-ministral3-full-ministral-3-3b-instruct-2512-q4_K_M \
  runs/cat78-gemma4-full-gemma4-e2b-it-q4_K_M \
  runs/cat78-phi4-full-phi4-mini-3.8b-q4_K_M \
  runs/cat78-qwen35-full-qwen3.5-4b-q4_K_M
```

## Operational and Graph Metrics

| Generator model | Success | Mean seconds/attempt | Peak GPU sampled | Peak system memory sampled | Mean final entities | Mean final relations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ministral 3 3B Q4_K_M | 8/10 | 325.2 | 4147 / 8192 MiB | 11.09 / 15 GiB | 25.4 (8 successes) | 52.3 (8 successes) |
| Gemma 4 E2B Q4_K_M | 9/10 | 294.9 | 3670 / 8192 MiB | 11.31 / 15 GiB | 7.4 (9 successes) | 21.0 (9 successes) |
| Phi-4-mini 3.8B Q4_K_M | 10/10 | 118.8 | 3704 / 8192 MiB | 11.48 / 15 GiB | 9.9 (10 successes) | 22.0 (10 successes) |
| Qwen3.5 4B Q4_K_M | 8/10 | 206.7 | 4406 / 8192 MiB | 11.82 / 15 GiB | 24.6 (8 successes) | 38.1 (8 successes) |

Peak GPU and system figures are once-per-second global samples, not isolated process peaks. Process peak RSS is also recorded in the run manifests. All four models remained under the 8 GiB VRAM and 16 GiB system RAM ceilings. Mean graph counts are descriptive, not quality scores. High counts can reflect useful coverage, verbosity, duplication, or extraction errors.

| Generator model | Mean local fidelity | Mean local coverage | Mean local relation quality | Mean local entity quality | Judged graphs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ministral 3 3B Q4_K_M | 3.88 | 3.88 | 3.00 | 3.88 | 8/10 |
| Gemma 4 E2B Q4_K_M | 3.89 | 3.67 | 2.89 | 3.89 | 9/10 |
| Phi-4-mini 3.8B Q4_K_M | 3.60 | 3.30 | 2.60 | 3.40 | 10/10 |
| Qwen3.5 4B Q4_K_M | 4.00 | 3.88 | 3.38 | 4.00 | 8/10 |

Means exclude failed generations, so model reliability must be considered separately. There was one local-judge pass per graph and no inter-rater agreement or calibration study. Given the demonstrated false criticism below, small differences in these scores should not be interpreted as reliable quality separation.

## Original MINE Retrieval QA

The upstream binary question-answering metric is reported for successful saved graphs only. Each graph was evaluated against 15 generated answers; a failed graph has no MINE score and is not counted as 0%.

| Generator model | MINE-correct answers / evaluated answers | Mean record accuracy | Successful graphs / 10 |
| --- | ---: | ---: | ---: |
| Ministral 3 3B Q4_K_M | 80 / 120 | 66.7% | 8/10 |
| Gemma 4 E2B Q4_K_M | 49 / 135 | 36.3% | 9/10 |
| Phi-4-mini 3.8B Q4_K_M | 87 / 150 | 58.0% | 10/10 |
| Qwen3.5 4B Q4_K_M | 84 / 120 | 70.0% | 8/10 |

Qwen3.5 has the highest observed answer accuracy but also failed to produce graphs for 2/10 records. Phi-4-mini has the highest end-to-end availability and second-highest total correct answers, despite lower accuracy on the subset of graphs that succeeded. Thus the operational baseline and the accuracy leader differ. This remains a 10-record pilot with one LLM judge and 15 questions per record; use the answer-level evidence and failure rates together, and do not generalize it to the full MINE dataset.

## Clustering and Graph Size

Means below compare extraction and final graphs only for records where both were successfully produced. Clustering reduction is `(mean extracted count - mean final count) / mean extracted count` and is descriptive; KGGen clustering can change labels and relations, so this is not a semantic quality measure.

| Generator model | Paired successes | Extracted -> final entities | Entity reduction | Extracted -> final relations | Relation reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ministral 3 3B Q4_K_M | 8 | 59.0 -> 25.4 | 57.0% | 55.1 -> 52.3 | 5.2% |
| Gemma 4 E2B Q4_K_M | 9 | 32.4 -> 7.4 | 77.1% | 27.6 -> 21.0 | 23.8% |
| Phi-4-mini 3.8B Q4_K_M | 10 | 48.3 -> 9.9 | 79.5% | 27.2 -> 22.0 | 19.1% |
| Qwen3.5 4B Q4_K_M | 8 | 29.5 -> 24.6 | 16.5% | 40.1 -> 38.1 | 5.0% |

The substantial variation in entity reduction shows that the final graph shape is model-dependent even with common KGGen settings. Ministral retains very large graphs; Gemma and Phi cluster entity lists much more aggressively; Qwen3.5 also retains most extracted entities. Graph size alone cannot decide which is most useful.

## Failures and Manual Inspection

All five CAT-78 failed records were preserved. Failures were structured-output parsing errors, not resource-limit failures:

| Generator model | Failed records | Observed failure stage/example |
| --- | --- | --- |
| Ministral 3 3B Q4_K_M | 7, 70 | Clustering returned an edge label outside KGGen's allowed literal set. |
| Gemma 4 E2B Q4_K_M | 40 | Clustering returned `Gluten structure` where the validated entity was `gluten structure`. |
| Qwen3.5 4B Q4_K_M | 40, 37 | Structured relation/extraction response failed DSPy parsing; record 37 omitted a required relation object field. |
| Phi-4-mini 3.8B Q4_K_M | None | All ten records completed clustering. |

Manual spot checks included the shared social-media essay (record 24) for all four models and the rocket-building essay (record 26) for Qwen3.5 and Phi-4-mini. On record 24, Gemma's final graph was unusually compressed (3 entities and 15 relations in the saved graph), while the other models retained 14-51 entities; this is a coverage risk, not inherently a defect. Ministral's record-24 graph retained many entities and broad, repeated relation labels, making it less compact. Qwen3.5's record-26 graph retained the ordered build steps and component relations, but also included peripheral/duplicative entities and claims not clearly central to the essay. Phi-4-mini's record-26 graph was smaller and easier to scan, but omitted much of the essay's step-by-step detail and used broad `provides` relations.

The local judge's record-26 Qwen3.5 review itself demonstrates why human checks matter: it listed `Building a Rocket -> has -> Step 2: Design the Rocket` as unsupported, although that exact triple appears in both the saved graph and source essay. Its record-26 Phi-4 review also described `provides` relations that do not match all of that graph's actual relation labels. The judge output is retained as evidence but those explanations are not treated as ground truth.

Retry counts are not available per attempt in the CAT-78 manifests; the configured DSPy retry budget is three. Do not infer actual retry counts from the final errors.

## Recommendation

1. **Operational baseline:** use Phi-4-mini 3.8B Q4_K_M when completion rate and turnaround are primary. It completed all 10 records and had the lowest runtime.
2. **Accuracy candidate:** prioritize Qwen3.5 4B Q4_K_M for a larger MINE follow-up because it had the highest original-procedure pilot answer accuracy (70.0%), while preserving and addressing its 2/10 generation failures. Its weaker completion rate means it is not yet an unconditional winner.
3. **Full-dataset decision:** this pilot now provides a useful ranking signal, but a 10-record subset is too small to justify immediate full-dataset expansion. First investigate Qwen3.5's structured-output failures and confirm the ranking on a larger identical sample, ideally with a second evaluator or manual audit. Preserve failures and resource measurements.

The original MINE evaluation makes Qwen3.5 the current quality candidate and Phi-4-mini the reliability-speed choice; no conclusion here establishes performance on MediaWiki content, conditional rules, or mathematical expressions.