# CAT-108: 20-Record Model Confirmation Benchmark

**Run date:** 2026-10-08
**Dataset:** `josancamon/kg-gen-MINE-evaluation-dataset`, revision `aacf51485de5cb6f4a411599f1d0846842c14009`
**Outcome:** **Experiment complete; no-go for the 101-record run.** Neither 4096-token baseline met the 19/20 reliability threshold (Phi-4-mini 18/20; Qwen3.5 13/20). The paired MINE estimate favors Qwen3.5 by 11.7 percentage points, but its record-bootstrap 95% CI is -6.7 to +28.9 pp and includes zero. Memory use was within the stated limits. The original 4K audit was not completed because Qwen baseline graphs for required records 37 and 40 were missing. A separate 8K audit was prepared and later replaced by the user-adopted CAT-114 AI-assisted evidence review; it is not an independent human audit and does not alter the 4K baseline. Separate 8192-token diagnostic recoveries do not modify or count toward the original baseline. **The full 101-record expansion is not authorized.**

## Frozen Cohort

The generated local manifest is `data/mine-confirmation-cohort.json`, created by `uv run scripts/prepare_pilot.py` followed by `uv run scripts/prepare_confirmation_cohort.py`. Dataset-derived JSON files are ignored and not distributed. The first ten IDs preserve the original pilot order; ten unused IDs were selected by `random.Random(20261008).sample(remaining_ids, 10)` in dataset order and appended:

`24, 7, 102, 40, 52, 37, 70, 26, 101, 96, 44, 6, 64, 35, 22, 42, 13, 32, 46, 100`

The cohort builder validates the original pilot contents against the immutable dataset revision before writing the manifest. Both model runs used that same manifest and exact ordered IDs.

## Generation Results

Both runs used KGGen 0.4.0, clustering, chunk size 3000, temperature 0, seed 42, 4096-token context and output limit, DSPy `litellm`, and the same host. Qwen3.5 retained its required Ollama `think=false` option. Model tags and digests, per-record results, final and intermediate graphs, complete logs, tracebacks, timings, and resource samples are preserved in the ignored `runs/` artifacts on the benchmark host.

| Model | Successful | Failed | Failed IDs (stage) | Mean seconds/attempt | Peak sampled VRAM | Peak sampled host RAM | Process peak RSS |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Phi-4-mini 3.8B Q4_K_M | 18/20 | 2 | 6 (extraction), 64 (clustering) | 85.5 | 3750 / 8192 MiB | 12.77 / 15.0 GiB | 1.03 GiB |
| Qwen3.5 4B Q4_K_M | 13/20 | 7 | 7 (clustering), 13 (extraction), 37 (extraction), 40 (extraction), 44 (clustering), 64 (clustering), 100 (clustering) | 243.3 | 4447 / 8192 MiB | 13.55 / 15.0 GiB | 1.04 GiB |

GPU and host-memory figures are one-second global samples. Host RAM is not process-only usage; process peak RSS is listed separately. Both candidates remained within the 8 GiB VRAM and 16 GB system RAM limits. Retry counts were not emitted by the inference stack and are unavailable; the configured DSPy retry budget was three. First-attempt success cannot be inferred from the final status.

The shared-valid subset contains 12 records: `22, 24, 26, 32, 35, 42, 46, 52, 70, 96, 101, 102`. All other records remain represented as generation failures for at least one candidate and are not assigned a MINE score.

## Qwen3.5 Failure Diagnosis

The preserved CAT-78 logs reproduce the record-37 failure in extraction relation generation: one relation object omitted the required `object` field. Record 40 also failed in extraction relation generation because four relation objects had `object: null`. Neither historical failure reached clustering. The new confirmation run failed again during extraction for both records 37 and 40, confirming the failure category under the unchanged configuration. The new record-40 response is retained in its error artifact. No output was normalized, and no retry or changed-output mitigation was included in the baseline comparison.

Other confirmation failures were also strict structured-output parse errors: malformed relation lists during extraction or unknown relation labels during clustering. These failures, rather than memory pressure, account for the reliability shortfall.

### Targeted Recovery Attempt: Records 37 and 40

On 2026-10-09, Qwen3.5 was run again on records `37, 40` as a separate targeted attempt using the unchanged CAT-108 settings and the same installed model digest. The subset cohort was ordered to match its declared IDs. Both records failed again during extraction with `AdapterParseError`; no graph was produced. The complete attempt is preserved under ignored `runs/cat108-qwen35-recovery-37-40-qwen3.5-4b-q4_K_M/`, including separate record manifests, full error traces, settings, and logs. The earlier command that produced the subset manifest in reverse record order was rejected by cohort validation before inference and produced no run artifacts. This targeted attempt does not meet the reliability gate or count as a successful recovery. The configured retry budget remains three; actual internal retries were not observed and are not inferred.

### Higher-Token Diagnostic Recovery: Records 37 and 40

On 2026-10-09, each record was retried separately with the same Qwen3.5 model digest and generation settings, except both `context_length` and `max_tokens` were raised from 4096 to 8192. Both runs succeeded through extraction and clustering: record 37 completed in 286.8 seconds and record 40 in 394.9 seconds. Their independent manifests and graphs are preserved under ignored `runs/cat108-qwen35-recovery-37-ctx8192-qwen3.5-4b-q4_K_M/` and `runs/cat108-qwen35-recovery-40-ctx8192-qwen3.5-4b-q4_K_M/`; diagnostic settings are retained in the corresponding `*-settings.json` files under `runs/`.


Sampled peaks were 4828 MiB VRAM and 12.34 GiB system RAM for record 37, and 4719 MiB VRAM and 12.14 GiB system RAM for record 40. The runs completed and unloaded the model successfully. This paired success is consistent with the larger context/output budget avoiding the earlier incomplete structured responses, but it does not isolate which limit was causal because both were increased together. These are changed-configuration recoveries, not baseline successes or a replacement for the 20-record reliability gate; the baseline remains Qwen3.5 13/20 and the 101-record decision remains no-go.

## CAT-108 Closeout

CAT-108 is closed as a completed confirmation experiment with a documented no-go, not as a successful model qualification. CAT-112's 4096-token MINE scoring and paired analysis are complete; its requested human audit was not completed because the required Qwen graphs were missing. No human findings are claimed. The audit is moved to CAT-114, where both models are run under matched 8192-token settings and a five-record source/graph packet is prepared. CAT-113's decision is no-go under the gates as measured. CAT-109, CAT-110, and CAT-111 were completed; the remaining follow-up belongs to CAT-114.

## CAT-114 Controlled 8K Follow-Up

These runs are separate from the CAT-108 baseline and the two-record diagnostic recovery. The 20-record cohort was selected after observing the original failures, so it is a **development-set confirmation**, not independent validation. Both candidates used the frozen cohort, KGGen 0.4.0, DSPy 3.4.0, clustering, chunk size 3000, temperature 0, seed 42, 8192 context and output limits, and at most two additional full-record attempts at unchanged settings. Each attempt has its own artifact directory. Configured DSPy request retries remain three; actual transport retry telemetry is unavailable. A completed graph from the initial full record attempt counts as first-attempt success; a graph after additional full-record attempts counts only as final success. No manual rescue was used.

| Set | Model | First-attempt success | Final success | Failed IDs | Mean seconds/record | Peak VRAM | Peak system RAM | Max process RSS |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Development, n=20 | Phi-4-mini | 14/20 | 14/20 | 24, 40, 6, 64, 42, 32 | 156.7 | 4356 / 8192 MiB | 12.73 / 15.55 GiB | 1053 MiB |
| Development, n=20 | Qwen3.5 | 19/20 | 19/20 | 96 | 304.0 | 4671 / 8192 MiB | 12.70 / 15.55 GiB | 1054 MiB |
| Held-out, n=20 | Phi-4-mini | 14/20 | 14/20 | 58, 95, 59, 54, 86, 74 | 120.8 | 4272 / 8192 MiB | 11.40 / 15.55 GiB | 1082 MiB |
| Held-out, n=20 | Qwen3.5 | 17/20 | 17/20 | 34, 16, 90 | 277.5 | 4676 / 8192 MiB | 12.04 / 15.55 GiB | 1092 MiB |

All successful records succeeded on their first full-record attempt; no full-record retry recovered a failed graph. Failures were `AdapterParseError`s: development Phi had two extraction and four clustering failures; development Qwen had one extraction failure. Held-out Phi had one extraction and five clustering failures; held-out Qwen had two extraction and one clustering failure. Neither set exceeded its sampled hardware ceilings (8192 MiB VRAM and 15,625,000 KiB system RAM, the 16 GB decimal limit); host `MemTotal` was 15.55 GiB. The final success rates meet the existing 19/20 gate only for Qwen on the development set; neither model met it on held-out validation. The 101-record expansion remains no-go.

The held-out subset was frozen before running either model: `34, 80, 23, 58, 14, 33, 89, 105, 41, 16, 95, 48, 90, 59, 54, 68, 86, 81, 74, 57`, sampled from the remaining 81 pinned dataset records with `random.Random(20261009).sample(..., 20)`. This set is non-overlapping with the development cohort.

The audit packet is at ignored `runs/cat114-human-audit/packet.md` for records 26, 37, 40, 7, and 13, with both 8K graphs and source text. The initially selected additional record 6 lacked a Phi development graph; 13 was the next-lowest shared-success record and replaced it for audit selection only. Record 6 remains a failure in reliability metrics. Phi record 40 was generated in a separate, one-attempt audit-only run with identical 8K settings because it failed all three development attempts; that supplemental graph is excluded from reliability and quality metrics. On 2026-10-11, the user supplied and explicitly authorized an AI-assisted evidence review as a substitute for a separate human audit. Its categorical findings and limitations are recorded in the packet. This satisfies the audit deliverable by explicit user authorization, but is not described as independent human review and does not change model-quality or reliability estimates.

The original MINE evaluator was run on both CAT-114 cohorts on 2026-10-09 using the same pinned settings as CAT-108. The key was loaded from the private user-level env file and was not written to the repository or logs. Answer-level judgments and retrieved contexts are preserved in ignored `runs/cat114-mine-development/` and `runs/cat114-mine-heldout/`.

| Set | Model | Evaluated graphs | Correct / answers | Pooled accuracy | Failed generations |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | Phi-4-mini | 14/20 | 121/210 | 57.6% | 6 |
| Development | Qwen3.5 | 19/20 | 198/285 | 69.5% | 1 |
| Held-out | Phi-4-mini | 14/20 | 106/210 | 50.5% | 6 |
| Held-out | Qwen3.5 | 13/20 | 119/195 | 61.0% | 3 |

Missing graphs are not assigned zero. Development paired-valid comparison uses 13 records; Qwen3.5 minus Phi-4-mini mean per-record difference is +9.2 pp, with a 50,000-resample percentile record-bootstrap 95% CI of -7.2 to +26.2 pp (seed 108). Held-out paired-valid comparison uses 9 records; mean difference is +11.9 pp, with 95% CI -15.6 to +33.3 pp. Both intervals include zero, so neither set establishes a quality winner. The held-out quality denominator excludes records with a failed graph for either model; counts and missing IDs are retained in each analyzer output. CAT-108 4K results remain separate and unchanged.

Reproduce the paired record bootstrap from the saved summaries:

```bash
uv run --locked scripts/evaluate_mine_saved_graphs.py \
  runs/cat114-8k-phi4-development-phi4-mini-3.8b-q4_K_M \
  runs/cat114-8k-qwen35-development-qwen3.5-4b-q4_K_M \
  --run-id cat114-mine-development --cohort data/cat114-development-cohort.json
uv run --locked scripts/evaluate_mine_saved_graphs.py \
  runs/cat114-8k-phi4-heldout-phi4-mini-3.8b-q4_K_M \
  runs/cat114-8k-qwen35-heldout-qwen3.5-4b-q4_K_M \
  --run-id cat114-mine-heldout --cohort data/cat114-heldout-cohort.json
uv run --locked python scripts/analyze_paired_mine.py \
  runs/cat114-mine-development/summary.json --cohort data/cat114-development-cohort.json
uv run --locked python scripts/analyze_paired_mine.py \
  runs/cat114-mine-heldout/summary.json --cohort data/cat114-heldout-cohort.json
```

CAT-114's audit deliverable has been adjudicated by explicit user acceptance of the AI-assisted review in place of a separate human audit. The 8K paired quality intervals remain inconclusive, and neither candidate met the 19/20 reliability gate on held-out validation (Phi 14/20, Qwen 17/20). Therefore the final recommendation remains no-go for the 101-record expansion. Keep the four generation runs, their separate MINE evaluations, the audit-only record 40 supplement, and the CAT-108 baseline distinct.

## Evaluation, Audit, and Gates

The original MINE retrieval-QA evaluator completed on 2026-10-09 using the pinned CAT-80 procedure: `openai/gpt-5`, reasoning effort `high`, temperature 1.0, max tokens 16000, and retrieval `all-MiniLM-L6-v2` with `k=8`. Per-answer judgments and retrieved contexts are preserved under ignored `runs/cat108-mine-gpt5-openai/`. All 31 successfully generated graphs were evaluated on 15 questions each; the nine generation failures remain separately reported and were not assigned zero accuracy.

| Model | Successful graphs evaluated | Correct answers / evaluated | Pooled answer accuracy | Mean per-record accuracy |
| --- | ---: | ---: | ---: | ---: |
| Phi-4-mini 3.8B Q4_K_M | 18/20 | 155/270 | 57.4% | 57.4% |
| Qwen3.5 4B Q4_K_M | 13/20 | 139/195 | 71.3% | 71.3% |

The paired-valid comparison includes 12 records (`22, 24, 26, 32, 35, 42, 46, 52, 70, 96, 101, 102`), with 15 answers per model per record:

| Record ID | Phi-4-mini | Qwen3.5 | Qwen3.5 minus Phi |
| ---: | ---: | ---: | ---: |
| 22 | 66.7% | 93.3% | +26.7 pp |
| 24 | 86.7% | 73.3% | -13.3 pp |
| 26 | 86.7% | 93.3% | +6.7 pp |
| 32 | 46.7% | 60.0% | +13.3 pp |
| 35 | 66.7% | 40.0% | -26.7 pp |
| 42 | 40.0% | 86.7% | +46.7 pp |
| 46 | 53.3% | 80.0% | +26.7 pp |
| 52 | 26.7% | 86.7% | +60.0 pp |
| 70 | 73.3% | 20.0% | -53.3 pp |
| 96 | 86.7% | 80.0% | -6.7 pp |
| 101 | 40.0% | 86.7% | +46.7 pp |
| 102 | 33.3% | 46.7% | +13.3 pp |

Across these 12 paired records, mean record accuracy is 58.9% for Phi-4-mini and 70.6% for Qwen3.5, a Qwen3.5 advantage of 11.7 percentage points. A percentile bootstrap resampling records with replacement (50,000 resamples, seed 108) gives a 95% interval of -6.7 to +28.9 percentage points for that paired mean difference. The interval includes zero and the per-record effects vary substantially, so the quality ranking is inconclusive; the pooled successful-graph scores are descriptive and should not override this paired uncertainty.

The required human audit was not completed. Qwen3.5 did not produce valid graphs for required audit records 37 and 40, precluding the specified five-record paired audit including 37, 40, and 26. No human correctness findings are claimed.

| Gate | Result |
| --- | --- |
| At least 19/20 completions without manual rescue | **Fail:** Phi-4-mini 18/20; Qwen3.5 13/20 |
| At most 8 GiB VRAM and 16 GB system RAM | **Pass:** both runs remained within sampled global limits |
| Paired original-MINE quality with record-level uncertainty | **Complete, inconclusive:** 12 paired records; 95% bootstrap interval for Qwen3.5 minus Phi-4-mini is -6.7 to +28.9 pp |
| Human audit with required records | **Not completed:** Qwen3.5 has no valid graphs for records 37 or 40 |
| Authorize the 101-record run | **No-go:** both candidates fail reliability; paired quality is inconclusive and audit is incomplete |

Do not launch the 101-record benchmark. Phi-4-mini remains the more reliable of the two in this sample, but its 18/20 result fails the specified gate and does not authorize expansion. Qwen3.5 has a higher pooled and paired mean MINE accuracy among successful graphs, but only 13/20 completions and inconclusive record-level uncertainty; this does not establish it as the winner. A bounded follow-up should first improve and independently measure structured-output reliability, then repeat the paired comparison and complete a feasible human audit. Keep any retry/schema intervention as a separate configuration with baseline and changed outputs preserved.

## Reproduction

Create the locked environment and regenerate the local, ignored cohort files:

```bash
uv sync --locked
uv run scripts/prepare_pilot.py
uv run scripts/prepare_confirmation_cohort.py
```

Run each candidate sequentially on the same frozen cohort:

```bash
uv run scripts/run_pilot.py --model phi4-mini:3.8b-q4_K_M --cohort data/mine-confirmation-cohort.json --run-id cat108-phi4-confirmation
uv run scripts/run_pilot.py --model qwen3.5:4b-q4_K_M --cohort data/mine-confirmation-cohort.json --run-id cat108-qwen35-confirmation
```

With `OPENAI_API_KEY` available to the process, run the original evaluator on the successful graphs and preserve failed generations separately:

```bash
uv run scripts/evaluate_mine_saved_graphs.py \
  runs/cat108-phi4-confirmation-phi4-mini-3.8b-q4_K_M \
  runs/cat108-qwen35-confirmation-qwen3.5-4b-q4_K_M \
  --run-id cat108-mine-gpt5-openai --resume
```

The benchmark run outputs are intentionally ignored and were not added to source control. This checked-in report records measured summary results and the outstanding gates; retain the host-side run directories for full artifact reproduction.