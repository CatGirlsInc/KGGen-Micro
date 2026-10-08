# CAT-81: Model Selection and Next-Phase Decision

**Decision date:** 2026-10-08  
**Evidence:** CAT-78 generation pilot and CAT-80 evaluation  
**Dataset:** `josancamon/kg-gen-MINE-evaluation-dataset`, revision `aacf51485de5cb6f4a411599f1d0846842c14009`

## Decision

**Do not start the full 101-record benchmark yet.** Proceed with a bounded, 20-record confirmation experiment using Phi-4-mini 3.8B Q4_K_M and Qwen3.5 4B Q4_K_M. The current 10-record pilot is useful for choosing finalists, but it is too small and uneven in successful-graph coverage to justify the full run.

There are two different current recommendations:

- **Operational baseline: Phi-4-mini 3.8B Q4_K_M.** It completed all 10 records, was fastest at 118.8 seconds per attempted record on average, and stayed within the observed hardware budget.
- **Quality challenger: Qwen3.5 4B Q4_K_M.** It had the highest original MINE answer accuracy among successful graphs, 84/120 (70.0%), but produced graphs for only 8/10 records. This makes it a candidate to validate, not the unconditional model recommendation.

The CAT-80 percentages use different denominators: failed generations have no MINE score. Phi-4-mini's 87/150 (58.0%) covers 10 successful graphs; Qwen3.5's 84/120 (70.0%) covers 8. Do not compare raw correct-answer totals as if both models were evaluated on the same number of answers. The separate local-judge results are provisional and are not used to decide the winner: the judge made a demonstrable graph-reading error, and it had no calibration or inter-rater study.

Gemma 4 E2B and Ministral 3 3B are not finalists for the next phase. They completed 9/10 and 8/10 records respectively, were slower than Phi-4-mini, and had lower original MINE pilot accuracy than Qwen3.5. Retain their results as baselines; do not spend the next experiment budget rerunning them.

## Evidence and Limits

All four CAT-78 runs used the same fixed 10 records, KGGen 0.4.0, 3000-token chunk size, clustering, temperature 0, seed 42, and 4096-token context. Qwen3.5 additionally required Ollama `think=false`. All models stayed below the project's 8 GiB VRAM and 16 GB system RAM ceilings. The reported GPU and host-memory maxima are once-per-second global samples, not guaranteed process peaks; process peak RSS is recorded separately in run manifests.

The original MINE retrieval-QA procedure was run against saved graphs for successful records only. It provides a more relevant quality signal than the local judge, but remains a 10-record sample with 15 generated answers per essay and a hosted GPT-5 evaluator. Evaluator token/cost telemetry was not captured. Neither this score nor the generation results establish performance on all 101 essays.

Structured-output failures, rather than memory exhaustion, caused all five failed generations. Qwen3.5 failed on records 37 and 40; its extraction response omitted a required relation field on record 37. Ministral and Gemma also had clustering parse failures. CAT-78's stale-cluster-progress guard bounds one non-progress loop, but it does not correct malformed model output. Retry budgets were configured, but actual per-attempt retry counts are unavailable and must not be inferred.

## Next Experiment

Run a fresh, reproducible 20-record comparison of Phi-4-mini and Qwen3.5 before deciding on the 101-record benchmark:

1. Use the same immutable dataset revision and a documented fixed seed to form a 20-record set. Include the original 10 pilot IDs and add 10 previously unused IDs selected deterministically from the remaining records. Save the exact ordered IDs in the run manifest.
2. Run both models on all 20 records with KGGen 0.4.0, clustering enabled, chunk size 3000, temperature 0, seed 42, and 4096-token context. Retain Qwen3.5's required `think=false` option and report it as a model-specific deviation. Do not change shared settings between models.
3. Before the comparison, reproduce and diagnose Qwen3.5's failures on records 37 and 40 from preserved logs. Test any proposed output-validation or targeted-retry change separately, preserve its baseline and changed outputs, and do not silently treat the changed configuration as the original CAT-78 result.
4. Apply the upstream MINE retrieval-QA evaluation to every successfully generated graph with the same evaluator and settings. Preserve answer-level judgments, retrieved contexts, failed records, logs, run settings, timings, and sampled global memory plus process RSS. Record retry counts only when directly observed.
5. Have a human audit at least five shared records, including records 37, 40, and 26, checking unsupported triples, omissions, duplicate or over-merged entities, and whether graph detail is useful for the source essay. Keep human findings separate from the automated MINE score.

### Predeclared Gates

- **Reliability:** a candidate must complete at least 19/20 records without manual rescue. Any recovered failures count as recovered only when the recovery mechanism and attempt are recorded; report first-attempt and final success separately.
- **Resources:** all measured runs must remain within 8 GiB VRAM and 16 GB system RAM. Report the measurement method and distinguish sampled global usage from process peak RSS.
- **Quality:** compare the models on paired records with valid graphs and report per-record results and uncertainty, not just pooled answer accuracy. Prefer Qwen3.5 for a full run only if its paired original-MINE advantage over Phi-4-mini is clear under a record-level uncertainty analysis and the human audit reveals no material correctness regression. If the ranking is inconclusive, extend the confirmation sample rather than launching the full benchmark.
- **Decision:** authorize the 101-record run only if at least one candidate passes reliability and resource gates and quality evidence supports the intended use. Otherwise, investigate structured-output reliability or evaluator validity first. Preserve all failures; do not convert missing graphs into zero MINE accuracy.

If one model must be used before this gate is complete, use Phi-4-mini for operational reliability. That is a pilot workflow choice, not a claim that it has the best extraction quality.

## Improvements to Investigate

1. **Structured-output reliability:** capture the raw response and parser failure stage; investigate schema-constrained or targeted retry behavior for missing fields and invalid cluster labels. Measure first-attempt success, recovery rate, and added latency. Do not silently normalize labels or accept incomplete graph objects.
2. **Cluster behavior:** retain the existing no-progress guard, and add targeted regression coverage for stale suggestions and malformed/unknown labels. The current guard prevents one stale-response loop but does not prevent parse errors.
3. **Evaluation confidence:** manually audit answer judgments and retrieved contexts on the confirmation set, especially where the automated explanation contradicts the graph. Keep the original evaluator results distinct from local-model judgments.
4. **Observability:** expose per-request attempts, retry counts, and latency where the inference stack supports it. Until then, mark these values unavailable and preserve raw logs.

These are hypotheses to test, not claims that the changes will improve quality. Do not modify the pinned KGGen version or introduce fine-tuning as part of the confirmation comparison.

## MediaWiki Implications

No CAT-78/80 run used MediaWiki pages, so the pilot does not validate extraction quality for that domain. In particular, the benchmark has not tested templates, tables, citation and reference structure, disambiguation, cross-page entity links, conflicting revisions, conditional rules, or mathematical expressions. Essay-level MINE scores cannot establish that these structures will survive extraction or clustering.

Treat MediaWiki as a separate later validation stage, not as part of the MINE full-run decision. After selecting a MINE configuration, prepare a small manually annotated page set covering prose, tables/templates, conditional rules, and mathematical content. Compare extracted entities and relations with the annotations, check provenance and unsupported claims, and evaluate page/chunk boundaries before considering broader ingestion. No production MediaWiki pipeline is authorized by this decision.

## Reproduction References

CAT-80's methods, evaluation details, result tables, and commands are in [cat80-evaluation.md](cat80-evaluation.md). Project setup and the fixed pilot procedure are in [README.md](../README.md). Run artifacts and evaluator outputs are stored under ignored `runs/` directories on the benchmark host; preserve their manifests and logs when executing the follow-up.