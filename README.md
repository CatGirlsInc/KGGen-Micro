# KGGen-Micro

Benchmark KGGen with small, locally hosted Ollama models under a consumer-hardware budget of **8 GB VRAM and 16 GB system RAM**. The goal is to establish a reproducible baseline on a fixed 10-record pilot from the published MINE dataset, then recommend whether and how to proceed with a larger benchmark and future MediaWiki extraction.

This is an experimental benchmark, not a production knowledge-graph service. The initial pilot does not include the full MINE dataset, model fine-tuning, or a custom MediaWiki extraction pipeline.

## Scope

1. **Set up a reproducible environment.** Install and configure Ollama, KGGen, and dependencies; inspect the original MINE evaluation requirements; pin versions and inference settings; record hardware; and select 10 records with a fixed random seed. Save record identifiers so every model receives identical inputs.
2. **Check model compatibility and resource use.** Test the initial shortlist below through KGGen extraction, clustering, and structured-output stages. Confirm actual memory use and document configuration changes or exclusions before finalizing the shortlist.
3. **Run the pilot.** Process the same 10 records with consistent KGGen settings. Save intermediate extractions, final graphs, execution logs, timings, resource measurements, retries, and failures.
4. **Evaluate and compare.** Reproduce the original MINE evaluation wherever possible, inspect representative outputs, and compare quality, reliability, graph size, speed, and memory use.
5. **Recommend the next phase.** Name a preferred model and configuration, explain the trade-offs and MediaWiki-relevant limitations, and make a reasoned go/no-go decision for the full MINE benchmark.

## Candidate Models

| Model | Approx. Ollama download size | Role |
| --- | ---: | --- |
| Qwen3.5 4B | 3.4 GB | Primary small-model candidate |
| Gemma 4 E2B | 4.6–7.5 GB | Architecture comparison |
| Ministral 3 3B | 3.0 GB | Compact structured-output candidate |
| Phi-4-mini 3.8B | 2.5 GB | Optional lightweight baseline |
| Qwen3.5 9B | 6.6 GB | Stretch candidate; test only if it fits the hardware budget |

Download sizes are approximate, not a guarantee of runtime fit. Final selection depends on measured peak VRAM/RAM and compatibility with every required KGGen stage. Use identical generation settings where supported and record any model-specific deviations.

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
