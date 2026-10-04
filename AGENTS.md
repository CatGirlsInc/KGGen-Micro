# Agent and Contributor Guidance

## Project Focus

- Keep work aligned with the local KGGen/Ollama MINE pilot described in `README.md`.
- Respect the target ceiling of 8 GB VRAM and 16 GB system RAM. Measure actual resource use; model download size alone does not establish compatibility.
- Keep the initial benchmark to the fixed 10-record pilot. Do not expand to the full dataset, fine-tune models, or build a production/MediaWiki pipeline without an explicit project decision.

## Reproducible Experiments

- Use identical source records and KGGen settings across model comparisons wherever supported. Record and explain model-specific deviations.
- Pin KGGen and dependency versions, Ollama model tags and quantization, inference settings, random seed, and relevant hardware details.
- Preserve intermediate outputs, final graphs, evaluation outputs, timings, peak memory, retries, failures, and error logs. Do not silently discard failed runs.
- Label results by evaluation method. Keep original-evaluator, paper-comparable scores separate from scores produced by a local substitute evaluator.
- Do not invent results, claim an experiment ran when it did not, or present scores from different evaluators as directly comparable.
- Keep generated datasets and large run artifacts out of source control unless they are intentionally small, required fixtures. Do not commit credentials or private data.

## Git Workflow

- Do independent work on separate, short-lived branches with concise names, such as `setup/ollama`, `bench/pilot`, or `eval/mine`.
- Use branches in the current worktree. Do **not** create or use separate Git worktrees.
- Keep changes focused; do not revert unrelated user changes.

## Changes and Validation

- Follow existing repository conventions and inspect the relevant files before changing them. Avoid adding tooling or commands that are not actually supported by the repository.
- Update the README when setup, benchmark procedure, evaluation methodology, or deliverables change.
- Run the narrowest relevant validation available after changes. For documentation-only changes, at minimum check whitespace errors with `git diff --check` and review the rendered Markdown structure.