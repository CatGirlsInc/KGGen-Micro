import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import dspy
from datasets import load_dataset
from kg_gen import KGGen


DATASET = "josancamon/kg-gen-MINE-evaluation-dataset"
DATASET_REVISION = "aacf51485de5cb6f4a411599f1d0846842c14009"
UPSTREAM_REVISION = "6259b4c71523ceae75dafb4acec2eb833bef8f8e"
PILOT = Path("data/mine-pilot.json")


class EvaluateResponse(dspy.Signature):
    """Determine whether the context contains the information stated in the correct answer. Respond with 1 if yes, 0 if no."""

    context: str = dspy.InputField(desc="The context to evaluate")
    correct_answer: str = dspy.InputField(desc="The correct answer to check for")
    evaluation: int = dspy.OutputField(
        desc="1 if context contains the correct answer, 0 otherwise"
    )


class ResponseEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(EvaluateResponse)

    def forward(self, context, correct_answer):
        return self.evaluate(context=context, correct_answer=correct_answer)


def evaluate_response(correct_answer, context):
    result = ResponseEvaluator().forward(context=context, correct_answer=correct_answer)
    if result.evaluation not in (0, 1):
        raise ValueError(f"Unexpected MINE evaluation value: {result.evaluation!r}")
    return result.evaluation


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def model_slug(model):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def graph_files(run_dir, manifest, selected_ids):
    outcomes = {outcome["record_id"]: outcome for outcome in manifest["outcomes"]}
    for record_id in selected_ids:
        outcome = outcomes.get(record_id)
        if not outcome or outcome.get("status") != "succeeded" or not outcome.get("graph"):
            yield record_id, None, outcome
        else:
            yield record_id, run_dir / outcome["graph"], outcome


def evaluate_graph(kggen, graph_data, queries, output_path, partial_path, resume):
    if output_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        if len(saved) == len(queries) + 1:
            return saved
        raise ValueError(f"Existing result is incomplete: {output_path}")

    completed = []
    if resume and partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        completed = partial.get("results", [])
        if len(completed) > len(queries):
            raise ValueError(f"Partial result has too many queries: {partial_path}")

    graph = kggen.from_dict(graph_data)
    network_graph = kggen.to_nx(graph)
    node_embeddings, _ = kggen.generate_embeddings(network_graph)

    for index in range(len(completed), len(queries)):
        query = queries[index]
        *_, context_text = kggen.retrieve(query, node_embeddings, network_graph)
        evaluation = evaluate_response(query, context_text)
        completed.append(
            {
                "correct_answer": query,
                "retrieved_context": context_text,
                "evaluation": evaluation,
            }
        )
        save_json(partial_path, {"results": completed})
        print(
            f"  query {index + 1}/{len(queries)}: {evaluation}", flush=True
        )

    accuracy = sum(item["evaluation"] for item in completed) / len(queries)
    results = completed + [{"accuracy": f"{accuracy * 100:.2f}%"}]
    save_json(output_path, results)
    partial_path.unlink(missing_ok=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the upstream MINE retrieval QA evaluator on saved pilot graphs"
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-graphs", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set in the process environment")

    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    if pilot["dataset"] != DATASET or pilot["revision"] != DATASET_REVISION:
        raise ValueError("Local pilot dataset does not match the pinned MINE revision")
    selected_ids = [record["id"] for record in pilot["records"]]

    manifests = [
        json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        for run_dir in args.run_dirs
    ]
    for manifest in manifests:
        if (
            manifest.get("dataset") != DATASET
            or manifest.get("dataset_revision") != DATASET_REVISION
        ):
            raise ValueError("Every run must use the pinned MINE dataset revision")

    dataset = load_dataset(DATASET, revision=DATASET_REVISION, split="train")
    reference_by_id = {
        record["id"]: record for record in dataset if record["id"] in selected_ids
    }
    if set(reference_by_id) != set(selected_ids):
        raise ValueError("Pinned MINE dataset is missing one or more pilot record IDs")
    for record_id, record in reference_by_id.items():
        queries = record.get("generated_queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"Record {record_id} has no generated MINE queries")

    evaluator_lm = dspy.LM(
        model="openai/gpt-5",
        api_key=os.environ["OPENAI_API_KEY"],
        reasoning_effort="high",
        temperature=1.0,
        max_tokens=16000,
    )
    dspy.configure(lm=evaluator_lm)
    kggen = KGGen(
        model="openai/gpt-5-nano",
        api_key=os.environ["OPENAI_API_KEY"],
        retrieval_model="all-MiniLM-L6-v2",
        temperature=1.0,
        max_tokens=64000,
    )

    output_dir = Path("runs") / args.run_id
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    settings_path = output_dir / "settings.json"
    if args.resume and settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if settings.get("upstream_revision") != UPSTREAM_REVISION:
            raise ValueError("Cannot resume with a different upstream evaluator revision")
    else:
        settings = {
            "evaluation_type": "upstream_mine_retrieval_qa_pilot",
            "upstream_repository": "https://github.com/stair-lab/kg-gen",
            "upstream_revision": UPSTREAM_REVISION,
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "record_ids": selected_ids,
            "evaluator_model": "openai/gpt-5",
            "evaluator_reasoning_effort": "high",
            "evaluator_temperature": 1.0,
            "evaluator_max_tokens": 16000,
            "evaluation_style": "DSPy ChainOfThought; binary supported-answer check",
            "retrieval_model": "all-MiniLM-L6-v2",
            "retrieval_k": 8,
            "retrieval_evaluator_kggen_model": "openai/gpt-5-nano",
            "retrieval_temperature": 1.0,
            "retrieval_max_tokens": 64000,
            "candidate_runs": [
                {
                    "run_id": run_dir.name,
                    "model": manifest.get("model"),
                    "model_digest": manifest.get("model_digest"),
                }
                for run_dir, manifest in zip(args.run_dirs, manifests)
            ],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        save_json(settings_path, settings)

    graph_jobs = []
    for run_dir, manifest in zip(args.run_dirs, manifests):
        for record_id, graph_path, outcome in graph_files(run_dir, manifest, selected_ids):
            graph_jobs.append((run_dir, manifest, record_id, graph_path, outcome))
    if args.max_graphs is not None:
        if args.max_graphs < 1:
            parser.error("--max-graphs must be positive")
        graph_jobs = graph_jobs[: args.max_graphs]

    summary_path = output_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if args.resume and summary_path.exists()
        else []
    )
    prior_keys = {(row["model"], row["record_id"]) for row in summary}

    for run_dir, manifest, record_id, graph_path, outcome in graph_jobs:
        model = manifest["model"]
        result_name = f"{model_slug(model)}_{record_id}.json"
        result_path = output_dir / result_name
        partial_path = output_dir / f"{result_name}.partial"
        key = (model, record_id)
        if key in prior_keys:
            print(f"already recorded: {model} record {record_id}", flush=True)
            continue

        if graph_path is None:
            row = {
                "model": model,
                "record_id": record_id,
                "status": "generation_failed",
                "generation_status": outcome.get("status") if outcome else "missing",
            }
        else:
            graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
            queries = reference_by_id[record_id]["generated_queries"]
            try:
                result = evaluate_graph(
                    kggen,
                    graph_data,
                    queries,
                    result_path,
                    partial_path,
                    args.resume,
                )
                accuracy = float(result[-1]["accuracy"].rstrip("%"))
                row = {
                    "model": model,
                    "record_id": record_id,
                    "status": "evaluated",
                    "query_count": len(queries),
                    "correct": sum(item["evaluation"] for item in result[:-1]),
                    "accuracy_percent": accuracy,
                    "result_file": result_name,
                }
            except Exception as error:
                message = str(error).replace(os.environ["OPENAI_API_KEY"], "[REDACTED]")
                row = {
                    "model": model,
                    "record_id": record_id,
                    "status": "evaluation_failed",
                    "error_type": type(error).__name__,
                    "error": message,
                    "partial_file": partial_path.name if partial_path.exists() else None,
                }
        summary.append(row)
        prior_keys.add(key)
        save_json(summary_path, summary)
        print(
            f"{row['status']}: {model} record {record_id}"
            + (
                f" accuracy={row['accuracy_percent']:.2f}%"
                if row["status"] == "evaluated"
                else ""
            ),
            flush=True,
        )

    settings["completed_at"] = datetime.now(timezone.utc).isoformat()
    settings["graphs_in_summary"] = len(summary)
    save_json(settings_path, settings)


if __name__ == "__main__":
    main()