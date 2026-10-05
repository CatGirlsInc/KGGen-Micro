import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PILOT = Path("data/mine-pilot.json")
DEFAULT_EVALUATOR = "qwen2.5:7b-instruct-q4_K_M"
SCORE_FIELDS = ("fidelity", "coverage", "relation_quality", "entity_quality")


def ollama_json(host, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{host.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def graph_counts(graph):
    return {"entities": len(graph.get("entities", [])), "relations": len(graph.get("relations", []))}


def validate_judgement(judgement):
    scores = judgement.get("scores")
    if not isinstance(scores, dict) or any(
        type(scores.get(field)) is not int or not 1 <= scores[field] <= 5
        for field in SCORE_FIELDS
    ):
        raise ValueError("Evaluator response must contain integer 1-5 scores for all four axes")
    for field in ("rationale", "supported_examples", "unsupported_examples", "missing_important_facts"):
        if field not in judgement:
            raise ValueError(f"Evaluator response is missing {field!r}")
    return judgement


def build_prompt(topic, source, graph):
    return f"""Evaluate this knowledge graph against its source essay. Judge only factual fidelity and usefulness; do not reward graph size by itself.

Score each axis from 1 to 5:
- fidelity: whether represented facts and relations are supported by the source
- coverage: whether the graph captures the essay's important entities and central claims
- relation_quality: whether relations are meaningful, specific, and directionally sensible
- entity_quality: whether entities are salient, clear, and appropriately normalized

Use 5 for strong and accurate with only minor omissions, 4 for useful with modest issues, 3 for mixed with material gaps or noise, 2 for mostly weak or misleading, and 1 for largely unusable. Base judgments on the supplied text only. Name concrete supported/unsupported graph examples and important omitted facts. Return only JSON with this shape:
{{"scores":{{"fidelity":1,"coverage":1,"relation_quality":1,"entity_quality":1}},"rationale":"...","supported_examples":["..."],"unsupported_examples":["..."],"missing_important_facts":["..."]}}

Topic: {topic}

Source essay:
{source}

Generated graph JSON:
{json.dumps(graph, ensure_ascii=False)}"""


def evaluate_graph(host, evaluator, topic, source, graph):
    started = datetime.now(timezone.utc)
    response = ollama_json(
        host,
        "/api/chat",
        {
            "model": evaluator,
            "messages": [{"role": "user", "content": build_prompt(topic, source, graph)}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 900},
        },
    )
    content = response["message"]["content"]
    judgement = validate_judgement(json.loads(content))
    ended = datetime.now(timezone.utc)
    return {
        "judgement": judgement,
        "started_at": started.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a separately labeled local LLM judgment of saved pilot graphs"
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--evaluator", default=DEFAULT_EVALUATOR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    pilot_by_id = {record["id"]: record for record in pilot["records"]}
    manifests = [json.loads((path / "run.json").read_text(encoding="utf-8")) for path in args.run_dirs]
    expected_ids = manifests[0]["record_ids"]
    if any(
        manifest.get("dataset_revision") != pilot["revision"]
        or manifest.get("record_ids") != expected_ids
        for manifest in manifests
    ):
        raise ValueError("All run manifests must use the same fixed pilot records and dataset revision")
    if any(record_id not in pilot_by_id for record_id in expected_ids):
        raise ValueError("Run manifests contain records outside the fixed pilot")
    record_ids = expected_ids[: args.limit] if args.limit else expected_ids
    if args.limit is not None and not 1 <= args.limit <= len(expected_ids):
        parser.error(f"--limit must be between 1 and {len(expected_ids)}")

    host = "http://localhost:11434"
    version = ollama_json(host, "/api/version").get("version")
    tags = ollama_json(host, "/api/tags")["models"]
    model_info = next((model for model in tags if model["name"] == args.evaluator), None)
    if model_info is None:
        raise RuntimeError(f"Evaluator model {args.evaluator!r} is not installed in Ollama")

    output = Path("runs") / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "evaluation_type": "local_llm_judgment_not_paper_comparable_mine_score",
        "evaluator": args.evaluator,
        "evaluator_digest": model_info.get("digest"),
        "ollama_version": version,
        "temperature": 0,
        "seed": 42,
        "context_length": 8192,
        "max_tokens": 900,
        "dataset": pilot["dataset"],
        "dataset_revision": pilot["revision"],
        "record_ids": record_ids,
        "candidate_runs": [
            {"run_id": path.name, "model": manifest.get("model"), "model_digest": manifest.get("model_digest")}
            for path, manifest in zip(args.run_dirs, manifests)
        ],
        "score_fields": list(SCORE_FIELDS),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    outcomes = []
    for run_dir, manifest in zip(args.run_dirs, manifests):
        by_id = {outcome["record_id"]: outcome for outcome in manifest["outcomes"]}
        model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", manifest["model"])
        for record_id in record_ids:
            outcome = by_id[record_id]
            row = {"model": manifest["model"], "record_id": record_id}
            if outcome["status"] != "succeeded" or not outcome.get("graph"):
                row.update({"status": "generation_failed", "generation_status": outcome["status"]})
            else:
                graph_path = run_dir / outcome["graph"]
                graph = json.loads(graph_path.read_text(encoding="utf-8"))
                record = pilot_by_id[record_id]
                row.update({"status": "judged", "graph": graph_counts(graph)})
                try:
                    row.update(evaluate_graph(host, args.evaluator, record["topic"], record["text"], graph))
                except Exception as error:
                    row.update({"status": "judge_failed", "error": str(error)})
            result_path = output / f"{model_slug}_{record_id}.json"
            result_path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            outcomes.append(row)
            print(f"{row['status']}: {row['model']} record {record_id}", flush=True)

    (output / "summary.json").write_text(json.dumps(outcomes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    settings["completed_at"] = datetime.now(timezone.utc).isoformat()
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()