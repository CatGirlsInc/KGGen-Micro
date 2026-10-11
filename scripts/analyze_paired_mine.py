import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize paired original-MINE evaluations with record bootstrap uncertainty"
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--phi-model", default="phi4-mini:3.8b-q4_K_M")
    parser.add_argument("--qwen-model", default="qwen3.5:4b-q4_K_M")
    parser.add_argument("--bootstrap", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=108)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    record_ids = cohort.get("record_ids", [record["id"] for record in cohort["records"]])
    by_model_record = {
        (item["model"], item["record_id"]): item
        for item in summary
        if item.get("status") == "evaluated"
    }
    paired_ids = [
        record_id
        for record_id in record_ids
        if (args.phi_model, record_id) in by_model_record
        and (args.qwen_model, record_id) in by_model_record
    ]
    if not paired_ids:
        raise ValueError("No paired-valid records are present in the evaluation summary")
    differences = [
        by_model_record[(args.qwen_model, record_id)]["accuracy_percent"]
        - by_model_record[(args.phi_model, record_id)]["accuracy_percent"]
        for record_id in paired_ids
    ]
    rng = random.Random(args.seed)
    boot_means = [
        sum(rng.choices(differences, k=len(differences))) / len(differences)
        for _ in range(args.bootstrap)
    ]
    boot_means.sort()
    lower = boot_means[int(0.025 * args.bootstrap)]
    upper = boot_means[min(args.bootstrap - 1, int(0.975 * args.bootstrap))]
    model_summaries = {}
    for model in (args.phi_model, args.qwen_model):
        evaluated = [
            item for item in summary
            if item.get("model") == model and item.get("status") == "evaluated"
        ]
        model_summaries[model] = {
            "evaluated_graphs": len(evaluated),
            "failed_graphs": sum(
                item.get("model") == model and item.get("status") == "generation_failed"
                for item in summary
            ),
            "correct_answers": sum(item["correct"] for item in evaluated),
            "evaluated_answers": sum(item["query_count"] for item in evaluated),
        }
    result = {
        "comparison": "Qwen3.5 minus Phi-4-mini; percentage points",
        "paired_valid_record_ids": paired_ids,
        "paired_valid_count": len(paired_ids),
        "total_cohort_records": len(record_ids),
        "excluded_or_missing_record_ids": [
            record_id for record_id in record_ids if record_id not in paired_ids
        ],
        "record_differences_pp": dict(zip(paired_ids, differences)),
        "mean_paired_difference_pp": sum(differences) / len(differences),
        "record_bootstrap": {
            "resamples": args.bootstrap,
            "seed": args.seed,
            "percentile_95_ci_pp": [lower, upper],
        },
        "model_summaries": model_summaries,
        "missing_graphs_are_not_scored_as_zero": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()