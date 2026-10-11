import json
import random
from pathlib import Path

from datasets import load_dataset


DATASET = "josancamon/kg-gen-MINE-evaluation-dataset"
REVISION = "aacf51485de5cb6f4a411599f1d0846842c14009"
SEED = 20261009
DEVELOPMENT = Path("data/cat114-development-cohort.json")
OUTPUT = Path("data/cat114-heldout-cohort.json")
SETTINGS = {
    "phi4": (Path("config/cat114-8k-phi4-settings.json"), Path("config/cat114-heldout-phi4-settings.json")),
    "qwen35": (Path("config/cat114-8k-qwen35-settings.json"), Path("config/cat114-heldout-qwen35-settings.json")),
}


def main():
    development = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    if development.get("dataset") != DATASET or development.get("revision") != REVISION:
        raise ValueError("Development cohort does not match the pinned dataset")
    development_ids = development.get("record_ids")
    if not isinstance(development_ids, list) or len(development_ids) != 20:
        raise ValueError("Development cohort must contain exactly 20 IDs")

    dataset = load_dataset(DATASET, revision=REVISION, split="train")
    source_by_id = {record["id"]: record for record in dataset}
    remaining_ids = [record_id for record_id in source_by_id if record_id not in development_ids]
    if len(remaining_ids) != 81:
        raise ValueError("Expected 81 records outside the development cohort")
    record_ids = random.Random(SEED).sample(remaining_ids, 20)
    records = [
        {
            "id": record_id,
            "topic": source_by_id[record_id]["essay_topic"],
            "text": source_by_id[record_id]["essay_content"],
        }
        for record_id in record_ids
    ]
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "experiment": "CAT-114 held-out validation",
        "selection_seed": SEED,
        "selection_procedure": "random.Random(20261009).sample(pinned dataset-order IDs excluding the frozen CAT-114 development cohort, k=20)",
        "development_cohort_manifest": str(DEVELOPMENT),
        "development_record_ids": development_ids,
        "record_ids": record_ids,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for source_path, output_path in SETTINGS.values():
        settings = json.loads(source_path.read_text(encoding="utf-8"))
        settings["cohort_manifest"] = str(OUTPUT)
        settings["record_ids"] = record_ids
        settings["experiment"] = "CAT-114 held-out validation; distinct from development set"
        settings["max_system_memory_kib"] = 15_625_000
        output_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote held-out cohort {record_ids} and separate settings files")


if __name__ == "__main__":
    main()