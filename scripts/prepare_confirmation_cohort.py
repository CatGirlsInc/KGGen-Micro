import json
import random
from pathlib import Path

from datasets import load_dataset


DATASET = "josancamon/kg-gen-MINE-evaluation-dataset"
REVISION = "aacf51485de5cb6f4a411599f1d0846842c14009"
SEED = 20261008
BASE_IDS = [24, 7, 102, 40, 52, 37, 70, 26, 101, 96]
PILOT = Path("data/mine-pilot.json")
OUTPUT = Path("data/mine-confirmation-cohort.json")


def main():
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    if pilot.get("dataset") != DATASET or pilot.get("revision") != REVISION:
        raise ValueError("Pilot fixture does not match the pinned MINE dataset")
    if [record["id"] for record in pilot["records"]] != BASE_IDS:
        raise ValueError("Pilot fixture IDs differ from the declared original cohort")

    dataset = load_dataset(DATASET, revision=REVISION, split="train")
    source_by_id = {record["id"]: record for record in dataset}
    if len(source_by_id) != len(dataset):
        raise ValueError("Pinned MINE dataset contains duplicate record IDs")
    if any(record_id not in source_by_id for record_id in BASE_IDS):
        raise ValueError("Pinned MINE dataset is missing an original pilot record")

    for pilot_record in pilot["records"]:
        source_record = source_by_id[pilot_record["id"]]
        if (
            pilot_record["topic"] != source_record["essay_topic"]
            or pilot_record["text"] != source_record["essay_content"]
        ):
            raise ValueError(f"Pilot record {pilot_record['id']} differs from source")

    remaining_ids = [record_id for record_id in source_by_id if record_id not in BASE_IDS]
    additional_ids = random.Random(SEED).sample(remaining_ids, 10)
    record_ids = BASE_IDS + additional_ids
    records = [
        {
            "id": record_id,
            "topic": source_by_id[record_id]["essay_topic"],
            "text": source_by_id[record_id]["essay_content"],
        }
        for record_id in record_ids
    ]
    if any(not record["text"].strip() for record in records):
        raise ValueError("Selected MINE record contains empty essay text")

    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "selection_seed": SEED,
        "selection_procedure": "random.Random(seed).sample(dataset-order IDs excluding the original pilot IDs, k=10); append after original pilot IDs",
        "base_record_ids": BASE_IDS,
        "additional_record_ids": additional_ids,
        "record_ids": record_ids,
        "records": records,
    }
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} records to {OUTPUT}: {record_ids}")


if __name__ == "__main__":
    main()