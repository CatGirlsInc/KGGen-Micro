import json
import random
from pathlib import Path

from datasets import load_dataset


DATASET = "josancamon/kg-gen-MINE-evaluation-dataset"
REVISION = "aacf51485de5cb6f4a411599f1d0846842c14009"
SEED = 20261004
PILOT_SIZE = 10
OUTPUT = Path("data/mine-pilot.json")


def main():
    dataset = load_dataset(DATASET, revision=REVISION, split="train")
    indices = random.Random(SEED).sample(range(len(dataset)), PILOT_SIZE)
    records = [
        {
            "id": dataset[index]["id"],
            "topic": dataset[index]["essay_topic"],
            "text": dataset[index]["essay_content"],
        }
        for index in indices
    ]
    if any(not record["text"].strip() for record in records):
        raise ValueError("Selected MINE record contains empty essay text")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "revision": REVISION,
                "seed": SEED,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} records to {OUTPUT}")


if __name__ == "__main__":
    main()