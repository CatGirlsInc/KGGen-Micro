import json
from pathlib import Path


SOURCE = Path("data/mine-confirmation-cohort.json")
OUTPUT = Path("data/cat114-development-cohort.json")
EXPECTED_IDS = [
    24, 7, 102, 40, 52, 37, 70, 26, 101, 96,
    44, 6, 64, 35, 22, 42, 13, 32, 46, 100,
]


def main():
    cohort = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = cohort.get("records", [])
    record_ids = [record.get("id") for record in records]
    if record_ids != EXPECTED_IDS or cohort.get("record_ids") != EXPECTED_IDS:
        raise ValueError("CAT-108 cohort differs from the predeclared CAT-114 IDs")

    manifest = {
        **cohort,
        "experiment": "CAT-114 controlled 8K development-set confirmation",
        "source_manifest": str(SOURCE),
        "selection_note": (
            "Exact CAT-108 20-record cohort; development set because 8K was chosen "
            "after observing failures on these records."
        ),
    }
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} frozen CAT-114 records to {OUTPUT}: {record_ids}")


if __name__ == "__main__":
    main()