import json
from pathlib import Path


COHORT = Path("data/cat114-development-cohort.json")
PHI = Path("runs/cat114-8k-phi4-development-phi4-mini-3.8b-q4_K_M")
QWEN = Path("runs/cat114-8k-qwen35-development-qwen3.5-4b-q4_K_M")
PHI_40 = Path("runs/cat114-audit-only-phi4-40-phi4-mini-3.8b-q4_K_M")
OUTPUT = Path("runs/cat114-human-audit/packet.md")
REQUIRED_IDS = [26, 37, 40]


def load_graph(run_dir, record_id):
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    outcome = next(
        (item for item in manifest["outcomes"] if item["record_id"] == record_id),
        None,
    )
    if not outcome or outcome.get("status") != "succeeded":
        raise ValueError(f"No successful graph for record {record_id} in {run_dir}")
    return json.loads((run_dir / outcome["graph"]).read_text(encoding="utf-8"))


def main():
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    cohort_ids = cohort["record_ids"]
    phi_manifest = json.loads((PHI / "run.json").read_text(encoding="utf-8"))
    qwen_manifest = json.loads((QWEN / "run.json").read_text(encoding="utf-8"))
    phi_success = {
        item["record_id"] for item in phi_manifest["outcomes"] if item["status"] == "succeeded"
    }
    qwen_success = {
        item["record_id"] for item in qwen_manifest["outcomes"] if item["status"] == "succeeded"
    }
    additional_ids = sorted((phi_success & qwen_success) - set(REQUIRED_IDS))[:2]
    audit_ids = REQUIRED_IDS + additional_ids
    if len(additional_ids) != 2:
        raise ValueError("Could not select two additional audit records")
    records = {record["id"]: record for record in cohort["records"]}

    sections = [
        "# CAT-114 Human Graph Audit Packet",
        "",
        "**Status:** Prepared for independent human review; no audit findings have been entered.",
        "**Configuration:** KGGen 0.4.0, both pinned model digests, 8192 context/output tokens, clustering enabled, chunk size 3000, temperature 0, seed 42. Qwen uses `think=false`.",
        f"**Selection:** Required IDs 26, 37, 40; two lowest-numbered shared-success IDs outside that set are {additional_ids}.",
        "**Selection deviation:** The original predeclared lowest-numbered candidates were 6 and 7, but Phi-4-mini had no valid development-run graph for record 6. Record 6 remains a generation failure in reliability results; record 13 is the next-lowest shared-success replacement for the audit only.",
        "**Audit-only exception:** Phi record 40 was absent from its 20-record run after two predeclared retries. A separate one-attempt 8K artifact was generated solely to provide a paired audit graph and is excluded from all reliability and quality metrics.",
        "",
        "Do not use automated model judgments as audit findings. Review source evidence against each graph and record concrete examples, including relation triples and source passages.",
    ]
    for record_id in audit_ids:
        source = records[record_id]
        phi_graph_dir = PHI_40 if record_id == 40 else PHI
        phi_graph = load_graph(phi_graph_dir, record_id)
        qwen_graph = load_graph(QWEN, record_id)
        sections.extend(
            [
                "",
                f"## Record {record_id}: {source['topic']}",
                "",
                "### Source text",
                "",
                source["text"],
                "",
                "### Phi-4-mini graph",
                "",
                f"Source run: `{phi_graph_dir}`",
                "",
                "```json",
                json.dumps(phi_graph, indent=2, ensure_ascii=False),
                "```",
                "",
                "### Qwen3.5 graph",
                "",
                f"Source run: `{QWEN}`",
                "",
                "```json",
                json.dumps(qwen_graph, indent=2, ensure_ascii=False),
                "```",
                "",
                "### Human reviewer notes",
                "",
                "Reviewer / date: ____________________",
                "",
                "| Dimension | Phi-4-mini evidence and rating | Qwen3.5 evidence and rating |",
                "| --- | --- | --- |",
                "| Unsupported or hallucinated triples |  |  |",
                "| Important omitted facts |  |  |",
                "| Entity merges / canonicalization |  |  |",
                "| Relationship correctness |  |  |",
                "| Overall usefulness for recovering source information |  |  |",
                "",
                "Material correctness regression? ____________________",
                "",
                "Specific source passage and graph evidence: ____________________",
            ]
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote human-review packet for records {audit_ids} to {OUTPUT}")


if __name__ == "__main__":
    main()