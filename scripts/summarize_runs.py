import argparse
import json
from pathlib import Path


def load_graph(path):
    graph = json.loads(path.read_text(encoding="utf-8"))
    return {
        "entities": len(graph.get("entities", [])),
        "relations": len(graph.get("relations", [])),
    }


def summarize_run(run_dir):
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []

    for outcome in manifest["outcomes"]:
        record = {
            "record_id": outcome["record_id"],
            "status": outcome["status"],
            "elapsed_seconds": outcome["elapsed_seconds"],
            "process_peak_rss_kib": outcome.get("process_peak_rss_kib"),
        }
        intermediate_value = outcome.get("intermediate")
        intermediate_path = Path(intermediate_value) if intermediate_value else None
        if intermediate_path and intermediate_path.is_file():
            record["extracted"] = load_graph(intermediate_path)
        else:
            fallback_path = run_dir / f"intermediate_{outcome['record_id']}" / "graph.json"
            if fallback_path.is_file():
                record["extracted"] = load_graph(fallback_path)

        graph_name = outcome.get("graph")
        graph_path = run_dir / graph_name if graph_name else None
        if graph_path and graph_path.is_file():
            record["final"] = load_graph(graph_path)
            if "extracted" in record:
                record["entity_reduction"] = reduction(
                    record["extracted"]["entities"], record["final"]["entities"]
                )
                record["relation_reduction"] = reduction(
                    record["extracted"]["relations"], record["final"]["relations"]
                )
        records.append(record)

    successful = [record for record in records if record["status"] == "succeeded"]
    resource_peaks = manifest.get("resource_peaks", {})
    return {
        "run_id": run_dir.name,
        "model": manifest.get("model"),
        "model_digest": manifest.get("model_digest"),
        "record_ids": manifest.get("record_ids", []),
        "attempted": manifest.get("attempted", len(records)),
        "succeeded": manifest.get("succeeded", len(successful)),
        "failed": manifest.get("failed", len(records) - len(successful)),
        "success_rate": len(successful) / len(records) if records else None,
        "total_record_seconds": sum(record["elapsed_seconds"] for record in records),
        "mean_seconds_per_record": (
            sum(record["elapsed_seconds"] for record in records) / len(records)
            if records
            else None
        ),
        "peak_gpu_used_mib": resource_peaks.get("peak_gpu_used_mib"),
        "gpu_total_mib": resource_peaks.get("gpu_total_mib"),
        "peak_system_used_gib": (
            resource_peaks["peak_system_used_kib"] / (1024 * 1024)
            if resource_peaks.get("peak_system_used_kib") is not None
            else None
        ),
        "max_process_peak_rss_gib": (
            max(
                (record["process_peak_rss_kib"] for record in records
                 if record["process_peak_rss_kib"] is not None),
                default=None,
            )
            / (1024 * 1024)
        ),
        "records": records,
    }


def reduction(before, after):
    return (before - after) / before if before else None


def main():
    parser = argparse.ArgumentParser(
        description="Summarize saved KGGen pilot run manifests and graphs as JSON"
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize_run(path) for path in args.run_dirs], indent=2))


if __name__ == "__main__":
    main()