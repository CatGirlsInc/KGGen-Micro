import argparse
import json
import logging
import os
import platform
import resource
import subprocess
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import dspy
from kg_gen import KGGen


PILOT = Path("data/mine-pilot.json")
DEFAULT_MODEL = "kggen-qwen25-7b:Q4_K_M"


def ollama_json(host, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{host.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def gpu_memory():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Run the fixed MINE pilot locally")
    parser.add_argument("--model", default=os.getenv("KGGEN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, help="Run only the first N fixed pilot records")
    parser.add_argument("--run-id", help="Stable output directory name")
    args = parser.parse_args()

    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    records = pilot["records"]
    if len(records) != 10:
        raise ValueError(f"Expected exactly 10 fixed records, found {len(records)}")
    if args.limit is not None and not 1 <= args.limit <= len(records):
        raise ValueError("--limit must be between 1 and 10")

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    tags = ollama_json(host, "/api/tags")["models"]
    model_info = next((item for item in tags if item["name"] == args.model), None)
    if model_info is None:
        raise RuntimeError(f"Model {args.model!r} is not installed in Ollama")

    stamp = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path("runs") / f"{stamp}-{args.model.replace(':', '-')}"
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(output / "kggen.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    settings = {
        "dataset": pilot["dataset"],
        "dataset_revision": pilot["revision"],
        "pilot_seed": pilot["seed"],
        "record_ids": [record["id"] for record in records],
        "model": args.model,
        "model_digest": model_info.get("digest"),
        "ollama_host": host,
        "temperature": 0,
        "seed": 42,
        "context_length": 4096,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "gpu_memory_before_mib": gpu_memory(),
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    kg = KGGen(
        model=f"ollama_chat/{args.model}",
        temperature=0,
        api_base=host,
        max_tokens=4096,
    )
    kg.lm = dspy.LM(
        model=f"ollama_chat/{args.model}",
        temperature=0,
        max_tokens=4096,
        api_base=host,
        engine="litellm",
    )
    outcomes = []
    for record in records[: args.limit]:
        started = time.perf_counter()
        result = {
            "record_id": record["id"],
            "topic": record["topic"],
            "status": "failed",
        }
        try:
            graph = kg.generate(
                input_data=record["text"],
                context=record["topic"],
                chunk_size=3000,
                cluster=True,
            )
            graph_path = output / f"graph_{record['id']}.json"
            graph_path.write_text(
                json.dumps(graph.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
            result["status"] = "succeeded"
            result["graph"] = graph_path.name
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            (output / f"error_{record['id']}.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["gpu_memory_mib"] = gpu_memory()
        (output / f"record_{record['id']}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        outcomes.append(result)
        print(f"record {record['id']}: {result['status']} ({result['elapsed_seconds']}s)")

    summary = {
        **settings,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "gpu_memory_after_mib": gpu_memory(),
        "attempted": len(outcomes),
        "succeeded": sum(item["status"] == "succeeded" for item in outcomes),
        "failed": sum(item["status"] == "failed" for item in outcomes),
        "outcomes": outcomes,
    }
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Run artifacts: {output}")
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()