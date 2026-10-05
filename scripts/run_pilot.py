import argparse
import json
import logging
import os
import platform
import resource
import subprocess
import threading
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import dspy
from kg_gen import KGGen


PILOT = Path("data/mine-pilot.json")
DEFAULT_MODEL = "kggen-qwen25-7b:Q4_K_M"


def ollama_json(host, path, body=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{host.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def unload_ollama_model(host, model):
    return ollama_json(
        host,
        "/api/generate",
        {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        timeout=60,
    )


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


def system_memory():
    try:
        values = {}
        with open("/proc/meminfo", encoding="utf-8") as memory_info:
            for line in memory_info:
                key, value = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable"}:
                    values[key] = int(value.strip().split()[0])
        return {
            "total_kib": values["MemTotal"],
            "available_kib": values["MemAvailable"],
            "used_kib": values["MemTotal"] - values["MemAvailable"],
        }
    except (OSError, KeyError, ValueError):
        return None


def install_kggen_cluster_progress_guard():
    from importlib import import_module

    cluster_module = import_module("kg_gen.steps._3_cluster_graph")
    if getattr(cluster_module, "_kggen_micro_progress_guard", False):
        return

    def cluster_items_with_progress(dspy_runtime, items, item_type="entities", context=""):
        context = f"{item_type} of a graph extracted from source text." + context
        remaining_items = items.copy()
        clusters = []
        no_progress_count = 0
        validate = None

        while remaining_items and no_progress_count < cluster_module.LOOP_N:
            extract_signature, _ = cluster_module.get_extract_cluster_sig(items)
            extract = dspy_runtime.Predict(extract_signature)
            suggested_cluster = set(
                extract(items=remaining_items, context=context).cluster
            ) & remaining_items
            if not suggested_cluster:
                no_progress_count += 1
                continue

            validate_signature, _ = cluster_module.get_validate_cluster_sig(
                suggested_cluster
            )
            validate = dspy_runtime.Predict(validate_signature)
            validated_cluster = set(
                validate(cluster=suggested_cluster, context=context).validated_items
            ) & suggested_cluster & remaining_items
            if not validated_cluster:
                no_progress_count += 1
                continue

            representative = cluster_module.choose_rep(
                cluster=validated_cluster, context=context
            ).representative
            clusters.append(
                cluster_module.Cluster(
                    representative=representative, members=validated_cluster
                )
            )
            remaining_items.difference_update(validated_cluster)
            no_progress_count = 0

        if remaining_items:
            items_to_process = list(remaining_items)
            for start in range(0, len(items_to_process), cluster_module.BATCH_SIZE):
                batch = items_to_process[
                    start : start + cluster_module.BATCH_SIZE
                ]
                cluster_module._process_batch(batch, clusters, context, validate)

        final_clusters = {cluster.representative: cluster.members for cluster in clusters}
        return set(final_clusters), final_clusters

    cluster_module.cluster_items = cluster_items_with_progress
    cluster_module._kggen_micro_progress_guard = True


class ResourceMonitor:
    def __init__(self, interval_seconds=1):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.peak_gpu_used_mib = None
        self.gpu_total_mib = None
        self.peak_system_used_kib = None
        self.minimum_system_available_kib = None

    def _sample(self):
        memory = system_memory()
        if memory is not None:
            used = memory["used_kib"]
            available = memory["available_kib"]
            self.peak_system_used_kib = max(self.peak_system_used_kib or 0, used)
            if self.minimum_system_available_kib is None:
                self.minimum_system_available_kib = available
            else:
                self.minimum_system_available_kib = min(
                    self.minimum_system_available_kib, available
                )

        gpu = gpu_memory()
        if gpu:
            try:
                used_mib, total_mib = (int(value.strip()) for value in gpu.split(","))
                self.peak_gpu_used_mib = max(self.peak_gpu_used_mib or 0, used_mib)
                self.gpu_total_mib = total_mib
            except ValueError:
                pass

    def _sample_loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        return {
            "sampling_interval_seconds": self.interval_seconds,
            "peak_gpu_used_mib": self.peak_gpu_used_mib,
            "gpu_total_mib": self.gpu_total_mib,
            "peak_system_used_kib": self.peak_system_used_kib,
            "minimum_system_available_kib": self.minimum_system_available_kib,
        }


def main():
    parser = argparse.ArgumentParser(description="Run the fixed MINE pilot locally")
    parser.add_argument(
        "--model", help="Ollama model tag; defaults to KGGEN_MODEL or the pilot model"
    )
    parser.add_argument("--limit", type=int, help="Run only the first N fixed pilot records")
    parser.add_argument("--run-id", help="Stable output directory name")
    parser.add_argument(
        "--cluster",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable graph clustering (default: saved setting or enabled)",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        help="Replay model and parameters from a saved settings.json or run.json",
    )
    args = parser.parse_args()

    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    pilot_records = pilot["records"]
    if len(pilot_records) != 10:
        raise ValueError(f"Expected exactly 10 fixed records, found {len(pilot_records)}")
    saved_settings = (
        json.loads(args.settings.read_text(encoding="utf-8")) if args.settings else {}
    )
    if args.settings and args.model:
        parser.error("--model cannot be combined with --settings")
    if args.settings and (
        saved_settings.get("dataset_revision") != pilot["revision"]
        or saved_settings.get("dataset") != pilot["dataset"]
    ):
        raise ValueError("Saved settings refer to a different pilot dataset revision")

    if args.settings:
        saved_ids = saved_settings.get("record_ids")
        records_by_id = {record["id"]: record for record in pilot_records}
        if (
            not isinstance(saved_ids, list)
            or not saved_ids
            or any(not isinstance(record_id, int) or record_id not in records_by_id for record_id in saved_ids)
            or len(saved_ids) != len(set(saved_ids))
        ):
            raise ValueError("Saved settings contain invalid pilot record IDs")
        records = [records_by_id[record_id] for record_id in saved_ids]
    else:
        records = pilot_records
    model = saved_settings.get("model") if args.settings else None
    model = model or args.model or os.getenv("KGGEN_MODEL", DEFAULT_MODEL)
    temperature = saved_settings.get("temperature", 0)
    seed = saved_settings.get("seed", 42)
    context_length = saved_settings.get("context_length", 4096)
    max_tokens = saved_settings.get("max_tokens", context_length)
    dspy_num_retries = saved_settings.get("dspy_num_retries", 3)
    chunk_size = saved_settings.get("chunk_size", 3000)
    cluster = saved_settings.get("cluster", True) if args.cluster is None else args.cluster
    lm_options = saved_settings.get("model_specific_options")
    if lm_options is None:
        lm_options = {"think": False} if model.startswith("qwen3.5:") else {}
    if args.limit is not None and not 1 <= args.limit <= len(records):
        raise ValueError(f"--limit must be between 1 and {len(records)}")
    records = records[: args.limit]

    host = os.getenv(
        "OLLAMA_HOST", saved_settings.get("ollama_host", "http://localhost:11434")
    )
    ollama_version = ollama_json(host, "/api/version").get("version")
    tags = ollama_json(host, "/api/tags")["models"]
    model_info = next((item for item in tags if item["name"] == model), None)
    if model_info is None:
        raise RuntimeError(f"Model {model!r} is not installed in Ollama")
    expected_digest = saved_settings.get("model_digest")
    if args.settings and expected_digest and model_info.get("digest") != expected_digest:
        raise RuntimeError("Installed model digest differs from the saved settings")

    stamp = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path("runs") / f"{stamp}-{model.replace(':', '-')}"
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
        "model": model,
        "model_digest": model_info.get("digest"),
        "ollama_host": host,
        "ollama_version": ollama_version,
        "temperature": temperature,
        "seed": seed,
        "context_length": context_length,
        "max_tokens": max_tokens,
        "dspy_num_retries": dspy_num_retries,
        "chunk_size": chunk_size,
        "cluster": cluster,
        "dspy_engine": "litellm",
        "model_specific_options": lm_options,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "gpu_memory_before_mib": gpu_memory(),
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    outcomes = []
    unload_status = "not_attempted"
    resource_monitor = ResourceMonitor()
    resource_monitor.start()
    try:
        install_kggen_cluster_progress_guard()
        kg = KGGen(
            model=f"ollama_chat/{model}",
            temperature=temperature,
            api_base=host,
            max_tokens=max_tokens,
        )
        kg.lm = dspy.LM(
            model=f"ollama_chat/{model}",
            temperature=temperature,
            max_tokens=max_tokens,
            num_retries=dspy_num_retries,
            num_ctx=context_length,
            api_base=host,
            engine="litellm",
            seed=seed,
            **lm_options,
        )
        for record in records:
            started = time.perf_counter()
            result = {
                "record_id": record["id"],
                "topic": record["topic"],
                "status": "failed",
            }
            intermediate_dir = output / f"intermediate_{record['id']}"
            try:
                result["stage"] = "extraction"
                extraction_started = time.perf_counter()
                extracted_graph = kg.generate(
                    input_data=record["text"],
                    context=record["topic"],
                    chunk_size=chunk_size,
                    cluster=False,
                    output_folder=str(intermediate_dir),
                )
                result["extraction_elapsed_seconds"] = round(
                    time.perf_counter() - extraction_started, 3
                )
                result["intermediate"] = str(intermediate_dir / "graph.json")
                if cluster:
                    result["stage"] = "clustering"
                    clustering_started = time.perf_counter()
                    graph = kg.cluster(extracted_graph, context=record["topic"])
                    result["clustering_elapsed_seconds"] = round(
                        time.perf_counter() - clustering_started, 3
                    )
                else:
                    graph = extracted_graph
                graph_path = output / f"graph_{record['id']}.json"
                graph_path.write_text(
                    json.dumps(graph.model_dump(mode="json"), indent=2) + "\n",
                    encoding="utf-8",
                )
                result["status"] = "succeeded"
                result.pop("stage")
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
    finally:
        try:
            unload_ollama_model(host, model)
            unload_status = "succeeded"
        except Exception as error:
            unload_status = f"failed: {type(error).__name__}: {error}"
            print(f"Ollama model cleanup failed: {unload_status}")
    resource_peaks = resource_monitor.stop()

    summary = {
        **settings,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "gpu_memory_after_mib": gpu_memory(),
        "resource_peaks": resource_peaks,
        "model_unload_status": unload_status,
        "attempted": len(outcomes),
        "succeeded": sum(item["status"] == "succeeded" for item in outcomes),
        "failed": sum(item["status"] == "failed" for item in outcomes),
        "outcomes": outcomes,
    }
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Run artifacts: {output}")
    if summary["failed"] or unload_status != "succeeded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()