"""Run independent DiT class/seed subsets in parallel, one process per GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True,
                        help="Comma-separated usable physical CUDA device IDs")
    parser.add_argument("--limit", type=int, help="Generate only the first N manifest conditions")
    args = parser.parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        parser.error("--gpus must contain distinct CUDA device IDs")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    log_dir = ROOT / "data" / "generated" / "dit_xl2_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    workers: list[tuple[int, subprocess.Popen, object]] = []
    try:
        for shard_index, gpu_id in enumerate(gpu_ids):
            log_path = log_dir / f"gpu_{gpu_id}.log"
            log = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            command = [
                sys.executable, str(ROOT / "scripts" / "sample_dit.py"),
                "--device", "cuda:0", "--num-shards", str(len(gpu_ids)),
                "--shard-index", str(shard_index),
            ]
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
            workers.append((shard_index, process, log))
            print(f"GPU {gpu_id}: PID {process.pid}, log {log_path}", flush=True)
        failures = []
        for shard_index, process, _log in workers:
            result = process.wait()
            if result:
                failures.append((shard_index, result))
        if failures:
            raise SystemExit(f"failed workers: {failures}; inspect data/generated/dit_xl2_logs/")
        print("All shards completed successfully")
    except KeyboardInterrupt:
        for _index, process, _log in workers:
            process.terminate()
        raise
    finally:
        for _index, _process, log in workers:
            log.close()


if __name__ == "__main__":
    main()
