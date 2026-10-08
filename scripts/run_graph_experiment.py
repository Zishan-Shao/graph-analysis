"""Reproduce E-2--E-7 from the saved DiT dataset, without using a GPU."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="Parallel spectral-analysis workers")
    parser.add_argument("--plot-workers", type=int, default=8, help="Parallel component-plot workers")
    parser.add_argument("--from-results", action="store_true", help="Reuse calculated tables and arrays; render reports only")
    parser.add_argument("--skip-component-plots", action="store_true", help="Render curated report figures without all component figures")
    args = parser.parse_args()
    if args.workers < 1 or args.plot_workers < 1:
        parser.error("Worker counts must be positive")
    environment = os.environ.copy()
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"
    environment.setdefault("MPLCONFIGDIR", str(ROOT / "data/raw/matplotlib-cache"))
    stages = []
    if not args.from_results:
        stages.extend([
            ["validate_dataset.py"],
            ["build_interactions.py"],
            ["analyze_graphs.py", "--workers", str(args.workers)],
            ["analyze_attributes.py"],
            ["compare_clustering.py"],
        ])
    plotting = ["plot_experiment.py", "--workers", str(args.plot_workers)]
    if args.skip_component_plots:
        plotting.append("--skip-component-plots")
    stages.extend([plotting, ["plot_clustering_comparison.py"], ["write_experiment_report.py"]])
    for stage in stages:
        command = [sys.executable, str(ROOT / "scripts" / stage[0]), *stage[1:]]
        print("Running: " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print("Finished: reports/experiment_report.pdf and reports/answers_e2_e7.tex", flush=True)


if __name__ == "__main__":
    main()
