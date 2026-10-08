"""Plot the E-6 optional k-means versus selected nodal-domain comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/raw/matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from PIL import Image

BLUE = "#3174ad"
ORANGE = "#e58c33"
GREEN = "#4b9778"
GRAY = "#b3b3b3"
GROUP_PALETTE = (BLUE, ORANGE, GREEN, "#9d6ab0", "#ca6462", "#797960")


def token_grid(axis, labels: np.ndarray, grid: tuple[int, int], title: str) -> None:
    maximum = max(int(labels.max()), 0)
    colors = [GRAY, *[GROUP_PALETTE[index % len(GROUP_PALETTE)] for index in range(maximum + 1)]]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-1.5, maximum + 1.5), cmap.N)
    axis.imshow(labels.reshape(grid), cmap=cmap, norm=norm, interpolation="nearest")
    axis.set(title=title, xlabel="Token column", ylabel="Token row")
    axis.set_xticks([0, 5, 10, 15])
    axis.set_yticks([0, 5, 10, 15])
    axis.set_xticks(np.arange(-0.5, grid[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, grid[0], 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.2, alpha=0.4)
    axis.tick_params(which="minor", bottom=False, left=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "data/generated/graph_analysis")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/generated/dit_xl2_imagenet256")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    metadata = json.loads((args.graph_dir / "metadata.json").read_text())
    baseline_metadata = json.loads((args.graph_dir / "optional_clustering_metadata.json").read_text())
    representative = metadata["representative"]
    with (args.graph_dir / "optional_clustering.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    match = lambda row: row["sample_id"] == representative["sample_id"] and all(
        int(row[key]) == representative[key] for key in ("denoiser_call", "block_1based"))
    row = next(row for row in rows if match(row))
    snapshots = [json.loads(line) for line in (args.graph_dir / "index.jsonl").read_text().splitlines()]
    snapshot = next(snapshot for snapshot in snapshots if match(snapshot))
    with np.load(args.graph_dir / row["baseline_array_path"], allow_pickle=False) as arrays:
        kmeans_labels = arrays["kmeans_labels"]
        nodal_labels = arrays["nodal_labels"]
    grid = tuple(snapshot["token_grid"])
    with (args.graph_dir / "component_summary.csv").open(newline="", encoding="utf-8") as source:
        component = next(item for item in csv.DictReader(source)
                         if match(item) and int(item["k"]) == int(row["k"]))
    eigenvector = component["selected_eigenvalue_index_1based"]
    comparable = [item for item in rows if item["comparable_two_group_partition"] == "True"]
    if not comparable:
        raise ValueError("no complete two-domain nodal partitions to compare")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure = plt.figure(figsize=(13.2, 7.5), layout="constrained")
    layout = figure.add_gridspec(2, 3, height_ratios=(1.1, 1))
    axes = [figure.add_subplot(layout[0, index]) for index in range(3)]
    with Image.open(args.dataset_dir / snapshot["image_path"]) as image:
        axes[0].imshow(image)
    axes[0].set_title(f"Generated image: {snapshot['class_name']}")
    axes[0].set_axis_off()
    token_grid(axes[1], kmeans_labels, grid,
               f"K-means, K=2\ncluster sizes {row['kmeans_cluster_0_size']} / {row['kmeans_cluster_1_size']}")
    if row["nodal_available"] == "True":
        token_grid(axes[2], nodal_labels, grid,
                   f"Selected nodal domains, q{eigenvector}\n{row['nodal_domain_count']} domains")
    else:
        axes[2].text(0.5, 0.5, "Nodal partition unavailable", ha="center", va="center")
        axes[2].set_axis_off()
    scatter = figure.add_subplot(layout[1, :2])
    for block, color in ((4, BLUE), (14, ORANGE), (24, GREEN)):
        selected = [item for item in comparable if int(item["block_1based"]) == block]
        scatter.scatter([float(item["kmeans_normalized_cut"]) for item in selected],
                        [float(item["nodal_normalized_cut"]) for item in selected],
                        color=color, s=18, alpha=0.55, edgecolors="none", label=f"Block {block}")
    all_values = [float(item[field]) for item in comparable
                  for field in ("kmeans_normalized_cut", "nodal_normalized_cut")]
    limit = max(all_values) * 1.05
    scatter.plot([0, limit], [0, limit], color="#555555", linestyle="--", lw=1, label="Equal cut value")
    scatter.set(xlabel="K-means normalized cut", ylabel="Selected nodal normalized cut",
                xlim=(0, limit), ylim=(0, limit),
                title=f"Descriptive comparison: {len(comparable)} complete two-group snapshots")
    scatter.grid(alpha=0.15)
    scatter.legend(fontsize=8, loc="upper left", ncols=2)
    if row["comparable_two_group_partition"] == "True":
        scatter.scatter([float(row["kmeans_normalized_cut"])], [float(row["nodal_normalized_cut"])],
                        marker="*", s=180, color="#222222", zorder=4)
        scatter.annotate("Representative", (float(row["kmeans_normalized_cut"]),
                                               float(row["nodal_normalized_cut"])),
                         xytext=(8, 8), textcoords="offset points", fontsize=9)
    notes = figure.add_subplot(layout[1, 2])
    notes.set_axis_off()
    lines = ["Representative comparison", "",
             f"K-means Ncut: {float(row['kmeans_normalized_cut']):.4f}"]
    if row["comparable_two_group_partition"] == "True":
        lines.extend([f"Nodal Ncut: {float(row['nodal_normalized_cut']):.4f}",
                      f"Partition ARI: {float(row['partition_ARI']):.4f}"])
    lines.extend(["", f"K-means: {baseline_metadata['initializations']} fixed starts,",
                  f"{baseline_metadata['iterations']} iterations, 1152 channels.", "",
                  "Nodal vector selected by eigenvalue gaps;",
                  "its partition does not optimize Ncut.", "",
                  "Blue: group 0; orange: group 1.",
                  "Labels are local; no semantic ground truth.", "",
                  "Scatter points share 50 trajectories;",
                  "they are descriptive observations."])
    notes.text(0, 1, "\n".join(lines), transform=notes.transAxes,
               ha="left", va="top", fontsize=9.5, linespacing=1.35)
    figure.suptitle(f"E-6: {representative['sample_id']} | call {representative['denoiser_call']} | "
                   f"block {representative['block_1based']} | k={row['k']}", fontsize=13)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    target = args.figure_dir / "e6_clustering_comparison"
    figure.savefig(target.with_suffix(".png"), dpi=160, bbox_inches="tight")
    figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"saved: {target.with_suffix('.png')} and {target.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
