"""Render E-2--E-7 figures and every large component at six detailed k values."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/raw/matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from PIL import Image


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def save_figure(figure, target: Path, pdf: bool = True) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target.with_suffix(".png"), dpi=160, bbox_inches="tight")
    if pdf:
        figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def domain_colors(labels: np.ndarray):
    n = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    if n <= 10:
        palette = plt.get_cmap("tab10")
        colors = np.array([palette(index) for index in range(max(n, 1))])
    elif n <= 20:
        palette = plt.get_cmap("tab20")
        colors = np.array([palette(index) for index in range(n)])
    else:
        colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, n))
    cmap = ListedColormap(np.vstack([[0.7, 0.7, 0.7, 1], colors]))
    norm = BoundaryNorm(np.arange(-1.5, max(n, 1) + 0.5), cmap.N)
    return cmap, norm


def component_figure(task: tuple[dict, str, str, bool]) -> str:
    row, graph_root_string, target_string, pdf = task
    graph_root, target = Path(graph_root_string), Path(target_string)
    with np.load(graph_root / row["arrays_file"]) as source:
        arrays = {key: source[key] for key in source.files}
    indices = arrays["token_indices"]
    labels = arrays["nodal_labels"]
    cmap, norm = domain_colors(labels)
    figure = plt.figure(figsize=(13.2, 7.6), layout="constrained")
    axes = [figure.add_subplot(2, 3, index + 1,
                             projection="3d" if index in (3, 4) else None)
            for index in range(6)]
    degrees = arrays["degrees"]
    axes[0].hist(degrees, bins=np.arange(degrees.min() - 0.5, degrees.max() + 1.5),
                 color="#3174ad", edgecolor="white")
    axes[0].set(xlabel="Unweighted degree", ylabel="Number of vertices",
                title="E-4: degree distribution")
    axes[1].hist(arrays["clustering"], bins=np.linspace(0, 1, 21),
                 color="#4b9778", edgecolor="white")
    axes[1].set(xlabel="Local clustering coefficient", ylabel="Number of vertices",
                xlim=(0, 1), title="E-4: clustering distribution")
    values = arrays["eigenvalues"]
    axes[2].plot(np.arange(1, len(values) + 1), values, "o-", markersize=3)
    selected_text = row.get("selected_eigenvalue_index", "")
    if selected_text:
        j = int(selected_text)
        axes[2].scatter([j], [values[j - 1]], color="#c74840", s=45, zorder=3)
        axes[2].annotate(f"selected j={j}", (j, values[j - 1]),
                         xytext=(5, -15), textcoords="offset points", fontsize=9)
    axes[2].set(xlabel="Eigenvalue index (1-based)", ylabel="Normalized Laplacian eigenvalue",
                title=f"E-5: smallest {len(values)} eigenvalues")
    embedding = arrays["embedding"]
    axes[3].scatter(*embedding.T, c=labels, cmap=cmap, norm=norm, s=9,
                    depthshade=False)
    for coordinate, setter in enumerate((axes[3].set_xlabel, axes[3].set_ylabel,
                                         axes[3].set_zlabel)):
        setter(f"q{coordinate + 2}")
    axes[3].set_title("E-5: invariant subspace (q2,q3,q4)")
    nodal_embedding = arrays["nodal_embedding"]
    if selected_text:
        axes[4].scatter(*nodal_embedding.T, c=labels, cmap=cmap, norm=norm, s=9,
                        depthshade=False)
        vector_indices = arrays["nodal_embedding_eigenvector_indices_1based"]
        for index, setter in zip(vector_indices, (axes[4].set_xlabel,
                                                  axes[4].set_ylabel, axes[4].set_zlabel)):
            setter(f"q{index}")
        axes[4].set_title(f"E-5: {int(labels.max()) + 1} strict nodal domains")
    else:
        axes[4].text2D(0.05, 0.5, "No well-separated simple\neigenvalue met the fixed rule",
                       transform=axes[4].transAxes)
        axes[4].set_title("E-5: candidate absent")
    grid = np.full(256, np.nan)
    grid[indices] = labels
    axes[5].imshow(grid.reshape(16, 16), cmap=cmap, norm=norm, interpolation="nearest")
    axes[5].set(xlabel="Token column", ylabel="Token row",
                title=("Nodal domains on token grid (gray: zero)" if selected_text
                       else "Nodal partition unavailable"))
    figure.suptitle(f"{row['sample_id']} | call {row['denoiser_call']} | block {row['block_1based']}"
                   f" | k={row['k']} | component {row['component_id']} | n={len(indices)}", fontsize=12)
    save_figure(figure, target, pdf=pdf)
    return str(target.with_suffix(".png"))


def plot_interactions(graph_root: Path, representative: dict, figure_root: Path) -> None:
    with np.load(graph_root / representative["interaction_path"]) as arrays:
        affinity = arrays["affinity"]
        distances = np.sqrt(arrays["distances_squared"])
    triangle = np.triu_indices(len(affinity), 1)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), layout="constrained")
    image = axes[0].imshow(affinity, vmin=0, vmax=1, cmap="viridis")
    figure.colorbar(image, ax=axes[0], label="Gaussian affinity")
    axes[0].set(xlabel="Token j", ylabel="Token i", title="256 x 256 weighted interaction")
    axes[1].hist(distances[triangle], bins=45, color="#3174ad")
    axes[1].axvline(representative["sigma"], color="#c74840", label="sigma: median distance")
    axes[1].set(xlabel="Euclidean activation distance", ylabel="Unordered pair count",
                title="All 32,640 unordered pairs")
    axes[1].legend(fontsize=8)
    axes[2].hist(affinity[triangle], bins=45, color="#4b9778")
    axes[2].set(xlabel="Affinity", ylabel="Unordered pair count",
                title="Off-diagonal interaction weights")
    save_figure(figure, figure_root / "e2_pairwise_interactions")


def plot_curve_grid(rows: list[dict], metric: str, title: str,
                    target: Path, connected_only: bool = False) -> None:
    groups = defaultdict(list)
    for row in rows:
        if row.get(metric, "") == "":
            continue
        groups[(int(row["denoiser_call"]), int(row["block_1based"]), int(row["k"]))].append(float(row[metric]))
    figure, axes = plt.subplots(3, 3, figsize=(11.2, 8), layout="constrained")
    for call_index, call in enumerate((10, 25, 40)):
        for block_index, block in enumerate((4, 14, 24)):
            axis = axes[call_index, block_index]
            k_values = sorted(k for c, b, k in groups if (c, b) == (call, block))
            samples = [np.array(groups[(call, block, k)]) for k in k_values]
            axis.plot(k_values, [x.mean() for x in samples], color="#3174ad", lw=1.6)
            axis.fill_between(k_values, [x.min() for x in samples], [x.max() for x in samples],
                              color="#3174ad", alpha=0.16)
            axis.set(title=f"Call {call}, block {block}", xlabel="k", ylabel=title)
            axis.set_xscale("log", base=2)
            axis.set_xticks([2, 4, 8, 16, 32, 64], labels=[2, 4, 8, 16, 32, 64])
            axis.grid(alpha=0.2)
    qualifier = "connected snapshots only; count varies with k" if connected_only else "50 trajectories per panel"
    figure.suptitle(f"Mean and observed range across {qualifier}", fontsize=12)
    save_figure(figure, target)


def plot_representative_curves(connectivity: list[dict], fiedler: list[dict],
                               representative: dict, figure_root: Path) -> None:
    match = lambda row: row["snapshot_id"] == representative["snapshot_id"]
    rows = [row for row in connectivity if match(row)]
    spectra = [row for row in fiedler if match(row) and row["fiedler_value"]]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), layout="constrained")
    axes[0].plot([int(row["k"]) for row in rows], [int(row["n_components"]) for row in rows], "o-", ms=3)
    axes[0].set(xlabel="k", ylabel="Connected components", title="E-3: representative connectivity")
    axes[1].plot([int(row["k"]) for row in spectra], [float(row["fiedler_value"]) for row in spectra], "o-", ms=3)
    axes[1].set(xlabel="k", ylabel="Fiedler value lambda2", title="E-5: representative connected range")
    for axis in axes:
        axis.grid(alpha=0.2)
    save_figure(figure, figure_root / "e3_e5_representative_curves")


def plot_attribute_summary(rows: list[dict], figure_root: Path) -> None:
    selected = [row for row in rows if int(row["k"]) == 16]
    metrics = ("update_norm_energy_ratio", "residual_energy_ratio", "grid_energy_ratio")
    titles = ("Update magnitude", "Full update vector", "Spatial coordinates")
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), layout="constrained")
    for axis, metric, title in zip(axes, metrics, titles):
        means = np.array([[np.mean([float(row[metric]) for row in selected
                                   if int(row["denoiser_call"]) == call
                                   and int(row["block_1based"]) == block and row[metric]])
                           for block in (4, 14, 24)] for call in (10, 25, 40)])
        upper = max(1.05, float(np.nanmax(means)))
        image = axis.imshow(means, vmin=0, vmax=upper, cmap="YlGnBu")
        for (i, j), value in np.ndenumerate(means):
            axis.text(j, i, f"{value:.3f}", ha="center", va="center",
                      color="white" if value > 0.6 * upper else "black")
        axis.set(xticks=[0, 1, 2], xticklabels=[4, 14, 24], xlabel="Block",
                 yticks=[0, 1, 2], yticklabels=[10, 25, 40], ylabel="Denoiser call", title=title)
        figure.colorbar(image, ax=axis, label="Energy / permutation expectation")
    figure.suptitle("E-7: graph signal smoothness, k=16 (ratio below 1 = smoother than shuffled)")
    save_figure(figure, figure_root / "e7_attribute_smoothness")


def plot_neighbor_overlap(rows: list[dict], figure_root: Path) -> None:
    selected = [row for row in rows if int(row["k"]) == 16]
    figure, axis = plt.subplots(figsize=(6.6, 3.8), layout="constrained")
    calls = (10, 25, 40)
    for source, target in ((4, 14), (14, 24), (4, 24)):
        samples = [np.array([float(row["mean_directed_neighbor_overlap"]) for row in selected
                            if int(row["denoiser_call"]) == call
                            and int(row["source_block"]) == source
                            and int(row["target_block"]) == target]) for call in calls]
        axis.plot(calls, [x.mean() for x in samples], "o-", label=f"Block {source} -> {target}")
    axis.axhline(16 / 255, color="gray", ls="--", label="Independent random neighbor sets")
    axis.set(xticks=calls, xlabel="Denoiser call", ylabel="Mean top-16 neighbor retention",
             ylim=(0, 1), title="E-7: cross-layer neighbor overlap")
    axis.legend(fontsize=8)
    save_figure(figure, figure_root / "e7_cross_layer_neighbors")


def plot_original_data(row: dict, graph_root: Path, dataset: Path, target: Path) -> None:
    with np.load(graph_root / row["arrays_file"]) as source:
        labels, indices, update_norm = source["nodal_labels"], source["token_indices"], source["update_norm"]
    cmap, norm = domain_colors(labels)
    label_grid, update_grid = np.full(256, np.nan), np.full(256, np.nan)
    label_grid[indices], update_grid[indices] = labels, update_norm
    figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), layout="constrained")
    with Image.open(dataset / row["sample_id"] / "image.png") as image:
        pixels = np.asarray(image)
    axes[0].imshow(pixels)
    axes[0].set_title("Final generated image")
    axes[1].imshow(label_grid.reshape(16, 16), cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Token nodal domains")
    axes[2].imshow(pixels)
    axes[2].imshow(label_grid.reshape(16, 16), cmap=cmap, norm=norm, alpha=0.35,
                   extent=(-0.5, 255.5, 255.5, -0.5), interpolation="nearest")
    axes[2].set_title("Nominal grid overlay")
    image = axes[3].imshow(update_grid.reshape(16, 16), cmap="magma", interpolation="nearest")
    figure.colorbar(image, ax=axes[3], label="Block-update L2 norm")
    axes[3].set_title("Measured update magnitude")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle("E-5/E-7: visual correspondence; token grid is not a semantic segmentation label")
    save_figure(figure, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "data/generated/graph_analysis")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/generated/dit_xl2_imagenet256")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "reports/figures")
    parser.add_argument("--main-k", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-component-plots", action="store_true")
    parser.add_argument("--main-k-only", action="store_true",
                        help="Render component plots only at main k; default covers all six detailed k values")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    metadata = json.loads((args.graph_dir / "metadata.json").read_text())
    index = [json.loads(line) for line in (args.graph_dir / "index.jsonl").read_text().splitlines()]
    rep = metadata["representative"]
    representative = next(row for row in index if all(row[key] == value for key, value in rep.items()))
    connectivity = csv_rows(args.graph_dir / "connectivity.csv")
    fiedler = csv_rows(args.graph_dir / "fiedler_curves.csv")
    components = csv_rows(args.graph_dir / "component_summary.csv")
    attributes = csv_rows(args.graph_dir / "attributes.csv")
    overlaps = csv_rows(args.graph_dir / "cross_layer_neighbors.csv")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    plot_interactions(args.graph_dir, representative, args.figure_dir)
    plot_curve_grid(connectivity, "n_components", "Connected components", args.figure_dir / "e3_connectivity_all")
    plot_curve_grid(fiedler, "fiedler_value", "Fiedler value", args.figure_dir / "e5_fiedler_all", connected_only=True)
    plot_representative_curves(connectivity, fiedler, representative, args.figure_dir)
    plot_attribute_summary(attributes, args.figure_dir)
    plot_neighbor_overlap(overlaps, args.figure_dir)
    rep_components = [row for row in components if row["snapshot_id"] == representative["snapshot_id"]]
    for row in rep_components:
        target = args.figure_dir / f"e4_e5_k{int(row['k']):02d}_component{int(row['component_id']):02d}"
        component_figure((row, str(args.graph_dir), str(target), True))
    main_rep = next(row for row in rep_components if int(row["k"]) == args.main_k)
    plot_original_data(main_rep, args.graph_dir, args.dataset_dir, args.figure_dir / "e5_e7_original_data")
    if not args.skip_component_plots:
        tasks = []
        for row in components:
            if args.main_k_only and int(row["k"]) != args.main_k:
                continue
            target = (args.graph_dir / "component_plots" / row["snapshot_id"]
                      / f"k_{int(row['k']):03d}_component_{int(row['component_id']):03d}")
            tasks.append((row, str(args.graph_dir), str(target), False))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            files = []
            for number, filename in enumerate(pool.map(component_figure, tasks), 1):
                files.append(filename)
                if number % 50 == 0 or number == len(tasks):
                    print(f"component_figures: {number}/{len(tasks)}", flush=True)
        (args.graph_dir / "component_plot_index.json").write_text(
            json.dumps({"main_k": args.main_k,
                        "detail_k": sorted({int(task[0]['k']) for task in tasks}), "count": len(files),
                        "files": [str(Path(file).relative_to(args.graph_dir)) for file in files]}, indent=2) + "\n")
    print(f"Report figures saved under {args.figure_dir}", flush=True)


if __name__ == "__main__":
    main()
