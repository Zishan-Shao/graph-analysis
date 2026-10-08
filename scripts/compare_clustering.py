"""E-6: optional reproducible two-cluster k-means baseline on raw activations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from scipy.cluster.vq import ClusterError, kmeans2

from graph_core import connected_components, union_knn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPHS = ROOT / "data/generated/graph_analysis"
DEFAULT_DATASET = ROOT / "data/generated/dit_xl2_imagenet256"


def stable_seed(base: int, snapshot: dict, initialization: int) -> int:
    key = (f"kmeans:{base}:{snapshot['sample_id']}:{snapshot['denoiser_call']}:"
           f"{snapshot['block_1based']}:{initialization}")
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    groups = sorted(np.unique(labels), key=lambda group: int(np.flatnonzero(labels == group)[0]))
    output = np.empty_like(labels)
    for index, group in enumerate(groups):
        output[labels == group] = index
    return output


def partition_inertia(features: np.ndarray, labels: np.ndarray) -> float:
    return float(sum(np.square(features[labels == group]
                              - features[labels == group].mean(axis=0)).sum()
                     for group in np.unique(labels)))


def normalized_cut(weights: np.ndarray, labels: np.ndarray) -> float:
    degree = weights.sum(axis=1)
    total = 0.0
    for group in np.unique(labels):
        selected = labels == group
        volume = float(degree[selected].sum())
        if volume <= 0:
            raise ValueError("normalized cut requires positive cluster volume")
        total += float(weights[np.ix_(selected, ~selected)].sum()) / volume
    return total


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    """Label-permutation-invariant pairwise partition agreement, no sklearn."""
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("ARI labels must be matching one-dimensional arrays")
    n = len(first)
    if n < 2:
        return 1.0
    _, first = np.unique(first, return_inverse=True)
    _, second = np.unique(second, return_inverse=True)
    n_second = int(second.max()) + 1
    contingency = np.bincount(first * n_second + second,
                              minlength=(int(first.max()) + 1) * n_second).reshape(-1, n_second)
    choose_two = lambda counts: float(np.sum(counts * (counts - 1) / 2))
    common_pairs = choose_two(contingency)
    first_pairs = choose_two(contingency.sum(axis=1))
    second_pairs = choose_two(contingency.sum(axis=0))
    expected = first_pairs * second_pairs / (n * (n - 1) / 2)
    maximum = (first_pairs + second_pairs) / 2
    if maximum == expected:
        return 1.0
    return float((common_pairs - expected) / (maximum - expected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPHS)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--initializations", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if args.initializations < 1 or args.iterations < 1:
        parser.error("initializations and iterations must be positive")
    index_path = args.graph_dir / "index.jsonl"
    snapshots = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    if not snapshots:
        raise ValueError("empty graph index")
    summary_path = args.graph_dir / "component_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as source:
        component_rows = [row for row in csv.DictReader(source) if int(row["k"]) == args.k]
    components = {(row["sample_id"], int(row["denoiser_call"]), int(row["block_1based"])): row
                  for row in component_rows if int(row["component_size"]) == 256}
    rows = []
    for number, snapshot in enumerate(snapshots, 1):
        key = (snapshot["sample_id"], snapshot["denoiser_call"], snapshot["block_1based"])
        trace_path = args.dataset_dir / snapshot["raw_trace_path"]
        if hashlib.sha256(trace_path.read_bytes()).hexdigest() != snapshot["source_trace_sha256"]:
            raise ValueError(f"trace hash mismatch: {trace_path}")
        with np.load(trace_path, allow_pickle=False) as trace:
            features = trace["h_in"].astype(np.float64)
        # Subtracting one constant vector from every node preserves all distances,
        # memberships, and within-cluster objectives; it improves numerical stability.
        center = features.mean(axis=0)
        centered = features - center
        interaction_path = args.graph_dir / snapshot["interaction_path"]
        if hashlib.sha256(interaction_path.read_bytes()).hexdigest() != snapshot["interaction_sha256"]:
            raise ValueError(f"interaction hash mismatch: {interaction_path}")
        with np.load(interaction_path, allow_pickle=False) as interaction:
            weights = union_knn(interaction["affinity"], interaction["neighbor_order"], args.k)
        _, component_sizes = connected_components(weights)
        if len(component_sizes) != 1:
            raise ValueError("optional baseline currently requires a connected full snapshot graph")
        best = None
        attempted_seeds = []
        failures = 0
        for initialization in range(args.initializations):
            seed = stable_seed(args.seed, snapshot, initialization)
            attempted_seeds.append(seed)
            try:
                _, labels = kmeans2(centered, 2, iter=args.iterations, minit="++",
                                    missing="raise", rng=np.random.default_rng(seed))
            except ClusterError:
                failures += 1
                continue
            if len(np.unique(labels)) != 2:
                failures += 1
                continue
            labels = canonical_labels(labels)
            inertia = partition_inertia(centered, labels)
            if best is None or inertia < best[0]:
                best = (inertia, labels, seed, initialization)
        if best is None:
            raise ValueError(f"all k-means initializations failed: {key}")
        inertia, labels, chosen_seed, chosen_initialization = best
        nodal_labels = np.full(len(labels), -1, dtype=np.int64)
        component = components.get(key)
        if component is not None:
            with np.load(args.graph_dir / component["component_array_path"], allow_pickle=False) as spectral:
                nodal_labels[spectral["token_indices"]] = spectral["nodal_labels"]
        nodal_groups = int(len(np.unique(nodal_labels[nodal_labels >= 0])))
        # An absent selected eigenvector has unavailable labels, not 256
        # numerical zero entries. Leave its zero count empty in the CSV.
        nodal_zero_count = int(np.count_nonzero(nodal_labels < 0)) if nodal_groups else None
        comparable = nodal_groups == 2 and nodal_zero_count == 0
        relative_path = (Path("baseline_clusters") / snapshot["sample_id"]
                         / f"step_{snapshot['denoiser_call']:03d}_block_{snapshot['block_1based']:02d}.npz")
        output_path = args.graph_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        centroids = np.stack([features[labels == group].mean(axis=0) for group in (0, 1)])
        np.savez_compressed(output_path, token_indices=np.arange(len(labels)),
                            kmeans_labels=labels, nodal_labels=nodal_labels,
                            kmeans_centroids=centroids, initialization_seeds=np.array(attempted_seeds, dtype=np.uint64))
        rows.append({
            **{field: snapshot[field] for field in ("sample_id", "class_id", "class_name", "generation_seed",
                                                    "denoiser_call", "diffusion_timestep", "block_1based")},
            "k": args.k, "K": 2, "initializations": args.initializations,
            "iterations": args.iterations, "failed_initializations": failures,
            "chosen_initialization": chosen_initialization, "chosen_seed": chosen_seed,
            "kmeans_cluster_0_size": int(np.count_nonzero(labels == 0)),
            "kmeans_cluster_1_size": int(np.count_nonzero(labels == 1)),
            "kmeans_inertia": inertia, "kmeans_normalized_cut": normalized_cut(weights, labels),
            "nodal_available": nodal_groups > 0,
            "nodal_domain_count": nodal_groups, "nodal_zero_count": nodal_zero_count,
            "comparable_two_group_partition": comparable,
            "nodal_inertia": partition_inertia(centered, nodal_labels) if comparable else None,
            "nodal_normalized_cut": normalized_cut(weights, nodal_labels) if comparable else None,
            "partition_ARI": adjusted_rand_index(labels, nodal_labels) if comparable else None,
            "baseline_array_path": relative_path.as_posix(),
        })
        if number % 50 == 0 or number == len(snapshots):
            print(f"optional_clustering: {number}/{len(snapshots)}", flush=True)
    csv_path = args.graph_dir / "optional_clustering.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "n_snapshots": len(rows), "K": 2, "k": args.k,
        "feature_dimension": 1152,
        "feature_transform": "constant per-channel mean subtraction; preserves all distances",
        "initialization": "k-means++, five stable per-snapshot seeds by default; choose lowest inertia",
        "initializations": args.initializations, "iterations": args.iterations,
        "seed": args.seed,
        "failed_initializations": sum(row["failed_initializations"] for row in rows),
        "nodal_available_snapshots": sum(row["nodal_available"] for row in rows),
        "comparable_two_group_snapshots": sum(row["comparable_two_group_partition"] for row in rows),
        "objective_comparison": "only compare inertia, normalized cut, and ARI when nodal partition has two groups and no zero nodes",
        "interpretation": "ARI compares partitions; no semantic ground-truth accuracy is measured",
        "numpy_version": np.__version__, "scipy_version": __import__("scipy").__version__,
        "graph_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "component_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.graph_dir / "optional_clustering_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in
                      ("n_snapshots", "failed_initializations", "nodal_available_snapshots",
                       "comparable_two_group_snapshots")}))


if __name__ == "__main__":
    main()
