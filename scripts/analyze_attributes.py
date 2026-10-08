"""Relate activation graphs to recorded token updates and spatial attributes.

This is descriptive graph analysis, not a token-correction benchmark. Graphs
are built from block inputs only; residual updates are held-out attributes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from graph_core import union_knn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "generated" / "dit_xl2_imagenet256"
DEFAULT_GRAPHS = ROOT / "data" / "generated" / "graph_analysis"
METRIC_FIELDS = (
    "update_norm_energy_ratio", "residual_energy_ratio",
    "weighted_update_cosine", "update_cosine_advantage", "weighted_grid_distance",
    "grid_energy_ratio", "weight_cv", "local_edge_fraction",
)


def stable_seed(base: int, snapshot: dict) -> int:
    key = (f"{base}:{snapshot['sample_id']}:"
           f"{snapshot['denoiser_call']}:{snapshot['block_1based']}")
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def finite_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= np.finfo(np.float64).tiny:
        return None
    return float(numerator / denominator)


def permutation_energy_expectation(values: np.ndarray,
                                   edge_weight_sum: float) -> float:
    """Exact mean energy over uniformly permuting node values.

    Energy is sum_{i<j} w_ij ||v_i-v_j||^2. For distinct randomly
    assigned values, E||v_i-v_j||^2 = 2*sum_i||v_i-mean(v)||^2/(n-1).
    """
    centered = values - values.mean(axis=0)
    return float(2 * edge_weight_sum * np.square(centered).sum()
                 / (len(values) - 1))


def graph_attribute_metrics(weights: np.ndarray, residual: np.ndarray,
                            grid: tuple[int, int], permutations: int,
                            seed: int) -> dict:
    n = len(weights)
    upper_i, upper_j = np.nonzero(np.triu(weights > 0, 1))
    edge_weights = weights[upper_i, upper_j]
    edge_weight_sum = float(edge_weights.sum())
    if edge_weight_sum <= 0:
        raise ValueError("attribute analysis requires at least one positive edge")
    residual = np.asarray(residual, dtype=np.float64)
    update_norm = np.linalg.norm(residual, axis=1)
    norm_edge_difference = update_norm[upper_i] - update_norm[upper_j]
    norm_energy = float(edge_weights @ np.square(norm_edge_difference))
    norm_null = permutation_energy_expectation(update_norm, edge_weight_sum)

    # Gram matrix avoids materializing an [n_edges,1152] difference array.
    residual_gram = residual @ residual.T
    squared_norm = np.diag(residual_gram)
    squared_difference = np.maximum(
        squared_norm[upper_i] + squared_norm[upper_j]
        - 2 * residual_gram[upper_i, upper_j], 0.0)
    residual_energy = float(edge_weights @ squared_difference)
    residual_null = permutation_energy_expectation(residual, edge_weight_sum)
    cosine_denominator = update_norm[upper_i] * update_norm[upper_j]
    cosine_valid = cosine_denominator > 0
    cosine = np.clip(
        residual_gram[upper_i[cosine_valid], upper_j[cosine_valid]]
        / cosine_denominator[cosine_valid], -1.0, 1.0)
    valid_cosine_weights = edge_weights[cosine_valid]
    weighted_cosine = (float(valid_cosine_weights @ cosine
                             / valid_cosine_weights.sum())
                       if valid_cosine_weights.size else None)
    all_i, all_j = np.triu_indices(n, 1)
    all_denominators = update_norm[all_i] * update_norm[all_j]
    all_valid = all_denominators > 0
    all_cosines = np.clip(residual_gram[all_i[all_valid], all_j[all_valid]]
                          / all_denominators[all_valid], -1.0, 1.0)
    all_pair_cosine = float(all_cosines.mean()) if all_cosines.size else None

    rows, columns = grid
    if rows * columns != n:
        raise ValueError("token grid does not match number of graph vertices")
    coords = np.column_stack(np.unravel_index(np.arange(n), grid)).astype(float)
    grid_squared_distance = np.square(coords[upper_i] - coords[upper_j]).sum(axis=1)
    grid_energy = float(edge_weights @ grid_squared_distance)
    grid_null = permutation_energy_expectation(coords, edge_weight_sum)
    result = {
        "n_tokens": n,
        "n_edges": len(edge_weights),
        "update_norm_mean": float(update_norm.mean()),
        "update_norm_std": float(update_norm.std()),
        "update_norm_energy": norm_energy,
        "update_norm_null_mean": norm_null,
        "update_norm_energy_ratio": finite_ratio(norm_energy, norm_null),
        "residual_energy": residual_energy,
        "residual_null_mean": residual_null,
        "residual_energy_ratio": finite_ratio(residual_energy, residual_null),
        "weighted_update_cosine": weighted_cosine,
        "all_pair_update_cosine": all_pair_cosine,
        "update_cosine_advantage": (weighted_cosine - all_pair_cosine
                                     if weighted_cosine is not None
                                     and all_pair_cosine is not None else None),
        "zero_update_tokens": int(np.count_nonzero(update_norm == 0)),
        "valid_cosine_edges": int(cosine_valid.sum()),
        "weighted_grid_distance": float(edge_weights @ np.sqrt(grid_squared_distance)
                                        / edge_weight_sum),
        "grid_energy_ratio": finite_ratio(grid_energy, grid_null),
        "local_edge_fraction": float(np.mean(grid_squared_distance <= 2)),
        "weight_cv": float(edge_weights.std() / edge_weights.mean()),
        "weight_min": float(edge_weights.min()),
        "weight_max": float(edge_weights.max()),
        "permutations": permutations,
        "permutation_seed": seed if permutations else None,
        "update_norm_permutation_p_lower": None,
        "update_norm_permutation_q05": None,
        "update_norm_permutation_q95": None,
    }
    if permutations:
        rng = np.random.default_rng(seed)
        shuffled = np.stack([rng.permutation(update_norm) for _ in range(permutations)])
        differences = shuffled[:, upper_i] - shuffled[:, upper_j]
        energies = np.square(differences) @ edge_weights
        result["update_norm_permutation_q05"] = float(np.quantile(energies, 0.05))
        result["update_norm_permutation_q95"] = float(np.quantile(energies, 0.95))
        result["update_norm_permutation_p_lower"] = float(
            (1 + np.count_nonzero(energies <= norm_energy)) / (permutations + 1))
    return result


def neighbor_overlap(first: np.ndarray, second: np.ndarray, k: int) -> dict:
    n = len(first)
    selected_first = np.zeros((n, n), dtype=bool)
    selected_second = np.zeros((n, n), dtype=bool)
    node_indices = np.arange(n)[:, None]
    selected_first[node_indices, first[:, :k]] = True
    selected_second[node_indices, second[:, :k]] = True
    common = np.count_nonzero(selected_first & selected_second, axis=1)
    return {
        "mean_directed_neighbor_overlap": float(np.mean(common / k)),
        "mean_directed_neighbor_jaccard": float(np.mean(common / (2 * k - common))),
        "random_overlap_expectation": float(k / (n - 1)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    return {"n_trajectories": len(x), "mean": float(x.mean()),
            "sd_across_trajectories": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max())}


def summarize(rows: list[dict], groups: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in groups)].append(row)
    summaries = []
    for group, entries in sorted(grouped.items()):
        summary = dict(zip(groups, group))
        summary["n_snapshots"] = len(entries)
        for metric in metrics:
            by_trajectory: dict[str, list[float]] = defaultdict(list)
            for row in entries:
                if row[metric] is not None:
                    by_trajectory[row["sample_id"]].append(row[metric])
            values = [float(np.mean(observations)) for observations in by_trajectory.values()]
            summary[metric] = describe(values) if values else None
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPHS)
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--permutation-k", type=int, default=16)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if args.permutations < 0:
        parser.error("--permutations must be nonnegative")
    if args.permutation_k not in args.k_values:
        parser.error("--permutation-k must occur in --k-values")
    index_path = args.graph_dir / "index.jsonl"
    snapshots = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    if not snapshots:
        raise ValueError("empty graph index")
    if len(set((row["sample_id"], row["denoiser_call"], row["block_1based"])
               for row in snapshots)) != len(snapshots):
        raise ValueError("duplicate snapshots in graph index")
    attrs = []
    neighbors: dict[tuple[str, int, int], np.ndarray] = {}
    context: dict[tuple[str, int], dict] = {}
    for index, snapshot in enumerate(snapshots, 1):
        key = (snapshot["sample_id"], snapshot["denoiser_call"], snapshot["block_1based"])
        trace_path = args.dataset_dir / snapshot["raw_trace_path"]
        trace_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        if trace_hash != snapshot["source_trace_sha256"]:
            raise ValueError(f"source trace hash mismatch: {trace_path}")
        with np.load(trace_path, allow_pickle=False) as trace:
            residual = trace["residual"].astype(np.float64)
        interaction_path = args.graph_dir / snapshot["interaction_path"]
        interaction_hash = hashlib.sha256(interaction_path.read_bytes()).hexdigest()
        if interaction_hash != snapshot["interaction_sha256"]:
            raise ValueError(f"interaction artifact hash mismatch: {interaction_path}")
        with np.load(interaction_path, allow_pickle=False) as interaction:
            affinity = interaction["affinity"].astype(np.float64)
            neighbor_order = interaction["neighbor_order"]
        neighbors[key] = neighbor_order.copy()
        context[key[:2]] = {field: snapshot[field] for field in
                           ("sample_id", "class_id", "class_name", "generation_seed",
                            "denoiser_call", "diffusion_timestep")}
        for k in args.k_values:
            if not 0 < k < len(affinity):
                raise ValueError(f"invalid k={k} for {len(affinity)} tokens")
            weights = union_knn(affinity, neighbor_order, k)
            counts = args.permutations if k == args.permutation_k else 0
            metrics = graph_attribute_metrics(weights, residual, tuple(snapshot["token_grid"]),
                                               counts, stable_seed(args.seed, snapshot))
            attrs.append({**context[key[:2]], "block_1based": key[2], "k": k, **metrics})
        if index % 50 == 0 or index == len(snapshots):
            print(f"attribute_snapshots: {index}/{len(snapshots)}", flush=True)
    cross_layer = []
    for pair, fields in sorted(context.items()):
        for source, target in ((4, 14), (14, 24), (4, 24)):
            if (*pair, source) not in neighbors or (*pair, target) not in neighbors:
                continue
            for k in args.k_values:
                metrics = neighbor_overlap(neighbors[(*pair, source)], neighbors[(*pair, target)], k)
                cross_layer.append({**fields, "source_block": source, "target_block": target,
                                    "k": k, **metrics})
    write_csv(args.graph_dir / "attributes.csv", attrs)
    write_csv(args.graph_dir / "cross_layer_neighbors.csv", cross_layer)
    overlap_metrics = ("mean_directed_neighbor_overlap", "mean_directed_neighbor_jaccard")
    summary = {
        "provenance": {
            "graph_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy_version": np.__version__,
            "seed": args.seed, "k_values": args.k_values,
            "permutation_k": args.permutation_k, "permutations": args.permutations,
            "feature": "original block-input activation, 1152 channels; no PCA",
            "energy_null": "uniform node-value permutation, exact analytic expectation",
            "scalar_permutation_test": "descriptive lower-tail test; no multiplicity correction",
            "aggregation_unit": "trajectory; average snapshots within trajectory before pooling",
            "scope": "50 trajectories conditional on 10 fixed class IDs; no semantic token labels",
            "local_edge_definition": "grid Euclidean distance <= sqrt(2), eight-neighbor locality",
            "interpretation": "association only; no correction accuracy or inference speed measured",
        },
        "attributes_by_condition": summarize(attrs, ("k", "denoiser_call", "block_1based"),
                                              METRIC_FIELDS),
        "attributes_pooled_by_trajectory": summarize(attrs, ("k",), METRIC_FIELDS),
        "cross_layer_by_condition": summarize(cross_layer,
                                                ("k", "denoiser_call", "source_block", "target_block"),
                                                overlap_metrics),
        "cross_layer_pooled_by_trajectory": summarize(cross_layer,
                                                       ("k", "source_block", "target_block"),
                                                       overlap_metrics),
    }
    (args.graph_dir / "attribute_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"attribute_rows: {len(attrs)}; cross_layer_rows: {len(cross_layer)}")


if __name__ == "__main__":
    main()
