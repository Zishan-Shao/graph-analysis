"""Compute E-4/E-5 connectivity distributions, spectra, and nodal domains.

Consumes the original-activation interaction matrices written by
build_interactions.py. No block update is used to construct an edge or select an
eigenvector. Updates are read only after graph analysis for interpretation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Avoid many BLAS threads in each independent snapshot worker.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from scipy.linalg import eigh

from graph_core import connected_components, union_knn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_DIR = ROOT / "data/generated/graph_analysis"
DEFAULT_DATA_DIR = ROOT / "data/generated/dit_xl2_imagenet256"
DETAIL_K = (2, 4, 8, 16, 32, 64)
MIN_LARGE_SIZE = 32
N_EIGENVALUES = 32
EIGENVALUE_ZERO_TOL = 1e-8
SIMPLE_ABSOLUTE_GAP = 1e-6
WELL_SEPARATED_RELATIVE_GAP = 0.05
SIGN_RELATIVE_TOL = 1e-10


def normalized_laplacian(weights: np.ndarray) -> np.ndarray:
    """Normalized Laplacian for an undirected component with no isolated node."""
    degree = weights.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("component normalized Laplacian requires positive weighted degrees")
    inverse_root = 1.0 / np.sqrt(degree)
    laplacian = -(inverse_root[:, None] * weights) * inverse_root[None, :]
    np.fill_diagonal(laplacian, 1.0)
    return laplacian


def local_clustering_coefficients(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unweighted degrees and triangles / possible neighbor pairs."""
    adjacency = (weights > 0).astype(np.float64)
    degree = adjacency.sum(axis=1).astype(np.int64)
    # diag(A^3) counts each incident triangle twice.
    twice_triangles = np.einsum("ij,ij->i", adjacency @ adjacency, adjacency)
    coefficient = np.zeros(len(degree), dtype=np.float64)
    mask = degree >= 2
    coefficient[mask] = twice_triangles[mask] / (degree[mask] * (degree[mask] - 1))
    if np.any(coefficient < -1e-12) or np.any(coefficient > 1.0 + 1e-12):
        raise ValueError("local clustering coefficient outside [0, 1]")
    return degree, coefficient


def describe_distribution(values: np.ndarray, prefix: str) -> dict:
    mean = float(np.mean(values))
    std = float(np.std(values))
    skewness = float(np.mean(((values - mean) / std) ** 3)) if std > 0 else 0.0
    if std == 0:
        shape = "constant"
    elif skewness > 0.5:
        shape = "right-skewed"
    elif skewness < -0.5:
        shape = "left-skewed"
    else:
        shape = "approximately symmetric by sample skewness"
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_mean": mean,
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": std,
        f"{prefix}_skewness": skewness,
        f"{prefix}_shape_description": shape,
    }


def orient_eigenvectors(vectors: np.ndarray) -> np.ndarray:
    """Fix arbitrary signs for stable plots; eigenspace degeneracy remains."""
    oriented = vectors.copy()
    for column in range(oriented.shape[1]):
        pivot = int(np.argmax(np.abs(oriented[:, column])))
        if oriented[pivot, column] < 0:
            oriented[:, column] *= -1
    return oriented


def select_nodal_eigenvalue(values: np.ndarray) -> dict:
    """Select a nonzero simple eigenvalue separated on BOTH sides.

    Candidate j is among the first 32 eigenvalues, and the full spectrum is
    retained here so candidate 32 is compared with eigenvalue 33. Numerically
    simple means both adjacent gaps exceed 1e-6. Well separated additionally
    requires min(left,right)/abs(lambda_j) >= 0.05. The candidate with largest
    minimum absolute gap is selected, with the smaller index breaking ties.
    """
    best = None
    for index in range(1, min(N_EIGENVALUES, len(values) - 1)):
        value = float(values[index])
        left_gap = float(value - values[index - 1])
        right_gap = float(values[index + 1] - value)
        min_gap = min(left_gap, right_gap)
        relative_gap = min_gap / max(abs(value), EIGENVALUE_ZERO_TOL)
        if value <= EIGENVALUE_ZERO_TOL or min_gap < SIMPLE_ABSOLUTE_GAP:
            continue
        candidate = {
            "selected_eigenvalue_index_1based": index + 1,
            "selected_eigenvalue": value,
            "left_eigenvalue_gap": left_gap,
            "right_eigenvalue_gap": right_gap,
            "min_eigenvalue_gap": min_gap,
            "relative_eigenvalue_gap": relative_gap,
            "numerically_simple": True,
            "well_separated": relative_gap >= WELL_SEPARATED_RELATIVE_GAP,
        }
        if candidate["well_separated"] and (best is None or min_gap > best["min_eigenvalue_gap"]):
            best = candidate
    if best is not None:
        return best
    return {
        "selected_eigenvalue_index_1based": None,
        "selected_eigenvalue": None,
        "left_eigenvalue_gap": None,
        "right_eigenvalue_gap": None,
        "min_eigenvalue_gap": None,
        "relative_eigenvalue_gap": None,
        "numerically_simple": False,
        "well_separated": False,
    }


def nodal_domains(weights: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Strict nodal domains = connected components of each sign-induced graph.

    Numerically zero entries receive label -1 and are never joined to a sign
    domain. Domain labels are ordered by their smallest original local index.
    """
    tolerance = SIGN_RELATIVE_TOL * float(np.max(np.abs(vector)))
    signs = np.zeros(len(vector), dtype=np.int8)
    signs[vector > tolerance] = 1
    signs[vector < -tolerance] = -1
    domains = []
    for sign in (1, -1):
        selected = np.flatnonzero(signs == sign)
        if not len(selected):
            continue
        local_labels, local_sizes = connected_components(weights[np.ix_(selected, selected)])
        for label in range(len(local_sizes)):
            domains.append(selected[local_labels == label])
    domains.sort(key=lambda indices: int(indices.min()))
    labels = np.full(len(vector), -1, dtype=np.int64)
    for label, indices in enumerate(domains):
        labels[indices] = label
    return labels, signs, tolerance


def node_group_statistics(labels: np.ndarray, update_norm: np.ndarray) -> tuple[list[dict], float]:
    groups = []
    valid = labels >= 0
    if not valid.any():
        return groups, float("nan")
    total_mean = float(np.mean(update_norm[valid]))
    total_squares = float(np.sum((update_norm[valid] - total_mean) ** 2))
    between_squares = 0.0
    for label in sorted(set(labels[valid].tolist())):
        values = update_norm[labels == label]
        mean = float(np.mean(values))
        between_squares += len(values) * (mean - total_mean) ** 2
        groups.append({
            "nodal_domain": label,
            "n_tokens": len(values),
            "update_norm_mean": mean,
            "update_norm_std": float(np.std(values)),
            "update_norm_min": float(np.min(values)),
            "update_norm_max": float(np.max(values)),
        })
    fraction = between_squares / total_squares if total_squares > 0 else 0.0
    return groups, float(fraction)


def validate_eigendecomposition(laplacian: np.ndarray, values: np.ndarray,
                              vectors: np.ndarray) -> dict:
    residual = laplacian @ vectors - vectors * values[None, :]
    residual_max = float(np.max(np.linalg.norm(residual, axis=0)))
    orthogonality = float(np.max(np.abs(vectors.T @ vectors - np.eye(len(values)))))
    zero_count = int(np.count_nonzero(np.abs(values) <= EIGENVALUE_ZERO_TOL))
    if residual_max > 1e-8 or orthogonality > 1e-8 or zero_count != 1:
        raise ValueError(f"invalid component eigendecomposition: {residual_max=}, {orthogonality=}, {zero_count=}")
    return {
        "max_eigenpair_residual": residual_max,
        "max_orthogonality_error": orthogonality,
        "zero_eigenvalue_count": zero_count,
    }


def analyze_snapshot(task: tuple[dict, str, str, bool]) -> dict:
    row, graph_dir_string, data_dir_string, validate = task
    graph_dir, data_dir = Path(graph_dir_string), Path(data_dir_string)
    with np.load(graph_dir / row["interaction_path"]) as matrix:
        affinity = matrix["affinity"].astype(np.float64)
        order = matrix["neighbor_order"].astype(np.int64)
    with np.load(data_dir / row["raw_trace_path"]) as trace:
        residual = trace["residual"].astype(np.float64)
    update_norm_all = np.linalg.norm(residual, axis=1)
    n_tokens = len(affinity)
    base = {key: row[key] for key in ("sample_id", "class_id", "class_name", "generation_seed",
                                     "denoiser_call", "block_1based", "diffusion_timestep")}
    base["snapshot_id"] = f"{row['sample_id']}/step_{row['denoiser_call']:03d}_block_{row['block_1based']:02d}"
    base["graph_id"] = base["snapshot_id"]
    output_dir = graph_dir / "components" / base["snapshot_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    curves, summaries, groups, checks = [], [], [], []
    connectivity_threshold = None
    previous_count = n_tokens
    with np.errstate(invalid="raise", over="raise"):
        for k in range(2, 65):
            weights = union_knn(affinity, order, k)
            component_labels, component_sizes = connected_components(weights)
            count = len(component_sizes)
            if count > previous_count:
                raise ValueError("nested union-kNN graphs increased component count")
            previous_count = count
            if count == 1 and connectivity_threshold is None:
                connectivity_threshold = k
            fiedler = None
            if count == 1:
                whole_laplacian = normalized_laplacian(weights)
                bottom = eigh(whole_laplacian, subset_by_index=(0, 1), eigvals_only=True,
                              check_finite=False, driver="evr")
                if abs(float(bottom[0])) > EIGENVALUE_ZERO_TOL or bottom[1] <= 0:
                    raise ValueError("connected graph does not have one zero and positive Fiedler value")
                fiedler = float(bottom[1])
            curves.append({**base, "k": k, "n_components": count,
                           "largest_component_size": int(component_sizes.max()),
                           "connected": count == 1, "fiedler_value": fiedler,
                           "connectivity_threshold_kc": connectivity_threshold})
            if k not in DETAIL_K:
                continue
            for component_id, component_size in enumerate(component_sizes):
                if component_size < MIN_LARGE_SIZE:
                    continue
                indices = np.flatnonzero(component_labels == component_id)
                component_weights = weights[np.ix_(indices, indices)]
                degree, clustering = local_clustering_coefficients(component_weights)
                laplacian = normalized_laplacian(component_weights)
                eigenvalues, eigenvectors = eigh(laplacian, check_finite=False, driver="evr")
                eigenvectors = orient_eigenvectors(eigenvectors)
                if int(np.count_nonzero(np.abs(eigenvalues) <= EIGENVALUE_ZERO_TOL)) != 1:
                    raise ValueError("component count and zero eigenvalue count disagree")
                selection = select_nodal_eigenvalue(eigenvalues)
                selected_index = selection["selected_eigenvalue_index_1based"]
                labels = np.full(len(indices), -1, dtype=np.int64)
                signs = np.zeros(len(indices), dtype=np.int8)
                selected_vector = np.full(len(indices), np.nan)
                nodal_embedding = np.full((len(indices), 3), np.nan)
                nodal_axes = np.full(3, -1, dtype=np.int64)
                sign_tolerance = None
                if selected_index is not None:
                    zero_index = selected_index - 1
                    selected_vector = eigenvectors[:, zero_index]
                    labels, signs, sign_tolerance = nodal_domains(component_weights, selected_vector)
                    other_axes = [column for column in range(1, len(indices)) if column != zero_index][:2]
                    nodal_axes = np.array([zero_index, *other_axes], dtype=np.int64) + 1
                    nodal_embedding = eigenvectors[:, nodal_axes - 1]
                update_norm = update_norm_all[indices]
                group_rows, explained_variance = node_group_statistics(labels, update_norm)
                component_base = {**base, "k": k, "component_id": component_id,
                                  "component_size": int(component_size)}
                array_path = output_dir / f"k_{k:03d}_component_{component_id:03d}.npz"
                np.savez_compressed(array_path, token_indices=indices, degrees=degree,
                                    clustering=clustering,
                                    eigenvalues=eigenvalues[:min(N_EIGENVALUES, len(eigenvalues))],
                                    embedding=eigenvectors[:, 1:4],
                                    embedding_eigenvector_indices_1based=np.array([2, 3, 4]),
                                    nodal_embedding=nodal_embedding,
                                    nodal_embedding_eigenvector_indices_1based=nodal_axes,
                                    nodal_labels=labels, nodal_signs=signs,
                                    selected_eigenvector=selected_vector,
                                    selected_eigenvalue_index_1based=np.array(selected_index or -1),
                                    nodal_zero_tolerance=np.array(sign_tolerance or 0.0),
                                    update_norm=update_norm)
                summary = {
                    **component_base,
                    "component_array_path": array_path.relative_to(graph_dir).as_posix(),
                    "arrays_file": array_path.relative_to(graph_dir).as_posix(),
                    "n_edges": int(degree.sum() // 2),
                    **describe_distribution(degree, "degree"),
                    **describe_distribution(clustering, "clustering"),
                    "fiedler_value": float(eigenvalues[1]),
                    "n_eigenvalues_plotted": min(N_EIGENVALUES, len(eigenvalues)),
                    **selection,
                    "selected_eigenvalue_index": selected_index,
                    "embedding_eigenvector_indices": "2;3;4",
                    "nodal_eigenvector_indices": ";".join(map(str, nodal_axes)) if selected_index else None,
                    "n_nodal_domains": len(group_rows),
                    "n_numerical_zero_nodes": int(np.count_nonzero(signs == 0)) if selected_index else None,
                    "nodal_zero_tolerance": sign_tolerance,
                    "update_norm_mean": float(update_norm.mean()),
                    "update_norm_std": float(update_norm.std()),
                    "nodal_update_norm_explained_variance": explained_variance if selected_index else None,
                }
                summaries.append(summary)
                groups.extend({**component_base, **group} for group in group_rows)
                if validate:
                    checks.append({**component_base,
                                   **validate_eigendecomposition(laplacian, eigenvalues, eigenvectors)})
    # kc belongs to the entire curve, including values before the threshold.
    for curve in curves:
        curve["connectivity_threshold_kc"] = connectivity_threshold
    return {"fiedler": curves, "components": summaries, "groups": groups, "checks": checks}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Analyze first N snapshots for a smoke run")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    rows = [json.loads(line) for line in (args.graph_dir / "index.jsonl").read_text().splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("empty graph snapshot index")
    tasks = [(row, str(args.graph_dir), str(args.data_dir), index == 0) for index, row in enumerate(rows)]
    all_results = {"fiedler": [], "components": [], "groups": [], "checks": []}
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        result_iterator = executor.map(analyze_snapshot, tasks, chunksize=1) if executor else map(analyze_snapshot, tasks)
        for index, result in enumerate(result_iterator, start=1):
            for key in all_results:
                all_results[key].extend(result[key])
            if index % 25 == 0 or index == len(tasks):
                print(f"Analyzed {index}/{len(tasks)} snapshots", flush=True)
    finally:
        if executor:
            executor.shutdown()
    write_csv(args.graph_dir / "fiedler_curves.csv", all_results["fiedler"])
    write_csv(args.graph_dir / "component_summary.csv", all_results["components"])
    write_csv(args.graph_dir / "nodal_groups.csv", all_results["groups"])
    metadata = {
        "n_snapshots": len(rows),
        "n_large_components": len(all_results["components"]),
        "k_scan": list(range(2, 65)),
        "detail_k": list(DETAIL_K),
        "large_component_min_size": MIN_LARGE_SIZE,
        "degree_and_clustering": "unweighted, computed inside each connected component",
        "laplacian": "I-D^(-1/2) W D^(-1/2), Gaussian union-kNN edge weights",
        "eigenvalues_plotted": N_EIGENVALUES,
        "baseline_embedding_eigenvectors_1based": [2, 3, 4],
        "eigenvalue_selection": {
            "candidate_indices_1based": [2, N_EIGENVALUES],
            "absolute_gap_min": SIMPLE_ABSOLUTE_GAP,
            "relative_gap_min": WELL_SEPARATED_RELATIVE_GAP,
            "relative_gap_definition": "min(left_gap,right_gap)/abs(eigenvalue)",
            "selection": "maximize minimum adjacent absolute gap among qualifying candidates",
            "both_neighbors_checked": True,
            "no_candidate_behavior": "report absent; no nodal partition fabricated",
        },
        "nodal_domains": "connected components of positive/negative induced subgraphs; numerical zeros label -1",
        "nodal_sign_relative_tolerance": SIGN_RELATIVE_TOL,
        "eigenvalue_zero_tolerance": EIGENVALUE_ZERO_TOL,
        "distribution_description": "empirical skewness only, no parametric family fit claimed",
        "validation": all_results["checks"],
        "software": {"numpy": np.__version__, "scipy": __import__("scipy").__version__},
    }
    (args.graph_dir / "spectral_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"snapshots": len(rows), "large_components": len(all_results["components"]),
                      "well_separated_components": sum(row["well_separated"] for row in all_results["components"]),
                      "validation_checks": len(all_results["checks"])}), flush=True)


if __name__ == "__main__":
    main()
