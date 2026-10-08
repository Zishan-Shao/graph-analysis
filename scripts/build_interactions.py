"""E-2/E-3: weighted pairwise interactions and union-kNN connectivity scans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from graph_core import connected_components, pairwise_affinity, union_knn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "conditions" / "imagenet_10_classes_5_seeds.jsonl"
DEFAULT_DATASET = ROOT / "data" / "generated" / "dit_xl2_imagenet256"
DEFAULT_OUTPUT = ROOT / "data" / "generated" / "graph_analysis"
EXPECTED_PAIRS = {(step, block) for step in (10, 25, 40) for block in (4, 14, 24)}
REPRESENTATIVE = {"sample_id": "imagenet-0207-r00", "denoiser_call": 25, "block_1based": 14}
CONFIGURATION = {
    "feature_attribute": "h_in",
    "feature_dimension": 1152,
    "feature_transform": "identity (all saved activation channels)",
    "arithmetic_dtype": "float64",
    "source_activation_dtype": "float16",
    "pairwise_quantity": "exp(-squared_euclidean_distance / (2 * sigma**2))",
    "bandwidth": "median positive Euclidean distance over unordered off-diagonal pairs",
    "diagonal": 0,
    "neighbor_ranking": "ascending squared Euclidean distance; ties by token index",
    "graph_symmetrization": "union: directed kNN OR its transpose",
    "edge_weight": "original Gaussian affinity on retained support",
    "k_values": list(range(2, 65)),
    "duplicate_feature_policy": "reject; datum-to-vertex equivalence mapping required",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_interactions(manifest: Path, dataset: Path, output: Path, limit: int | None = None) -> list[dict]:
    manifest_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(bool(manifest_rows), "condition manifest is empty")
    require(len({row["sample_id"] for row in manifest_rows}) == len(manifest_rows), "duplicate sample IDs")
    manifest_sha256 = sha256_file(manifest)
    if limit is not None:
        manifest_rows = manifest_rows[:limit]
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for sample_number, condition in enumerate(manifest_rows, 1):
        sample_id = condition["sample_id"]
        sample_directory = dataset / sample_id
        metadata_path = sample_directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("sample_id", "class_id", "class_name", "generation_seed", "replicate"):
            require(metadata[key] == condition[key], f"{sample_id}: mismatch in {key}")
        require(metadata["manifest_sha256"] == manifest_sha256, f"{sample_id}: wrong manifest hash")
        pairs = {(trace["denoiser_call"], trace["block_1based"]) for trace in metadata["traces"]}
        require(len(metadata["traces"]) == 9 and pairs == EXPECTED_PAIRS, f"{sample_id}: incomplete traces")
        for trace in sorted(metadata["traces"], key=lambda row: (row["denoiser_call"], row["block_1based"])):
            trace_path = sample_directory / trace["path"]
            with np.load(trace_path, allow_pickle=False) as source:
                features = source["h_in"]
            require(features.shape == (256, 1152) and features.dtype == np.float16,
                    f"{trace_path}: expected float16 [256,1152] block-input activations")
            d2, affinity, sigma, neighbor_order = pairwise_affinity(features)
            filename = f"step_{trace['denoiser_call']:03d}_block_{trace['block_1based']:02d}.npz"
            relative_target = Path("interactions") / sample_id / filename
            target = output / relative_target
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".npz.part")
            with temporary.open("wb") as destination:
                np.savez_compressed(destination, distances_squared=d2, affinity=affinity,
                                    neighbor_order=neighbor_order, sigma=np.float64(sigma))
            temporary.replace(target)
            index.append({
                "snapshot_id": f"{sample_id}/{filename[:-4]}",
                "sample_id": sample_id,
                "class_id": condition["class_id"],
                "class_name": condition["class_name"],
                "generation_seed": condition["generation_seed"],
                "replicate": condition["replicate"],
                "denoiser_call": trace["denoiser_call"],
                "block_1based": trace["block_1based"],
                "diffusion_timestep": trace["diffusion_timestep"],
                "branch": trace["branch"],
                "token_grid": trace["token_grid"],
                "n_tokens": int(features.shape[0]),
                "feature_dimension": int(features.shape[1]),
                "sigma": sigma,
                "unique_feature_count": int(features.shape[0]),
                "interaction_path": relative_target.as_posix(),
                "raw_trace_path": (Path(sample_id) / trace["path"]).as_posix(),
                "image_path": (Path(sample_id) / metadata["image_path"]).as_posix(),
                "source_trace_sha256": sha256_file(trace_path),
                "source_metadata_sha256": sha256_file(metadata_path),
                "interaction_sha256": sha256_file(target),
            })
        print(f"interactions: {sample_number}/{len(manifest_rows)} samples ({len(index)} snapshots)", flush=True)
    index_path = output / "index.jsonl"
    temporary_index = index_path.with_suffix(".jsonl.part")
    temporary_index.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in index), encoding="utf-8")
    temporary_index.replace(index_path)
    write_json(output / "metadata.json", {
        "schema_version": 1,
        "input_manifest": str(manifest.resolve()),
        "input_manifest_sha256": manifest_sha256,
        "dataset_root": str(dataset.resolve()),
        "snapshot_count": len(index),
        "sample_count": len(manifest_rows),
        "token_observation_count": sum(row["n_tokens"] for row in index),
        "configuration": CONFIGURATION,
        "representative": REPRESENTATIVE,
        "software": {"numpy": np.__version__},
        "script_sha256": sha256_file(Path(__file__)),
        "graph_core_sha256": sha256_file(Path(__file__).with_name("graph_core.py")),
        "index_sha256": sha256_file(index_path),
    })
    return index


def scan_connectivity(index: list[dict], output: Path) -> dict:
    require(bool(index), "snapshot index is empty")
    k_values = CONFIGURATION["k_values"]
    columns = ["snapshot_id", "sample_id", "class_id", "generation_seed", "denoiser_call",
               "block_1based", "diffusion_timestep", "k", "n_vertices", "n_edges",
               "n_components", "largest_component_size", "second_largest_component_size"]
    target = output / "connectivity.csv"
    temporary = target.with_suffix(".csv.part")
    snapshot_summaries = []
    groups = defaultdict(list)
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for number, row in enumerate(index, 1):
            with np.load(output / row["interaction_path"], allow_pickle=False) as source:
                affinity, order = source["affinity"], source["neighbor_order"]
            require(affinity.shape == (row["n_tokens"], row["n_tokens"]), "artifact/index shape mismatch")
            counts = []
            kc = None
            for k in k_values:
                weights = union_knn(affinity, order, k)
                labels, sizes = connected_components(weights)
                sizes_descending = np.sort(sizes)[::-1]
                n_components = len(sizes)
                if n_components == 1 and kc is None:
                    kc = k
                require(not counts or n_components <= counts[-1], "components increased for nested union-kNN graphs")
                counts.append(n_components)
                writer.writerow({
                    **{key: row[key] for key in columns[:7]},
                    "k": k,
                    "n_vertices": len(labels),
                    "n_edges": int(np.count_nonzero(weights) // 2),
                    "n_components": n_components,
                    "largest_component_size": int(sizes_descending[0]),
                    "second_largest_component_size": int(sizes_descending[1]) if len(sizes) > 1 else 0,
                })
            summary = {
                "snapshot_id": row["snapshot_id"],
                "sample_id": row["sample_id"],
                "denoiser_call": row["denoiser_call"],
                "block_1based": row["block_1based"],
                "k_c_tested": kc,
                "component_counts": counts,
            }
            snapshot_summaries.append(summary)
            groups[(row["denoiser_call"], row["block_1based"])].append(summary)
            if number % 25 == 0 or number == len(index):
                print(f"connectivity: {number}/{len(index)} snapshots", flush=True)
    temporary.replace(target)
    group_summaries = []
    for (call, block), rows in sorted(groups.items()):
        counts = np.asarray([row["component_counts"] for row in rows])
        connected_kc = [row["k_c_tested"] for row in rows if row["k_c_tested"] is not None]
        group_summaries.append({
            "denoiser_call": call, "block_1based": block, "n_snapshots": len(rows),
            "mean_component_counts": counts.mean(axis=0).tolist(),
            "min_component_counts": counts.min(axis=0).tolist(),
            "max_component_counts": counts.max(axis=0).tolist(),
            "k_c_tested_min": min(connected_kc) if connected_kc else None,
            "k_c_tested_median": float(np.median(connected_kc)) if connected_kc else None,
            "k_c_tested_max": max(connected_kc) if connected_kc else None,
            "not_connected_by_k64": len(rows) - len(connected_kc),
        })
    summary = {
        "k_values": k_values,
        "k_c_definition": "smallest tested k in [2,64] whose union-kNN graph is connected; may also be connected at k=1",
        "configuration": CONFIGURATION,
        "snapshots": snapshot_summaries,
        "by_call_block": group_summaries,
        "connectivity_csv_sha256": sha256_file(target),
    }
    write_json(output / "connectivity_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("interactions", "connectivity", "all"), default="all")
    parser.add_argument("--limit", type=int, help="use the first N manifest samples for a small verification run")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.stage == "connectivity" and args.limit is not None:
        parser.error("--limit only applies when building interactions")
    if args.stage in ("interactions", "all"):
        index = build_interactions(args.manifest, args.dataset_dir, args.output_dir, args.limit)
    else:
        index = [json.loads(line) for line in (args.output_dir / "index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.stage in ("connectivity", "all"):
        scan_connectivity(index, args.output_dir)
    print(f"finished: {len(index)} snapshots; output: {args.output_dir}")


if __name__ == "__main__":
    main()
