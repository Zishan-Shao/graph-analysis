"""Check every DiT image and activation trace against the class/seed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "conditions" / "imagenet_10_classes_5_seeds.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "generated" / "dit_xl2_imagenet256"
EXPECTED_PAIRS = {(step, block) for step in (10, 25, 40) for block in (4, 14, 24)}
EXPECTED_TIMESTEPS = {10: 800, 25: 500, 40: 200}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    require(bool(rows), "empty manifest")
    require(len({row["sample_id"] for row in rows}) == len(rows), "duplicate sample IDs")
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    missing = []
    checked = 0
    for row in rows:
        directory = args.output_dir / row["sample_id"]
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            missing.append(row["sample_id"])
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("sample_id", "class_id", "class_name", "replicate", "generation_seed"):
            require(metadata[key] == row[key], f"{row['sample_id']}: mismatched {key}")
        require(metadata["manifest_sha256"] == manifest_sha256,
                f"{row['sample_id']}: wrong manifest SHA-256")
        require(metadata["model_id"] == "facebook/DiT-XL-2-256",
                f"{row['sample_id']}: wrong model")
        require(metadata["model_revision"] == "eab87f77abd5aef071a632f08807fbaab0b704d0",
                f"{row['sample_id']}: wrong model revision")
        require(metadata["sampler"] == "DDIMScheduler" and metadata["eta"] == 0.0,
                f"{row['sample_id']}: wrong sampler")
        require(metadata["num_inference_steps"] == 50 and metadata["guidance_scale"] == 4.0,
                f"{row['sample_id']}: wrong inference settings")
        with Image.open(directory / metadata["image_path"]) as image:
            require(image.size == (256, 256), f"{row['sample_id']}: wrong image size")

        traces = metadata["traces"]
        pairs = {(item["denoiser_call"], item["block_1based"]) for item in traces}
        require(len(traces) == 9 and pairs == EXPECTED_PAIRS,
                f"{row['sample_id']}: incomplete traces")
        for item in traces:
            require(item["shape"] == [256, 1152], f"{row['sample_id']}: wrong trace shape")
            require(item["token_grid"] == [16, 16], f"{row['sample_id']}: wrong token grid")
            require(item["branch"] == "conditional", f"{row['sample_id']}: wrong CFG branch")
            require(item["diffusion_timestep"] == EXPECTED_TIMESTEPS[item["denoiser_call"]],
                    f"{row['sample_id']}: wrong diffusion timestep")
            with np.load(directory / item["path"]) as arrays:
                for name in ("h_in", "residual"):
                    values = arrays[name]
                    require(values.shape == (256, 1152) and values.dtype == np.float16,
                            f"{row['sample_id']}: wrong {name} shape or dtype")
                    require(np.isfinite(values).all(), f"{row['sample_id']}: non-finite {name}")
        checked += 1

    print(f"validated_samples: {checked}/{len(rows)}")
    print(f"missing_samples: {len(missing)}")
    if missing:
        print("missing_ids:", ", ".join(missing))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
