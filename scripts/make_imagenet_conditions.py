"""Write the fixed 10-class, five-seed condition manifest for DiT-XL/2."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/"
    "c3beaae7d32fca2a23fec30aa7938ef5c9b6e5d5/imagenet_classes.txt"
)
LABELS_SHA256 = "1f386e0d1cb6e28b9c2dac651c3dea6801e98ad1b41a14ce6bb1a093d72069f5"
DEFAULT_LABELS = ROOT / "data" / "raw" / "imagenet_classes.txt"
DEFAULT_OUTPUT = ROOT / "data" / "conditions" / "imagenet_10_classes_5_seeds.jsonl"
CLASS_IDS = (207, 281, 340, 386, 402, 404, 620, 779, 817, 963)
REPLICATES = 5
SEED_BASE = 2026


def get_labels(path: Path) -> list[str]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(LABELS_URL, headers={"User-Agent": "graph-analysis/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            path.write_bytes(response.read())
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != LABELS_SHA256:
        raise ValueError(f"ImageNet class-index SHA-256 mismatch: {digest}")
    labels = content.decode("utf-8").splitlines()
    if len(labels) != 1000:
        raise ValueError(f"expected 1000 ImageNet classes, found {len(labels)}")
    return labels


def make_rows(labels: list[str]) -> list[dict]:
    rows = []
    for class_id in CLASS_IDS:
        for replicate in range(REPLICATES):
            key = f"dit-xl2-256:{SEED_BASE}:{class_id}:{replicate}".encode("ascii")
            seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
            rows.append({
                "sample_id": f"imagenet-{class_id:04d}-r{replicate:02d}",
                "condition_type": "imagenet_class_id",
                "class_id": class_id,
                "class_name": labels[class_id],
                "replicate": replicate,
                "generation_seed": seed,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    labels = get_labels(args.labels)
    rows = make_rows(labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    metadata = {
        "condition_source": "ImageNet-1K class indices; no source images are sampled",
        "class_index_url": LABELS_URL,
        "class_index_sha256": LABELS_SHA256,
        "selected_class_ids": list(CLASS_IDS),
        "replicates_per_class": REPLICATES,
        "generation_seed_base": SEED_BASE,
        "count": len(rows),
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} conditions to {args.output}")


if __name__ == "__main__":
    main()
