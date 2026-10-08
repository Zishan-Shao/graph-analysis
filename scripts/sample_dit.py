"""Generate DiT-XL/2 images and selected token-activation observations."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "raw" / "hf-cache"))

import numpy as np
import torch
from diffusers import DiTPipeline


DEFAULT_MANIFEST = ROOT / "data" / "conditions" / "imagenet_10_classes_5_seeds.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "generated" / "dit_xl2_imagenet256"
MODEL_ID = "facebook/DiT-XL-2-256"
MODEL_REVISION = "eab87f77abd5aef071a632f08807fbaab0b704d0"
STEPS = 50
CAPTURE_STEPS = (10, 25, 40)  # 1-based denoiser call indices
BLOCKS = (4, 14, 24)  # 1-based transformer block indices
GUIDANCE_SCALE = 4.0
SAMPLER = "DDIMScheduler"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows:
        raise ValueError(f"empty condition manifest: {path}")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample_id in manifest")
    for row in rows:
        if row.get("condition_type") != "imagenet_class_id":
            raise ValueError(f"unsupported condition type for {row['sample_id']}")
        if not 0 <= row["class_id"] < 1000 or not isinstance(row["generation_seed"], int):
            raise ValueError(f"invalid class ID or seed for {row['sample_id']}")
    return rows


class TraceCollector:
    """Read block residual-stream tensors from the conditional CFG branch."""

    def __init__(self, transformer: torch.nn.Module) -> None:
        self.call_number = 0
        self.timestep = None
        self.records: dict[tuple[int, int], dict] = {}
        self.handles = [
            transformer.register_forward_pre_hook(self._on_denoiser_call, with_kwargs=True)
        ]
        blocks = transformer.transformer_blocks
        if len(blocks) < max(BLOCKS):
            raise ValueError(f"checkpoint has only {len(blocks)} transformer blocks")
        for block_number in BLOCKS:
            self.handles.append(
                blocks[block_number - 1].register_forward_hook(self._make_hook(block_number))
            )

    def _on_denoiser_call(self, _module, _args, kwargs) -> None:
        self.call_number += 1
        timestep = kwargs["timestep"]
        self.timestep = int(timestep.reshape(-1)[0].item())
        if not torch.all(timestep == timestep.reshape(-1)[0]):
            raise ValueError("mixed diffusion timesteps in a denoiser call")

    def _make_hook(self, block_number: int):
        def hook(_module, inputs, output):
            if self.call_number not in CAPTURE_STEPS:
                return
            hidden_in = inputs[0]
            hidden_out = output[0] if isinstance(output, tuple) else output
            if hidden_in.ndim != 3 or hidden_out.shape != hidden_in.shape:
                raise ValueError("expected matching [batch, spatial_tokens, channels] tensors")
            if hidden_in.shape[0] != 2:
                raise ValueError(f"expected conditional/unconditional batch of 2, got {hidden_in.shape[0]}")
            # DiTPipeline concatenates [conditional, unconditional] in that order.
            h_in = hidden_in[0].detach().float().cpu().numpy()
            residual = (
                hidden_out[0].detach().float() - hidden_in[0].detach().float()
            ).cpu().numpy()
            key = (self.call_number, block_number)
            if key in self.records:
                raise ValueError(f"duplicate trace for call {key[0]}, block {key[1]}")
            self.records[key] = {
                "h_in": h_in.astype(np.float16),
                "residual": residual.astype(np.float16),
                "timestep": self.timestep,
            }

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def save_sample(output_root: Path, row: dict, image, collector: TraceCollector,
                manifest_sha256: str, duration_seconds: float) -> None:
    sample_dir = output_root / row["sample_id"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = sample_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    expected = {(step, block) for step in CAPTURE_STEPS for block in BLOCKS}
    if set(collector.records) != expected:
        raise ValueError("incomplete activation capture")

    trace_rows = []
    for call_number, block_number in sorted(expected):
        record = collector.records[(call_number, block_number)]
        n_tokens, channels = record["h_in"].shape
        grid = math.isqrt(n_tokens)
        if grid * grid != n_tokens:
            raise ValueError(f"cannot map {n_tokens} tokens to a square grid")
        filename = f"step_{call_number:03d}_block_{block_number:02d}.npz"
        target = traces_dir / filename
        temporary = target.with_suffix(".npz.part")
        try:
            with temporary.open("wb") as output:
                np.savez_compressed(output, h_in=record["h_in"], residual=record["residual"])
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        trace_rows.append({
            "path": f"traces/{filename}",
            "denoiser_call": call_number,
            "diffusion_timestep": record["timestep"],
            "block_1based": block_number,
            "shape": [n_tokens, channels],
            "token_grid": [grid, grid],
            "dtype": "float16",
            "branch": "conditional",
        })

    image_target = sample_dir / "image.png"
    image_temporary = sample_dir / "image.png.part"
    try:
        image.save(image_temporary, format="PNG")
        image_temporary.replace(image_target)
    finally:
        image_temporary.unlink(missing_ok=True)
    metadata = {
        **row,
        "manifest_sha256": manifest_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "sampler": SAMPLER,
        "eta": 0.0,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "image_path": "image.png",
        "traces": trace_rows,
        "duration_seconds": round(duration_seconds, 3),
        "software": {
            "torch": torch.__version__,
            "diffusers": __import__("diffusers").__version__,
            "numpy": np.__version__,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    }
    metadata_target = sample_dir / "metadata.json"
    metadata_temporary = sample_dir / "metadata.json.part"
    try:
        metadata_temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        metadata_temporary.replace(metadata_target)
    finally:
        metadata_temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="Generate only the first N conditions")
    selection.add_argument("--sample-id", help="Generate one exact sample_id from the manifest")
    selection.add_argument(
        "--manifest-index", type=int,
        help="Generate one zero-based manifest row (0 through 49 in the default manifest)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require --num-shards > 0 and 0 <= --shard-index < --num-shards")
    if (args.sample_id is not None or args.manifest_index is not None) and args.num_shards != 1:
        parser.error("individual sample selection cannot be combined with sharding")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu or use a CUDA environment")

    rows = load_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[:args.limit]
    elif args.sample_id is not None:
        rows = [row for row in rows if row["sample_id"] == args.sample_id]
        if not rows:
            parser.error(f"sample ID not found in manifest: {args.sample_id}")
    elif args.manifest_index is not None:
        if not 0 <= args.manifest_index < len(rows):
            parser.error(f"--manifest-index must be between 0 and {len(rows) - 1}")
        rows = [rows[args.manifest_index]]
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    pending = [row for row in rows if args.overwrite or not (
        args.output_dir / row["sample_id"] / "metadata.json"
    ).exists()]
    if not pending:
        print("All selected samples already have metadata.json; nothing to do")
        return

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pipe = DiTPipeline.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_safetensors=False
    )
    if type(pipe.scheduler).__name__ != SAMPLER:
        raise ValueError(f"expected the pinned checkpoint's {SAMPLER} scheduler")
    if inspect.signature(pipe.scheduler.step).parameters["eta"].default != 0.0:
        raise ValueError("DDIM scheduler must default to eta=0.0")
    pipe = pipe.to(args.device, dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    manifest_sha256 = sha256_file(args.manifest)

    for index, row in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {row['sample_id']}: {row['class_name']}", flush=True)
        generator = torch.Generator(device="cpu").manual_seed(row["generation_seed"])
        collector = TraceCollector(pipe.transformer)
        started = time.perf_counter()
        try:
            result = pipe(
                class_labels=[row["class_id"]],
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                num_inference_steps=STEPS,
                output_type="pil",
            )
        finally:
            collector.close()
        if collector.call_number != STEPS:
            raise ValueError(f"expected {STEPS} denoiser calls; got {collector.call_number}")
        save_sample(
            args.output_dir, row, result.images[0], collector,
            manifest_sha256, time.perf_counter() - started,
        )
        print(f"  saved {args.output_dir / row['sample_id']}", flush=True)


if __name__ == "__main__":
    main()
