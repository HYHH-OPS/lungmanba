#!/usr/bin/env python3
"""
Lightweight efficiency probe for Medical VLM inference.

Measures:
- latency (s/sample) from wall-clock over N samples
- throughput (samples/s)
- peak GPU memory (GB) via torch.cuda.max_memory_allocated()

It reuses inference.run_val_sample so we do not touch the main inference entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from inference import run_val_sample, _infer_vision_bridge_type_and_config  # type: ignore
from dataset_pkg.medical_vlm_dataset import load_paths_config  # type: ignore


def measure_run(
    *,
    checkpoint: str,
    mamba_model: str,
    csv_val: str | None,
    out_dir: str,
    num_val: int,
    model_variant: str,
    structural_constraint: bool,
) -> dict:
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = perf_counter()
    run_dir = run_val_sample(
        out_dir=Path(out_dir).resolve(),
        checkpoint=checkpoint,
        mamba_model=mamba_model,
        num_val=num_val,
        csv_val=csv_val,
        max_visual_tokens=164,
        max_new_tokens=512,
        num_beams=4,
        length_penalty=1.1,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        suppress_eos_steps=128,
        do_sample=False,
        temperature=0.1,
        top_p=0.95,
        draw_nodule_contour=False,
        model_variant=model_variant,
        diffmamba_checkpoint=None,
        structural_constraint=structural_constraint,
    )
    if use_cuda:
        torch.cuda.synchronize(device)
    t1 = perf_counter()

    elapsed = t1 - t0
    n = max(1, num_val)
    latency = elapsed / n
    throughput = n / elapsed if elapsed > 0 else 0.0
    if use_cuda:
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gb = peak_bytes / (1024**3)
    else:
        peak_bytes = 0
        peak_gb = 0.0

    return {
        "run_dir": str(run_dir),
        "num_samples": n,
        "elapsed_seconds": elapsed,
        "latency_seconds_per_sample": latency,
        "throughput_samples_per_second": throughput,
        "peak_gpu_memory_bytes": int(peak_bytes),
        "peak_gpu_memory_gb": peak_gb,
        "device": str(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure latency/throughput/memory for a Medical VLM checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to vision_bridge_vlm_final.pt")
    parser.add_argument("--mamba_model", type=str, default="state-spaces/mamba-2.8b-hf")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Val CSV; if omitted, uses caption_csv_val from config/paths.yaml",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save run_* outputs (same as inference.py --out_dir).",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=8,
        help="Number of validation samples to process for timing.",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="e1_base",
        choices=["e1_base", "e1_diffmamba", "e1_mfflite"],
    )
    parser.add_argument(
        "--structural_constraint",
        action="store_true",
        help="Enable StructAlign-lite constraint during decode.",
    )
    parser.add_argument(
        "--paths_yaml",
        type=str,
        default=None,
        help="Optional override for config/paths.yaml (for csv default).",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if args.csv:
        csv_val = str(Path(args.csv).resolve())
    else:
        repo = Path(__file__).resolve().parent
        paths_yaml = Path(args.paths_yaml) if args.paths_yaml else (repo / "config" / "paths.yaml")
        cfg = load_paths_config(paths_yaml)
        csv_val = cfg.get("caption_csv_val") or cfg.get("caption_csv_train")
        if not csv_val:
            raise ValueError("caption_csv_val / caption_csv_train not found in paths.yaml; please pass --csv explicitly.")
        csv_val = str(Path(csv_val).resolve())

    if not Path(csv_val).exists():
        raise FileNotFoundError(f"CSV not found: {csv_val}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = measure_run(
        checkpoint=str(ckpt),
        mamba_model=args.mamba_model,
        csv_val=csv_val,
        out_dir=str(out_dir),
        num_val=args.num_val,
        model_variant=args.model_variant,
        structural_constraint=args.structural_constraint,
    )

    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

