#!/usr/bin/env python3
"""
Measure parameter counts and (approximate) FLOPs for the Medical VLM vision+bridge stack.

- Reuses forward_medical_vlm.build_medical_vlm_from_config and inference._infer_vision_bridge_type_and_config
- Only considers the MedicalVLM module (encoder + bridge + optional MFF-lite / CMI connector)
- Does NOT include the Mamba LLM.

FLOPs:
- Tries THOP first; if unavailable or fails, tries fvcore.
- If both are unavailable, prints parameter counts only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset_pkg.medical_vlm_dataset import load_paths_config  # type: ignore
from inference import _infer_vision_bridge_type_and_config  # type: ignore
from model.forward_medical_vlm import build_medical_vlm_from_config  # type: ignore


def build_vision_bridge(checkpoint: str, paths_yaml: str | None):
    repo = Path(__file__).resolve().parent
    if paths_yaml:
        cfg = load_paths_config(paths_yaml)
    else:
        cfg = load_paths_config(repo / "config" / "paths.yaml")
    cfg.setdefault("encoder_output_stage", 4)
    cfg.setdefault("encoder_target_spatial", 28)
    cfg["bridge_d_model"] = 2560

    cfg = _infer_vision_bridge_type_and_config(checkpoint, cfg)
    model = build_medical_vlm_from_config(cfg)
    model.eval()

    spatial_dims = int(cfg.get("spatial_dims", 3))
    if spatial_dims == 3:
        dummy = torch.randn(1, 1, 64, 256, 256)
    else:
        dummy = torch.randn(1, 1, 256, 256)

    return model, dummy


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_flops_thop(model: torch.nn.Module, example: torch.Tensor):
    try:
        from thop import profile  # type: ignore
    except Exception:
        return None
    try:
        macs, _ = profile(model, inputs=(example,), verbose=False)
        return int(2 * macs)
    except Exception:
        return None


def measure_flops_fvcore(model: torch.nn.Module, example: torch.Tensor):
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
    except Exception:
        return None
    try:
        flops = FlopCountAnalysis(model, example).total()
        return int(flops)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure params and FLOPs for MedicalVLM (vision+bridge).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to vision_bridge_vlm_final.pt")
    parser.add_argument(
        "--paths_yaml",
        type=str,
        default=None,
        help="Optional override for config/paths.yaml",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model, dummy = build_vision_bridge(str(ckpt), args.paths_yaml)

    params = count_params(model)
    flops = measure_flops_thop(model, dummy)
    if flops is None:
        flops = measure_flops_fvcore(model, dummy)

    print(f"checkpoint: {ckpt}")
    print(f"spatial_dims: {model.spatial_dims}")
    print(f"params_total: {params}")
    if flops is not None:
        print(f"flops_per_forward (vision+bridge, dummy input): {flops}")
    else:
        print("flops_per_forward: <not available; thop/fvcore not installed or unsupported>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

