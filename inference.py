#!/usr/bin/env python3
"""
Minimal inference entry for 3D VLM: val_sample (batch run) and single-image report generation.
Exports: load_vision_bridge, load_image_tensor, generate_from_image, _build_template_force_words_ids,
         infer_grade_from_queries, _pool_visual_tokens.
Compatible with run_private_eval_bundle.py (subprocess --val_sample) and run_clinical_pipeline.py (imports).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dataset_pkg.medical_vlm_dataset import MedicalVLMDataset, load_paths_config

# --- Constants ---
GRADE_LABELS = ["AAH", "AIS", "MIA", "IAC"]
DEFAULT_PROMPT = "Please generate a chest CT report with 4 sections: Findings, Conclusion, Recommendation, Pathologic Tendency.\n"


def _pool_visual_tokens(vis_tokens: torch.Tensor, max_visual_tokens: int) -> torch.Tensor:
    """Truncate visual tokens to max_visual_tokens (for training/inference consistency)."""
    if vis_tokens.shape[1] <= max_visual_tokens:
        return vis_tokens
    return vis_tokens[:, :max_visual_tokens]


# StructAlign-lite: four section headers (order matches _check_sections in run_private_eval_bundle.py)
STRUCTURAL_PHRASES = ["所见：", "结论：", "建议：", "病理倾向："]


def _build_template_force_words_ids(tokenizer) -> Optional[list]:
    """
    Build token-id lists for the four section headers for StructAlign-lite constrained decode.
    Returns list of 4 lists of token ids, or None if tokenizer fails to encode any phrase.
    """
    out = []
    try:
        for phrase in STRUCTURAL_PHRASES:
            ids = tokenizer.encode(phrase, add_special_tokens=False)
            if isinstance(ids, list):
                out.append(ids)
            else:
                out.append(ids.tolist() if hasattr(ids, "tolist") else list(ids))
        if not out or all(len(p) == 0 for p in out):
            return None
        return out
    except Exception:
        return None


def _infer_spatial_dims_from_checkpoint(checkpoint_path: str) -> int:
    """Infer 2 vs 3 from first encoder conv weight in checkpoint (Conv2d -> 4D, Conv3d -> 5D)."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = state.get("model_state_dict", state)
    k = "encoder.stages.0.0.weight"
    if k in sd:
        return 3 if sd[k].ndim == 5 else 2
    return 3  # default 3 for unknown


def _infer_vision_bridge_type_and_config(checkpoint_path: str, config: dict) -> dict:
    """Infer vision_bridge_type and spatial_dims from checkpoint path and state_dict; merge with config.
    If checkpoint is a file path, load ablation_mode from same-dir stage2_config.json when present (for global_only/local_only)."""
    c = dict(config)
    path_lower = checkpoint_path.lower()
    if "transformer" in path_lower:
        c["vision_bridge_type"] = "transformer"
    else:
        c["vision_bridge_type"] = "vim"
    # Infer spatial_dims from checkpoint weights so encoder matches saved model (avoids 2D encoder + 3D data or vice versa).
    c["spatial_dims"] = _infer_spatial_dims_from_checkpoint(checkpoint_path)
    c.setdefault("ablation_mode", "full")
    # Load ablation_mode, vision_bridge_type, max_visual_tokens from checkpoint dir's stage2_config.json
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_file():
        stage2_json = ckpt_path.parent / "stage2_config.json"
        if stage2_json.exists():
            try:
                stage2 = json.loads(stage2_json.read_text(encoding="utf-8"))
                if "ablation_mode" in stage2:
                    c["ablation_mode"] = str(stage2["ablation_mode"]).strip().lower()
                if "vision_bridge_type" in stage2:
                    c["vision_bridge_type"] = str(stage2["vision_bridge_type"]).strip().lower()
                if "max_visual_tokens" in stage2:
                    c["max_visual_tokens"] = int(stage2["max_visual_tokens"])
            except Exception:
                pass
    c.setdefault("patch_size_3d", (32, 128, 128))
    if isinstance(c.get("patch_size_3d"), list):
        c["patch_size_3d"] = tuple(c["patch_size_3d"])
    return c


def load_vision_bridge(checkpoint_path: str, config: dict, device: torch.device):
    """Build MedicalVLM from config (infer bridge type from ckpt path if needed), load state_dict."""
    from model.forward_medical_vlm import build_medical_vlm_from_config

    config = _infer_vision_bridge_type_and_config(checkpoint_path, config)
    vision_bridge = build_medical_vlm_from_config(config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = state.get("model_state_dict", state)
    vision_bridge.load_state_dict(sd, strict=False)
    vision_bridge = vision_bridge.to(device)
    vision_bridge.eval()
    return vision_bridge


def load_grade_head(checkpoint_path: str, config: dict, device: torch.device):
    """Load grade head from checkpoint if present."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    grade_sd = state.get("grade_head_state_dict") or state.get("cls_head_state_dict")
    d_model = config.get("bridge_d_model", 2560)
    ordinal_bins = 3  # 4 classes -> 3 thresholds
    grade_head = torch.nn.Linear(d_model, ordinal_bins, device=device)
    if isinstance(grade_sd, dict) and grade_sd:
        cur_sd = grade_head.state_dict()
        safe_sd = {k: v for k, v in grade_sd.items() if k in cur_sd and v.shape == cur_sd[k].shape}
        if safe_sd:
            grade_head.load_state_dict(safe_sd, strict=False)
    grade_head.eval()
    return grade_head


def load_image_tensor(
    image_path: str,
    mask_path: Optional[str] = None,
    config: Optional[dict] = None,
    spatial_dims: int = 3,
    patch_size_3d: tuple = (32, 128, 128),
) -> torch.Tensor:
    """
    Load one 3D volume as tensor [1, D, H, W] using same preprocessing as MedicalVLMDataset.
    Uses a one-row temporary dataset to avoid code duplication.
    """
    import pandas as pd
    import tempfile

    config = config or load_paths_config(REPO / "config" / "paths.yaml")
    csv_val = config.get("caption_csv_val") or config.get("caption_csv_train")
    if not Path(csv_val).exists():
        raise FileNotFoundError(f"CSV not found: {csv_val}")

    df = pd.read_csv(csv_val, nrows=1)
    if "image_path" not in df.columns:
        raise ValueError("CSV must have image_path")
    # Build a minimal CSV with single row
    row = {"image_path": image_path, "question": DEFAULT_PROMPT.strip(), "answer": ""}
    if mask_path:
        row["mask_path"] = mask_path
    if "grade" in df.columns:
        row["grade"] = 0
    tmp_df = pd.DataFrame([row])
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_df.to_csv(f.name, index=False)
        try:
            ds = MedicalVLMDataset(
                f.name,
                spatial_dims=spatial_dims,
                patch_size_3d=patch_size_3d,
            )
            out = ds[0]
        finally:
            Path(f.name).unlink(missing_ok=True)
    img = out["image"]
    if img.dim() == 3:
        img = img.unsqueeze(0)
    return img.float()


def generate_from_image(
    image_tensor: torch.Tensor,
    vision_bridge: torch.nn.Module,
    llm_model,
    tokenizer,
    *,
    prompt: Optional[str] = None,
    max_new_tokens: int = 512,
    device: torch.device = None,
    max_visual_tokens: int = 164,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    length_penalty: float = 1.1,
    no_repeat_ngram_size: int = 0,
    suppress_eos_steps: int = 128,
    num_beams: int = 4,
    force_words_ids: Optional[list] = None,
    min_chars: int = 0,
    max_retries: int = 1,
    force_template: bool = False,
    raw_out: Optional[list] = None,
    debug_vision: bool = False,
    roi_center_3d: Optional[torch.Tensor] = None,
    diff_block: Optional[torch.nn.Module] = None,
    use_structural_constraint: bool = False,
    force_words_phrase_ids: Optional[list] = None,
) -> str:
    """
    Run vision_bridge on image, then autoregressive generation with LLM.
    If diff_block is not None (e1_diffmamba), apply differential correction to embeddings before each LLM step.
    If use_structural_constraint and force_words_phrase_ids (list of 4 token-id lists), force four section headers in order (StructAlign-lite).
    """
    if device is None:
        device = next(vision_bridge.parameters()).device
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    # Only add channel for 3D: [1,D,H,W] -> [1,1,D,H,W]. For 2D, [1,1,H,W] stays 4D.
    if image_tensor.dim() == 4 and getattr(vision_bridge, "spatial_dims", 2) == 3:
        image_tensor = image_tensor.unsqueeze(1)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        if roi_center_3d is not None and roi_center_3d.dim() == 1:
            roi_center_3d = roi_center_3d.unsqueeze(0)
        vis_tokens = vision_bridge(image_tensor, roi_center_3d=roi_center_3d)
    vis = _pool_visual_tokens(vis_tokens, max_visual_tokens)
    L_vis = vis.shape[1]
    d_model = vis.shape[-1]

    prompt = prompt or DEFAULT_PROMPT
    if not prompt.endswith("\n"):
        prompt = prompt + "\n"
    q_enc = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    prompt_ids = q_enc["input_ids"].to(device)
    embed_fn = llm_model.get_input_embeddings() if hasattr(llm_model, "get_input_embeddings") else getattr(llm_model, "backbone", None) and getattr(llm_model.backbone, "embeddings", None)
    if embed_fn is None:
        embed_fn = getattr(llm_model, "model", None) and getattr(llm_model.model, "embed_tokens", None)
    if embed_fn is None:
        raise RuntimeError("Cannot get LLM embedding layer")
    text_emb = embed_fn(prompt_ids)
    E = text_emb.shape[-1]
    if E != d_model:
        if E < d_model:
            pad = torch.zeros(1, text_emb.shape[1], d_model - E, device=text_emb.device, dtype=text_emb.dtype)
            text_emb = torch.cat([text_emb, pad], dim=-1)
        else:
            text_emb = text_emb[:, :, :d_model]
    vis = vis.to(text_emb.device)
    inputs_embeds = torch.cat([vis, text_emb], dim=1)
    prompt_len = text_emb.shape[1]

    # CMI if present
    cmi = getattr(vision_bridge, "cmi_connector", None)
    if cmi is not None:
        vis = cmi(vis, text_emb)
        L_vis = vis.shape[1]
        inputs_embeds = torch.cat([vis, text_emb], dim=1)

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id

    # StructAlign-lite: force four section headers in order
    MIN_TOKENS_BETWEEN_SECTIONS = 40
    use_struct = bool(use_structural_constraint and force_words_phrase_ids)
    current_section = 0
    current_phrase_token_idx = 0
    free_tokens_since_section = 0

    generated_ids = []
    cur_embeds = inputs_embeds
    for step in range(max_new_tokens):
        if step < suppress_eos_steps:
            attn = torch.ones(1, cur_embeds.shape[1], device=cur_embeds.device, dtype=torch.long)
        else:
            attn = None
        # Diff-Mamba: optional residual correction before LLM forward (model_variant=e1_diffmamba)
        embeds_for_llm = cur_embeds
        if diff_block is not None:
            embeds_for_llm = diff_block(cur_embeds)
        out = llm_model(inputs_embeds=embeds_for_llm, attention_mask=attn)
        logits = out.logits[:, -1, :]
        if repetition_penalty != 1.0 and generated_ids:
            for gid in generated_ids[-20:]:
                logits[:, gid] = logits[:, gid] / repetition_penalty
        if use_struct and current_section < len(force_words_phrase_ids):
            phrase = force_words_phrase_ids[current_section]
            if phrase and current_phrase_token_idx < len(phrase):
                forced_id = phrase[current_phrase_token_idx]
                next_id = torch.tensor([[forced_id]], device=logits.device, dtype=torch.long)
                current_phrase_token_idx += 1
                if current_phrase_token_idx >= len(phrase):
                    free_tokens_since_section = 0
            else:
                free_tokens_since_section += 1
                if free_tokens_since_section >= MIN_TOKENS_BETWEEN_SECTIONS and current_section + 1 < len(force_words_phrase_ids):
                    next_phrase = force_words_phrase_ids[current_section + 1]
                    if next_phrase:
                        current_section += 1
                        current_phrase_token_idx = 1
                        next_id = torch.tensor([[next_phrase[0]]], device=logits.device, dtype=torch.long)
                    else:
                        next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_id.item())
        if next_id.item() == eos_token_id:
            break
        next_emb = embed_fn(next_id)
        if next_emb.shape[-1] != d_model:
            if next_emb.shape[-1] < d_model:
                pad = torch.zeros(1, 1, d_model - next_emb.shape[-1], device=next_emb.device, dtype=next_emb.dtype)
                next_emb = torch.cat([next_emb, pad], dim=-1)
            else:
                next_emb = next_emb[:, :, :d_model]
        cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if min_chars and len(text.strip()) < min_chars and max_retries > 0:
        pass  # Could retry with do_sample=True; simplified here
    return text.strip()


def infer_grade_from_queries(vision_bridge: torch.nn.Module, grade_head: Optional[torch.nn.Module] = None) -> dict:
    """Get grade prediction from bridge's latest_queries_out and optional grade_head."""
    bridge = getattr(vision_bridge, "bridge", None)
    queries = getattr(bridge, "latest_queries_out", None) if bridge is not None else None
    if grade_head is None or queries is None:
        return {"label": "N/A", "index": -1, "probs": []}
    with torch.no_grad():
        q = queries.mean(dim=1)
        logits = grade_head(q)
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        pred = (torch.sigmoid(logits) > 0.5).long().sum(dim=1).item()
        pred = min(pred, 3)
    return {
        "label": GRADE_LABELS[pred],
        "index": pred,
        "probs": probs[0] if probs else [],
        "mode": "ordinal_bce",
    }


def run_val_sample(
    out_dir: Path,
    checkpoint: str,
    mamba_model: str,
    num_val: int = 23,
    csv_val: Optional[str] = None,
    max_visual_tokens: int = 164,
    max_new_tokens: int = 512,
    num_beams: int = 4,
    length_penalty: float = 1.1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    suppress_eos_steps: int = 128,
    do_sample: bool = False,
    temperature: float = 0.1,
    top_p: float = 0.95,
    draw_nodule_contour: bool = False,
    model_variant: str = "e1_base",
    diffmamba_checkpoint: Optional[str] = None,
    structural_constraint: bool = False,
) -> Path:
    """Run inference on first num_val validation samples; write run_YYYYMMDD_HHMMSS with gen/ref and meta.json."""
    from llm.mamba_loader import load_mamba_lm

    config = load_paths_config(REPO / "config" / "paths.yaml")
    config.setdefault("encoder_output_stage", 4)
    config.setdefault("encoder_target_spatial", 28)
    config["bridge_d_model"] = 2560
    config = _infer_vision_bridge_type_and_config(checkpoint, config)
    if (model_variant or "").strip().lower() == "e1_mfflite":
        config["use_mff_lite"] = True

    if not csv_val:
        csv_val = config.get("caption_csv_val") or config.get("caption_csv_train")
    csv_val = str(csv_val)
    if not Path(csv_val).exists():
        raise FileNotFoundError(f"Val CSV not found: {csv_val}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_bridge = load_vision_bridge(checkpoint, config, device)
    grade_head = load_grade_head(checkpoint, config, device)

    # Use the same spatial_dims as the loaded model so data and encoder always match (avoids 5D->conv2d).
    spatial_dims = getattr(vision_bridge, "spatial_dims", config.get("spatial_dims", 3))

    llm_model, tokenizer = load_mamba_lm(
        mamba_model,
        device_map="auto" if device.type == "cuda" else None,
    )
    llm_model.eval()
    if hasattr(llm_model, "backbone") and hasattr(llm_model.backbone, "embeddings") and hasattr(llm_model, "lm_head"):
        llm_model.lm_head.weight = llm_model.backbone.embeddings.weight

    # Diff-Mamba: optional differential block (model_variant=e1_diffmamba)
    diff_block = None
    if (model_variant or "").strip().lower() == "e1_diffmamba":
        from model.diff_mamba import build_differential_block
        d_model = config.get("bridge_d_model", 2560)
        diff_block = build_differential_block(d_model, scale=0.01, device=device)
        if diffmamba_checkpoint and Path(diffmamba_checkpoint).exists():
            state = torch.load(diffmamba_checkpoint, map_location=device, weights_only=True)
            if isinstance(state, dict) and "diff_block" in state:
                diff_block.load_state_dict(state["diff_block"], strict=False)
            elif isinstance(state, dict):
                diff_block.load_state_dict(state, strict=False)
            print(f"[Diff-Mamba] loaded weights from {diffmamba_checkpoint}", flush=True)
        else:
            print("[Diff-Mamba] using near-identity init (no checkpoint)", flush=True)

    force_words_phrase_ids = _build_template_force_words_ids(tokenizer) if structural_constraint else None
    if structural_constraint:
        print("[StructAlign-lite] four-section constraint enabled (所见/结论/建议/病理倾向)", flush=True)
        if not force_words_phrase_ids:
            print("[StructAlign-lite] warning: tokenizer failed to encode phrases; constraint disabled", flush=True)

    ds = MedicalVLMDataset(
        csv_val,
        spatial_dims=spatial_dims,
        patch_size_3d=config.get("patch_size_3d", (32, 128, 128)),
    )
    n = min(num_val, len(ds))
    run_dir = out_dir / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    contour_fn = None
    if draw_nodule_contour:
        try:
            from vision.nodule_contour import generate_nodule_contour_outputs
            contour_fn = generate_nodule_contour_outputs
        except ImportError:
            contour_fn = None

    samples_meta = []
    for i in range(n):
        idx = i + 1
        batch = ds[i]
        image = batch["image"].unsqueeze(0).to(device)
        question = batch.get("question") or DEFAULT_PROMPT.strip()
        answer = batch.get("answer") or ""
        image_path = batch.get("image_path", "")
        mask_path = batch.get("mask_path") or ""
        grade_gt = int(batch.get("grade", -1))
        if grade_gt < 0 or grade_gt > 3:
            grade_gt = -1
        roi_3d = batch.get("roi_center_3d")
        if roi_3d is not None and isinstance(roi_3d, torch.Tensor):
            roi_3d = roi_3d.unsqueeze(0).to(device)

        with torch.no_grad():
            _ = vision_bridge(image, roi_center_3d=roi_3d)
        grade_info = infer_grade_from_queries(vision_bridge, grade_head)

        text = generate_from_image(
            image.squeeze(0),
            vision_bridge,
            llm_model,
            tokenizer,
            prompt=question + "\n" if not question.endswith("\n") else question,
            max_new_tokens=max_new_tokens,
            device=device,
            max_visual_tokens=max_visual_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            suppress_eos_steps=suppress_eos_steps,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            roi_center_3d=roi_3d,
            diff_block=diff_block,
            use_structural_constraint=structural_constraint,
            force_words_phrase_ids=force_words_phrase_ids,
        )

        (run_dir / f"sample_{idx}_gen.txt").write_text(text, encoding="utf-8")
        (run_dir / f"sample_{idx}_ref.txt").write_text(answer, encoding="utf-8")

        nodule_contour = None
        if contour_fn and image_path and mask_path and Path(mask_path).exists():
            cont_dir = run_dir / f"sample_{idx}_nodule_contour"
            try:
                nodule_contour = contour_fn(image_path, mask_path, str(cont_dir))
                if nodule_contour:
                    nodule_contour["stats_csv"] = str(Path(nodule_contour.get("stats_csv", "")))
                    nodule_contour["overlay_png"] = str(Path(nodule_contour.get("overlay_png", "")))
            except Exception:
                nodule_contour = None

        samples_meta.append({
            "idx": idx,
            "image_path": image_path,
            "mask_path": mask_path,
            "prompt": question,
            "grade_gt": grade_gt,
            "grade_gt_label": GRADE_LABELS[grade_gt] if 0 <= grade_gt <= 3 else "",
            "grade": grade_info,
            "nodule_contour": nodule_contour,
        })
        print(f"  sample {idx}/{n} done", flush=True)

    meta = {
        "checkpoint": checkpoint,
        "num_val": n,
        "model_variant": model_variant or "e1_base",
        "structural_constraint": structural_constraint,
        "use_mff_lite": (model_variant or "").strip().lower() == "e1_mfflite",
        "samples": samples_meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Run dir: {run_dir}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="VLM inference: --val_sample for batch, --image for single.")
    parser.add_argument("--val_sample", action="store_true", help="Run on val set")
    parser.add_argument("--num_val", type=int, default=23)
    parser.add_argument("--out_dir", type=str, required=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mamba_model", type=str, default="state-spaces/mamba-2.8b-hf")
    parser.add_argument("--max_visual_tokens", type=int, default=164)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=1.1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--suppress_eos_steps", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--constrained_decode", action="store_true")
    parser.add_argument("--draw_nodule_contour", action="store_true")
    parser.add_argument("--model_variant", type=str, default="e1_base", choices=["e1_base", "e1_diffmamba", "e1_mfflite"], help="e1_base=original; e1_diffmamba=Diff-Mamba; e1_mfflite=MFF-lite bridge-after fusion")
    parser.add_argument("--diffmamba_checkpoint", type=str, default=None, help="Optional .pt with diff_block state_dict for e1_diffmamba")
    parser.add_argument("--structural_constraint", action="store_true", help="StructAlign-lite: force four section headers (所见/结论/建议/病理倾向) in order during decode")
    parser.add_argument("--csv", type=str, default=None, help="Override val CSV path (e.g. luna_val.csv for LUNA16)")
    parser.add_argument("--image", type=str, default=None, help="Single image path (optional)")
    args = parser.parse_args()

    if args.val_sample:
        if not args.out_dir:
            parser.error("--val_sample requires --out_dir")
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = args.checkpoint
        if not ckpt or not Path(ckpt).exists():
            parser.error("--checkpoint required and must exist for --val_sample")
        run_val_sample(
            out_dir=out_dir,
            checkpoint=ckpt,
            mamba_model=args.mamba_model,
            num_val=args.num_val,
            csv_val=args.csv,
            max_visual_tokens=args.max_visual_tokens,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            suppress_eos_steps=args.suppress_eos_steps,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            draw_nodule_contour=args.draw_nodule_contour,
            model_variant=args.model_variant,
            diffmamba_checkpoint=args.diffmamba_checkpoint,
            structural_constraint=getattr(args, "structural_constraint", False),
        )
        return 0

    if args.image:
        # Single-image mode (for run_clinical_pipeline compatibility; full impl would need mask_path arg)
        print("Single-image mode: use run_clinical_pipeline.py or pass --val_sample for batch.", file=sys.stderr)
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
