# Official implementation for paper reproducibility

This repository is a **clean, public-release version** of our paper codebase, prepared for **reviewing and reproduction**.
It contains training / inference / evaluation code, plus **template configs**. Private cohorts and model checkpoints are **not** included.

## Method overview (high level)

- **Vision encoder**: nnU-Net style encoder with light downsampling to keep higher spatial resolution.
- **GL-FPB**: a global-local feature pooling / bridging design to preserve both global context and local evidence.
- **Vision Mamba (Vim Bridge)**: converts 2D/3D features into sequences and applies Mamba blocks as the bridge.
- **SFC / FAB**: decoding-time or evaluation-time components used in our pipeline for structure/factual consistency (see `paper_eval/` and `paper_eval_extra/`).

For an architecture sketch, see `ARCHITECTURE.md`.

## Repository structure

```text
.
├── config/                 # config templates (edit before running)
├── dataset_pkg/            # lightweight dataset loader used by entrypoints
├── model/ bridge/ vision/  # core modules
├── llm/                    # Mamba/LLM loader and adapters
├── train.py                # Stage-1 (vision+bridge) entry
├── train_vlm.py            # Stage-2 (VLM) entry
├── inference.py            # inference entry
├── scripts/                # utilities + public dataset preparation
├── paper_eval/             # evaluation pipeline (paper-facing)
├── paper_eval_extra/       # additional analysis scripts (no private outputs)
├── tests/
└── docs/
```

## Environment (mamba-a800)

The original experiments were run in a conda env similar to:

- Python 3.10
- PyTorch (CUDA)

Install minimal dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data preparation

### Public datasets

We provide preparation scripts under `scripts/`:

- LUNA16: `scripts/prepare_luna16_public.py`
- KiTS19: `scripts/prepare_kits19_public.py`
- ReX (requires `HF_TOKEN` as an environment variable): `scripts/prepare_public_ablation_data.py`

See `docs/README_public_validation.md` for the expected output CSV locations.

### Private cohort

The private cohort used in the paper is **not released** due to privacy restrictions.
To reproduce the training pipeline, create your own CSV files following the schema described in `config/paths.yaml`.

## Configs

Edit `config/paths.yaml` (template, no private paths are included).
Key fields:

- `caption_csv_train`, `caption_csv_val`
- `mamba_hf_model` (HuggingFace model id or local path)

## Training

### Stage 1 (vision + bridge sanity check)

```bash
python train.py --epochs 3 --batch_size 2 --lr 1e-5 --output_dir outputs/stage1
```

### Stage 2 (VLM training)

```bash
python train_vlm.py --epochs 20 --batch_size 1 --lr 2e-5 --output_dir outputs/stage2
```

## Inference

```bash
python inference.py --val_sample --num_val 5 --out_dir outputs/infer_demo
```

## Evaluation

Paper-oriented evaluation entrypoints live in:

- `paper_eval/`
- `paper_eval_extra/`

Example (use your existing run directory):

```bash
python paper_eval/run_full_evaluation.py --out_root outputs/eval --split expanded_audit90
```

## Reproducibility scope

- This repo reproduces **the training/inference/evaluation pipeline** and code structure used in the paper.
- Private cohort data, raw reports, and trained checkpoints are **not included**.

## Citation

Add your BibTeX here after camera-ready:

```bibtex
@inproceedings{your_paper_2026,
  title     = {TODO},
  author    = {TODO},
  booktitle = {TMI},
  year      = {2026}
}
```

## License

See `LICENSE`.
