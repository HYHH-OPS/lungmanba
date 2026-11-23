# LungMamba-EAR

**Evidence-Aligned Global–Local Modeling for Structured Pulmonary Nodule Reporting from 3D CT**

This independent reference implementation generates a target-ROI-conditioned report draft. It implements the architecture specified for the JBHI revision; it is **not verified as the source used to produce the manuscript results**. Earlier code is preserved under `legacy/` and in Git history.

## Architecture and features

A randomly initialized five-stage 3D CNN supplies Stage-5 global context (64 tokens) and annotated-ROI-centered Stage-3 local evidence (100 tokens). Independent projections map both streams to 2560 dimensions. Independent forward/backward Mamba selective scans process these tokens plus 32 learned queries. Only the 164 visual outputs condition the frozen `state-spaces/mamba-2.8b-hf` decoder. Token-FAB supervises the visual outputs; Query-FAB changes only the supervision location. Five masked binary field losses have coefficient 0.02. No LoRA, grade head, or pretrained nnU-Net weights are used. An optional parameter-free SFC controls four report sections.

## Installation and base model

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

Training requires a BF16-capable CUDA device with sufficient memory for the frozen 2.8B decoder and differentiable visual prefix. CPU tests use a small randomly initialized real Mamba decoder and never download the backbone. Transformers downloads `state-spaces/mamba-2.8b-hf` on first use; supply `--base-model /path/to/local/model` for an independently downloaded copy. Users must comply with the backbone license and Hugging Face access terms. Optional CUDA acceleration is not required by the reference implementation.

## Data preparation

Provide private JSONL manifests with `volume` (relative path to a preprocessed float32 NumPy array of shape 32×128×128), `roi_center` (annotated z,y,x voxel coordinates in that array), `report`, and optional `fields`. Fields are `location`, `count`, `size`, `density`, and `recommendation`, each a binary vector in a user-defined vocabulary. Omit a field or use null when unavailable; an explicitly supplied all-zero vector is a valid negative target. A YAML field-schema maps each field name to its vector length. Train and validation sets must be separated at patient level.

Users must preprocess their appropriately licensed CT and transform ROI coordinates together. HU windows, resampling spacing, orientation, and physical crop size from the final experiments could not be verified; this repository deliberately does not invent these settings. Record your preprocessing and vocabulary alongside your local experiment. Private manifests and arrays must remain outside Git.

## Training

```bash
python -m training.train --config configs/lungmamba_ear.yaml --train /private/train.jsonl --validation /private/val.jsonl --field-schema /private/fields.yaml --output outputs/run --seed 20261
```

AdamW uses LR 1e-5, weight decay 0.01, batch size 3, BF16, clipping 1.0, and 20 epochs. Repeat for seeds 20261–20265. Best validation caption NLL selects the checkpoint. Checkpoints contain newly trained visual modules and heads; the backbone is excluded. Use `configs/query_fab.yaml` for Query-FAB or set `fab: none` for no auxiliary loss.

## Inference

```bash
python -m inference.generate --checkpoint outputs/run/best.pt --volume /private/ct.npy --roi-center 16 64 64
# Add --no-sfc to disable structured control.
```

Replace the example center with the actual annotation. SFC uses Imaging Findings, Conclusion, Recommendation, and Pathological Tendency, a repetition window of 16, and at most 1024 generated tokens. This reference controller caps each body at 200 tokens, transitions on EOS or the cap, and inserts headings. These controller details are reference choices, not verified original experiment settings. Generated drafts require clinical review.

## Evaluation

`evaluation.field_metrics.score` computes availability-masked binary TP/FP/FN and micro F1 for each of the five fields. Unavailable entries are excluded; true negatives remain available but do not enter the F1 numerator. Undefined F1 is returned as null. Supply independently annotated or extracted binary predictions using the same vocabulary; this interface is not the original paper's report parser or adjudication protocol. The code cannot independently reproduce all paper numbers without original inputs, outputs, and evaluation records.

## Pretrained weights

The trained LungMamba-EAR checkpoints are not publicly distributed. The source code is provided to support independent training and methodological inspection. Users should obtain the publicly available Mamba-2.8B backbone separately and train the visual encoder and evidence bridge on appropriately licensed datasets.

## Data availability

Private clinical CT and reports are not distributed because of privacy and institutional restrictions. No author-trained weights or patient records are included.

## Citation

LungMamba-EAR: Evidence-Aligned Global–Local Modeling for Structured Pulmonary Nodule Reporting from 3D CT. JBHI revision manuscript. Publication metadata will be added when verified; no DOI or publication details are asserted here.
