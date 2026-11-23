## Release checklist (public_repo)

This file documents what was kept/removed when preparing a **public GitHub release** for paper reproducibility.

### What is included (core)

- **Core model code**: `model/`, `bridge/`, `vision/`, `llm/`
- **Entrypoints**: `train.py`, `train_vlm.py`, `inference.py`
- **Evaluation code**: `paper_eval/`, `paper_eval_extra/` (scripts only; no run outputs)
- **Config templates**: `config/paths.yaml`, `config/paths_public.yaml`
- **Dataset loader (public)**: `dataset_pkg/medical_vlm_dataset.py`
- **Docs**: `docs/` (public-facing guidance)

### What was removed / not copied (not suitable for public release)

- **Private cohort data** and any derivatives:
  - CT volumes, masks, private CSVs, pathology text tables, doctor scoring sheets, etc.
- **Run outputs / intermediate artifacts**:
  - `mamba-res/`, `outputs/`, `results/`, `project_runs/`, prediction dumps, generated reports
- **Model checkpoints and large binaries**:
  - `*.pt`, `*.pth`, `*.safetensors`, `*.bin`
- **Backups / temporary files**:
  - `*.bak*`, `.ipynb_checkpoints/`
- **Sensitive or private-only utilities** (examples):
  - Scripts that build training CSVs directly from private report spreadsheets
  - Scripts that align private CT-mask pairs

### Renames / restructures

- Kept the overall layout close to the original `mamba-sci` to avoid changing algorithmic logic.
- Added a lightweight dataset module under `dataset_pkg/` to make entrypoints self-contained.

### What still needs your manual confirmation (before pushing)

1. **Licensing**:
   - Confirm MIT is acceptable for your institute/collaborators.
   - Check any third-party code usage in `external_models/` (not included here) and in scripts.
2. **Privacy**:
   - Ensure no remaining files contain private cohort identifiers or raw report text.
3. **Paths**:
   - Confirm `config/paths.yaml` contains only template paths.
4. **Reproducibility**:
   - Optionally run a tiny smoke test on a public dataset (LUNA16/KiTS19) to confirm imports work.
5. **Paper claims**:
   - This repo should not embed private results tables; keep paper results in the manuscript, not in code.

### Known limitations (public)

- Without private cohort + trained checkpoints, users can reproduce **the pipeline**, not necessarily the exact paper numbers.

