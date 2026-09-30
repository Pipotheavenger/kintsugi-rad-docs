# kintsugi-rad, documented

**What it is.** Kintsugi Health R&D code that predicts depression (**PHQ-9**, 0–27) and anxiety
(**GAD-7**, 0–21) from speech. One Whisper-based model scores 30 s audio windows; a CORAL
ordinal loss, threshold tuning and Sn/Sp metrics with an indeterminate band sit on top.
`kirad/` is the library, `research/stable/*/` holds the three model versions (DAM 1, 2, 3),
`scripts/launcher.py` runs everything from one YAML. This copy is documented for retraining on
data from Congo, Colombia, Jordan and Ukraine.

**How to read this site.**

| You want | Go to |
|---|---|
| The order in which files call each other, with every function | [Code map](code-map.md) |
| What the CSV, WAVs and batches must look like | [Data contract](data-contract.md) |
| Architecture, CORAL loss, thresholds, metrics | [Model, loss, metrics](model.md) |
| DAM 1 vs 2 vs 3 and their hyperparameters | [Experiments](experiments.md) |
| Every fixed value in the code | [Hardcoded values](hardcoded.md) |
| What blocks a run today, English-only assumptions | [Limitations](limitations.md) |
| Signatures and full docstrings | API reference (left menu) |

Every box and every function name links to its line in the code. Box descriptions are the
first line of each docstring, read at build time: to change them, edit the code.

## Files, in execution order

{{ codemap:overview }}

## Quick start

```bash
pip install -e .                    # env: envs/kipy-gpu-full.yaml
# 1. scripts/launcher.py: set wandb_entity / wandb_project (L158-159), drop "kipy" (L236)
# 2. edit research/stable/ordinal_regression_mtl_mar_2024/config.yaml (DAM 1), or a copy in the
#    same folder (or pass --experiment-dir), and set, per split,
#    df_or_dataset_version: <split>.csv and audio_root: <dir>; remove merge_rows_by_subject
python scripts/launcher.py train research/stable/ordinal_regression_mtl_mar_2024/config.yaml
```

Minimal data: `train.csv`, `val.csv`, `test.csv` with `uuid, filename, phq, gad` and a duration
column in seconds (set `duration_column`; the config uses `prompt_vad_trimmed_duration` for train,
`prompt_vad_cut_duration` for val/test), plus `<audio_root>/<filename>.wav` at 16 kHz, ≥ 30 s after VAD. W&B login is
required. Details: [Data contract](data-contract.md), blockers: [Limitations](limitations.md).
