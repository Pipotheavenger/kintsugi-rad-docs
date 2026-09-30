# Limitations

Found by reading the code; `file:line` links open the exact line.

## US-English assumptions

| Assumption | Where |
|---|---|
| ASR: Whisper `large-v3` with `language="en"` | [`scripts/run_whisper_asr.py:35`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/scripts/run_whisper_asr.py#L35), [`:61`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/scripts/run_whisper_asr.py#L61) |
| Audio encoder `openai/whisper-small.en` (English-only) | [`kirad/base_models.py:810`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/base_models.py#L810), [`kirad/datasets_and_dataloaders.py:373`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/datasets_and_dataloaders.py#L373), all YAMLs |
| Text model `google-bert/bert-base-uncased` | [`kirad/base_models.py:881`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/base_models.py#L881), [`kirad/datasets_and_dataloaders.py:533`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/datasets_and_dataloaders.py#L533) |
| LLaMA prompt in English (PHQ-9 only) | `research/stable/ordinal_regression_mtl_jun_2024/config_llama.yaml` |
| Cutoffs PHQ-9 (9, 14), GAD-7 (4, 9, 14); maxima 27 / 21 | [`kirad/constants.py:35`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/constants.py#L35), [`:39`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/constants.py#L39) |
| Fixed 80-bin log-mel profile `IDEAL_LOGMEL_ENERGIES` | [`kirad/constants.py:168`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/constants.py#L168), used at [`kirad/feature_extraction_whisper.py:92`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/feature_extraction_whisper.py#L92) |
| US survey schema for subject merge (`zip_code`, `state`, `english_preferred`, …) | [`kirad/dataset_utils.py:637`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/dataset_utils.py#L637) |
| 16 kHz assumed, never resampled (only a warning) | [`kirad/constants.py:14`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/constants.py#L14), [`kirad/datasets_and_dataloaders.py:467`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/datasets_and_dataloaders.py#L467) |

## What blocks a run today

| # | Blocker | Where | Fix |
|---|---|---|---|
| 1 | `wandb_entity` / `wandb_project` are `None` and raise | [`scripts/launcher.py:158`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/scripts/launcher.py#L158) | set them; W&B is also required by the score tables ([`kirad/base_models.py:430`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/base_models.py#L430), `wandb.run.id`) |
| 2 | `get_package_info("kipy")`: private package, `find_spec` returns `None` | [`scripts/launcher.py:236`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/scripts/launcher.py#L236) → [`kirad/utils.py:689`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/utils.py#L689) | drop `"kipy"` from the list |
| 3 | `df_or_dataset_version: "v2.x"` downloads private splits with `gsutil`; URIs are `None` | [`kirad/dataset_utils.py:170`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/dataset_utils.py#L170), [`kirad/constants.py:113`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/constants.py#L113) | point it to a CSV per split |
| 4 | DAM 1 `config.yaml` train uses `merge_rows_by_subject: many_to_one`; the merge drops `audio_filename`, which the audio dataset then reads | merge: [`kirad/dataset_utils.py:665`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/dataset_utils.py#L665); read: [`kirad/datasets_and_dataloaders.py:455`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/datasets_and_dataloaders.py#L455) | remove the option (or recompute the path list after merging) |
| 5 | `audio_dir_or_feature_store_bucket` is not a field of `KintsugiAudioDataset` (`extra="forbid"`) → validation error | `config_dam1.0.yaml`, `config_dam2.0.yaml`, `config_llama.yaml`, `config_dam3.0.yaml`, `config_llma.yaml`, `config_llma_dam3.0.yaml` | use `audio_root` under `metadata=` |
| 6 | `text_dir` is not a field of `KintsugiTextDataset` (`extra="forbid"`) | every config with `text=` (DAM 2 `config.yaml`, `config_dam2.0.yaml`, `config_llama.yaml`, `config_llma_dam3.0.yaml`) | use `text_root` under `metadata=` |
| 7 | `ideal_logmel_energies: "logmel_energies.npy"` is not in the repo → `np.load` fails | [`kirad/datasets_and_dataloaders.py:395`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/datasets_and_dataloaders.py#L395), also [`scripts/launcher.py:254`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/scripts/launcher.py#L254); all DAM 2/3 configs and `config_dam1.0.yaml` | provide the file or remove the key |
| 8 | Placeholders `<wandb_url_to_…>` (LLMA, LLaMA, backbone cache, DAM 2 checkpoint) | DAM 3 configs, `config_llama.yaml` | train the upstream stage first and paste its W&B URL |
| 9 | Text runs read `row.asr_output_id` | [`kirad/datasets_and_dataloaders.py:586`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/datasets_and_dataloaders.py#L586) | add the column (any value) |

Only DAM 1 `config.yaml` avoids 5–8; with 1–4 fixed it is the shortest path to a first run.

## Missing data and metrics

- **Data**: metadata CSVs, audio buckets and `logmel_energies.npy` are private (GCS/W&B); no
  checkpoints or teacher scores (`score_targets_csv`) are included.
- **Preprocessing**: VAD and resampling are upstream and not in the repo; only the VAD
  parameters remain ([`kirad/constants.py:21`](https://github.com/Pipotheavenger/kintsugi-rad-docs/blob/main/kirad/constants.py#L21)).
- **Outcomes**: only PHQ-9 and GAD-7 are wired in configs; a binary `real` task (`exact2`,
  cutoff 0) exists in `kirad/constants.py` but no config uses it (candidate for SCID).
- **Metrics**: no calibration, no per-country / per-language breakdown by default
  (`analyze_dataframe_with_scores` groups by any column, but is not called in training); no
  confidence intervals (`metrics.bootstrap_confusion_matrix` exists but is never called).
- **Tests**: none in this copy.
