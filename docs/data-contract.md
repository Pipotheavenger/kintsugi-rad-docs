# Data contract

What the code expects on disk and what reaches the model. All defaults below are read from
`kirad/datasets_and_dataloaders.py` and `kirad/constants.py`.

## Metadata CSV columns

One CSV per split: with a CSV path, `dataset_splits`, `data_sources` and `dataset_fold` are
ignored (`load_dataset`), so set `df_or_dataset_version` in the `train:` and `val_test:` (or
`val:`/`test:`) sections of the config.

| Column | Needed for | Used by | Notes |
|---|---|---|---|
| `uuid` | always | row index, sampler, W&B score tables | unique per recording |
| `filename` | always | `load_relative_dataset` | → `<audio_root>/<filename>.wav`, `<text_root>/<filename>.json` |
| `phq` | depression task | `label_column.depression` | PHQ-9 total, 0–27 (raw score, not a class) |
| `gad` | anxiety task | `label_column.anxiety` | GAD-7 total, 0–21 |
| duration column | always | `num_windows`, min/max filter | seconds after VAD; name set by `duration_column` (configs: `prompt_vad_trimmed_duration` train, `prompt_vad_cut_duration` val/test) |
| `asr_output_id` | text runs: DAM 2 (DAM 3 only indirectly, via the cached DAM 2 embeddings; `config_llma_dam3.0.yaml` reads text directly) | `KintsugiTextDataset.__getitem__` | must exist; value is not used for the path |
| `user` (+ demographics) | `merge_rows_by_subject` only | `merge_df_rows_by_subject` | rows merge when all present Kintsugi columns match |
| `scores_<task>` | knowledge distillation only | `KintsugiMetadata.__getitem__` → `label.<task>.kd` | from `score_targets_csv` (`uuid, scores_depression, scores_anxiety`); teacher-model scores, not human ratings |

## File layout

```text
<audio_root>/<filename>.wav    16 kHz (not resampled: only a warning), channel 0 used, VAD already applied
<text_root>/<filename>.json    whisper_timestamped output; only its "text" field is read
train.csv  val.csv  test.csv   metadata above (paths relative to the config folder or cwd)
```

`audio_root` / `text_root` are `KintsugiMetadata` params (default: current directory). VAD is
not in this repo; recordings shorter than `min_duration` (30 s) are dropped.

## Windows (30 s)

| Quantity | Value | Where |
|---|---|---|
| Window length | 30 s = 480,000 samples at 16 kHz | `INFERENCE_WINDOW_SIZE`, `EXPECTED_SAMPLE_RATE` |
| Encoder positions per window | 1500 × 320 samples | `mar_2024/model.py` (`self.padding`) |
| Log-mel per window | 80 bins × 3000 frames (hop 160, n_fft 400) | `WhisperTorchFeatureExtractor` |
| Windows per recording (`all`, overlap 0) | ⌊duration / 30⌋, spread from start to end (leftover audio falls between windows) | `get_window_starts` |
| … with `pad_last_chunk_to_full: true` | ⌈duration / 30⌉ (last one zero-padded) | `get_preprocessor_with_audio_normalization` |

`window_method`: `all` (every window; val/test), `single_random` (1 random), `single_start`
(first 30 s), `random_adjacent_pair` (2 consecutive; DAM 1 train, feeds the variance loss).

## Batch

`broadcast_collate` concatenates the windows of all B recordings; `length[i]` = windows of
recording *i*, so ΣW = `length.sum()`.

| Key | Shape | Content |
|---|---|---|
| `uuid` | [B] | recording ids |
| `features.audio` | [ΣW, 480000] | raw waveform (`preprocessor: false`, DAM 1 current config) |
| `features.audio` | [ΣW, 80, 3000] | HF log-mel (`preprocessor: openai/whisper-small.en`, DAM 2/3) |
| `features.text.input_ids`, `.attention_mask` | [ΣW, L] | tokens, tiled to every window; L = 512 for BERT |
| `features.backbone_cache` | [ΣW, D] | cached embeddings, tiled like the other modalities (DAM 3 phase 1 target) |
| `label.<task>.primary` | [B] | raw PHQ / GAD score |
| `label.<task>.kd` | [B] | teacher score (only with `score_targets_csv`) |
| `length` | [B] | windows per recording |

`batch_size` in the config counts **recordings**, not windows.

## Data parameters and defaults

`KintsugiMetadata` (config key `metadata=kirad.datasets_and_dataloaders.KintsugiMetadata`):

| Param | Default | Notes |
|---|---|---|
| `df_or_dataset_version` | required | CSV path, DataFrame, `{metadata, audio_root, text_root}` or a private Kintsugi version name |
| `label_column` | `{depression: phq, anxiety: gad}` | task → CSV column |
| `duration_column` | `None` | effectively required (used for `num_windows`) |
| `min_duration` / `max_duration` | 30 / `None` s | inclusive filters |
| `window_duration` | 30 s | keep equal to the audio dataset's |
| `max_overlap_frac` | 0.0 | keep equal to the audio dataset's |
| `fraction` | 1.0 | subsample, `random_state=1` |
| `merge_rows_by_subject` | `False` | `many_to_one` / `many_to_many` concatenate a subject's audio |
| `score_targets_csv` | `None` | KD targets |
| `audio_root` / `text_root` | `None` (cwd) | |

`KintsugiAudioDataset` (`audio=...`):

| Param | Default | Current configs |
|---|---|---|
| `preprocessor` | `openai/whisper-small.en` | DAM 1: `false` (log-mel on GPU in the model) |
| `normalize_audio` | `False` | `true` (DC removal + peak scale to [-1, 1]) |
| `normalize_features` | `False` | `true` in DAM 2/3 |
| `window_method` | `all` | train `random_adjacent_pair` (DAM 1) / `single_random` (DAM 2) |
| `window_duration`, `max_overlap_frac` | 30, 0.0 | |
| `ideal_logmel_energies` | `None` | DAM 2/3: `logmel_energies.npy` (not in repo) |
| `pad_last_chunk_to_full` | `False` | `true` in DAM 3 |
| `augmentations` | `()` | `[{class_name, params, aug_prob}]`, e.g. RawBoost |

`KintsugiTextDataset` (`text=...`): `preprocessor` `google-bert/bert-base-uncased`,
`text_output_per_sample` `merge`, `prompt` `None` (LLaMA only), `max_tokens` `None`
(tokenizer max). `KintsugiCacheDataset` (`backbone_cache=...`): `backbone_cache` required,
`feat_start_idx` 0, `feat_end_idx` `None`.

Per-split merge order: `default` < `val_test` (val and test) < `train`/`val`/`test`
(`LauncherConfig.get_params_by_split`).

{{ codemap:data }}
