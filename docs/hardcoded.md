# Hardcoded values

Every fixed value a replicator may need to change is marked in the code with `# HARDCODED:`.
The table below is regenerated from those comments on every build, so it always matches the
code; click a location to open it. To add a row, add a `# HARDCODED:` comment in the code.

{{ hardcoded }}

## Move to YAML for the new countries

| Value | Now in | Why it matters for Congo, Colombia, Jordan, Ukraine |
|---|---|---|
| ASR model and `language="en"` | `scripts/run_whisper_asr.py` | transcripts must be in French/Lingala, Spanish, Arabic, Ukrainian |
| Whisper `openai/whisper-small.en` | `WhisperBackbone`, `KintsugiAudioDataset` defaults and every YAML | English-only encoder; a multilingual checkpoint (e.g. `whisper-small`) is needed |
| BERT `google-bert/bert-base-uncased` | `BERTBackbone`, `KintsugiTextDataset` defaults | English tokenizer/model |
| `IDEAL_LOGMEL_ENERGIES` | `kirad/constants.py` | fixed log-mel profile of unknown origin (Kintsugi data); recompute per device/country or disable |
| `METRIC_TARGET_CUTOFFS`, `MAX_SCORE` | `kirad/constants.py` | local validated PHQ-9 / GAD-7 cutoffs may differ; SCID outcome needs its own entry |
| Task names `("anxiety", "depression")` | `kirad/utils.py` (`indet_analysis`; default of `analyze_dataframe_with_scores`) | new outcomes are silently skipped in analysis |
| Merge columns (`zip_code`, `english_preferred`, …) | `merge_df_rows_by_subject` | US survey schema |
| `EXPECTED_SAMPLE_RATE` 16 kHz, no resampling | `kirad/constants.py` | field audio at 8/44.1/48 kHz must be resampled upstream |
| Indeterminate budgets (0.0, 0.4) | `model.py` of each DAM | already overridable per task (`indet_budgets`); keep explicit in YAML |
| W&B entity/project, `kipy` | `scripts/launcher.py` | must be set for any run |
