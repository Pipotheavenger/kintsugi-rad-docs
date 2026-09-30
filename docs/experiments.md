# Experiments: DAM 1, 2, 3

Each folder in `research/stable/` has a `model.py` and YAML configs; `model_path` in the YAML
picks the class. `config.yaml` is the current config of each folder (dataset `v2.11.4`, 4
sources); `config_dam*.yaml` and `config_llma_dam3.0.yaml` are older snapshots (`v2.9.4`, 2 sources). Only DAM 1 `config.yaml`
validates as-is; see [Limitations](limitations.md) 5–8.

## Comparison

| | DAM 1 | DAM 2 | DAM 3 |
|---|---|---|---|
| Folder | `ordinal_regression_mtl_mar_2024` | `ordinal_regression_mtl_jun_2024` | `ordinal_regression_mtl_aug_2024` |
| Class | `DepAnxClassifier` | `DepAnxClassifierWhisperBERT` | phase 1 `WhisperLLMA`, phase 2 `DepAnxClassifier` |
| Input at inference | audio | audio + transcript | audio |
| Backbone(s) | Whisper small.en + LoRA | Whisper + LoRA, BERT base uncased + LoRA | frozen Whisper + frozen WhisperLLMA |
| Head input | 768 | 768 + 768 = 1536 | 768 + 768 = 1536 |
| Log-mel | in model, on GPU | HF extractor in dataset | HF extractor in dataset |
| Trained | encoder LoRA + conv1/conv2, head, CORAL biases | both LoRAs, head, biases | phase 1: Whisper (MSE to BERT half `[768:1536]` of a cached DAM 2); phase 2: head + biases only |
| Needs ASR | no | yes (`scripts/run_whisper_asr.py`) | phase 1 only (via DAM 2 cache) |

Other classes in `jun_2024/model.py`: `DepAnxClassifierWhisperLLAMA` (Whisper + LLaMA 3.2 3B
cut to 15 layers, `config_llama.yaml`) and `DepAnxClassifierLLAMA` (text only).

## Hyperparameters (current configs)

| Key | DAM 1 `config.yaml` | DAM 2 `config.yaml` | DAM 3 ph. 1 `config_llma.yaml` | DAM 3 ph. 2 `config.yaml` |
|---|---|---|---|---|
| seed | 44 | 44 | 42 | 46 |
| batch_size / effective | 2 / 128 | 2 / 128 | 1 / 64 | 1 / 4 |
| max_epochs / patience | 32 / 10 | 32 / 10 | 18 / 10 | 5 / 10 |
| val every | 16,000 recordings | epoch | epoch | 16,000 recordings |
| monitor | max `sn_eq_sp@0%i` | max `sn_eq_sp@0%i` | min `total_loss` | max `sn_eq_sp@0%i` |
| lr backbone / head | 1.6e-4 / 8e-4 | 1.6e-4 / 8e-4 | 4e-5 / — | unused (frozen) / 4e-4 |
| weight decay | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| MultiStepLR γ, milestones | 0.5, [20, 26] | 0.5, [20, 26] | 0.5, [14] | 0.5, [20, 26] |
| LoRA (Whisper) | r 32, α 64, drop 0.4 | same (+ BERT same) | none (full fine-tune) | r 32, α 64 (frozen) |
| shared / task proj / dropout | [256, 64] / 128 / 0.4 | same | — | same |
| targets | exact28 / exact22 | exact28 / exact22 | — | exact28 / exact22 |
| score_variance weight | 40 | 0 | — | 0 |
| train window_method | `random_adjacent_pair` | `single_random` | `all` | `all` |
| val/test window_method | `all` | `all` | `all` | `all` |
| merge_rows_by_subject (train) | `many_to_one` | no | no | no |
| audio `preprocessor` | `false` | `openai/whisper-small.en` | same | same |
| `ideal_logmel_energies` | built-in constant | `logmel_energies.npy` | same | same |
| other | | text: `bert-base-uncased` | `max_duration` 90 s (train), `feat_start_idx` 768 | `pad_last_chunk_to_full` |

Phase 2 loads WhisperLLMA from `llma_ckpt` (a W&B URL). Its audio Whisper gets trained weights only
from `ckpt_path`, which is empty in `config.yaml`, so it runs as the pretrained HF encoder.

Tasks in every config: `depression` (label `phq`, 28 classes) and `anxiety` (label `gad`, 22
classes). Indeterminate budgets default to (0.0, 0.4) in every `model.py`.

{{ codemap:variants }}
