"""DAM 1 model (stage: model): audio-only depression/anxiety classifier.

Raw 16 kHz windows -> GPU log-mel (kirad/feature_extraction_whisper.py) -> Whisper encoder
with LoRA -> MultitaskHead -> one CORAL score per window and task (kirad/base_models.py).
Selected by research/stable/ordinal_regression_mtl_mar_2024/config*.yaml.
"""

import itertools
from typing import Any, Mapping, Optional

import torch
from torch.optim.lr_scheduler import MultiStepLR

from kirad.base_models import (
    MultitaskHead,
    MultiTaskOrdinalRegressionPLModule,
    OrdinalRegressionPLModule,
    WhisperBackbone,
)
from kirad.constants import LOSS_TARGET_CUTOFFS, METRIC_TARGET_CUTOFFS
from kirad.feature_extraction_whisper import WhisperTorchFeatureExtractor
from kirad.launcher_utils import TrainingParams


class DepAnxClassifier(MultiTaskOrdinalRegressionPLModule):
    """DAM 1: log-mel -> Whisper small.en encoder (LoRA) -> mean pool -> multitask CORAL head.

    A Whisper backbone -> mean pool -> full-connected layer.
    The whole network is trained end-to-end. Note that the whisper backbone
    is loaded from HuggingFace every time this module is called. Loading the
    backbone from `kipy` repeatedly ran into a bug, but this is tech debt.
    """

    def __init__(
        self,
        classifier_config: Mapping[str, Any],
        featex_config: Optional[Mapping[str, Any]] = None,
        whisper_config: Optional[Mapping[str, Any]] = None,
        training_params: Optional[TrainingParams] = None,
        cached_backbone: bool = False,
    ):
        """Build per-task CORAL modules, GPU log-mel extractor, Whisper backbone and head.

        Tasks come from training_params.tasks; cutoffs from kirad/constants.py.
        """
        task_modules = {
            task: OrdinalRegressionPLModule(
                num_classes=task_config["num_classes"],
                loss_target_cutoffs=LOSS_TARGET_CUTOFFS[task][
                    task_config["ordinal_loss_target_type"]
                ],
                metric_target_cutoffs=METRIC_TARGET_CUTOFFS[task],
                loss_name=f"loss_{task_config['ordinal_loss_target_type']}",
                score_table_name=f"scores-{task}",
                # HARDCODED: default indeterminate budgets 0% and 40%
                indet_budgets=task_config.get("indet_budgets", (0.0, 0.4)),
                score_variance_loss_weight=task_config.get(
                    "score_variance_loss_weight", 0.0
                ),
                kd_loss_weight=task_config.get("kd_loss_weight", 0.0),
                coral_loss_weight=task_config.get("coral_loss_weight", 1.0),
            )
            for task, task_config in training_params.tasks.items()
        }
        super().__init__(task_modules)
        whisper_config = whisper_config or {}
        whisper_config.setdefault("hf_config", {})
        # HARDCODED: 1500 encoder positions = 30 s windows (Whisper default)
        whisper_config["hf_config"].setdefault("max_source_positions", 1500)
        featex_config = featex_config or {}
        # HARDCODED: 320 samples per encoder position (hop 160 x conv stride 2 at 16 kHz);
        # 1500 x 320 = 480,000 samples = 30 s. Audio is assumed 16 kHz, never resampled.
        self.padding = whisper_config["hf_config"]["max_source_positions"] * 320
        self.feature_extractor = WhisperTorchFeatureExtractor(**featex_config)
        self.backbone = WhisperBackbone(**whisper_config)
        self.head = MultitaskHead(self.backbone.backbone_dim, **classifier_config)

        self.training_params = training_params
        self.cached_backbone = cached_backbone

    def forward(self, x, lengths):
        """Backbone then MultitaskHead -> ({task: [ΣW, 1] scores}, lengths)."""
        backbone_output, lengths = self.forward_backbone(x, lengths)
        return self.head(backbone_output), lengths

    def forward_backbone(self, x, lengths):
        """Pad raw audio to 480,000 samples, log-mel on GPU, Whisper encoder -> [ΣW, 768].

        With cached_backbone, returns x["backbone_cache"] (embeddings saved by `cache`).
        Shapes: x["audio"] [ΣW, ≤480000] -> log-mel [ΣW, 80, 3000] -> [ΣW, 768].
        """
        if self.cached_backbone:
            return x["backbone_cache"], lengths
        else:
            audio = x["audio"]
            audio = torch.nn.functional.pad(audio, (0, self.padding - audio.shape[-1]))
            return self.backbone({"audio": self.feature_extractor(audio)}), lengths

    def compute_features_to_cache(self, x, lengths):
        """Backbone embeddings split per recording (tuple of [W_i, 768]) for `cache`."""
        x, lengths = self.forward_backbone(x, lengths)
        x = torch.split(x, lengths.tolist(), dim=0)
        return x

    def configure_optimizers(self):
        """AdamW with two groups (backbone, head + CORAL biases) and MultiStepLR.

        lr / weight_decay from training_params.optimizer["backbone"] and ["classifier"].
        """
        if self.training_params is None:
            return None
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.backbone.parameters(),
                    "lr": self.training_params.optimizer["backbone"].lr,
                    "weight_decay": self.training_params.optimizer[
                        "backbone"
                    ].weight_decay,
                    "name": "backbone",
                },
                {
                    "params": itertools.chain(
                        self.head.parameters(),
                        self.tasks.parameters(),  # need self.tasks to train ordinal regression thresholds
                    ),
                    "lr": self.training_params.optimizer["classifier"].lr,
                    "weight_decay": self.training_params.optimizer[
                        "classifier"
                    ].weight_decay,
                    "name": "classifier",
                },
            ]
        )
        scheduler = MultiStepLR(
            optimizer, **self.training_params.scheduler.model_dump()
        )
        return [optimizer], [scheduler]
