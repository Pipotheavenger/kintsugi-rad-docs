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
    """A Whisper backbone -> mean pool -> full-connected layer.
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
        task_modules = {
            task: OrdinalRegressionPLModule(
                num_classes=task_config["num_classes"],
                loss_target_cutoffs=LOSS_TARGET_CUTOFFS[task][
                    task_config["ordinal_loss_target_type"]
                ],
                metric_target_cutoffs=METRIC_TARGET_CUTOFFS[task],
                loss_name=f"loss_{task_config['ordinal_loss_target_type']}",
                score_table_name=f"scores-{task}",
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
        whisper_config["hf_config"].setdefault("max_source_positions", 1500)
        featex_config = featex_config or {}
        self.padding = whisper_config["hf_config"]["max_source_positions"] * 320
        self.feature_extractor = WhisperTorchFeatureExtractor(**featex_config)
        self.backbone = WhisperBackbone(**whisper_config)
        self.head = MultitaskHead(self.backbone.backbone_dim, **classifier_config)

        self.training_params = training_params
        self.cached_backbone = cached_backbone

    def forward(self, x, lengths):
        backbone_output, lengths = self.forward_backbone(x, lengths)
        return self.head(backbone_output), lengths

    def forward_backbone(self, x, lengths):
        if self.cached_backbone:
            return x["backbone_cache"], lengths
        else:
            audio = x["audio"]
            audio = torch.nn.functional.pad(audio, (0, self.padding - audio.shape[-1]))
            return self.backbone({"audio": self.feature_extractor(audio)}), lengths

    def compute_features_to_cache(self, x, lengths):
        x, lengths = self.forward_backbone(x, lengths)
        x = torch.split(x, lengths.tolist(), dim=0)
        return x

    def configure_optimizers(self):
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
