import itertools
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from kirad.base_models import (
    BERTBackbone,
    LLAMABackbone,
    MultitaskHead,
    MultiTaskOrdinalRegressionPLModule,
    OrdinalRegressionPLModule,
    WhisperBackbone,
)
from kirad.constants import LOSS_TARGET_CUTOFFS, METRIC_TARGET_CUTOFFS
from kirad.launcher_utils import TrainingParams


def create_task_modules_from_params(tasks):
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
        for task, task_config in tasks
    }
    return task_modules


class DepAnxClassifierBase(MultiTaskOrdinalRegressionPLModule):
    def __init__(
        self,
        backbone: nn.ModuleDict,
        classifier_config: Mapping[str, Any],
        training_params: Optional[TrainingParams] = None,
        cached_backbone: bool = False,
    ):
        task_modules = create_task_modules_from_params(training_params.tasks.items())
        super().__init__(task_modules)

        self.backbone = backbone
        backbone_dim = sum(
            [backbone.backbone_dim for backbone in self.backbone.values()]
        )
        self.head = MultitaskHead(backbone_dim, **classifier_config)

        self.training_params = training_params
        self.cached_backbone = cached_backbone

    def forward(self, x, lengths):
        backbone_output, lengths = self.forward_backbone(x, lengths)
        return self.head(backbone_output), lengths

    def forward_backbone(self, x, lengths):
        if self.cached_backbone:
            return x["backbone_cache"], lengths
        else:
            outputs = [module(x) for module in self.backbone.values()]
            return torch.cat(outputs, dim=1), lengths

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


class DepAnxClassifierWhisperBERT(DepAnxClassifierBase):
    """A Whisper backbone -> mean pool -> full-connected layer.
    The whole network is trained end-to-end. Note that the whisper backbone
    is loaded from HuggingFace every time this module is called. Loading the
    backbone from `kipy` repeatedly ran into a bug, but this is tech debt.
    """

    def __init__(
        self,
        whisper_config: Mapping[str, Any],
        bert_config: Mapping[str, Any],
        **kwargs,
    ):
        # ordering in dict is preserved when concatenating
        # the backbone embeddings
        backbone = nn.ModuleDict(
            {
                "audio": WhisperBackbone(**whisper_config),
                "text": BERTBackbone(**bert_config),
            }
        )
        super().__init__(backbone=backbone, **kwargs)


class DepAnxClassifierWhisperLLAMA(DepAnxClassifierBase):
    def __init__(
        self,
        whisper_config: Mapping[str, Any],
        llama_config: Mapping[str, Any],
        **kwargs,
    ):
        # ordering in dict is preserved when concatenating
        # the backbone embeddings
        backbone = nn.ModuleDict(
            {
                "audio": WhisperBackbone(**whisper_config),
                "text": LLAMABackbone(**llama_config),
            }
        )
        super().__init__(backbone=backbone, **kwargs)


class DepAnxClassifierLLAMA(DepAnxClassifierBase):
    def __init__(self, llama_config: Mapping[str, Any], **kwargs):
        backbone = nn.ModuleDict(
            {
                "text": LLAMABackbone(**llama_config),
            }
        )
        super().__init__(backbone=backbone, **kwargs)
