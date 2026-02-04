import itertools
from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from kirad.base_models import (
    MultitaskHead,
    MultiTaskOrdinalRegressionPLModule,
    OrdinalRegressionPLModule,
    WhisperBackbone,
)
from kirad.constants import LOSS_TARGET_CUTOFFS, METRIC_TARGET_CUTOFFS, DatasetFields
from kirad.launcher_utils import TrainingParams
from kirad.utils import (
    average_tensor_in_segments,
    get_wandb_run_name_tag_from_artifact_url,
    localize_wandb_artifact,
)


class WhisperLLMA(pl.LightningModule):
    """Whisper-based LLM approximation model.
    gt_dim – ground truth dimensionality, can be different for
    different models. If not specified, defaults to dimensionality of the
    approximator
    """

    def __init__(
        self,
        whisper_config: Optional[Mapping[str, Any]] = None,
        gt_dim: Optional[int] = None,
        training_params: Optional[TrainingParams] = None,
        cached_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = nn.ModuleDict(
            {
                "llma": WhisperBackbone(**whisper_config),
            }
        )
        self.backbone_dim = self.backbone["llma"].backbone_dim

        #
        # ground truth dimensionality can differ from Whisper one.
        # so we do a random linear projection from ground truth to
        # Whisper dimensionality. We do not optimize/train the random projector.
        gt_dim = self.backbone_dim if gt_dim is None else gt_dim
        self.random_projector = (
            nn.Linear(gt_dim, self.backbone_dim)
            if self.backbone_dim != gt_dim
            else nn.Identity()
        )

        self.training_params = training_params
        self.cached_backbone = cached_backbone

    def forward(self, x, lengths):
        x = self.backbone["llma"](x)
        x = average_tensor_in_segments(x, lengths)
        return x

    def forward_ground_truth(self, x, lengths):
        x_truth = x["backbone_cache"]
        x_truth = average_tensor_in_segments(x_truth, lengths)
        x_truth = self.random_projector(x_truth)
        return x_truth

    def training_step(self, batch, batch_idx):
        total_loss = self.step("train", batch, batch_idx)
        return total_loss

    def validation_step(self, batch, batch_idx):
        self.step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.step("test", batch, batch_idx)

    def step(self, split, batch, batch_idx):
        x = batch[DatasetFields.FEATURES]
        lengths = batch[DatasetFields.LENGTH]

        pred = self.forward(x, lengths)
        ground_truth = self.forward_ground_truth(x, lengths)
        total_loss = F.mse_loss(pred, ground_truth)

        self.log(
            f"metrics/{split}/total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return total_loss

    def configure_optimizers(self):
        if self.training_params is None:
            return None
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.backbone["llma"].parameters(),
                    "lr": self.training_params.optimizer["backbone"].lr,
                    "weight_decay": self.training_params.optimizer[
                        "backbone"
                    ].weight_decay,
                    "name": "llma_backbone",
                },
            ]
        )
        scheduler = MultiStepLR(
            optimizer, **self.training_params.scheduler.model_dump()
        )
        return [optimizer], [scheduler]


class DepAnxClassifier(MultiTaskOrdinalRegressionPLModule):
    """A Whisper backbone -> mean pool -> full-connected layer.
    The whole network is trained end-to-end. Note that the whisper backbone
    is loaded from HuggingFace every time this module is called. Loading the
    backbone from `kipy` repeatedly ran into a bug, but this is tech debt.
    """

    def __init__(
        self,
        classifier_config: Mapping[str, Any],
        llma_ckpt: str,
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

        self.backbone = nn.ModuleDict(
            {
                "audio": WhisperBackbone(**whisper_config),
            }
        )

        # load pretrained llma model
        run, name, tag = get_wandb_run_name_tag_from_artifact_url(llma_ckpt)
        ckpt = localize_wandb_artifact(run, name, tag)
        self.llma = WhisperLLMA(
            **run.config["model_params"]["config"], training_params=training_params
        )
        self.llma.load_state_dict(torch.load(ckpt)["state_dict"], strict=True)

        backbone_dim = self.backbone["audio"].backbone_dim + self.llma.backbone_dim
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
            with torch.no_grad():
                audio_output = self.backbone["audio"](x)
                audio_output = average_tensor_in_segments(audio_output, lengths)
                llma_output = self.llma(x, lengths)
            # due to averaging, there is always one vector per uuid/utterance
            lengths = torch.ones_like(lengths)
            return torch.cat((audio_output, llma_output), dim=1), lengths

    def compute_features_to_cache(self, x, lengths):
        x, lengths = self.forward_backbone(x, lengths)
        x = torch.split(x, lengths, dim=0)
        return x

    def configure_optimizers(self):
        if self.training_params is None:
            return None
        optimizer = torch.optim.AdamW(
            [
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
