"""Model building blocks.

Stage: model + loss + thresholds/metrics. Pretrained backbones (Whisper, BERT, LLaMA),
the shared MultitaskHead, and the Lightning modules that compute the CORAL loss per task,
tune thresholds on validation and log Sn/Sp metrics. Used by research/stable/*/model.py.
"""

from collections import OrderedDict
from dataclasses import asdict
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from loguru import logger
from peft import LoraConfig, get_peft_model
from pytorch_lightning.utilities import rank_zero_only
from scipy.stats import hmean
from torch import nn
from transformers import (
    BertConfig,
    BertModel,
    LlamaConfig,
    LlamaForCausalLM,
    ResNetConfig,
    ResNetForImageClassification,
    Wav2Vec2Config,
    Wav2Vec2Model,
    WhisperConfig,
    WhisperModel,
)

from .constants import DatasetFields
from .losses import CORALLoss
from .metrics import (
    IndetSnEqSpGraph,
    IndetSnSpArray,
    IndetSnSpTensorModule,
    MultiCatMetric,
)
from .ordinal_thresholding import (
    CORALThresholding,
    MaxAccuracyOrdinalThresholding,
    MaxMacroF1OrdinalThresholding,
    MaxMacroPrecisionOrdinalThresholding,
    MaxMacroRecallOrdinalThresholding,
    MinAbsoluteErrorOrdinalThresholding,
    OptimalOrdinalThresholdingViaDynamicProgramming,
)
from .utils import (
    average_and_variance_tensor_in_segments,
    average_tensor_in_segments,
    dataframe_to_wandb_table,
    get_wandb_run_name_tag_from_artifact_url,
    localize_wandb_artifact,
    strs_to_tensor,
    tensor_to_strs,
)

DATA_LOADER_ATTR_NAMES = {
    "train": "train_dataloader",
    "val": "val_dataloaders",
    "test": "test_dataloaders",
}

rank_zero_log_info = rank_zero_only(logger.info)


class OrdinalRegressionPLModule(pl.LightningModule):
    """An ordinal regression model with CORAL loss and threshold tuning.

    This class can be used in two ways:
    (1) subclass and override the `forward` method to get a class whose instances can be passed directly to `trainer.fit`
    (2) instantiate within a class to customize the behavior, e.g. having multiple ordinal regression heads whose inputs
    and labels are different, but which share some backbone computation.

    In case (1) the following methods are called automatically
    * `on_validation_epoch_end`
    * `on_test_epoch_end`
    * `training_step`
    * `validation_step`
    * `test_step`
    to compute losses, tune thresholds, and compute metrics. In case (2) these methods must be called manually,
    presumably by methods with the same name in the parent object.

    Case (2) requires pytorch-lightning 1.7+ for self.log to work correctly with nested `LightningModule`s.

    """

    # HARDCODED: above 10,000 scores, Sn/Sp thresholds use a 10,000-point linspace grid
    THRESH_LIM: int = 10_000

    def __init__(
        self,
        num_classes: int,
        loss_target_cutoffs: Sequence[int],
        metric_target_cutoffs: OrderedDict[int, str] | Sequence[int],
        loss_name: str = "loss",
        score_table_name: str = "scores",
        indet_budgets: Sequence[float] = (0.0, 0.4),  # HARDCODED: 0% / 40% indet budgets
        coral_loss_weight: float = 1.0,
        score_variance_loss_weight: float = 0.0,
        kd_loss_weight: float = 0.0,
        mean_before_loss: Optional[dict[str, bool]] = None,
    ):
        """Build CORAL loss, six threshold tuners, Sn=Sp buffers and val/test accumulators.

        Initialize the model.

        Parameters
        ----------
        num_classes : number of classes to classify inputs into
        loss_target_cutoffs : for each element, include a CORAL loss classifying whether inputs are greater than that
            (typically `range(num_classes)`)
        metric_target_cutoffs: Either (a) mapping of increasing of input labels which are upper bounds for each
            output label to names of corresponding divisions of output labels, e.g. {9: "LvMH", 14: "LMvH"} for
            depression, or (b) increasing sequence of input labels, e.g. (9, 14), from which the labels
            {9: ">9", 14: ">14"} would be inferred.
        increasing sequence of input labels which are upper bounds for each output label
            (e.g. [9, 14] to indicate that the upper limit of a "low depression" (0) output is PHQ of 9 and the upper
            limit of "medium depression" (1) is PHQ of 14.
        loss_name : string describing `loss_target_cutoffs`, so experiments with different losses are displayed on
            separate plots in wandb
        indet_budgets : fractions of outputs allowed to be indeterminate in metric computations (will compute separate
            metrics for each entry)
        score_variance_loss_weight : Weight to apply to variance of scores over chunks within each stream when computing
            loss, relative to weight of CORAL loss
        mean_before_loss : mapping from split to bool. If True, compute loss for split as loss(mean(scores)); if False,
            compute loss as mean(loss(scores)). Defaults to False for "train" and True for "val" and "test".

        """
        super().__init__()
        if isinstance(metric_target_cutoffs, Sequence):
            metric_target_cutoffs = OrderedDict(
                (i, f">{i}") for i in metric_target_cutoffs
            )
        self.metric_target_cutoff_names = metric_target_cutoffs
        self.loss = CORALLoss(
            num_classes=num_classes,
            weights=[float(i in loss_target_cutoffs) for i in range(num_classes - 1)],
        )
        self.kd_loss = nn.MSELoss(reduction="none")
        num_targets = len(self.metric_target_cutoff_names) + 1
        self.thresholding = nn.ModuleDict(
            dict(
                accuracy=MaxAccuracyOrdinalThresholding(num_targets),
                absolute_error=MinAbsoluteErrorOrdinalThresholding(num_targets),
                macro_recall=MaxMacroRecallOrdinalThresholding(num_targets),
                macro_precision=MaxMacroPrecisionOrdinalThresholding(num_targets),
                macro_f1=MaxMacroF1OrdinalThresholding(num_targets),
            )
        )
        try:
            metric_target_indices = [
                loss_target_cutoffs.index(i) for i in self.metric_target_cutoff_names
            ]
            self.thresholding["coral"] = CORALThresholding(
                self.loss, metric_target_indices
            )
        except ValueError:
            logger.info(
                "Skipping CORAL-derived thresholding logic because metric_target_cutoffs are not all contained"
                "in loss_target_cutoffs, so not all metric targets have a CORAL-learned threshold / bias."
            )
        self.indet_thresholding = nn.ModuleDict(
            {
                str(c_idx): IndetSnSpTensorModule()  # keys must be strings
                for c_idx in self.metric_target_cutoff_names
            }
        )
        # 4 because we will accumulate (1) uuid, (2) score mean, (3) score variance, and (4) label for each sample in the val and test sets
        self.accumulators = nn.ModuleDict(
            {key: MultiCatMetric(4) for key in ("val", "test")}
        )
        self.loss_name = loss_name
        self.score_table_name = score_table_name
        self.indet_budget_names = {
            budget: f"@{budget:.0%}i" for budget in indet_budgets
        }
        self.score_variance_loss_weight = score_variance_loss_weight
        self.kd_loss_weight = kd_loss_weight
        self.coral_loss_weight = coral_loss_weight
        if mean_before_loss is None:
            # HARDCODED: per-window loss in train; loss of the per-recording mean in val/test
            self.mean_before_loss = {"train": False, "val": True, "test": True}
        else:
            self.mean_before_loss = mean_before_loss

    def load_state_dict(self, state_dict, strict=True):
        """Resize indet-threshold buffers to the checkpoint's shapes, then load weights."""
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        for c_idx in self.metric_target_cutoff_names:
            for k, v in self.indet_thresholding[str(c_idx)].named_buffers():
                v.resize_(state_dict[f"indet_thresholding.{c_idx}.{k}"].shape)
        super().load_state_dict(state_dict, strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity: the multi-task parent computes scores; this module only adds loss/metrics.

        Defaults to identity for case where this class is instantiated directly rather than subclassed.

        This default is to enable the case where a model contains multiple ordinal regression tasks (e.g. depression
        and anxiety) sharing some computation, which is therefore implemented elsewhere. If there is only a single
        task, it is simpler to subclass this class and override `forward`.

        """
        return x

    def quantize_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Map input labels (e.g. PHQ-9 sum) to output labels (e.g. 0/1/2 for L/M/H)."""
        return torch.stack(
            [torch.gt(labels, l) for l in self.metric_target_cutoff_names], dim=0
        ).sum(dim=0)

    def log_ordinal_metrics(
        self,
        split: str,
        issas: Mapping[int, IndetSnSpArray],
        *,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        """Log AUROC and min(Sn, Sp) per binary cut and indet budget, plus their mean.

        On test, also Sn/Sp/indet fraction at the thresholds tuned on validation.
        Returns the dict of logged metrics (keys bucketed_metrics/... and metrics/...).
        """
        # Log ordinal bucketed metrics
        auroc_sum = {budget: 0.0 for budget in self.indet_budget_names}
        sn_eq_sp_sum = {budget: 0.0 for budget in self.indet_budget_names}
        sn_sum = {budget: 0.0 for budget in self.indet_budget_names}
        sp_sum = {budget: 0.0 for budget in self.indet_budget_names}
        indet_sum = {budget: 0.0 for budget in self.indet_budget_names}
        labels_numpy = labels.numpy(force=True)
        scores_numpy = scores.numpy(force=True)
        metrics = dict()
        for c_idx, c_name in self.metric_target_cutoff_names.items():
            labels_2class = (labels_numpy > c_idx).astype(int)
            for indet_budget, indet_budget_str in self.indet_budget_names.items():
                roc_curve = issas[c_idx].roc_curve(indet_budget)
                auroc = roc_curve.auc()
                auroc_sum[indet_budget] += auroc
                metrics[
                    f"bucketed_metrics/{split}/auroc_{c_name}{indet_budget_str}"
                ] = auroc
                sn_eq_sp = roc_curve.sn_eq_sp().min_sn_sp
                sn_eq_sp_sum[indet_budget] += sn_eq_sp
                metrics[
                    f"bucketed_metrics/{split}/sn_eq_sp_{c_name}{indet_budget_str}"
                ] = sn_eq_sp

                if split == "test":
                    try:
                        val_ops = IndetSnEqSpGraph(
                            **asdict(self.indet_thresholding[str(c_idx)].numpy())
                        )
                        test_at_val_op = val_ops.at_budget(indet_budget).eval(
                            y_true=labels_2class,
                            y_score=scores_numpy,
                        )
                        metrics[
                            f"bucketed_metrics/test/sn_at_val_eq_thresh_{c_name}{indet_budget_str}"
                        ] = test_at_val_op.sn
                        metrics[
                            f"bucketed_metrics/test/sp_at_val_eq_thresh_{c_name}{indet_budget_str}"
                        ] = test_at_val_op.sp
                        metrics[
                            f"bucketed_metrics/test/indet_frac_at_val_eq_thresh_{c_name}{indet_budget_str}"
                        ] = test_at_val_op.indet_frac
                        sn_sum[indet_budget] += test_at_val_op.sn
                        sp_sum[indet_budget] += test_at_val_op.sp
                        indet_sum[indet_budget] += test_at_val_op.indet_frac
                    except Exception as e:
                        logger.info(f"Validation threshold is ill-conditioned: {e}")

        denom = len(self.metric_target_cutoff_names)

        # Log ordinal macro average metrics
        for indet_budget, indet_budget_str in self.indet_budget_names.items():
            metrics[f"metrics/{split}/auroc{indet_budget_str}"] = (
                auroc_sum[indet_budget] / denom
            )
            metrics[f"metrics/{split}/sn_eq_sp{indet_budget_str}"] = (
                sn_eq_sp_sum[indet_budget] / denom
            )
            if split == "test":
                metrics[f"metrics/test/sn_at_val_eq_thresh{indet_budget_str}"] = (
                    sn_sum[indet_budget] / denom
                )
                metrics[f"metrics/test/sp_at_val_eq_thresh{indet_budget_str}"] = (
                    sp_sum[indet_budget] / denom
                )
                metrics[
                    f"metrics/test/indet_frac_at_val_eq_thresh{indet_budget_str}"
                ] = (indet_sum[indet_budget] / denom)
        self.log_dict(metrics, sync_dist=True)
        return metrics

    def log_thresholding_metrics(
        self, split: str, *, scores: torch.Tensor, quantized_labels: torch.tensor
    ) -> dict[str, float]:
        """Log each tuned method's cost when scored with every method's thresholds.

        Also logs the cost of the best constant-output classifier as a baseline.
        """
        metrics = dict()
        for key1, value1 in self.thresholding.items():
            if isinstance(value1, OptimalOrdinalThresholdingViaDynamicProgramming):
                metric_key = f"thresh_metrics/{split}/{key1}"
                baseline_cost, baseline_index = value1.best_constant_output_classifier(
                    quantized_labels
                )
                metrics[
                    f"{metric_key}/baseline_const_{baseline_index}"
                ] = baseline_cost.item()
                for key2, value2 in self.thresholding.items():
                    if value2.is_valid():
                        metric = value1.mean_cost(
                            labels=quantized_labels, preds=value2(scores)
                        ).item()
                        metrics[f"{metric_key}/tuned_for_{key2}"] = metric
                    else:
                        logger.info(
                            f"Skipping evaluation of threshold method {key2!r} because its "
                            f"thresholds {value2.thresholds.tolist()} are not sorted."
                        )

        self.log_dict(metrics, sync_dist=True)
        return metrics

    def log_test_indet_analysis(
        self, *, scores: torch.Tensor, labels: torch.tensor
    ) -> None:
        """W&B line plots of test Sn, Sp, indet fraction vs the val indeterminate budget.

        One plot per binary cut, plus the average over cuts. Test split only.
        Hardcoded: 1000 interpolation points for the averaged curve.
        """
        task = self.score_table_name.split("-")[-1]
        # only do this for test because wandb doesn't give a good way to navigate val results by epoch at this time
        dfs = []
        for c_idx in self.metric_target_cutoff_names:
            # evaluated test data on thresholds for val sn=sp as a function of indet budget
            sn_eq_sp_on_val = self.indet_thresholding[str(c_idx)].numpy()
            val_ops_on_test = sn_eq_sp_on_val.eval(
                y_score=scores.numpy(force=True),
                y_true=(labels.numpy(force=True) > c_idx).astype(int),
            )
            df = pd.DataFrame(
                data=dict(
                    indet_budget_val=sn_eq_sp_on_val.indet_frac,
                    indet_frac_test=val_ops_on_test.indet_frac,
                    sn_test=val_ops_on_test.sn,
                    sp_test=val_ops_on_test.sp,
                )
            )
            dfs.append(df)
            table = dataframe_to_wandb_table(dataframe=df)
            for field in ("indet_frac_test", "sn_test", "sp_test"):
                wandb.run.log(
                    {
                        f"indet_analysis/test/{task}/{field}_>{c_idx}": wandb.plot.line(
                            table,
                            x="indet_budget_val",
                            y=field,
                            title=f"{task}>{c_idx}: {field} as a function of indeterminate budget on val set",
                            split_table=True,
                        )
                    }
                )
        max_indet = min(df.indet_budget_val.max() for df in dfs)
        # HARDCODED: 1000 interpolation points
        indet_budget_val = np.linspace(0.0, max_indet, num=1000)
        data = dict(indet_budget_val=indet_budget_val)
        for field in ("indet_frac_test", "sn_test", "sp_test"):
            value = np.zeros_like(indet_budget_val)
            for df in dfs:
                value += np.interp(indet_budget_val, df.indet_budget_val, df[field])
            value /= len(dfs)
            data[field] = value
        table = dataframe_to_wandb_table(dataframe=pd.DataFrame(data=data))
        for field in ("indet_frac_test", "sn_test", "sp_test"):
            wandb.run.log(
                {
                    f"indet_analysis/test/{task}/{field}": wandb.plot.line(
                        table,
                        x="indet_budget_val",
                        y=field,
                        title=f"{task}: {field} as a function of indeterminate budget on val set",
                        split_table=True,
                    )
                }
            )

    def on_validation_epoch_end(self) -> None:
        """Lightning hook: run the val epoch-end logic and free CUDA cache."""
        self.on_validation_test_epoch_end("val")
        torch.cuda.empty_cache()

    def on_test_epoch_end(self) -> None:
        """Lightning hook: run the test epoch-end logic."""
        self.on_validation_test_epoch_end("test")

    def on_validation_test_epoch_end(self, split: str) -> dict[str, float]:
        """Gather scores, log W&B table, tune thresholds on val, compute Sn/Sp metrics.

        Dedups uuids (DDP repeats samples), uploads scores-<task>-<split>-<run> artifact,
        builds one IndetSnSpArray per binary cut. Only on "val": tunes the six threshold
        methods and stores the Sn=Sp operating points; "test" reuses them.
        """
        tensor_uuids, *rests = self.accumulators[split].compute()
        uuids = tensor_to_strs(tensor_uuids)
        # Deduplicate -- pytorch runs some items twice to keep all replicas busy
        uuid_to_all = {uuid: (uuid, *rest) for uuid, *rest in zip(uuids, *rests)}
        uuids, scores, variances, labels = zip(*uuid_to_all.values())
        scores = torch.stack(scores)
        variances = torch.stack(variances)
        labels = torch.stack(labels)
        self.accumulators[split].reset()
        quantized_labels = self.quantize_labels(labels)

        global_rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        if global_rank == 0 and not self.trainer.sanity_checking:
            artifact = wandb.Artifact(
                f"{self.score_table_name}-{split}-{wandb.run.id}", "scores"
            )
            table = dataframe_to_wandb_table(
                dataframe=pd.DataFrame(
                    dict(
                        uuid=uuids,
                        scores=scores.cpu(),
                        score_variances=variances.cpu(),
                        labels=labels.cpu(),
                        quantized_labels=quantized_labels.cpu(),
                    )
                )
            )
            artifact.add(table, "scores")
            wandb.run.log_artifact(artifact)
        if scores.numel() > self.THRESH_LIM:
            threshes = np.linspace(
                scores.min().cpu().numpy(), scores.max().cpu().numpy(), self.THRESH_LIM
            )
        else:
            threshes = None
        issas = {
            c_idx: IndetSnSpArray.build(
                threshes,
                y_score=scores.double().numpy(force=True),
                y_true=(labels.numpy(force=True) > c_idx).astype(int),
            )
            for c_idx in self.metric_target_cutoff_names
        }
        sn_eq_sp_graphs = {key: value.sn_eq_sp_graph() for key, value in issas.items()}
        if split == "val":
            for method in self.thresholding.values():
                method.tune_thresholds(scores=scores, labels=quantized_labels)
            for c_idx, sn_eq_sp_graph in sn_eq_sp_graphs.items():
                self.indet_thresholding[str(c_idx)].update(
                    sn_eq_sp_graph, scores.device
                )
        if global_rank == 0 and not self.trainer.sanity_checking and split == "test":
            try:
                self.log_test_indet_analysis(scores=scores, labels=labels)
            except Exception as e:
                logger.info(
                    f"Validation threshold is ill-conditioned, skipping log_test_indet_analysis: {e}"
                )

        return {
            **self.log_ordinal_metrics(split, issas, scores=scores, labels=labels),
            **self.log_thresholding_metrics(
                split, scores=scores, quantized_labels=quantized_labels
            ),
        }

    def compute_and_log_loss(self, logits, y, lengths, split):
        """Loss = coral_w·CORAL + var_w·Var(window scores) + kd_w·MSE(teacher), logged.

        CORAL is loss(mean(scores)) if mean_before_loss[split] else mean(loss(per-window)).
        KD term only if the label dict has the KD field (teacher score per recording).
        Shapes: logits [ΣW, 1], y[PRIMARY] [B], lengths [B] -> scalar.
        """
        logits_mean, logits_variance = average_and_variance_tensor_in_segments(
            logits, lengths
        )

        try:
            y_kd = y[DatasetFields.LabelFields.KD].float()
        except KeyError:
            y_kd = None
        y = y[DatasetFields.LabelFields.PRIMARY]

        y_broadcast = torch.repeat_interleave(y, repeats=lengths, dim=0)
        if self.mean_before_loss[split]:
            loss = self.loss(logits_mean, y)
        else:
            loss = average_tensor_in_segments(self.loss(logits, y_broadcast), lengths)

        loss_mean = loss.mean()
        loss_variance_mean = logits_variance.mean()
        # include loss type in name of loss to avoid comparing different loss types in WandB
        self.log(
            f"metrics/{split}/{self.loss_name}",
            loss_mean,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"metrics/{split}/score_variance_{self.loss_name}",
            loss_variance_mean,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        loss_mean = (
            self.coral_loss_weight * loss_mean
            + self.score_variance_loss_weight * loss_variance_mean
        )

        if y_kd is not None:
            y_kd_broadcast = torch.repeat_interleave(y_kd, repeats=lengths, dim=0)
            loss_kd_by_chunk = self.kd_loss(logits, y_kd_broadcast[..., None])
            loss_kd = average_tensor_in_segments(loss_kd_by_chunk, lengths)
            loss_kd_mean = loss_kd.mean()
            self.log(
                f"metrics/{split}/kd_{self.loss_name}",
                loss_kd_mean,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            loss_mean = loss_mean + self.kd_loss_weight * loss_kd_mean

        return loss_mean

    def step(self, split, batch, batch_idx):
        """Accumulate per-recording mean/var scores for val/test, then return the loss.

        Shapes: batch features -> forward -> [ΣW, 1]; accumulated per recording [B].
        """
        logits = self.forward(batch[DatasetFields.FEATURES])
        if split in self.accumulators:
            logits_mean, logits_var = average_and_variance_tensor_in_segments(
                logits, batch[DatasetFields.LENGTH]
            )
            self.accumulators[split].update(
                strs_to_tensor(batch[DatasetFields.UUID]).to(logits.device),
                torch.squeeze(logits_mean, 1),
                torch.squeeze(logits_var, 1),
                batch[DatasetFields.LABEL][DatasetFields.LabelFields.PRIMARY],
            )
        return self.compute_and_log_loss(
            logits,
            batch[DatasetFields.LABEL],
            batch[DatasetFields.LENGTH],
            split,
        )

    def training_step(self, batch, batch_idx):
        """Lightning train step for a single-task subclass."""
        return self.step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """Lightning val step; loss is logged, not returned."""
        self.step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """Lightning test step; loss is logged, not returned."""
        self.step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Per-recording score: forward, then mean over its windows. Shapes: [B, 1]."""
        logits = average_tensor_in_segments(
            self.forward(batch[DatasetFields.FEATURES]),
            batch[DatasetFields.LENGTH],
        )
        torch.cuda.empty_cache()
        return logits


class MultiTaskOrdinalRegressionPLModule(pl.LightningModule):
    """Multi-task wrapper: one OrdinalRegressionPLModule per task on a shared model.

    Subclasses (the DAM models) define forward(features, lengths) -> ({task: scores}, lengths).
    """
    def __init__(self, tasks: Mapping[str, OrdinalRegressionPLModule]):
        """Store the per-task modules (loss, thresholds, accumulators) in a ModuleDict."""
        super().__init__()
        self.tasks = nn.ModuleDict(tasks)

    def load_state_dict(self, state_dict, strict=True):
        """Resize every task's indet-threshold buffers, then load weights.

        Missing/unexpected keys are only logged (with strict=False).
        """
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        for task_name, task in self.tasks.items():
            for c_idx in task.metric_target_cutoff_names:
                for k, v in task.indet_thresholding[str(c_idx)].named_buffers():
                    name = f"tasks.{task_name}.indet_thresholding.{c_idx}.{k}"
                    try:
                        v.resize_(state_dict[name].shape)
                    except KeyError:
                        if strict:
                            raise KeyError(
                                f"Missing {name} in state_dict with keys {list(state_dict.keys())}"
                            )
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict)
        if missing_keys:
            logger.info(
                f"The following tensors were missing from the loaded "
                f"state dict and will be initialized in the default "
                f"way: {missing_keys}"
            )
        if unexpected_keys:
            logger.info(
                f"The following keys in the loaded state dict do not "
                f"correspond to tensors in the model, so have been "
                f"ignored : {unexpected_keys}"
            )

    def configure_model(self) -> None:
        """Give each task the trainer and a logger that prefixes names with '<task>/'."""
        for key, task in self.tasks.items():
            # This has two effects:
            #   (1) installs a logger in the tasks, which current version of lightning does not do automatically
            #   (2) prepends f"{key}/" to everything logged by the task
            task.log = self.get_sub_logger(key)
            task.trainer = self.trainer

    def get_sub_logger(self, key):
        """Return a log function that prefixes metric names with '<key>/'."""
        def sub_logger(name, *args, **kwargs):
            """self.log with the task prefix."""
            self.log(f"{key}/{name}", *args, **kwargs)

        return sub_logger

    def on_validation_test_epoch_end(self, split) -> None:
        """Run each task's epoch end, then log harmonic mean of shared metrics across tasks.

        The combined metric (e.g. metrics/val/sn_eq_sp@0%i) drives checkpointing.
        """
        all_tasks_metrics = [
            task.on_validation_test_epoch_end(split) for task in self.tasks.values()
        ]
        common_metric_names = set.intersection(
            *(set(metrics) for metrics in all_tasks_metrics)
        )
        # For pytorch-lightning to behave correctly, it is crucial that the log statements happen in the same order
        # on all replicas. The `sorted` guarantees this.
        for metric_name in sorted(common_metric_names):
            values = [metrics[metric_name] for metrics in all_tasks_metrics]
            if all(value >= 0 for value in values):
                m = hmean(values)
                self.log(metric_name, m, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook: val epoch end for all tasks."""
        self.on_validation_test_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        """Lightning hook: test epoch end for all tasks."""
        self.on_validation_test_epoch_end("test")

    def compute_features_to_cache(self, x, lengths) -> list:
        """Backbone embeddings split per recording, for the cache command (abstract).

        Returns list of tensors with shapes N_ExD,
        where N_E - number of embedding vectors associated with each sample and
        D - embedding dimensionality.
        N_E depends on a mode (training,validation) as well as model architecture
        """
        raise NotImplementedError(
            "This method must be properly implemented in the inherited classes"
        )

    def training_step(self, batch, batch_idx):
        """Lightning train step: returns the step dict with total_loss."""
        return self.step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """Lightning val step; losses are logged, not returned."""
        self.step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """Lightning test step; losses are logged, not returned."""
        self.step("test", batch, batch_idx)

    def step(self, split, batch, batch_idx):
        """Run model once, sum each task's loss (task.step) into total_loss.

        Each task gets its own scores and labels: logits[task], batch[label][task].
        """
        logits, lengths = self.forward(
            batch[DatasetFields.FEATURES], batch[DatasetFields.LENGTH]
        )

        total_loss = sum(
            task.step(
                split,
                {
                    DatasetFields.UUID: batch[DatasetFields.UUID],
                    DatasetFields.FEATURES: logits[task_name],
                    DatasetFields.LABEL: batch[DatasetFields.LABEL][task_name],
                    DatasetFields.LENGTH: lengths,
                },
                batch_idx,
            )
            for task_name, task in self.tasks.items()
        )
        self.log(
            f"metrics/{split}/total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {
            DatasetFields.UUID: batch[DatasetFields.UUID],
            DatasetFields.FEATURES: logits,
            DatasetFields.LABEL: batch[DatasetFields.LABEL],
            DatasetFields.LENGTH: lengths,
            "loss": total_loss,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Per-recording scores for each task: {task: [B, 1]} (mean over windows)."""
        logits, lengths = self.forward(
            batch[DatasetFields.FEATURES], batch[DatasetFields.LENGTH]
        )
        logits = {
            task_name: average_tensor_in_segments(tensor, lengths)
            for task_name, tensor in logits.items()
        }
        torch.cuda.empty_cache()
        return logits


class MultiTaskOrdinalRegressionBackboneDictPLModule(
    MultiTaskOrdinalRegressionPLModule
):
    """Multi-task module whose backbone is a ModuleDict; loads DAM 1 weights into "audio".

    Version of `MultiTaskOrdinalRegressionPLModule` that correctly loads DAM 1 weights into the audio backbone."""

    backbone: nn.ModuleDict

    def load_state_dict(self, state_dict, strict=True):
        """Rename backbone.base_model.* keys to backbone.audio.base_model.* before loading."""
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # If trying to load DAM 1 weights, load them into the "audio" backbone
        if "audio" in self.backbone and "base_model" not in self.backbone:
            for k in state_dict.copy():
                if k.startswith("backbone.base_model."):
                    state_dict[
                        k.replace("backbone.base_model.", "backbone.audio.base_model.")
                    ] = state_dict.pop(k)
        super().load_state_dict(state_dict, strict)


class KintsugiBackboneBase(pl.LightningModule):
    """Base for pretrained backbones: holds self.backbone and adds optional LoRA."""
    backbone: nn.Module

    def __init__(self):
        """No parameters; subclasses build self.backbone."""
        super().__init__()

    def apply_lora(self, lora_params: Optional[Mapping[str, Any]] = None):
        """Wrap self.backbone with PEFT LoRA from lora_params; no-op (full fine-tune) if empty.

        lora_params go straight to peft.LoraConfig (r, lora_alpha, target_modules,
        modules_to_save, ...); modules_to_save layers (e.g. conv1/conv2) train fully.
        """
        if lora_params is not None and len(lora_params) > 0:
            lora_config = LoraConfig(**lora_params)
            self.backbone = get_peft_model(self.backbone, lora_config)
            rank_zero_log_info(f"Using LoRA params: {lora_config}")
            sep = "\n - "
            rank_zero_log_info(
                f"Backbone layers tuning with LoRA:\n - "
                f"{sep.join(self.backbone.base_model.targeted_module_names)}"
            )
            rank_zero_log_info(
                f"Backbone layers tuning without LoRA:\n - "
                f"{sep.join(self.backbone.modules_to_save or [])}"
            )
        else:
            rank_zero_log_info(
                f"No LoRA params specified for backbone {self.__class__.__name__}; "
                "fine tuning full backbone."
            )


class WhisperBackbone(KintsugiBackboneBase):
    """Whisper encoder only (decoder dropped), optionally mean-pooled over time."""
    def __init__(
        self,
        model: str = "openai/whisper-small.en",  # HARDCODED: English-only Whisper (DAM 1-3)
        hf_config: Optional[Mapping[str, Any]] = None,
        lora_params: Optional[Mapping[str, Any]] = None,
        mean_pool: bool = True,
    ):
        """Load the pretrained Whisper encoder from HuggingFace and apply LoRA.

        Note: if max_source_positions != 1500, mismatched weights (positional table)
        are re-initialized.
        """
        super().__init__()
        hf_config = hf_config if hf_config is not None else dict()
        backbone_config = WhisperConfig.from_pretrained(model, **hf_config)
        self.backbone = (
            WhisperModel.from_pretrained(
                model,
                config=backbone_config,
                # HARDCODED: 1500 encoder positions = 30 s of audio (Whisper default)
                ignore_mismatched_sizes=backbone_config.max_source_positions != 1500,
            )
            .get_encoder()
            .train()
        )
        self.apply_lora(lora_params)
        self.backbone_dim = backbone_config.hidden_size
        self.mean_pool = mean_pool

    def forward(self, x, *args, **kwargs):
        """Whisper encoder on log-mel windows, mean over time -> [ΣW, D] (768 for small.en).

        Shapes: x["audio"] [ΣW, 80, 3000] -> hidden [ΣW, 1500, D] -> [ΣW, D] if mean_pool.
        """
        output = self.backbone(x["audio"]).last_hidden_state
        if self.mean_pool:
            output = output.mean(dim=1)
        return output


class Wav2Vec2Backbone(KintsugiBackboneBase):
    """wav2vec2 encoder on raw audio, mean-pooled (not used by the stable DAM configs)."""
    def __init__(
        self,
        model: str = "facebook/wav2vec2-xls-r-300m",  # HARDCODED: default model id
        hf_config: Optional[Mapping[str, Any]] = None,
        lora_params: Optional[Mapping[str, Any]] = None,
        mean_pool: bool = True,
    ):
        """Load the pretrained wav2vec2 model from HuggingFace and apply LoRA."""
        super().__init__()
        hf_config = hf_config if hf_config is not None else dict()
        backbone_config = Wav2Vec2Config.from_pretrained(model, **hf_config)
        self.backbone = Wav2Vec2Model.from_pretrained(
            model,
            config=backbone_config,
        ).train()
        self.apply_lora(lora_params)
        self.backbone_dim = backbone_config.hidden_size
        self.mean_pool = mean_pool

    def forward(self, x, *args, **kwargs):
        """Encode raw audio x["audio"] [ΣW, samples] -> [ΣW, D] (mean over frames if mean_pool)."""
        output = self.backbone(x["audio"]).last_hidden_state
        if self.mean_pool:
            output = output.mean(dim=1)
        return output


class BERTBackbone(KintsugiBackboneBase):
    """BERT on transcript tokens; returns the pooler output [ΣW, 768] (DAM 2 text branch)."""
    def __init__(
        self,
        model: str = "google-bert/bert-base-uncased",  # HARDCODED: English BERT (DAM 2 text)
        hf_config: Optional[Mapping[str, Any]] = None,
        lora_params: Optional[Mapping[str, Any]] = None,
    ):
        """Load pretrained BERT from HuggingFace and apply LoRA."""
        super().__init__()
        hf_config = hf_config if hf_config is not None else dict()
        backbone_config = BertConfig.from_pretrained(model, **hf_config)
        self.backbone = BertModel.from_pretrained(model, config=backbone_config).train()
        self.apply_lora(lora_params)
        self.backbone_dim = backbone_config.hidden_size

    def forward(self, x, *args, **kwargs):
        """Pooler output ([CLS] + dense + tanh) of x["text"] tokens. Shapes: -> [ΣW, 768]."""
        _, pooler_output = self.backbone(**x["text"], return_dict=False)
        return pooler_output


class LLAMABackbone(KintsugiBackboneBase):
    """LLaMA cut to num_layers layers; embeds text as the last real token's hidden state."""
    param_group_name = "llama_backbone"

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",  # HARDCODED: default model id
        hf_config: Optional[Mapping[str, Any]] = None,
        lora_params: Optional[Mapping[str, Any]] = None,
        num_layers: int = 15,  # HARDCODED: LLaMA truncated to its first 15 layers
    ):
        """Load LLaMA (HF id or W&B artifact URL), keep the first num_layers layers, LoRA.

        Note: LLaMA is gated on HuggingFace, so configs pass a W&B artifact URL.
        """
        super().__init__()
        hf_config = hf_config if hf_config is not None else dict()

        # we always fetch from wandb because LLAMA is a gated model(requires additional huggingface authentication)
        if model.startswith("https:"):
            run, name, tag = get_wandb_run_name_tag_from_artifact_url(model)
            model = localize_wandb_artifact(run, name, tag, return_dir=True)
        backbone_config = LlamaConfig.from_pretrained(model, **hf_config)
        self.backbone = LlamaForCausalLM.from_pretrained(
            model, config=backbone_config
        ).train()

        # by default cut LLAMA at 15-th layer
        self.backbone.model.layers = self.backbone.model.layers[:num_layers]
        self.apply_lora(lora_params)
        self.backbone_dim = self.backbone.config.hidden_size

    def forward(self, x, *args, **kwargs):
        """Last hidden state at the last non-padded token of each sequence.

        Shapes: x["text"] input_ids/attention_mask [ΣW, L] -> [ΣW, D] (3072 for Llama-3.2-3B).
        """
        output = self.backbone(
            input_ids=x["text"]["input_ids"],
            attention_mask=x["text"]["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        )

        # [-1] selects the last hidden state
        # tensor "indices" lists indices of the last valid(unpadded) token of each batch exemplar,
        # which corresponds to the unseen, (n+1)-st token, given n input token
        indices = torch.sum(x["text"]["attention_mask"], dim=1) - 1
        output = output["hidden_states"][-1]

        # torch.arange(len(indices)) is required to avoid unneeded broadcasting behaviour
        output = output[torch.arange(len(indices)), indices, :]
        return output


class MultitaskHead(pl.LightningModule):
    """Shared MLP followed by one small head per task (depression, anxiety).

    e.g. D -> 256 (Mish) -> 64, then per task 64 -> 128 (Mish, dropout) -> 1.
    """
    class SharedLayers(nn.Module):
        """Shared MLP: Linear+Mish per proj_dim, last Linear without activation (e.g. D->256->64).

        The activation-free last layer acts as shared low-rank B in W_t = H_t·B.
        With a single proj_dim, Mish is kept after it (no factorization).
        """
        def __init__(self, input_dim, proj_dims):
            """Build Linear(+Mish) layers input_dim -> proj_dims[0] -> ... -> proj_dims[-1]."""
            super().__init__()

            # Stack linear layers with activation layers in between. When more than one
            # linear layer is specified (len(proj_dims) > 1), don't put an activation
            # layer after the final linear layer; this is so that the last linear layer
            # acts as a low-rank shared matrix B in a low-rank factorization of the
            # task-specific weights (W_t = H_t * B). If only one linear layer is
            # specified, then include an activation layer after it so that the low-rank
            # factorization isn't performed.
            num_layers = len(proj_dims)
            modules = []
            for output_dim in proj_dims[:-1]:
                modules.extend([nn.Linear(input_dim, output_dim), nn.Mish()])
                input_dim = output_dim
            modules.append(nn.Linear(input_dim, proj_dims[-1]))
            if num_layers == 1:
                modules.append(nn.Mish())
            self.shared_layers = nn.Sequential(*modules)

        def forward(self, x):
            """Shapes: [ΣW, D] -> [ΣW, proj_dims[-1]]."""
            return self.shared_layers(x)

    class TaskHead(nn.Module):
        """Per-task head: Linear(in, proj_dim) -> Mish -> Dropout -> Linear(proj_dim, 1, no bias)."""
        def __init__(self, input_dim, proj_dim, dropout):
            """No bias on the final layer: CORAL learns the per-cutoff biases instead."""
            super().__init__()

            self.linear = nn.Linear(input_dim, proj_dim)
            self.activation = nn.Mish()
            self.dropout = nn.Dropout(dropout)
            self.final_layer = nn.Linear(proj_dim, 1, bias=False)

        def forward(self, x):
            """Shapes: [ΣW, in] -> [ΣW, 1] (one ordinal score per window)."""
            x = self.linear(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.final_layer(x)
            return x

    def __init__(
        self,
        backbone_dim: int,
        shared_projection_dim: int | list[int],
        tasks: Mapping[Literal["depression", "anxiety"], Mapping[str, Any]],
    ):
        """Build SharedLayers(backbone_dim, shared_projection_dim) and a TaskHead per task config."""
        super().__init__()

        if not isinstance(shared_projection_dim, list):
            # Convert a single value into a 1-length list
            shared_projection_dim = [shared_projection_dim]

        # Initialize the shared network and task-specific networks
        self.shared_layers = self.SharedLayers(backbone_dim, shared_projection_dim)
        self.classifier_head = nn.ModuleDict(
            {
                task: self.TaskHead(shared_projection_dim[-1], **task_config)
                for task, task_config in tasks.items()
            }
        )

    def forward(self, x):
        """Shared layers, then one TaskHead per task -> {task: [ΣW, 1] score}."""
        x = self.shared_layers(x)
        return {task: head(x) for task, head in self.classifier_head.items()}


class ResNetHead(pl.LightningModule):
    """Alternative head: 2-D ResNet over [N, T, D] features, one logit per task (unused)."""
    def __init__(
        self,
        resnet_model: str,
        tasks: list[str],
    ):
        """ResNet from a HF config, randomly initialized, 1 input channel, len(tasks) outputs."""
        super().__init__()
        resnet_config = ResNetConfig.from_pretrained(
            resnet_model, num_labels=len(tasks), num_channels=1
        )
        self.resnet = ResNetForImageClassification(resnet_config)
        self.tasks = tasks

    def forward(self, x):
        """Shapes: [N, T, D] -> {task: [N, 1]}."""
        x = self.resnet(x[:, None, ...]).logits
        return {task: x[..., i : i + 1] for i, task in enumerate(self.tasks)}


class ResNetHead1d(pl.LightningModule):
    """Alternative head: ResNet with feature dim as channels over time (unused)."""
    def __init__(
        self,
        resnet_model: str,
        tasks: list[str],
        backbone_dim: int,
    ):
        """ResNet from a HF config, randomly initialized, backbone_dim channels, len(tasks) outputs."""
        super().__init__()
        resnet_config = ResNetConfig.from_pretrained(
            resnet_model, num_labels=len(tasks), num_channels=backbone_dim
        )
        self.resnet = ResNetForImageClassification(resnet_config)
        self.tasks = tasks

    def forward(self, x):
        """Shapes: [N, T, D] -> transposed to [N, D, T, 1] -> {task: [N, 1]}."""
        x = self.resnet(torch.transpose(x, 1, 2)[..., None]).logits
        return {task: x[..., i : i + 1] for i, task in enumerate(self.tasks)}
