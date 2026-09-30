"""Tune ordinal thresholds that turn continuous scores into severity classes.

Stage: thresholds/metrics. After each validation epoch base_models tunes six threshold
sets (accuracy, absolute_error, macro_recall, macro_precision, macro_f1 by DP; coral from
the CORAL biases), stores them as buffers in the checkpoint and reuses them on test.
"""
from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence

import torch

from kirad.losses import CORALLoss


class OrdinalThresholding(torch.nn.Module):
    """Base module: K-1 sorted thresholds (buffer) that map scores to K ordinal classes.

    Basic 1d thresholding logic.
    """

    def __init__(self, num_classes: int):
        """Create num_classes - 1 zero thresholds as a buffer (saved in the checkpoint).

        Init thresholding module with the specified number of classes (one more than the number of thresholds)."""
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("thresholds", torch.zeros(num_classes - 1))
        self.thresholds: torch.Tensor

    def is_valid(self) -> bool:
        """Check whether the thresholds are monotone non-decreasing."""
        return all(torch.greater_equal(self.thresholds[1:], self.thresholds[:-1]))

    def forward(self, scores) -> torch.Tensor:
        """Class index per score = number of thresholds strictly below it (searchsorted).

        Find which thresholds each score lies between.
        Note: a score equal to a threshold falls in the lower class.
        """
        return torch.searchsorted(self.thresholds, scores)

    def tune_thresholds(
        self,
        *,
        scores: torch.Tensor,
        labels: torch.Tensor,
        available_thresholds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tune thresholds on validation scores/labels; no-op here, overridden by subclasses.

        Adapt the thresholds to the given data.

        This is essentially an abstract method, but for testing purposes it's helpful to be able to instantiate the
        class with a no-op version.

        Parameters
        ----------
        scores : a vector of `float` scores for each example in the validation set
        labels : a vector of `int` labels having the same shape as `scores` containing the corresponding labels
        available_thresholds : a vector of `float` score values over which to optimize choice of thresholds;
            `None`, then thresholds between every score in the validation set are allowed. +/- inf are always allowed.

        Returns
        -------
        scalar `float` mean cost on the validation set using optimal thresholds

        """


class CORALThresholding(OrdinalThresholding):
    """Thresholds = -CORAL bias at the chosen cutoff indices (learned on train, no tuning).

    Thresholding using the biases learned on the TRAINING set from the CORAL loss.
    """

    def __init__(self, coral_loss: CORALLoss, threshold_indices: Sequence[int]):
        """Keep a reference to the CORAL biases and the cutoff indices to read.

        Init module taking thresholds at specified indices from a given CORALLoss instance
        """
        super().__init__(len(threshold_indices) + 1)
        self.biases = coral_loss.biases
        self.threshold_indices = torch.tensor(threshold_indices)

    def tune_thresholds(
        self,
        *,
        scores: torch.Tensor,
        labels: torch.Tensor,
        available_thresholds: Optional[torch.Tensor] = None,
    ):
        """Copy -biases[threshold_indices] from the CORAL loss; scores and labels are ignored.

        Ignore the data, pull the specified biases from the CORAL loss, and multiply by -1 to get thresholds.
        """
        self.thresholds[:] = -torch.gather(
            self.biases, 0, self.threshold_indices.to(self.biases.device)
        )


class OptimalOrdinalThresholdingViaDynamicProgramming(OrdinalThresholding, ABC):
    """Base for DP threshold tuning: pick K-1 thresholds from candidates to optimize a metric.

    Super-class for general dynamic programming implementations of ordinal threshold tuning.

    Subclasses implement different ways of computing the mean cost and corresponding DP step.

    """

    direction: Literal["min", "max"]  # provided by subclasses

    def __init__(self, num_classes: int):
        """Check that the subclass set `direction` to "min" or "max"."""
        super().__init__(num_classes=num_classes)
        if self.direction not in ("min", "max"):
            raise ValueError(
                f"Got direction {self.direction!r}, expected 'min' or 'max'."
            )

    @abstractmethod
    def mean_cost(
        self, *, labels: torch.Tensor, preds: int | torch.Tensor
    ) -> torch.Tensor:
        """Compute the mean cost of assigning label(s) `preds` when the ground truth is `labels`."""

    def best_constant_output_classifier(self, labels: torch.Tensor):
        """Best metric of a constant-class predictor and that class (baseline_const_* in logs).

        Find the optimal mean cost of a constant-output classifier for given `labels` and the associated constant.
        """
        if self.direction == "min":
            optimize = torch.min
        else:
            optimize = torch.max
        optimum = optimize(
            torch.tensor(
                [
                    self.mean_cost(labels=labels, preds=c)
                    for c in range(self.num_classes)
                ],
                device=labels.device,
            ),
            0,
        )
        return optimum.values, optimum.indices

    @abstractmethod
    def dp_step(
        self,
        c_idx: int,
        *,
        scores: torch.Tensor,
        labels: torch.Tensor,
        available_thresholds: torch.Tensor,
        prev_cost: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, Optional[torch.Tensor]):
        """One DP step: best cost of classes <= c_idx for each candidate upper threshold of c_idx.

        Given optimal cost `prev_cost` of classes < `c_idx`, optimize cost of `c_idx` as a function of threshold.

        Arguments
        ---------
            c_idx : current class index
            scores, labels, available_thresholds : see `tune_thresholds`
            prev_cost (optional float tensor) : optimal cost of classes < `c_idx` as a function of upper threshold
                for class `c_idx - 1`; ignored if `c_idx == 0`

        Returns
        -------
            cost: `cost[i]` is for choosing upper threshold of class `c_idx` equal to `available_thresholds[i]`
                when thresholds for lower classes are chosen optimally
            indices : to achieve `cost[i]`, optimal upper threshold for class `c_idx - 1` is
                `available_thresholds[indices[i]]`; `None` if `c_idx == 0`

        """

    def tune_thresholds(
        self,
        *,
        scores: torch.Tensor,
        labels: torch.Tensor,
        available_thresholds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Set thresholds by DP over midpoints of unique scores (+/-inf); return the best metric.

        Set `self.thresholds` to optimize mean cost of given `scores` and `labels`.

        Arguments
        ---------
            scores (1d float tensor) : scores of examples on tuning dataset
            labels (1d int tensor) : labels in {0, ..., self.num_classes - 1} of same shape as scores
            available_thresholds (optional 1d float tensor) : thresholds which will be considered when tuning.
                +/-inf will be added automatically to ensure all examples are classified. If omitted, will
                insert thresholds between each element of sorted(unique(scores)).

        Returns
        -------
            float tensor : optimal mean cost achieved on the provided dataset at the tuned `self.thresholds`

        """
        inf = torch.tensor([torch.inf], device=scores.device)
        if available_thresholds is None:  # use all possible thresholds
            unique_scores = torch.unique(scores)
            available_thresholds = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        # Always allow some classes to be omitted entirely by setting thresholds to +/- inf.
        # This simplifies the algorithm and also guarantees that the baseline constant-output
        # classifiers are feasible choices for tuning, which is needed to assure that the
        # optimum is at least as good as a constant-output classifier.
        available_thresholds = torch.concatenate(
            [
                -inf,
                available_thresholds,
                inf,
            ]
        )
        indices = torch.empty(
            (self.num_classes - 2, len(available_thresholds)),
            dtype=torch.int,
            device=scores.device,
        )

        # cost[j] = optimal total cost of items assigned pred <= c if the
        # threshold between class c and c+1 is available_thresholds[j] (by appropriate choice of lower thresholds).
        cost, _ = self.dp_step(
            c_idx=0,
            scores=scores,
            labels=labels,
            available_thresholds=available_thresholds,
        )
        for c in range(1, self.num_classes - 1):
            cost, indices[c - 1, :] = self.dp_step(
                c_idx=c,
                scores=scores,
                labels=labels,
                available_thresholds=available_thresholds,
                prev_cost=cost,
            )
        cost, best_index = self.dp_step(
            c_idx=self.num_classes - 1,
            scores=scores,
            labels=labels,
            available_thresholds=available_thresholds,
            prev_cost=cost,
        )
        if self.direction == "min":
            cost *= -1

        # Follow DP path backwards to find thresholds which optimized cost
        self.thresholds[self.num_classes - 2] = available_thresholds[
            best_index
        ]  # final threshold
        for c in range(self.num_classes - 3, -1, -1):  # counting down to zero
            best_index = indices[c, best_index.long()]
            self.thresholds[c] = available_thresholds[best_index.long()]

        return cost


def cumsum_with_0(t: torch.Tensor):
    """Cumulative sum along dim 0 with a leading 0 (length n + 1)."""
    return torch.nn.functional.pad(torch.cumsum(t, dim=0), (1, 0))


class OptimalCostPerSampleOrdinalThresholding(
    OptimalOrdinalThresholdingViaDynamicProgramming, ABC
):
    """DP tuning for metrics that are a mean of per-sample costs (accuracy, MAE, macro recall).

    Optimal 1d thresholding based on tuning thresholds to optimize the mean of a sample-wise cost function.
    """

    @abstractmethod
    def cost(self, *, labels: torch.Tensor, preds: int | torch.Tensor):
        """Per-sample cost of predicting `preds` for `labels` (subclasses define the metric).

        Compute the sample-wise cost of assigning label(s) `preds` when the ground truth is `labels`."""

    def mean_cost(
        self, *, labels: torch.Tensor, preds: int | torch.Tensor
    ) -> torch.Tensor:
        """Compute the mean cost of assigning label(s) `preds` when the ground truth is `labels`."""
        return torch.mean(self.cost(labels=labels, preds=preds))

    def dp_step(
        self,
        c_idx: int,
        *,
        scores: torch.Tensor,
        labels: torch.Tensor,
        available_thresholds: torch.Tensor,
        prev_cost: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, Optional[torch.Tensor]):
        """O(n) DP step: histogram per-sample costs over candidates, cumsum, running max.

        O(len(scores)) implementation for per-sample cost.
        Note: torch.histogram runs on CPU (no CUDA kernel); tensors are moved back and forth.
        """
        # Compute running_cost[i] = sum of costs of elements with score less than available_thresholds[i] if assigned label c
        item_costs = self.cost(labels=labels, preds=c_idx) / len(scores)
        if self.direction == "min":
            item_costs *= -1
        # move tensors to and from CPU because histogram has no CUDA implementation
        cost_new_class_by_thresh, _ = torch.histogram(
            scores.cpu().float(),
            weight=item_costs.cpu().float(),
            bins=available_thresholds.cpu().float(),
        )
        running_cost = cumsum_with_0(cost_new_class_by_thresh.to(labels.device))

        # Combine with running_cost with prev_cost
        if c_idx == 0:
            return running_cost, None
        diff = prev_cost - running_cost
        cummax = torch.cummax(diff, dim=0)
        cost = running_cost + cummax.values
        if c_idx == self.num_classes - 1:
            # -1 to always set the *upper* threshold for class `num_classes - 1` to include the rest of the data
            return cost[-1], cummax.indices[-1]
        return cost, cummax.indices


class MaxAccuracyOrdinalThresholding(OptimalCostPerSampleOrdinalThresholding):
    """Threshold to maximize accuracy."""

    direction = "max"

    def cost(self, *, labels: torch.Tensor, preds: int | torch.Tensor):
        """1.0 where pred == label, else 0.0."""
        return torch.eq(labels, preds).float()


class MaxMacroRecallOrdinalThresholding(OptimalCostPerSampleOrdinalThresholding):
    """Threshold to maximize macro-averaged recall."""

    direction = "max"

    def cost(self, *, labels: torch.Tensor, preds: int | torch.Tensor):
        """Correct = N / (K * count[label]), so the mean equals macro-averaged recall."""
        counts = torch.bincount(labels, minlength=self.num_classes).float()
        ratios = counts.sum() / (self.num_classes * counts)
        return torch.eq(labels, preds).float() * torch.gather(
            ratios, 0, labels.type(torch.int64)
        )


class MinAbsoluteErrorOrdinalThresholding(OptimalCostPerSampleOrdinalThresholding):
    """Threshold to minimize mean absolute error."""

    direction = "min"

    def cost(self, *, labels: torch.Tensor, preds: int | torch.Tensor):
        """|pred - label| in class units."""
        return torch.abs(preds - labels).float()


class ClassWeightedOptimalCostPerSampleOrdinalThresholding(
    OptimalCostPerSampleOrdinalThresholding
):
    """Wrap a per-sample cost so each class weighs equally; not used by base_models.

    Compute cost weighted equally over classes instead of equally over samples.

    This class takes another instance of OptimalCostPerSampleOrdinalThresholding
    which computes its cost independently for each sample and reweights the cost
    based on label frequencies.

    Note: this class depends on an implementation detail of its superclass:
      namely calling `self.cost` with the full tuning or eval set of labels,
      rather than a single label. This is required to do the re-weighting properly.

    """

    def __init__(self, unweighted_instance: OptimalCostPerSampleOrdinalThresholding):
        """Copy direction and num_classes from the wrapped instance."""
        self.direction = unweighted_instance.direction
        super().__init__(unweighted_instance.num_classes)
        self.unweighted_instance = unweighted_instance

    def cost(self, *, labels: torch.Tensor, preds: int | torch.Tensor):
        """Wrapped cost times N / (K * count[label]); raises if any class is missing."""
        counts = torch.bincount(labels, minlength=self.num_classes)
        (indices,) = torch.where(counts == 0)
        if len(indices) > 0:
            raise ValueError(
                f"Cannot compute class-weighted cost because classes {set(indices.tolist())} are missing."
            )
        unweighted_cost = self.unweighted_instance.cost(labels=labels, preds=preds)
        weights = len(labels) / (self.num_classes * counts[labels].float())
        return weights * unweighted_cost


class OptimalCostPerClassOrdinalThresholding(
    OptimalOrdinalThresholdingViaDynamicProgramming, ABC
):
    """DP tuning for per-class metrics (macro precision, macro F1) via a T x T cost matrix.

    General DP case for when the linear algorithm for per-sample costs is not applicable.

    Complexity depends on the implementation of `cost_matrix`.

    """

    @abstractmethod
    def cost_matrix(
        self,
        c_idx: int,
        *,
        scores: torch.Tensor,
        labels: torch.Tensor,
        available_thresholds: torch.Tensor,
        start: bool,
        end: bool,
    ) -> torch.Tensor:
        """Cost matrix [T, T]: metric of class c_idx if scores in [thr[i], thr[j]) get c_idx.

        Each output[i, j] = cost for when scores in range `available_thresholds[i:j]` are assigned label `c_idx`."""

    def mean_cost(
        self, *, labels: torch.Tensor, preds: int | torch.Tensor
    ) -> torch.Tensor:
        """Macro metric of fixed `preds`, reusing cost_matrix with thresholds c +/- 0.5.

        Compute the mean cost of assigning label(s) `preds` when the ground truth is `labels`.
        """

        if isinstance(preds, int) or preds.numel() == 1:
            preds = preds * torch.ones_like(labels, dtype=torch.int)

        total_cost = torch.tensor(0.0, device=labels.device)
        for c_idx in range(self.num_classes):
            thresholds = torch.tensor([c_idx - 0.5, c_idx + 0.5], device=labels.device)
            total_cost += self.cost_matrix(
                c_idx, preds.float(), labels, thresholds, start=True, end=True
            )[0, 0]
        return total_cost / self.num_classes

    def dp_step(
        self,
        c_idx: int,
        *,
        scores: torch.Tensor,
        labels: torch.Tensor,
        available_thresholds: torch.Tensor,
        prev_cost: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, Optional[torch.Tensor]):
        """O(T^2) DP step: add prev_cost to each lower-threshold row and max over it.

        T = number of candidate thresholds. Cost is divided by K, giving a macro average.
        """
        cost_matrix = (
            self.cost_matrix(
                c_idx,
                scores=scores,
                labels=labels,
                available_thresholds=available_thresholds,
                start=c_idx == 0,
                end=c_idx == self.num_classes - 1,
            )
            / self.num_classes
        )
        if self.direction == "min":
            cost_matrix *= -1
        if prev_cost is not None:
            cost_matrix += prev_cost[:, None]
        max_ = torch.max(cost_matrix, dim=0)
        return max_.values, max_.indices


def _compute_metrics_matrices(
    scores: torch.Tensor,
    binary_labels: torch.Tensor,
    thresholds: torch.Tensor,
    start: bool = False,
    end: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TP and predicted-positive counts for every [thresholds[i], thresholds[j]) range.

    Each output[i, j] = stats for when scores between thresholds[i] and thresolds[j] are assigned `True`.

    Helper function for `MaxMacroPrecisionOrdinalThresholding` and `MaxMacroF1OrdinalThresholding`

    Computed in O(len(thresholds)**2 + len(scores)*log(len(thresholds))) operations instead of the naive
    O(len(scores)*len(thresholds)**2) operations to compute each element of the output independently.

    Arguments
    ---------
        scores (float Tensor) : scores of labeled examples for which to compute metrics
        binary_labels (bool Tensor) : corresponding binary labels of same shape as `scores`
        thresholds (float Tensor) : thresholds between which to compute metrics
        start : compute only the first row of the output (lower threshold at its minimum value)
        end : compute only the last column of the output (upper threshold at its maximum value)

    Returns
    -------
        tp : tp[i, j] = number of true positives if scores between thresholds[i:j] are classified as positive
        tp_plus_fp: tp_plus_fp[i, j] = number of scores between thresholds[i:j]

    """
    # move tensors to and from CPU because histogram has no CUDA implementation
    scores = scores.float().cpu()
    thresholds = thresholds.float().cpu()
    labeled_true_by_thresh, _ = torch.histogram(
        scores,
        weight=binary_labels.float().cpu(),
        bins=thresholds,
    )
    count_by_thresh, _ = torch.histogram(
        scores,
        bins=thresholds,
    )
    running_labeled_true_by_thresh = cumsum_with_0(
        labeled_true_by_thresh.to(binary_labels.device)
    )
    running_count_by_thresh = cumsum_with_0(
        count_by_thresh.to(binary_labels.device).float()
    )

    def start_slice(t):
        """Lower-threshold axis as rows (only row 0 when `start`)."""
        return t[: (1 if start else None), None]

    def end_slice(t):
        """Upper-threshold axis as columns (only the last column when `end`)."""
        return t[None, (-1 if end else None) :]

    tp = end_slice(running_labeled_true_by_thresh) - start_slice(
        running_labeled_true_by_thresh
    )
    tp_plus_fp = end_slice(running_count_by_thresh) - start_slice(
        running_count_by_thresh
    )
    return tp, tp_plus_fp


class MaxMacroPrecisionOrdinalThresholding(OptimalCostPerClassOrdinalThresholding):
    """Threshold to maximize macro-averaged precision."""

    direction = "max"

    def cost_matrix(
        self,
        c_idx: int,
        scores: torch.Tensor,
        labels: torch.Tensor,
        available_thresholds: torch.Tensor,
        start: bool,
        end: bool,
    ) -> torch.Tensor:
        """Precision of class c_idx for each (lower, upper) threshold pair; empty range = 0."""
        tp, tp_plus_fp = _compute_metrics_matrices(
            scores, torch.eq(labels, c_idx), available_thresholds, start=start, end=end
        )
        safe_tp_plus_fp = torch.maximum(
            tp_plus_fp, torch.ones(1, device=tp_plus_fp.device)
        )
        return torch.where(torch.ge(tp_plus_fp, 0.0), tp / safe_tp_plus_fp, -torch.inf)


class MaxMacroF1OrdinalThresholding(OptimalCostPerClassOrdinalThresholding):
    """Threshold to maximize macro-averaged F1 score."""

    direction = "max"

    def cost_matrix(
        self,
        c_idx: int,
        scores: torch.Tensor,
        labels: torch.Tensor,
        available_thresholds: torch.Tensor,
        start: bool,
        end: bool,
    ) -> torch.Tensor:
        """F1 of class c_idx = 2TP / (TP+FP + TP+FN) for each (lower, upper) threshold pair."""
        tp, tp_plus_fp = _compute_metrics_matrices(
            scores, torch.eq(labels, c_idx), available_thresholds, start=start, end=end
        )
        tp_plus_fn = torch.eq(labels, c_idx).float().sum()  # scalar
        safe_tp_plus_fp = torch.maximum(
            tp_plus_fp, torch.ones(1, device=tp_plus_fp.device)
        )
        return torch.where(
            torch.ge(tp_plus_fp, 0.0),
            2 * tp / (safe_tp_plus_fp + tp_plus_fn),
            -torch.inf,
        )
