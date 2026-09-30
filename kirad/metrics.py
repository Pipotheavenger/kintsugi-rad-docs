"""Binary Sn/Sp metrics with an indeterminate band, plus a DDP-safe score accumulator.

Stage: thresholds/metrics. base_models gathers val/test scores with MultiCatMetric,
builds an IndetSnSpArray per binary cut, and reports AUROC and sn_eq_sp at each
indeterminate budget (default 0% and 40%).

Contains various metrics used by researchers.
"""
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import auc, confusion_matrix
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from typing_extensions import Self  # in typing in python3.11


def pad_cat(tensors):
    """Zero-pad non-batch dims to the largest shape, then concat on dim 0 (DDP reduce)."""
    shapes = [torch.tensor(t.shape[1:]) for t in tensors]
    padded_shape = torch.stack(shapes, dim=0).max(dim=0).values
    padded_tensors = [
        torch.nn.functional.pad(
            t,
            tuple(
                i
                for pad_dim, t_dim in zip(padded_shape, t.shape[1:])
                for i in (0, pad_dim - t_dim)
            ),
        )
        for t in tensors
    ]
    return torch.cat(padded_tensors, dim=0)


class MultiCatMetric(Metric):
    """Accumulate parallel tensors over batches and ranks (uuids, scores, variances, labels).

    Container metric to concatenate parallel lists of tensors, e.g. corresponding scores and labels.
    Note: base_models uses MultiCatMetric(4); uuids are uint8 tensors, padded by pad_cat.
    """

    is_differentiable = None
    higher_is_better = None
    full_state_update: bool = False

    def __init__(
        self,
        num_states: int = 1,
        **kwargs: Any,
    ) -> None:
        """Register `num_states` list states, reduced across ranks with pad_cat.

        Create a container metric to concatenate parallel lists of tensors, e.g. corresponding scores and labels.

        Arguments
        ---------
        num_states : number of arguments which will be passed to `update` on each call. The i^th arguments from all
            calls to `update` will be concatenated along the batch dimension and the result returned as the i^th
            element of the tuple returned by `compute`.

        """
        super().__init__(**kwargs)

        self.state_names = [f"state_{i}" for i in range(num_states)]
        for state_name in self.state_names:
            self.add_state(state_name, default=[], dist_reduce_fx=pad_cat)

    def update(self, *values: Tensor) -> None:
        """Append one batch of `num_states` tensors.

        Update state with data.

        Args:
            values: tuple `num_states` tensors to accumulate over batches. For fixed `i`, `values[i]` must have the
                same non-batch dimensions and type in all calls to this function, but `values[i]` and `values[j]`
                need not have the same dimensions or type.

        """
        if len(values) != len(self.state_names):
            raise ValueError(
                f"Got {len(values)=}, which does not match num_states={len(self.state_names)}."
            )
        for state_name, value in zip(self.state_names, values):
            getattr(self, state_name).append(value)

    def compute(self) -> tuple[Tensor, ...]:
        """Concatenate each of the `num_states` tensors over all calls to `update`.

        Returns
        -------
            tuple of length `num_states` whose i^th element is the concatenation of the i^th elements of all calls
                to `update`.

        """
        return tuple(
            dim_zero_cat(getattr(self, state_name)) for state_name in self.state_names
        )


def running_argmax_indices(a):
    """Return indices of a where the value is larger than all previous values.

    >>> running_argmax_indices([1, 0, 3, 4, 4, 2, 5, 7, 1])
    array([0, 2, 3, 6, 7])

    """
    m = np.maximum.accumulate(a)
    return np.flatnonzero(np.r_[True, m[:-1] < m[1:]])


def pareto_2d_indices(x, y):
    """Indices of the 2-d Pareto frontier maximizing x and y.

    Compute indices of the Pareto frontier maximizing x and y, sorted in increasing x and decreasing y.

    e.g. the Pareto frontier of the point set below is [A, G]

    B A
     C
    E D
    F  G
     H

    >>> u = [2, 0, 1, 2, 0, 0, 3, 1]
    >>> v = [4, 4, 3, 2, 2, 1, 1, 0]
    >>> pareto_2d_indices(np.array(u), np.array(v))
    array([0, 6])

    """
    sort_indices = np.lexsort((-x, -y))  # last element is primary sort key
    return sort_indices[running_argmax_indices(x[sort_indices])]


def midpoints_with_infs(x):
    """Return the midpoints between the sorted unique elements of x, along with +/-inf."""
    unique_scores = np.unique(np.r_[-np.inf, x, np.inf])
    return (unique_scores[1:] + unique_scores[:-1]) / 2


@dataclass
class IndetSnSpArray:
    """Sn, Sp and indeterminate fraction for many (lower, upper) threshold pairs.

    An array of metrics at different lower and upper threshold values.

    Each member `lower_thresh`, `upper_thresh`, `sn`, `sp`, and `indet_frac` must be a numpy array, and they all must
    have the same shape. Corresponding entries of these arrays specify a pair of thresholds and the metrics when a
    common dataset is evaluated using those thresholds. The thresholding logic is that scores less than the lower
    threshold count as negative outputs, scores greater than or equal to the upper threshold count as positive outputs,
    and scores in between are indeterminate outputs.

    """

    lower_thresh: np.ndarray
    upper_thresh: np.ndarray
    sn: np.ndarray
    sp: np.ndarray
    indet_frac: np.ndarray

    @property
    def min_sn_sp(self):
        """min(Sn, Sp) per threshold pair; the quantity maximized for sn_eq_sp."""
        return np.minimum(self.sn, self.sp)

    @classmethod
    def build(
        cls,
        lower_thresh: Optional[np.ndarray] = None,
        upper_thresh: Optional[np.ndarray] = None,
        *,
        y_true: np.ndarray,
        y_score: np.ndarray,
        weights: Optional[np.ndarray] = None,
        eps: float = 1e-8,
    ) -> Self:
        """Sn/Sp/indet_frac for all threshold pairs (or the given ones) via histogram cumsums.

        Find `IndetSnSpArray` values for given truth and scores as thresholds vary (à la sklearn.metrics.roc_curve).

        The output object contains arrays for `sn`, `sp`, `indet_frac`, `lower_thresh`, and `upper_thresh`, all with the
        same shape. What these arrays contain and what their common shape is depends on the input as follows.

        If both lower_thresh and upper_thresh are provided, they must have the same shape and this method computes
        metrics at the pairs given by corresponding entries in these arrays. The common output shape will be the same as
        this common input shape.

        If only one set of thresholds is provided, this method computes metrics at all sorted pairs of these thresholds
        (along with +/- inf). If neither is provided, sort scores and allow thresholds between each pair (along with
        +/- inf). In both of these cases, the common output shape is a 1-d vector of length equal to the number of such
        pairs.

        """
        weights = weights if weights is not None else np.ones_like(y_true)
        y_true = y_true[weights > 0]
        y_score = y_score[weights > 0]
        weights = weights[weights > 0]

        # Find all threshes and include +/- inf so np.histogram does the right thing
        if lower_thresh is not None and upper_thresh is not None:
            threshes = np.unique(np.r_[-np.inf, lower_thresh, upper_thresh, np.inf])
            lower_indices = np.searchsorted(threshes, lower_thresh)
            upper_indices = np.searchsorted(threshes, upper_thresh)
        else:
            if lower_thresh is not None:
                threshes = np.unique(np.r_[-np.inf, lower_thresh, np.inf])
            elif upper_thresh is not None:
                threshes = np.unique(np.r_[-np.inf, upper_thresh, np.inf])
            else:
                unique_scores = np.unique(np.r_[-np.inf, y_score, np.inf])
                threshes = (unique_scores[1:] + unique_scores[:-1]) / 2
            lower_indices, upper_indices = np.triu_indices(len(threshes))

        count_by_bin = np.histogram(y_score, bins=threshes, weights=weights)[0]
        pos_by_bin = np.histogram(y_score, bins=threshes, weights=y_true * weights)[0]
        count_by_thresh = np.pad(np.cumsum(count_by_bin), (1, 0))
        pos_by_thresh = np.pad(np.cumsum(pos_by_bin), (1, 0))
        tn_plus_fn = count_by_thresh[lower_indices]
        total_minus_tp_minus_fp = count_by_thresh[upper_indices]
        tp_plus_fp = count_by_thresh[-1] - total_minus_tp_minus_fp
        fn = pos_by_thresh[lower_indices]
        total_pos = pos_by_thresh[-1]  # last thresh is +inf
        tp = total_pos - pos_by_thresh[upper_indices]
        fp = tp_plus_fp - tp
        tn = tn_plus_fn - fn
        min_weight = weights.min()
        sn = tp / np.maximum(tp + fn, min_weight)
        sp = tn / np.maximum(tn + fp, min_weight)
        for name, value in (("sensitivity", sn), ("specificity", sp)):
            if value.max() > 1 + eps:
                raise ValueError(
                    f"Numerical precision issues produced invalid value {name} = {value.max()}."
                )
            if value.min() < -eps:
                raise ValueError(
                    f"Numerical precision issues produced invalid value {name} = {value.min()}."
                )
        return cls(
            lower_thresh=threshes[lower_indices],
            upper_thresh=threshes[upper_indices],
            sn=np.clip(sn, 0.0, 1.0),
            sp=np.clip(sp, 0.0, 1.0),
            indet_frac=(total_minus_tp_minus_fp - tn_plus_fn)
            / max(count_by_thresh[-1], min_weight),
        )

    def eval(
        self,
        *,
        y_true,
        y_score,
        weights: Optional[np.ndarray] = None,
    ) -> "IndetSnSpArray":
        """Evaluate the given data on the thresholds of `self` (e.g. val thresholds on test)."""
        return IndetSnSpArray.build(
            lower_thresh=self.lower_thresh,
            upper_thresh=self.upper_thresh,
            y_true=y_true,
            y_score=y_score,
            weights=weights,
        )

    def __getitem__(self, item) -> "IndetSnSpArray":
        """Extract a subarray with numpy-style indexing."""
        return IndetSnSpArray(
            lower_thresh=self.lower_thresh[item],
            upper_thresh=self.upper_thresh[item],
            sn=self.sn[item],
            sp=self.sp[item],
            indet_frac=self.indet_frac[item],
        )

    def roc_curve(self, indet_budget=0.0) -> "IndetRocCurve":
        """Pareto ROC curve of (Sn, Sp) using only pairs with indet_frac <= indet_budget.

        Compute ROC curve with indeterminate budget, sorted by increasing sn and decreasing sp.

        Restrict `self` to Pareto-optimal pairs (sn, sp) for which `indet_frac <= indet_budget`. Other points are worse
        than the points on the curve in the sense of having worse Sn, worse Sp, or not meeting the indeterminate budget.

        """
        within_budget = self[self.indet_frac <= indet_budget]
        frontier = pareto_2d_indices(within_budget.sn, within_budget.sp)
        return IndetRocCurve(**asdict(within_budget[frontier]))

    def sn_eq_sp_graph(self) -> "IndetSnEqSpGraph":
        """Best min(Sn, Sp) per indeterminate fraction; these thresholds are saved in the ckpt.

        Compute sn=sp as a function of indet_frac, returning both sorted in increasing order.

        Method: restrict to Pareto-optimal pairs (s, indet_frac) where s = min(sn, sp).

        Pareto-optimality means that if (s, indet_frac) is in the output, there is no point (sn', sp', indet_frac') in
        the input with indet_frac' <= indet_frac and sn', sp' > s. In other words, s is the maximum value such that
        the quadrant { sn, sp >= s} intersects `self.roc_curve(indet_frac)`. This maximum occurs where the ROC curve
        intersects the diagonal, up to an error bounded by the distance between points on the ROC curve.

        """
        frontier = pareto_2d_indices(self.min_sn_sp, -self.indet_frac)
        return IndetSnEqSpGraph(**asdict(self[frontier]))


class IndetSnSpTensorModule(torch.nn.Module):
    """Buffers holding an IndetSnSpArray (val sn_eq_sp thresholds) in the checkpoint.

    A torch container for the contents of an `IndetSnSpArray` to be saved along with a model's `state_dict`."""

    def __init__(self):
        """One empty buffer per IndetSnSpArray field."""
        super().__init__()
        for field in fields(IndetSnSpArray):
            self.register_buffer(field.name, torch.tensor(()))

    def update(self, array: IndetSnSpArray, device: torch.device):
        """Update the stored values to those given in the array."""
        for key, value in asdict(array).items():
            setattr(self, key, torch.tensor(value, device=device))

    def numpy(self) -> IndetSnSpArray:
        """Return an `IndetSnSpArray` with values taken from the stored tensors."""
        return IndetSnSpArray(
            **{key: value.numpy(force=True) for key, value in self.named_buffers()}
        )


class IndetRocCurve(IndetSnSpArray):
    """ROC curve (Pareto Sn/Sp points) within an indeterminate budget, with thresholds.

    Sn, Sp achievable within some indeterminate budget and associated lower and upper thresholds.

    `sn` is assumed to be sorted in increasing order and `sp` decreasing.

    """

    def sn_eq_sp(self) -> IndetSnSpArray:
        """Locate the point on the ROC curve closest to the diagonal (max min(Sn, Sp))."""
        return self[np.argmax(self.min_sn_sp)]

    def auc(self) -> float:
        """Compute the area under the ROC curve."""
        # `auc` does not automatically include the trivial points (0, 1) and (1, 0)
        # and will underestimate the AUC if these are not explicitly added
        return auc(1 - np.r_[1.0, self.sp, 0.0], np.r_[0.0, self.sn, 1.0])

    @classmethod
    def build(
        cls,
        thresh=None,
        *,
        y_true: np.ndarray,
        y_score: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Self:
        """Plain ROC curve (no indeterminate band) in O(n log n).

        Build an indeterminate=0 ROC curve in n log n time (vs n**2 for IndetSnSpArray.build().roc_curve())."""
        if thresh is None:
            thresh = midpoints_with_infs(y_score)[
                ::-1
            ]  # reverse for proper output sorting
        issa = IndetSnSpArray.build(
            lower_thresh=thresh,
            upper_thresh=thresh,
            y_true=y_true,
            y_score=y_score,
            weights=weights,
        )
        return cls(**asdict(issa))


class IndetSnEqSpGraph(IndetSnSpArray):
    """Best Sn=Sp for each indeterminate budget, with its lower/upper thresholds.

    Sn=Sp achievable as a function of indeterminate budget and associated lower and upper thresholds.

    Both min(self.sn, self.sp) and self.indet_frac are assumed to be sorted in non-decreasing order.

    """

    def at_budget(self, indet_budget: float = 0.0) -> IndetSnSpArray:
        """Locate the best point on the graph within the given budget."""
        return self[np.searchsorted(self.indet_frac, indet_budget, side="right") - 1]


def binary_indet_confusion_matrix(
    scores: np.ndarray, y_true: np.ndarray, thresh_low: float, thresh_high: float
) -> np.ndarray:
    """Flat 2x3 confusion matrix (neg/pos truth x neg/pos/indet output) for one threshold pair.

    Compute flattened confusion matrix for binary ground truth, with indeterminate score range.
    Note: not called inside kirad; standalone analysis helper.

    Parameters
    ----------
    scores: vector of model scores
    y_true: vector of true labels; either boolean or 0/1
    thresh_low: scores below this are mapped to model output "negative"
    thresh_high: scores above this are mapped to model output "positive" (scores between
        thresh_low and thresh_high are mapped to model output "indeterminate")

    Returns
    -------
    Integer vector of length 6 representing flattened confusion matrix as follows:
        (true negatives, false positives, negative ground truth with indeterminate output,
         false negatives, true positives, positive ground truth with indeterminate output)
    suitable for passing to `cms_to_stats` or `bootstrap_confusion_matrix`.

    """
    y_true = y_true.astype(int)
    not_false = np.greater_equal(scores, thresh_low)
    indet = np.logical_and(not_false, np.less(scores, thresh_high))
    # false -> 0, true -> 1, indet -> 2 (so it doesn't match anything in y_true)
    y_pred = not_false.astype(int) + indet.astype(int)
    # row is true label, col is pred, order for each is false, true, indet
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    return cm[:2, :].ravel()


def bootstrap_confusion_matrix(cm: np.ndarray, num_bootstrap: int) -> np.ndarray:
    """Multinomial bootstrap of a flat confusion matrix; returns len(cm) x num_bootstrap.

    Resample from flat confusion matrix with replacement `num_bootstrap` times.

    Parameters
    ----------
    cm: one dimensional vector of non-negative integers containing a flattened
        confusion matrix
    num_bootstrap: number of bootstrap resampled confusion matrices to create

    Returns
    -------
    Integer matrix of shape len(cm) x num_bootstrap with each column giving the
    confusion matrix for an independent bootstrap resample of `cm`.

    """
    n = cm.sum()
    return np.random.multinomial(n, cm / n, size=num_bootstrap).T
