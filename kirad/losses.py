"""CORAL ordinal-regression loss (stage: loss).

Turns one score per sample into K ordered "is y > k?" binary problems with learned
biases b_k. Instantiated by OrdinalRegressionPLModule in kirad/base_models.py.
"""

from typing import Optional, Sequence

import pytorch_lightning as pl
import torch


class CORALLoss(pl.LightningModule):
    """CORAL loss: score + learned per-cutoff biases, BCE against the thermometer code of y.

    Implements the CORAL loss function for ordinal regression introduced in https://arxiv.org/abs/1901.07884."""

    def __init__(
        self,
        num_classes: int,
        weights: Optional[Sequence[float]] = None,
        *args,
        **kwargs,
    ):
        """One learned bias per cutoff (num_classes - 1, e.g. 27 for PHQ-9) and a 0/1 weight each.

        weights come from loss_target_cutoffs (all 1.0 for "exact28"/"exact22"); default all 1.0.
        """
        super().__init__(*args, **kwargs)
        self.weights = weights or [1.0 for _ in range(num_classes - 1)]
        if len(self.weights) != num_classes - 1:
            raise ValueError(
                f"Got {weights=} with {len(weights)=}; expected len(weights)={num_classes-1=}."
            )
        self.biases = torch.nn.Parameter(
            torch.zeros(len(self.weights), dtype=torch.float)
        )
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x, y):
        """Weighted sum of BCE over cutoffs: sigmoid(s + b_k) vs [y > k] for each cutoff k.

        Shapes: x [N, 1] score, y [N] integer label -> [N] loss per sample (not averaged).
        Note: sum over cutoffs, not mean; the caller averages over samples.
        """
        x_plus_biases = x + self.biases
        weights = torch.tensor(self.weights, device=y.device)
        y_thermometer_code = torch.gt(
            y[..., None],
            torch.arange(len(self.weights), dtype=torch.int, device=y.device),
        )
        return torch.matmul(
            self.loss(input=x_plus_biases, target=y_thermometer_code.float()), weights
        )
