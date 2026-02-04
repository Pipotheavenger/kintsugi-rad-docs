from typing import Optional, Sequence

import pytorch_lightning as pl
import torch


class CORALLoss(pl.LightningModule):
    """Implements the CORAL loss function for ordinal regression introduced in https://arxiv.org/abs/1901.07884."""

    def __init__(
        self,
        num_classes: int,
        weights: Optional[Sequence[float]] = None,
        *args,
        **kwargs,
    ):
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
        x_plus_biases = x + self.biases
        weights = torch.tensor(self.weights, device=y.device)
        y_thermometer_code = torch.gt(
            y[..., None],
            torch.arange(len(self.weights), dtype=torch.int, device=y.device),
        )
        return torch.matmul(
            self.loss(input=x_plus_biases, target=y_thermometer_code.float()), weights
        )
