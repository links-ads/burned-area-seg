# original source from https://github.com/qubvel/segmentation_models.pytorch

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class HuberLoss(nn.Module):
    __constants__ = [
        "weight",
        "ignore_index",
    ]

    def __init__(
        self,
        weight: float = 0.1,
        reduction: str = 'mean',
        delta: float = 1.0,
    ):
        """Huber Loss with weights for zero values

        Args:
            @weight: Specifies a weight value for zero pixels
            @reduction: 'mean' or 'sum'
            @delta: Specifies the threshold at which the Huber loss function should change from a quadratic to linear.

        """
        super().__init__()
        self.weight = weight
        self.criterion = nn.HuberLoss(reduction='none', delta=delta)
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """
        
       
        loss = self.criterion(y_pred, y_true)
        
        zeros_y_true = y_true == 0
        loss[zeros_y_true] = loss[zeros_y_true]  * self.weight
        loss[~zeros_y_true] = loss[~zeros_y_true] * abs(1-self.weight)
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
        