# original source from https://github.com/qubvel/segmentation_models.pytorch

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class RMSELoss(nn.Module):
    __constants__ = [
        "weight",
        "ignore_index",
    ]

    def __init__(
        self,
        weight: float = 0.1,
    ):
        """Root MSE Loss with weight for zero pixels

        Args:
            weight: Specifies a weight value for zero pixels

        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        super().__init__()
        self.weight = weight
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """
        zeros_y_true = y_true == 0
        mse = self.criterion(y_pred, y_true)
        mse[zeros_y_true] = mse[zeros_y_true] * self.weight
        mse[~zeros_y_true] = mse[~zeros_y_true] * abs(1-self.weight)
        mse = mse.mean()
        return torch.sqrt(mse)