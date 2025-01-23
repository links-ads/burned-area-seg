from typing import Any, Callable

import torch
import torch.nn as nn
from torchmetrics import F1Score, JaccardIndex, MeanSquaredError, MeanAbsoluteError

from baseg.losses import DiceLoss, SoftBCEWithLogitsLoss, RMSELoss, SoftCrossEntropyLoss, LogCoshLoss, HuberLoss
from baseg.modules.base import BaseModule


class SingleTaskModule(BaseModule):
    def __init__(
        self,
        config: dict,
        tiler: Callable[..., Any] | None = None,
        predict_callback: Callable[..., Any] | None = None,
        loss: str = "bce",
        severity: bool = False,
        img_label: str = "S2L2A",
        gt_label: str = "DEL"
    ):
        self.severity = severity
        if self.severity:
            task = "multiclass"
            num_classes = 5
        else:
            task = "binary"
            num_classes = None

        super().__init__(config, tiler, predict_callback, task = task, num_classes = num_classes)
        self.img_label = img_label
        self.gt_label = gt_label
        if severity:
            self.criterion_decode = nn.CrossEntropyLoss(ignore_index=255)
            self.train_metrics = nn.ModuleDict(
                {
                    "train_f1": F1Score(task="multiclass", ignore_index=255, num_classes=num_classes, average="macro"),
                    "train_iou": JaccardIndex(
                        task="multiclass", ignore_index=255, num_classes=num_classes, average="macro"
                    ),
                }
            )
            self.val_metrics = nn.ModuleDict(
                {
                    "val_f1": F1Score(task="multiclass", ignore_index=255, num_classes=num_classes, average="macro"),
                    "val_iou": JaccardIndex(
                        task="multiclass", ignore_index=255, num_classes=num_classes, average="macro"
                    ),
                }
            )
            self.test_metrics = nn.ModuleDict(
                {
                    "test_f1": F1Score(task="multiclass", ignore_index=255, num_classes=num_classes, average="macro"),
                    "test_iou": JaccardIndex(
                        task="multiclass", ignore_index=255, num_classes=num_classes, average="macro"
                    ),
                }
            )

        else:
            if loss == "bce":
                self.criterion_decode = SoftBCEWithLogitsLoss(ignore_index=255, pos_weight=torch.tensor(3.0))
            else:
                self.criterion_decode = DiceLoss(mode="binary", from_logits=True, ignore_index=255)

    def training_step(self, batch: Any, batch_idx: int):
        x = batch[self.img_label]
        y_del = batch[self.gt_label]

        # lc = batch["ESA_LC"]
        # x = torch.cat([x, lc.unsqueeze(1)], dim=1)
        decode_out = self.model(x)
        if self.severity:
            loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.long())
        else:
            loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss = loss_decode

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        for metric_name, metric in self.train_metrics.items():
            if self.severity:
                metric(decode_out.squeeze(1), y_del.long())
            else:
                metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x = batch[self.img_label]
        y_del = batch[self.gt_label]
        # lc = batch["ESA_LC"]
        # x = torch.cat([x, lc.unsqueeze(1)], dim=1)
        decode_out = self.model(x)
        
        if self.severity:
            loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.long())
        else:
            loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss = loss_decode

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for metric_name, metric in self.val_metrics.items():
            if self.severity:
                metric(decode_out.squeeze(1), y_del.long())
            else:
                metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        x = batch[self.img_label]
        y_del = batch[self.gt_label]
        # lc = batch["ESA_LC"]
        # x = torch.cat([x, lc.unsqueeze(1)], dim=1)
        decode_out = self.model(x)
        if self.severity:
            loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.long())
        else:
            loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss = loss_decode

        # self.log("test_loss", loss, on_epoch=True, logger=True)
        for metric_name, metric in self.test_metrics.items():
            if self.severity:
                metric(decode_out.squeeze(1), y_del.long())
            else:
                metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, logger=True, on_step=False)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        full_image = batch[self.img_label]

        def callback(batch: Any):
            del_out = self.model(batch)  # [b, 1, h, w]
            if del_out.shape[1] > 1:
                return del_out.argmax(1)
            return del_out.squeeze(1)  # [b, h, w]

        full_pred = self.tiler(full_image[0], callback=callback)
        if self.severity:   
            batch["pred"] = full_pred
        else:
            batch["pred"] = torch.sigmoid(full_pred)
        return batch

    def on_predict_batch_end(self, outputs: Any | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.predict_callback(batch)




class SingleTaskRegressionModule(BaseModule):
    def __init__(
        self,
        config: dict,
        tiler: Callable[..., Any] | None = None,
        predict_callback: Callable[..., Any] | None = None,
        loss: str = "rmse",
        img_label: str = "post",
        gt_label: str = "dNBR"
    ):
    
        super().__init__(config, tiler, predict_callback)
        self.img_label = img_label
        self.gt_label = gt_label
        self.tanh = nn.Tanh()
        self.criterion_decode = HuberLoss(weight=0.1, reduction='mean', delta=1.0)
        # self.criterion_decode = RMSELoss(weight=0.1)
        self.train_metrics = nn.ModuleDict(
            {
                "train_mse": MeanSquaredError(),
                "train_mae": MeanAbsoluteError(),
            }
        )
        self.val_metrics = nn.ModuleDict(
            {
                "val_mse": MeanSquaredError(),
                "val_mae": MeanAbsoluteError(),
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                "test_mse": MeanSquaredError(),
                "test_mae": MeanAbsoluteError(),
            }
        )

    def training_step(self, batch: Any, batch_idx: int):
        x = batch[self.img_label]
        y_reg = batch[self.gt_label].nan_to_num()
        decode_out = self.model(x).squeeze(1).sigmoid()
        loss_decode = self.criterion_decode(decode_out, y_reg.float())
        loss = loss_decode

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        for metric_name, metric in self.train_metrics.items():
            metric(decode_out, y_reg.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x = batch[self.img_label]
        y_reg = batch[self.gt_label].nan_to_num()
        decode_out = self.model(x).squeeze(1).sigmoid()
        loss_decode = self.criterion_decode(decode_out, y_reg.float())
        loss = loss_decode

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for metric_name, metric in self.val_metrics.items():
            metric(decode_out, y_reg.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        x = batch[self.img_label]
        y_reg = batch[self.gt_label].nan_to_num()
        decode_out = self.model(x).squeeze(1).sigmoid()
        loss_decode = self.criterion_decode(decode_out, y_reg.float())
        loss = loss_decode

        for metric_name, metric in self.test_metrics.items():
            metric(decode_out, y_reg.long())
            self.log(metric_name, metric, on_epoch=True, logger=True, on_step=False)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        full_image = batch[self.img_label]

        def callback(batch: Any):
            del_out = self.model(batch)  # [b, 1, h, w]
            if del_out.shape[1] > 1:
                return del_out.argmax(1)
            return del_out.squeeze(1)  # [b, h, w]

        full_pred = self.tiler(full_image[0], callback=callback)
        batch["pred"] = torch.sigmoid(full_pred)
        return batch

    def on_predict_batch_end(self, outputs: Any | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.predict_callback(batch)
