from typing import Any, Callable

import torch
from loguru import logger
from torch import nn
from torchmetrics import F1Score, JaccardIndex

from baseg.losses import DiceLoss, SoftBCEWithLogitsLoss
from baseg.modules.base import BaseModule


class MultiTaskModule(BaseModule):
    def __init__(
        self,
        config: dict,
        tiler: Callable[..., Any] | None = None,
        predict_callback: Callable[..., Any] | None = None,
        mask_lc: bool = False,
        loss: str = "bce",
        severity: bool = False,
        img_label: str = "S2L2A",
        gt_label: str = "DEL",
        lc_label: str = "ESA_LC",
        
    ):
        self.severity = severity
        super().__init__(config, tiler, predict_callback)
        
        if self.severity:
            self.criterion_decode = nn.CrossEntropyLoss(ignore_index=255)
        else:
            if loss == "bce":
                self.criterion_decode = SoftBCEWithLogitsLoss(ignore_index=255)
            else:
                self.criterion_decode = DiceLoss(mode="binary", from_logits=True, ignore_index=255)

        self.mask_lc = mask_lc
        self.criterion_auxiliary = nn.CrossEntropyLoss(ignore_index=255)
        self.aux_factor = config.decode_head.pop("aux_factor", 1.0)
        num_classes = config.decode_head.aux_classes
        self.img_label = img_label
        self.gt_label = gt_label
        self.lc_label = lc_label
        logger.info(f"auxiliary factor: {self.aux_factor}")
        logger.info(f"auxiliary classes: {num_classes}")
        
        self.train_metrics_aux = nn.ModuleDict(
            {
                "train_f1_aux": F1Score(task="multiclass", ignore_index=255, num_classes=num_classes, average="macro"),
                "train_iou_aux": JaccardIndex(
                    task="multiclass", ignore_index=255, num_classes=num_classes, average="macro"
                ),
            }
        )
        self.val_metrics_aux = nn.ModuleDict(
            {
                "val_f1_aux": F1Score(task="multiclass", ignore_index=255, num_classes=num_classes, average="macro"),
                "val_iou_aux": JaccardIndex(
                    task="multiclass", ignore_index=255, num_classes=num_classes, average="macro"
                ),
            }
        )
        self.test_metrics_aux = nn.ModuleDict(
            {
                "test_f1_aux": F1Score(task="multiclass", ignore_index=255, num_classes=num_classes, average="macro"),
                "test_iou_aux": JaccardIndex(
                    task="multiclass", ignore_index=255, num_classes=num_classes, average="macro"
                ),
            }
        )

        if self.severity:
            self.train_metrics = nn.ModuleDict(
            {
                "train_f1": F1Score(task="multiclass", ignore_index=255, num_classes=5, average="macro"),
                "train_iou": JaccardIndex(
                    task="multiclass", ignore_index=255, num_classes=5, average="macro"
                ),
            }
            )
            self.val_metrics = nn.ModuleDict(
                {
                    "val_f1": F1Score(task="multiclass", ignore_index=255, num_classes=5, average="macro"),
                    "val_iou": JaccardIndex(
                        task="multiclass", ignore_index=255, num_classes=5, average="macro"
                    ),
                }
            )
            self.test_metrics = nn.ModuleDict(
                {
                    "test_f1": F1Score(task="multiclass", ignore_index=255, num_classes=5, average="macro"),
                    "test_iou": JaccardIndex(
                        task="multiclass", ignore_index=255, num_classes=5, average="macro"
                    ),
                }
            )

    def training_step(self, batch: Any, batch_idx: int):
        x = batch[self.img_label]
        y_del = batch[self.gt_label]
        y_lc = batch[self.lc_label]
        if self.mask_lc:
            y_lc[y_del == 1] = 255
        decode_out, auxiliary_out = self.model(x)
        if self.severity:
            loss_decode = self.criterion_decode(decode_out, y_del.long())
        else:
            loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss_auxiliary = self.criterion_auxiliary(auxiliary_out, y_lc.long())
        loss = loss_decode + self.aux_factor * loss_auxiliary

        self.log("train_loss_del", loss_decode, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_aux", loss_auxiliary, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # compute delineation metrics
        for metric_name, metric in self.train_metrics.items():
            if self.severity:
                metric(decode_out, y_del.long())
            else:
                metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        # compute auxiliary metrics
        for metric_name, metric in self.train_metrics_aux.items():
            metric(auxiliary_out, y_lc.long())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x = batch[self.img_label]
        y_del = batch[self.gt_label]
        y_lc = batch[self.lc_label]
        if self.mask_lc:
            y_lc[y_del == 1] = 255
        decode_out, auxiliary_out = self.model(x)
        if self.severity:
            loss_decode = self.criterion_decode(decode_out, y_del.long())
        else:
            loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss_auxiliary = self.criterion_auxiliary(auxiliary_out, y_lc.long())
        loss = loss_decode + self.aux_factor * loss_auxiliary

        self.log("val_loss_del", loss_decode, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss_aux", loss_auxiliary, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # compute delineation metrics
        for metric_name, metric in self.val_metrics.items():
            if self.severity:
                metric(decode_out, y_del.long())
            else:
                metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        # compute auxiliary metrics
        for metric_name, metric in self.val_metrics_aux.items():
            metric(auxiliary_out, y_lc.long())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        x = batch[self.img_label]
        y_del = batch[self.gt_label]
        # y_lc = batch[self.lc_label]
        # if self.mask_lc:
        #     y_lc[y_del == 1] = 255
        decode_out, auxiliary_out = self.model(x)
        if self.severity:
            loss_decode = self.criterion_decode(decode_out, y_del.long())
        else:
            loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        # loss_auxiliary = self.criterion_auxiliary(auxiliary_out, y_lc.long())
        # loss = loss_decode + self.aux_factor * loss_auxiliary
        loss = loss_decode

        # self.log("test_loss_del", loss_decode, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_loss_aux", loss_auxiliary, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # compute delineation metrics
        for metric_name, metric in self.test_metrics.items():
            if self.severity:
                metric(decode_out, y_del.long())
            else:
                metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True, on_step=False)
        # compute auxiliary metrics
        # for metric_name, metric in self.test_metrics_aux.items():
        #     metric(auxiliary_out, y_lc.long())
        #     self.log(metric_name, metric, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        full_image = batch[self.img_label]

        def callback(batch: Any):
            del_out, _ = self.model(batch)  # [b, 1, h, w]
            if self.severity:
                return del_out.argmax(1).squeeze(1)
            return del_out.squeeze(1)  # [b, h, w]

        full_pred = self.tiler(full_image[0], callback=callback)
        batch["pred"] = torch.sigmoid(full_pred)
        return batch

    def on_predict_batch_end(self, outputs: Any | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.predict_callback(batch)
