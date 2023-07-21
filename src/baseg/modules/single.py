from typing import Any, Callable

import torch

from baseg.losses import DiceLoss, SoftBCEWithLogitsLoss
from baseg.modules.base import BaseModule


class SingleTaskModule(BaseModule):
    def __init__(
        self,
        config: dict,
        tiler: Callable[..., Any] | None = None,
        predict_callback: Callable[..., Any] | None = None,
        loss: str = "bce",
    ):
        super().__init__(config, tiler, predict_callback)
        if loss == "bce":
            self.criterion_decode = SoftBCEWithLogitsLoss(ignore_index=255, pos_weight=torch.tensor(3.0))
        else:
            self.criterion_decode = DiceLoss(mode="binary", from_logits=True, ignore_index=255)

    def training_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]

        # lc = batch["ESA_LC"]
        # x = torch.cat([x, lc.unsqueeze(1)], dim=1)
        decode_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss = loss_decode

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        for metric_name, metric in self.train_metrics.items():
            metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        # lc = batch["ESA_LC"]
        # x = torch.cat([x, lc.unsqueeze(1)], dim=1)
        decode_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss = loss_decode

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for metric_name, metric in self.val_metrics.items():
            metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        # lc = batch["ESA_LC"]
        # x = torch.cat([x, lc.unsqueeze(1)], dim=1)
        decode_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss = loss_decode

        # self.log("test_loss", loss, on_epoch=True, logger=True)
        for metric_name, metric in self.test_metrics.items():
            metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, logger=True, on_step=False)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        full_image = batch["S2L2A"]

        def callback(batch: Any):
            del_out = self.model(batch)  # [b, 1, h, w]
            return del_out.squeeze(1)  # [b, h, w]

        full_pred = self.tiler(full_image[0], callback=callback)
        batch["pred"] = torch.sigmoid(full_pred)
        return batch

    def on_predict_batch_end(self, outputs: Any | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.predict_callback(batch)
