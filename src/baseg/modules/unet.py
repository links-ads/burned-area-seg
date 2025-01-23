from typing import Any, Callable
from frozenlist import FrozenList
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from baseg.losses.huber import HuberLoss
from baseg.losses.rmse import RMSELoss


class UnetModel(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = True
        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels),
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                else:
                    self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)
            
        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


class UnetModule(pl.LightningModule):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        tiler: Callable[..., Any] | None = None,
        predict_callback: Callable[..., Any] | None = None,
        loss: str = "rmse",
        img_label: str = "post",
        gt_label: str = "dNBR"
    ):

        super().__init__()
        
        self.model = UnetModel(n_channels=n_channels, n_classes=n_classes)
        self.tiler = tiler
        self.predict_callback = predict_callback
        self.img_label = img_label
        self.gt_label = gt_label
        self.tanh = nn.Tanh()
        assert loss in ["rmse", "huber"]
        if loss == "huber":
            self.criterion_decode = HuberLoss(weight=0.1, reduction='mean', delta=1.0)
        else:
            self.criterion_decode = RMSELoss(weight=0.1)
        
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

    def configure_optimizers(self) -> Any:
        return AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
