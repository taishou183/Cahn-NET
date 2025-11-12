import torch
import os
import torch.optim as optim
import torch.nn as nn
from torch.nn import utils
import torch.nn.functional as f

from data_utils import real_Dataset, RESIDE_Dataset, outRESIDE_Dataset
from torch.utils.data import DataLoader
from option import opt
import numpy as np
import random

import lightning.pytorch as pl
from model.cahnet.cahnet import CahnNet

from C2R import C2R
from utils.schedulers import CosineAnnealingRestartCyclicLR


class ChanNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = CahnNet()
        self.criterion = nn.L1Loss()
        self.MSELoss = nn.MSELoss(reduction='mean')
        self.clcr_criterion = C2R()
        self.save_hyperparameters()
        self.lr = 0

        self.losses = []
        self.val_losses = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        input_img, label_img, _ = batch

        out = self(input_img)

        # label_img2 = f.interpolate(label_img, scale_factor=0.5, mode='bilinear')
        # label_img4 = f.interpolate(label_img, scale_factor=0.25, mode='bilinear')

        # l1 = self.criterion(out[0], label_img4)
        # l2 = self.criterion(out[1], label_img2)
        # l3 = self.criterion(out[2], label_img)
        # loss_content = l1 + l2 + l3

        # ---------- FFT 损失 ----------
        # def make_fft_pair(pred, target):
        #     fft_pred = torch.fft.fft2(pred, dim=(-2, -1))
        #     fft_target = torch.fft.fft2(target, dim=(-2, -1))
        #     return torch.stack((fft_pred.real, fft_pred.imag), -1), \
        #         torch.stack((fft_target.real, fft_target.imag), -1)
        #
        # pred_fft1, label_fft1 = make_fft_pair(out[0], label_img4)
        # pred_fft2, label_fft2 = make_fft_pair(out[1], label_img2)
        # pred_fft3, label_fft3 = make_fft_pair(out[2], label_img)
        #
        # f1 = self.criterion(pred_fft1, label_fft1)
        # f2 = self.criterion(pred_fft2, label_fft2)
        # f3 = self.criterion(pred_fft3, label_fft3)
        # loss_fft = f1 + f2 + f3
        #
        # pixel_loss = loss_content + 0.1 * loss_fft
        # loss2 = 0
        # if opt.clcrloss:
        #     loss2 = self.clcr_criterion(out, y, x)
        # loss = pixel_loss + opt.loss_weight * loss2
        loss = self.criterion(out, label_img)

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        self.losses.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        factor = 4
        degrad_patch, clean_patch, _ = batch

        h, w = degrad_patch.shape[2], degrad_patch.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        degrad_patch = f.pad(degrad_patch, (0, padw, 0, padh), 'reflect')

        # restored = self(degrad_patch)[2]
        restored = self(degrad_patch)

        restored = restored[:, :, :h, :w]
        mse = self.MSELoss(restored, clean_patch)
        val_loss = 10 * torch.log10(1 / mse)
        # val_loss = self.criterion(restored, clean_patch)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)

        self.val_losses.append(val_loss.item())

        return val_loss

    def lr_scheduler_step(self, scheduler, metric):
        if self.trainer.is_global_zero:
            before = scheduler.optimizer.param_groups[0]["lr"]
        scheduler.step()
        if self.trainer.is_global_zero:
            after = scheduler.optimizer.param_groups[0]["lr"]
            print(f"Epoch {self.current_epoch}: {before:.7e} → {after:.7e}")
            step = self.current_epoch
            self.logger.experiment.add_scalar("lr", after, step)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=2e-4,
                                betas=(0.9, 0.999),
                                weight_decay=1e-4)
        scheduler = CosineAnnealingRestartCyclicLR(
            optimizer=optimizer,
            periods=[200, 50, 50],
            restart_weights=[1, 0.5, 0.5],
            eta_mins=[1e-6, 1e-6, 1e-7],
            last_epoch=-1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_after_backward(self):
        if opt.clip:
            utils.clip_grad_norm_(self.parameters(), 0.2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    seed_value = 1949
    set_seed(seed_value)

    device = torch.device('cuda:0')
    map_location = torch.device('cpu')

    net = ChanNetModel()

    train_loader = DataLoader(
        dataset=RESIDE_Dataset(os.path.join('/media/StudentGroup/LZ/Dataset/', 'Haze4K/train/'), train=True,
                               size=256,
                               format='.png'),
        batch_size=4, shuffle=True, num_workers=8)

    val_loader = DataLoader(
        dataset=RESIDE_Dataset(os.path.join('/media/StudentGroup/LZ/Dataset/', 'Haze4K/test/'), train=False,
                               size=400,
                               format='.png'),
        batch_size=8, shuffle=False, num_workers=8)


    # 配置 Trainer
    trainer = pl.Trainer(
        # limit_train_batches=1,
        # limit_train_batches=0.5,
        # limit_val_batches=0.1,
        num_sanity_val_steps=2,
        max_epochs=300,
        # precision=16,
        accelerator="gpu",
        devices=2,
        strategy='ddp',
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath='./trained_models/ckpt',
            filename="haze-{epoch}-{step}-{loss:.7f}",
            every_n_epochs=1,
            save_top_k=-1,
        )
        ],
        log_every_n_steps=10
    )

    with torch.autograd.set_detect_anomaly(True):
        trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)
