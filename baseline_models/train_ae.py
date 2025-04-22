import os
import random
from typing import Any

import SimpleITK
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from baseline_models.common.dataset import setup_datasets
from datetime import datetime
from monai.networks.nets import AutoEncoder


class AEModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoEncoder(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2, 2),
            num_res_units=1
        )
        if config['loss'] == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif config['loss'] == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['data']
        y_pred = self.model(x)
        loss = self.criterion(y_pred, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        y_pred = self.model(x)
        loss = self.criterion(y_pred, x)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        os.makedirs(f"{self.logger.log_dir}/samples/", exist_ok=True)
        if batch_idx == 0 or batch_idx == 10:
            y_pred = torch.nn.functional.sigmoid(y_pred)
            i = random.randint(0, self.hparams.config['batch_size'] - 1)
            sitk_img = SimpleITK.GetImageFromArray(y_pred[i][0].cpu().float().numpy())
            sitk_img.SetSpacing((1, 1, 2))
            SimpleITK.WriteImage(sitk_img,
                                 f"{self.logger.log_dir}/samples/recon_{self.current_epoch}_{batch_idx}.nii.gz")

            sitk_img = SimpleITK.GetImageFromArray(x[i][0].cpu().float().numpy())
            sitk_img.SetSpacing((1, 1, 2))
            SimpleITK.WriteImage(sitk_img,
                                 f"{self.logger.log_dir}/samples/orign_{self.current_epoch}_{batch_idx}.nii.gz")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.config['lr'], amsgrad=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.hparams.config['lr'],
                                                        epochs=self.hparams.config['epochs'],
                                                        steps_per_epoch=self.hparams.config['steps_per_epoch'])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class CBCTDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dir = img_dir
        self.trn_dataset, self.val_dataset = setup_datasets(self.img_dir, test=False)

    def train_dataloader(self):
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          drop_last=False)


if __name__ == "__main__":
    img_dir = "/home/portafaixa/PycharmProjects/CausalVAE/data/HNC_Survival/"
    batch_size = 32
    num_workers = os.cpu_count()

    data_module = CBCTDataModule(img_dir, batch_size, num_workers)

    config = {
        "batch_size": batch_size,
        "lr": 1e-3,
        "epochs": 100,
        "steps_per_epoch": len(data_module.trn_dataset) // batch_size,
        "loss": "mse"
    }

    print(config)

    torch.set_float32_matmul_precision('medium')
    dt = datetime.now()
    os.makedirs(f'output/ae_training_{dt}', exist_ok=True)

    model = AEModel(config)
    trainer = pl.Trainer(max_epochs=config['epochs'], default_root_dir=f'output/ae_training_{dt}', precision='16-mixed')
    trainer.fit(model, data_module)
