import os
import random
from datetime import datetime

import SimpleITK
import pytorch_lightning as pl
import torch
from monai.networks.nets import VarAutoEncoder
from torch.utils.data import DataLoader

from baseline_models.common.dataset import setup_datasets


class VAEModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = VarAutoEncoder(
            spatial_dims=3,
            latent_size=config['latent_size'],
            in_shape=(1, 128, 128, 128),
            out_channels=1,
            channels=(64, 64, 64, 128, 128, 128),  # (64, 64, 64, 128, 128, 128)
            strides=(2, 2, 2, 2, 2, 2),  # (2, 2, 2, 2, 2, 2)
            num_res_units=1,
            use_sigmoid=(self.hparams.config['loss'] == 'mse')
        )

    def loss_function(self, recon_x, x, mu, log_var):
        if self.hparams.config['loss'] == 'bce':
            rec = torch.nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
        elif self.hparams.config['loss'] == 'mse':
            rec = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
        elif self.hparams['config']['loss'] == 'gaussian':
            rec_std = torch.exp(0.5 * log_var)  # instead of the sqrt(exp(log_var))
            normal = torch.distributions.Normal(recon_x, rec_std)
            log_prob = normal.log_prob(x)
            rec = -torch.sum(log_prob)
        else:
            raise NotImplementedError()

        rec /= self.hparams.config['batch_size']
        kld = -0.5 * self.hparams.config['beta'] * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return rec + kld

    def training_step(self, batch, batch_idx):
        x = batch['data']
        recon_batch, mu, log_var, _ = self.model(x)
        loss = self.loss_function(recon_batch, x, mu, log_var)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        recon_batch, mu, log_var, _ = self.model(x)
        loss = self.loss_function(recon_batch, x, mu, log_var)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        os.makedirs(f"{self.logger.log_dir}/samples/", exist_ok=True)
        if batch_idx == 0 or batch_idx == 10:
            y_pred = torch.nn.functional.sigmoid(recon_batch)
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
        d = {'optimizer': torch.optim.AdamW(self.parameters(), lr=self.hparams.config['lr'], amsgrad=True)}

        d['lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(d['optimizer'], verbose=True)
        d['monitor'] = 'train_loss'

        return d


class CBCTDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dir = img_dir
        self.trn_dataset, self.val_dataset, self.test_dataset = setup_datasets(self.img_dir, test=False)

    def train_dataloader(self):
        return DataLoader(self.trn_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          drop_last=False)


def train():
    pl.seed_everything(42)
    # img_dir = "/home/portafaixa/PycharmProjects/CausalVAE/data/HNC_Survival"
    img_dir = "/home/portafaixa/PycharmProjects/CausalVAE/data/HNC_Survival_v2"
    batch_size = 96
    num_workers = os.cpu_count() // 2

    data_module = CBCTDataModule(img_dir, batch_size, num_workers)

    config = {
        "batch_size": batch_size,
        "lr": 1e-4,
        "epochs": 1000,
        "steps_per_epoch": len(data_module.trn_dataset) // batch_size,
        "loss": "mse",
        "beta": 1,
        "latent_size": 512,
        "sched": 'plateau'
    }

    print(config)

    torch.set_float32_matmul_precision('high')

    model = VAEModel(config)
    trainer = pl.Trainer(max_epochs=config['epochs'],
                         default_root_dir=f'output/vae_training_{datetime.now()}',
                         precision='16-mixed',
                         benchmark=True,
                         gradient_clip_val=1.0)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
