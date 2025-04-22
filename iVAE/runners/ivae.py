import os
import random
import time

import SimpleITK
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import setup_datasets
from metrics import mean_corr_coef as mcc
from models import Discriminator, permute_dims
from models.nets import iVAEforCBCT


def runner(args, config):
    st = time.time()

    print('Executing script on: {}\n'.format(config.device))

    factor = config.gamma > 0
    d_latent = None

    dset, val_dset, _ = setup_datasets("/home/portafaixa/PycharmProjects/CausalVAE/data/HNC_Survival/", test=False)

    loader_params = {'num_workers': 16, 'pin_memory': True} if torch.cuda.is_available() else {}
    trn_dataloader = DataLoader(dset, batch_size=config.batch_size, shuffle=True, drop_last=True, **loader_params)
    val_dataloader = DataLoader(val_dset, batch_size=config.batch_size, shuffle=False, drop_last=True, **loader_params)

    perfs = []
    loss_hists = []
    perf_hists = []

    for seed in range(args.seed, args.seed + args.n_sims):
        model = iVAEforCBCT(latent_dim=512).to(config.device)

        # if config.ica:
        #     model = cleanIVAE(data_dim=d_data, latent_dim=d_latent, aux_dim=d_aux, hidden_dim=config.hidden_dim,
        #                       n_layers=config.n_layers, activation=config.activation, slope=.1).to(config.device)
        # else:
        #     model = cleanVAE(data_dim=d_data, latent_dim=d_latent, hidden_dim=config.hidden_dim,
        #                      n_layers=config.n_layers, activation=config.activation, slope=.1).to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0, verbose=True)

        if factor:
            D = Discriminator(d_latent).to(config.device)
            optim_D = optim.Adam(D.parameters(), lr=config.lr,
                                 betas=(.5, .9))

        loss_hist = []
        perf_hist = []
        for epoch in range(1, config.epochs + 1):
            model.train()

            if config.anneal:
                a = config.a
                d = config.d
                b = config.b
                c = 0
                if epoch > config.epochs / 1.6:
                    b = 1
                    c = 1
                    d = 1
                    a = 2 * config.a
            else:
                a = config.a
                b = config.b
                c = config.c
                d = config.d

            train_loss = 0
            train_perf = 0
            for data in tqdm(trn_dataloader):
                if not factor:
                    x, u, s_true = data
                else:
                    x, x2, u, s_true = data
                x, u = x.to(config.device), u.to(config.device)
                optimizer.zero_grad()
                loss, z = model.elbo(x, u, len(dset), a=a, b=b, c=c, d=d)

                if factor:
                    D_z = D(z)
                    vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
                    loss += config.gamma * vae_tc_loss

                loss.backward(retain_graph=factor)

                train_loss += loss.item()
                try:
                    perf = mcc(s_true.numpy(), z.cpu().detach().numpy())
                except:
                    perf = 0
                train_perf += perf

                optimizer.step()

                if factor:
                    ones = torch.ones(config.batch_size, dtype=torch.long, device=config.device)
                    zeros = torch.zeros(config.batch_size, dtype=torch.long, device=config.device)
                    x_true2 = x2.to(config.device)
                    _, _, _, z_prime = model(x_true2)
                    z_pperm = permute_dims(z_prime).detach()
                    D_z_pperm = D(z_pperm)
                    D_tc_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

                    optim_D.zero_grad()
                    D_tc_loss.backward()
                    optim_D.step()

            train_perf /= len(trn_dataloader)
            perf_hist.append(train_perf)
            train_loss /= len(trn_dataloader)
            loss_hist.append(train_loss)
            print('==> Epoch {}/{}:\ttrain loss: {:.6f}\ttrain perf: {:.6f}'.format(epoch, config.epochs, train_loss,
                                                                                    train_perf))

            # Validation loss
            val_loss = 0
            for data in tqdm(val_dataloader):
                with torch.no_grad():
                    model.eval()
                    x, u, s_true = data
                    x, u = x.to(config.device), u.to(config.device)

                    y_pred = model(x, u)[0]
                    val_loss += torch.nn.functional.mse_loss(y_pred, x)
            print(f"Validation loss (MSE): {val_loss / len(val_dataloader)}")

            # Check the reconstruction loss on the validation set
            os.makedirs("run/samples", exist_ok=True)
            i = random.randint(0, config.batch_size - 1)
            sitk_img = SimpleITK.GetImageFromArray(y_pred[i][0].cpu().float().numpy())
            sitk_img.SetSpacing((1, 1, 2))
            SimpleITK.WriteImage(sitk_img,
                                 f"run/samples/recon_{epoch}_0.nii.gz")

            sitk_img = SimpleITK.GetImageFromArray(x[i][0].cpu().float().numpy())
            sitk_img.SetSpacing((1, 1, 2))
            SimpleITK.WriteImage(sitk_img,
                                 f"run/samples/orign_{epoch}.nii.gz")

            torch.save(model.state_dict(), "run/checkpoints/chkpt.pth")

            if not config.no_scheduler:
                scheduler.step(train_loss)
        print('\ntotal runtime: {}'.format(time.time() - st))

        # evaluate perf on full dataset
        Xt, Ut, St = dset.x.to(config.device), dset.u.to(config.device), dset.s
        if config.ica:
            _, _, _, s, _ = model(Xt, Ut)
        else:
            _, _, _, s = model(Xt)
        full_perf = mcc(dset.s.numpy(), s.cpu().detach().numpy())
        perfs.append(full_perf)
        loss_hists.append(loss_hist)
        perf_hists.append(perf_hist)

    return perfs, loss_hists, perf_hists
