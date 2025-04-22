import glob
import itertools
import os

import numpy as np
import torch
import SimpleITK as sitk

from torch.utils.data import Dataset
from monai import transforms as T


class CBCTDataset(Dataset):
    def __init__(self, split_keys, eval):
        self.eval = eval
        files = [glob.glob(f"{key}/cbct_raw/*.nii.gz") for key in split_keys]
        self.data_map = list(itertools.chain(*files))
        self.trn_transforms = T.Compose([
            # T.Resize((64, 128, 128), mode='area', anti_aliasing=False),
            T.RandFlip(spatial_axis=(1, 2), prob=0.5),
        ])

        self.cases_max_time_idx = self.find_max_x_in_folders(split_keys)

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        fpath = self.data_map[idx]
        vol = sitk.GetArrayFromImage(sitk.ReadImage(fpath))
        vol = torch.FloatTensor(vol).unsqueeze(0)
        time_idx = int(fpath.split("/")[-1].split(".")[1])
        max_time_idx = self.cases_max_time_idx[os.path.normpath(os.path.dirname(fpath))]

        if not self.eval:
            vol = self.trn_transforms(vol)

        X = torch.clip(vol, 0, 1)
        U = torch.FloatTensor([time_idx / max_time_idx])
        S = X

        return X, U, S

    @staticmethod
    def find_max_x_in_folders(case_folders):
        max_dict = {}
        for case_path in case_folders:
            cbct_path = os.path.join(case_path, 'cbct_raw')
            if os.path.exists(cbct_path):
                files = [f for f in os.listdir(cbct_path) if f.startswith('cbct.') and f.endswith('.nii.gz')]
                max_x = max([int(f.split('.')[1]) for f in files], default=-1)
                if max_x != -1:
                    case_folder_name = cbct_path
                    max_dict[case_folder_name] = 30 if max_x < 30 else max_x
        return max_dict


def setup_datasets(img_dir, test=False):
    patients = sorted(glob.glob(f"{img_dir}/*/"))

    np.random.shuffle(patients)
    if not test:
        trn_keys = patients[:int(0.8 * len(patients))]
        val_keys = patients[int(0.8 * len(patients)):int(0.9 * len(patients))]
        tst_keys = patients[int(0.9 * len(patients)):]
    else:
        trn_keys = patients[:int(0.05 * len(patients))]
        val_keys = patients[int(0.05 * len(patients)):int(0.075 * len(patients))]
        tst_keys = patients[int(0.075 * len(patients)):int(0.1 * len(patients))]

    trn_dataset = CBCTDataset(trn_keys, eval=False)
    val_dataset = CBCTDataset(val_keys, eval=True)
    tst_dataset = CBCTDataset(tst_keys, eval=True)

    return trn_dataset, val_dataset, tst_dataset


if __name__ == "__main__":
    trn_datasets, _, _ = setup_datasets("/home/portafaixa/PycharmProjects/CausalVAE/data/HNC_Survival/")

    for i in range(len(trn_datasets)):
        ds = trn_datasets[i]
        print(ds[0].shape, ds[1].shape)

        if i > 10:
            break
