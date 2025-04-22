import glob
import itertools
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

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        fpath = self.data_map[idx]
        vol = sitk.GetArrayFromImage(sitk.ReadImage(fpath))
        vol = torch.FloatTensor(vol).unsqueeze(0)

        if not self.eval:
            vol = self.trn_transforms(vol)

        return {"data": torch.clip(vol, 0, 1)}


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


# if __name__ == "__main__":
#     trn_datasets, _ = setup_datasets("/store/datasets/HNC_Survival/")
#
#     os.makedirs("samples", exist_ok=True)
#     for i in range(len(trn_datasets)):
#         print(np.min(trn_datasets[i]['data'].numpy()), np.max(trn_datasets[i]['data'].numpy()))
#         img = sitk.GetImageFromArray(trn_datasets[i]['data'][0].numpy())
#         img.SetSpacing((1., 1., 2.))
#         sitk.WriteImage(img, f"samples/{i:04d}.nii.gz")
#
#         if i > 10:
#             break
