import glob
import os

import SimpleITK as sitk
import h5py
import itk
import numpy as np
from tqdm import tqdm


def crop(image):
    z = 96 - image.shape[0]
    w1, w2 = np.ceil(np.abs((128 - image.shape[1]) / 2)), np.floor(np.abs((128 - image.shape[1]) / 2))
    d1, d2 = np.ceil(np.abs((128 - image.shape[2]) / 2)), np.floor(np.abs((128 - image.shape[2]) / 2))
    return image[:-np.abs(int(z)), int(w1):-int(w2), int(d1):-int(d2)]


def get_binary_mask(image):
    smth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    smth_filter.SetSigma(7)
    image = smth_filter.Execute(image)

    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    seg = otsu_filter.Execute(image)

    hole_filler = sitk.BinaryMorphologicalOpeningImageFilter()
    seg = hole_filler.Execute(seg)

    op = sitk.BinaryMorphologicalClosingImageFilter()
    seg = op.Execute(seg)

    return sitk.GetArrayFromImage(seg)


if __name__ == "__main__":
    input_dir = "/store/datasets/HNC_Tox/2016-2022-HNC/"
    case_exclu = np.load("artefacts/case_exclusion.npy", allow_pickle=True).item()

    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(default_rigid_parameter_map)

    with h5py.File("/store/datasets/HNC_Survival/cbct.h5", "w") as f:
        for case_dir in sorted(glob.glob(f"{input_dir}/*")):
            if case_dir.split("/")[-1] in case_exclu or len(glob.glob(f"{case_dir}/cbct_aligned/*.nii.gz")) == 0:
                print(f"Skipping case {case_dir.split('/')[-1]}")
                continue

            print(f"Processsing case {case_dir.split('/')[-1]}")
            grp = f.create_group(case_dir.split("/")[-1])

            os.makedirs("tmp", exist_ok=True)

            for img_fpath in sorted(glob.glob(f"{case_dir}/cbct_aligned/*.nii.gz")):
                img = sitk.ReadImage(img_fpath)
                seg = get_binary_mask(img)
                arr = sitk.GetArrayFromImage(img)
                arr[seg == 0] = np.min(arr)
                arr = np.clip(arr, np.percentile(arr, 0.5), np.percentile(arr, 99.5))
                arr = (arr - arr.min()) / (arr.max() - arr.min())
                arr = crop(arr)

                fixed_img = sitk.GetImageFromArray(arr)
                fixed_img.SetOrigin(img.GetOrigin())
                fixed_img.SetDirection(img.GetDirection())
                fixed_img.SetSpacing(img.GetSpacing())
                sitk.WriteImage(fixed_img, f'tmp/{img_fpath.split(".")[-3]}.nii.gz')

                grp.create_dataset(img_fpath.split("/")[-1].split(".")[1], data=arr, compression='lzf',
                                   chunks=arr.shape)

            import sys

            sys.exit(0)
