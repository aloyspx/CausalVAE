import glob
import os
import multiprocessing
import shutil
from functools import partial
import SimpleITK as sitk
import numpy as np


def resample_img(itk_image, val, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    """
    Utility function to resample a function to a new spacing
    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2]))),
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(val)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkCosineWindowedSinc)

    return resample.Execute(itk_image)


def crop(image):
    shape = list(reversed(image.GetSize()))
    z = 128 - shape[0]
    w1, w2 = np.ceil(np.abs((256 - shape[1]) / 2)), np.floor(np.abs((256 - shape[1]) / 2))
    d1, d2 = np.ceil(np.abs((256 - shape[2]) / 2)), np.floor(np.abs((256 - shape[2]) / 2))
    return image[int(w1):-int(w2), int(d1):-int(d2), np.abs(int(z)):]


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


def process_image(case_dir, img_fpath, case_exclu, output_base_dir):
    print(f"Processing {img_fpath}...")
    if case_dir.split("/")[-1] in case_exclu or not img_fpath.endswith('.nii.gz'):
        return

    # Construct the output path
    relative_path = os.path.relpath(img_fpath, case_dir)
    output_path = os.path.join(output_base_dir, case_dir.split('/')[-1], relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # if os.path.exists(output_path):
    #     return

    # print(img_fpath)
    img = sitk.ReadImage(img_fpath)
    seg = get_binary_mask(img)
    arr = sitk.GetArrayFromImage(img)
    arr[seg == 0] = np.min(arr)
    arr = np.clip(arr, -1024, np.percentile(arr, 99.5))
    arr = (arr - arr.mean()) / arr.std()
    arr = (arr - arr.min()) / (arr.max() - arr.min())

    processed_img = sitk.GetImageFromArray(arr)
    processed_img.SetOrigin(img.GetOrigin())
    processed_img.SetDirection(img.GetDirection())
    processed_img.SetSpacing(img.GetSpacing())

    processed_img = crop(processed_img)

    # resample
    processed_img = resample_img(processed_img, val=0.)

    # Save the processed image
    sitk.WriteImage(processed_img, output_path)

    assert processed_img.GetSize() == (128, 128, 128), f"IMAGE SIZE IS WRONG, {img_fpath}"
    # assert np.count_nonzero(arr) / np.size(arr) > 0.25, f"VERY FEW NON-ZERO PIXELS, {img_fpath}"


def main(input_dir, output_base_dir, case_exclu):
    case_dirs = [dir for dir in sorted(glob.glob(f"{input_dir}/*")) if os.path.isdir(dir)]
    img_paths = [(case_dir, img_fpath)
                 for case_dir in case_dirs
                 for img_fpath in sorted(glob.glob(f"{case_dir}/cbct_raw/*.nii.gz"))]

    # for img_path in img_paths:
    #     process_image(img_path[0], img_path[1], case_exclu=case_exclu, output_base_dir=output_base_dir)

    # Set up multiprocessing
    pool = multiprocessing.Pool(processes=100)
    func = partial(process_image, case_exclu=case_exclu, output_base_dir=output_base_dir)
    pool.starmap(func, img_paths)
    pool.close()
    pool.join()

    print("done.")


if __name__ == "__main__":
    input_dir = "/store/datasets/HNC_Tox/2016-2022-HNC/"
    output_base_dir = "/home/portafaixa/PycharmProjects/CausalVAE/data/HNC_Survival_v2/"
    case_exclu = np.load("artefacts/case_exclusion_raw.npy", allow_pickle=True).item()

    main(input_dir, output_base_dir, case_exclu)
