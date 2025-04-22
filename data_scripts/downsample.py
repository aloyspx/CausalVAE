import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def find_nii_gz_files(directory):
    nii_gz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_gz_files.append(os.path.join(root, file))
    return nii_gz_files


def resample_img(itk_image, val, out_spacing=[1.0, 1.0, 1.0], is_label=False):
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
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


if __name__ == "__main__":
    directory_to_search = '/home/portafaixa/PycharmProjects/CausalVAE/data/HNC_Survival/'

    for file in tqdm(find_nii_gz_files(directory_to_search)):
        img = sitk.ReadImage(file)
        img = resample_img(img, val=0., out_spacing=[2., 2., 4.])
        sitk.WriteImage(img, file)
