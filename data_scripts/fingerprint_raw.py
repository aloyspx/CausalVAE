import glob
import os.path
import nibabel as nib
import numpy as np

from collections import Counter
from tqdm import tqdm


def get_image_properties(fpath):
    img = nib.load(fpath, mmap=True)
    size = img.header.get_data_shape()
    spacing = img.header.get_zooms()[:3]
    return fpath, size, spacing


if __name__ == "__main__":
    input_dir = "/store/datasets/HNC_Tox/2016-2022-HNC/"

    if os.path.exists("artefacts/info_raw.npy"):
        data = np.load("artefacts/info_raw.npy", allow_pickle=True).item()
        sizes = data['sizes']
        spacings = data['spacings']
    else:
        input_dir = "/store/datasets/HNC_Tox/2016-2022-HNC/"
        cbct_image_fpaths = sorted(glob.glob(f"{input_dir}/*/cbct_raw/*.nii.gz"))

        sizes, spacings = {}, {}

        results = [get_image_properties(img_fpath) for img_fpath in tqdm(cbct_image_fpaths)]

        for key, size, spacing in results:
            sizes[key] = size
            spacings[key] = spacing

        np.save("artefacts/info_raw.npy", {"sizes": sizes, "spacings": spacings}, allow_pickle=True)

    print("Check all spacing is isotropic 1mm.")
    for spacing in spacings.items():
        if spacing[1] != (1., 1., 2.):
            print(spacing[0], spacing[1], sizes[spacing[0]])

    print("Count the different types of sizes")
    size_counter = Counter()
    for key, size in sizes.items():
        size_counter[size] += 1

    print(dict(size_counter))

    print("Median sizes: ", np.median(list(sizes.values()), axis=0))
    print("Median spacing: ", np.median(list(spacings.values()), axis=0))

    # Excluding cases
    file_exclu = []

    for spacing, size in zip(spacings.items(), sizes.items()):
        if size[1] != (270, 270, 132) or spacing[1] != (1., 1., 2.):
            file_exclu.append(spacing[0].split("/")[-3])

    file_exclu = set(file_exclu)
    np.save('artefacts/case_exclusion_raw.npy', file_exclu, allow_pickle=True)

