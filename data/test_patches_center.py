import os
import glob
import cv2
import json
import numpy as np
import h5py
from torchvision.transforms import transforms
from tqdm import tqdm

"""
Script Purpose:
Extract and process the test set based on the 'split.json' file generated during training patch creation.
It copies the full-size test images/masks and generates sliding-window patches for CT and PET modalities.
The resulting label patches are saved in an HDF5-formatted .mat file.
"""

PATCH_SIZE = 256
STRIDE = 64
SELECTED_CENTER = ""

BASE_DIR = os.path.join(".", "pre_process", f"dataset_{SELECTED_CENTER}")
DATA_ROOT = os.path.join(BASE_DIR, "data")

SRC_CT = os.path.join(DATA_ROOT, "test_images", "CT")
SRC_PET = os.path.join(DATA_ROOT, "test_images", "PET")
SRC_MASK = os.path.join(DATA_ROOT, "test_masks", "PET")
SPLIT_FILE = os.path.join(BASE_DIR, "split.json")

TEST_ROOT = os.path.join(BASE_DIR, "test")
FS_IMG_CT = os.path.join(TEST_ROOT, "full_size_images", "CT")
FS_IMG_PET = os.path.join(TEST_ROOT, "full_size_images", "PET")
FS_MASK_CT = os.path.join(TEST_ROOT, "full_size_masks", "CT")
FS_MASK_PET = os.path.join(TEST_ROOT, "full_size_masks", "PET")
PATCH_IMG_CT = os.path.join(TEST_ROOT, "image_patches", "CT")
PATCH_IMG_PET = os.path.join(TEST_ROOT, "image_patches", "PET")

transform = transforms.ToTensor()


def pad_to_min(img, min_size=PATCH_SIZE):
    h, w = img.shape
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)

    if pad_h == 0 and pad_w == 0:
        return img

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )


def sorted_list(dir_path):
    files = glob.glob(os.path.join(dir_path, "*.jpg"))
    return sorted(
        files, key=lambda x: int(os.path.basename(x).split("_")[-1].split(".jpg")[0])
    )


def extract_patches(image_dir, out_dir, modality):
    idx = 0
    files = sorted_list(image_dir)
    for f in tqdm(files, desc=f"Extracting patches for {modality}"):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = pad_to_min(img)
        t = transform(img)
        patches = t.unfold(1, PATCH_SIZE, STRIDE).unfold(2, PATCH_SIZE, STRIDE)
        patches = patches.reshape(1, -1, PATCH_SIZE, PATCH_SIZE).squeeze(0)

        for p in patches:
            cv2.imwrite(
                os.path.join(out_dir, f"image_{idx}.jpg"), np.uint8(255 * p.numpy())
            )
            idx += 1
    return idx


def extract_label_patches(mask_dir, label_list):
    files = sorted_list(mask_dir)
    for f in tqdm(files, desc="Extracting label patches"):
        m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        m = pad_to_min(m)
        t = transform(m)
        patches = t.unfold(1, PATCH_SIZE, STRIDE).unfold(2, PATCH_SIZE, STRIDE)
        patches = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE)

        for p in patches:
            label_list.append(p.numpy().astype(np.uint8))


if __name__ == "__main__":
    for d in [
        FS_IMG_CT,
        FS_IMG_PET,
        FS_MASK_CT,
        FS_MASK_PET,
        PATCH_IMG_CT,
        PATCH_IMG_PET,
    ]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(
            "split.json not found. Please run the training patch generation script first."
        )

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    test_indices = split_info.get("test")
    if test_indices is None:
        raise RuntimeError("Key 'test' is missing in split.json.")

    ct_files = sorted_list(SRC_CT)
    pet_files = sorted_list(SRC_PET)
    mask_files = sorted_list(SRC_MASK)
    N = min(len(ct_files), len(pet_files), len(mask_files))

    print(f"Total file pairs: {N}, Test indices count: {len(test_indices)}")
    print(f"Copying {len(test_indices)} full-size test items...")

    for new_idx, orig_i in enumerate(sorted(test_indices)):
        if orig_i >= N:
            continue

        ct_img = cv2.imread(ct_files[orig_i], cv2.IMREAD_GRAYSCALE)
        pet_img = cv2.imread(pet_files[orig_i], cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(mask_files[orig_i], cv2.IMREAD_GRAYSCALE)

        if ct_img is None or pet_img is None or mask_img is None:
            continue

        cv2.imwrite(os.path.join(FS_IMG_CT, f"image_{new_idx}.jpg"), ct_img)
        cv2.imwrite(os.path.join(FS_IMG_PET, f"image_{new_idx}.jpg"), pet_img)
        cv2.imwrite(os.path.join(FS_MASK_CT, f"label_{new_idx}.jpg"), mask_img)
        cv2.imwrite(os.path.join(FS_MASK_PET, f"label_{new_idx}.jpg"), mask_img)

    LABEL_PATCHES = []
    ct_patch_num = extract_patches(FS_IMG_CT, PATCH_IMG_CT, "CT")
    pet_patch_num = extract_patches(FS_IMG_PET, PATCH_IMG_PET, "PET")
    extract_label_patches(FS_MASK_PET, LABEL_PATCHES)

    LABEL_PATCHES_ARRAY = np.array(LABEL_PATCHES, dtype=np.uint8)
    print(
        f"Test split - CT patches: {ct_patch_num}, PET patches: {pet_patch_num}, Label patches: {LABEL_PATCHES_ARRAY.shape[0]}"
    )

    mat_path = os.path.join(TEST_ROOT, "label_patches.mat")
    with h5py.File(mat_path, "w") as f:
        f.create_dataset("data", data=LABEL_PATCHES_ARRAY)
        f.attrs["label"] = "test_labels"

    print(f"Successfully saved {mat_path} (HDF5 format)")
