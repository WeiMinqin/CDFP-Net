import os
import glob
import cv2
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm

"""
Script Purpose: 
Generate unlabelled image patches from extracted JPG images for self-supervised learning.
Inputs: CT and PET images from the training set.
Outputs: Extracted image patches saved as JPG files.
Note: Images smaller than PATCH_SIZE (256) are center-padded to meet the minimum size requirement.
"""

PATCH_SIZE = 256
STRIDE = 64
SELECTED_CENTER = ""

BASE_DIR = os.path.join(".", "pre_process", f"dataset_{SELECTED_CENTER}")
DATA_ROOT = os.path.join(BASE_DIR, "data", "train_images")
CT_SRC = os.path.join(DATA_ROOT, "CT")
PET_SRC = os.path.join(DATA_ROOT, "PET")

OUT_ROOT = os.path.join(BASE_DIR, "unlabelled_img_patches")
OUT_CT = os.path.join(OUT_ROOT, "CT")
OUT_PET = os.path.join(OUT_ROOT, "PET")

for d in [OUT_CT, OUT_PET]:
    os.makedirs(d, exist_ok=True)

transform_to_tensor = transforms.ToTensor()


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


def make_patch_tensor(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = pad_to_min(img)
    t = transform_to_tensor(img)

    patches = t.unfold(1, PATCH_SIZE, STRIDE).unfold(2, PATCH_SIZE, STRIDE)
    patches = patches.reshape(1, -1, PATCH_SIZE, PATCH_SIZE).squeeze(0)

    return patches


def process_modality(src_dir, out_dir):
    files = sorted(
        glob.glob(os.path.join(src_dir, "*.jpg")),
        key=lambda x: int(os.path.basename(x).split("_")[-1].split(".jpg")[0]),
    )

    idx = 0
    modality_name = os.path.basename(src_dir)

    for f in tqdm(files, desc=f"Generating unlabelled patches for {modality_name}"):
        patches = make_patch_tensor(f)
        if patches is None:
            continue

        for i in range(patches.shape[0]):
            save_path = os.path.join(out_dir, f"image_{idx}.jpg")
            cv2.imwrite(save_path, np.uint8(255 * patches[i].numpy()))
            idx += 1

    return idx


if __name__ == "__main__":
    print("Starting patch extraction...")
    ct_count = process_modality(CT_SRC, OUT_CT)
    pet_count = process_modality(PET_SRC, OUT_PET)
    print(f"Extraction complete. CT patches: {ct_count}, PET patches: {pet_count}")
