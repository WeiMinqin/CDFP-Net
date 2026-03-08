import os
import glob
import cv2
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm

"""
Generate unlabelled patches from extracted JPG images for self-supervised learning.
Inputs: CT and PET images from the training set.
Outputs: Extracted image patches saved as JPG files.
Note: Images smaller than PATCH_SIZE (256) are center-padded to meet the minimum size requirement.
"""

PATCH_SIZE = 256
STRIDE = 64
SELECTED_CENTER = ""

# Define base paths (Using relative paths for repository portability)
BASE_DIR = os.path.join(".", "ct_inf_hecktor", "pre_process_hecktor25")
DATA_ROOT = os.path.join(BASE_DIR, f"dataset{SELECTED_CENTER}", "data", "train_images")
CT_SRC = os.path.join(DATA_ROOT, "CT")
PET_SRC = os.path.join(DATA_ROOT, "PET")

OUT_ROOT = os.path.join(BASE_DIR, f"dataset{SELECTED_CENTER}", "unlabelled_img_patches")
OUT_CT = os.path.join(OUT_ROOT, "CT")
OUT_PET = os.path.join(OUT_ROOT, "PET")

# Create output directories if they don't exist
for d in [OUT_CT, OUT_PET]:
    os.makedirs(d, exist_ok=True)

transform_to_tensor = transforms.ToTensor()  # (H,W) -> (1,H,W)


def pad_to_min(img, min_size=PATCH_SIZE):
    """Center pad the image to min_size if any dimension is smaller."""
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
    """Read an image, apply padding, and extract sliding-window patches."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = pad_to_min(img)
    t = transform_to_tensor(img)  # (1,H,W) with values in range [0, 1]

    # Extract patches using unfolding
    patches = t.unfold(1, PATCH_SIZE, STRIDE).unfold(
        2, PATCH_SIZE, STRIDE
    )  # Resulting shape: (1, nH, nW, ph, pw)

    patches = patches.reshape(1, -1, PATCH_SIZE, PATCH_SIZE).squeeze(
        0
    )  # Shape: (N, ph, pw)
    return patches


def process_modality(src_dir, out_dir):
    """Process all images in the source directory and save patches to the output directory."""
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
            # Convert back to uint8 format for saving
            cv2.imwrite(save_path, np.uint8(255 * patches[i].numpy()))
            idx += 1

    return idx


if __name__ == "__main__":
    print("Starting patch extraction...")
    ct_count = process_modality(CT_SRC, OUT_CT)
    pet_count = process_modality(PET_SRC, OUT_PET)
    print(f"Extraction complete. CT patches: {ct_count}, PET patches: {pet_count}")
