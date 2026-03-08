import os
import glob
import random
import cv2
import numpy as np
import nibabel as nib
import json
from tqdm import tqdm

SOURCE_PATTERN = r"/path/to/dataset/HECKTOR25/**.nii.gz"
MAX_FILES = None
RANDOM_SEED_FILES = 2026
RANDOM_SEED_SPLIT = 1216
TRAIN_RATIO = 0.7

TARGET_ROOT = os.path.join(
    "./project_root", "pre_process_hecktor25", "datasetNew_512", "data"
)
DIRS = {
    "train_images_CT": os.path.join(TARGET_ROOT, "train_images", "CT"),
    "train_images_PET": os.path.join(TARGET_ROOT, "train_images", "PET"),
    "train_masks_CT": os.path.join(TARGET_ROOT, "train_masks", "CT"),
    "train_masks_PET": os.path.join(TARGET_ROOT, "train_masks", "PET"),
    "test_images_CT": os.path.join(TARGET_ROOT, "test_images", "CT"),
    "test_images_PET": os.path.join(TARGET_ROOT, "test_images", "PET"),
    "test_masks_CT": os.path.join(TARGET_ROOT, "test_masks", "CT"),
    "test_masks_PET": os.path.join(TARGET_ROOT, "test_masks", "PET"),
}
for p in DIRS.values():
    os.makedirs(p, exist_ok=True)

ROOT_DIR = "/path/to/dataset/HECKTOR25"
angle_dirs = [d for d in glob.glob(os.path.join(ROOT_DIR, "*")) if os.path.isdir(d)]
records = []
for ad in tqdm(angle_dirs, desc="Scanning angles"):
    angle_name = os.path.basename(ad)
    if "-A" not in angle_name:
        continue
    patient_id = angle_name.rsplit("-A", 1)[0]
    angle_tag = "A" + angle_name.rsplit("-A", 1)[1]

    pet_list = glob.glob(os.path.join(ad, f"{patient_id}_PET10.nii.gz"))
    ct_list = glob.glob(os.path.join(ad, f"{patient_id}_CT.nii.gz"))
    seg_list = glob.glob(os.path.join(ad, f"{patient_id}_SEG.nii.gz"))

    if not (pet_list and ct_list and seg_list):
        continue
    pet_path, ct_path, seg_path = pet_list[0], ct_list[0], seg_list[0]

    try:
        pet_shape = nib.load(pet_path).shape
        ct_shape = nib.load(ct_path).shape
        seg_shape = nib.load(seg_path).shape
        if len(pet_shape) > 3 or len(ct_shape) > 3 or len(seg_shape) > 3:
            continue
    except Exception:
        continue

    records.append(
        {
            "patient_id": patient_id,
            "angle": angle_tag,
            "pet_path": pet_path,
            "ct_path": ct_path,
            "seg_path": seg_path,
        }
    )

if not records:
    raise ValueError("No valid projection records found.")


def norm_to_uint8(arr):
    if np.max(arr) == np.min(arr):
        return np.zeros_like(arr, dtype=np.uint8)
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


patient_ids = sorted(list(set([r["patient_id"] for r in records])))
random.seed(RANDOM_SEED_SPLIT)
random.shuffle(patient_ids)
split_idx = int(len(patient_ids) * TRAIN_RATIO)
train_patients = set(patient_ids[:split_idx])
test_patients = set(patient_ids[split_idx:])

train_records = [r for r in records if r["patient_id"] in train_patients]
test_records = [r for r in records if r["patient_id"] in test_patients]


def export_angle_records(rec_list, phase):
    img_idx = 0
    for r in tqdm(rec_list, desc=f"Exporting {phase}"):
        try:
            pet_ar = np.squeeze(nib.load(r["pet_path"]).get_fdata())
            ct_ar = np.squeeze(nib.load(r["ct_path"]).get_fdata())
            seg_ar = np.squeeze(nib.load(r["seg_path"]).get_fdata())
        except Exception:
            continue

        if pet_ar.ndim == 3:
            pet_ar = np.max(pet_ar, axis=-1)
        if ct_ar.ndim == 3:
            ct_ar = np.max(ct_ar, axis=-1)
        if seg_ar.ndim == 3:
            seg_ar = np.max(seg_ar, axis=-1)

        seg_bin = (seg_ar != 0).astype(np.uint8) * 255
        pet_uint8 = norm_to_uint8(pet_ar)
        ct_uint8 = norm_to_uint8(ct_ar)

        cv2.imwrite(
            os.path.join(DIRS[f"{phase}_images_PET"], f"image_{img_idx}.jpg"), pet_uint8
        )
        cv2.imwrite(
            os.path.join(DIRS[f"{phase}_images_CT"], f"image_{img_idx}.jpg"), ct_uint8
        )
        cv2.imwrite(
            os.path.join(DIRS[f"{phase}_masks_PET"], f"label_{img_idx}.jpg"), seg_bin
        )
        cv2.imwrite(
            os.path.join(DIRS[f"{phase}_masks_CT"], f"label_{img_idx}.jpg"), seg_bin
        )
        img_idx += 1


export_angle_records(train_records, "train")
export_angle_records(test_records, "test")
