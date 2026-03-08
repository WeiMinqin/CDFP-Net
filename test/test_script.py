import os
import sys
import pickle
import io as pyio
import numpy as np
import cv2
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.nn import functional as F
from skimage import io, measure
from scipy.io import loadmat
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

from metrics import (
    Aggregated_jaccard_index,
    Hausdorff_distance,
    Jaccard_score,
    precision,
    sensitivity,
    accuracy,
    F1_score,
    conf_matrix,
)

# Insert the path to the model architecture
sys.path.insert(0, r"/path/to/project/model_directory")
from model import GuidedSegmentationModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ==============================================================================
# Hyperparameters & Paths
# ==============================================================================
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
IMG_CHANNELS = 3
BATCH_SIZE = (
    1  # Note: Batch size must be 1 to properly calculate certain spatial metrics
)
NUM_CLASSES = 2
SELECTED_CENTER = "New"

TEST_PET_DIR = f"/path/to/dataset_{SELECTED_CENTER}/test/image_patches/PET"
TEST_CT_DIR = f"/path/to/dataset_{SELECTED_CENTER}/test/image_patches/CT"
TEST_LABEL_PATH = f"/path/to/dataset_{SELECTED_CENTER}/test/label_patches.mat"

start = "\033[1m"
end = "\033[0;0m"


# ==============================================================================
# Data Preprocessing (Must match training exactly)
# ==============================================================================
def ensure_three_channel(t):
    if t.dim() == 3 and t.shape[0] == 1:
        return t.repeat(3, 1, 1)
    return t


def normalize_m11(t):
    return t * 2 - 1


transformations = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(ensure_three_channel),
        transforms.Lambda(normalize_m11),
    ]
)


def class_weights(target_labl):
    _, target_label1 = torch.max(target_labl, dim=0)
    weights = np.ones(NUM_CLASSES)
    target_label = target_label1.reshape(-1)
    all_labels = target_label.cpu().numpy()
    labels, label_counts = np.unique(np.array(all_labels), return_counts=True)
    w = 1 - np.round(label_counts / np.sum(label_counts), 4)
    if len(labels) == 1:
        w = 1.0
    weights[labels] = w

    print(labels)
    print(label_counts)

    return weights


def get_images_list(path1, k=None):
    total_list1 = os.listdir(path1)
    total_list1 = sorted(total_list1, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return np.array(total_list1) if k is None else np.array(total_list1[:k])


class Histo_Dataset(Dataset):
    def __init__(self, pet_dir, ct_dir, image_list, label_list, transform=None):
        self.pet_dir = pet_dir
        self.ct_dir = ct_dir
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_name = self.image_list[index]
        pet_path = os.path.join(self.pet_dir, img_name)
        ct_path = os.path.join(self.ct_dir, img_name)

        image_pet = io.imread(pet_path)
        image_ct = io.imread(ct_path)

        # Unified preprocessing function
        def _preprocess_img(img):
            if img.ndim == 2:
                return np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[2] == 1:
                return np.concatenate([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[2] > 3:
                return img[:, :, :3]
            return img

        image_pet = _preprocess_img(image_pet)
        image_ct = _preprocess_img(image_ct)
        mask = self.label_list[index]

        if self.transform is not None:
            image_pet = self.transform(image_pet)
            image_ct = self.transform(image_ct)

        return image_pet, image_ct, mask


# ==============================================================================
# Evaluation Loop
# ==============================================================================
def eval_epoch(test_loader, model, num_classes, device):
    model_type = "diffusion_ct_guided"
    loss_type = "SSFL_A"
    info_txt_path = f"/path/to/project/results/inf_{SELECTED_CENTER}.txt"

    os.makedirs(os.path.dirname(info_txt_path), exist_ok=True)
    with open(info_txt_path, "w", encoding="utf-8") as f_clear:
        f_clear.write(
            f"PETSEG Inference Results (model_type={model_type}, loss={loss_type})\n"
        )
        f_clear.write(
            "================ SUMMARY PER IMAGE (raw lines mirror console) ================\n\n"
        )

    with torch.no_grad():
        model.eval()
        aji_scores = 0.0
        jaccard_scores = np.zeros(num_classes - 1)
        fscores = np.zeros(num_classes - 1)
        hds = np.zeros(num_classes - 1)
        accuracies = np.zeros(num_classes - 1)
        senstivities = np.zeros(num_classes - 1)
        precisions = np.zeros(num_classes - 1)
        ious = np.zeros(num_classes - 1)
        dice_scores = np.zeros(num_classes - 1)  # Dice (Foreground class)
        cm = np.zeros((num_classes, num_classes))

        # False Positive/Negative Volume (Pixels & Ratio) Tracking
        fpv_pix_sum = 0.0
        fnv_pix_sum = 0.0
        fpv_ratio_sum = 0.0
        fnv_ratio_sum = 0.0

        total_items = 0
        p_bar = tqdm(test_loader)

        save_path1 = f"/path/to/project/results/images/true_labels_{SELECTED_CENTER}"
        save_path2 = (
            f"/path/to/project/results/images/predicted_labels_{SELECTED_CENTER}"
        )
        save_path3 = f"/path/to/project/results/images/raw_images_{SELECTED_CENTER}"

        for sp in (save_path1, save_path2, save_path3):
            if os.path.exists(sp):
                shutil.rmtree(sp)
            os.makedirs(sp, exist_ok=True)

        indexing = 0
        for pet_img, ct_img, target_label in p_bar:
            pet_img = pet_img.to(device)
            ct_img = ct_img.to(device)
            target_label = target_label.squeeze(1).to(device)
            target_label1 = target_label.long()
            target_label = F.one_hot(target_label1, num_classes)
            target_label = torch.permute(target_label, (0, 3, 1, 2))

            predicted_label = model(pet_img, ct_img)
            batch = predicted_label.shape[0]

            aji_score = Aggregated_jaccard_index(target_label, predicted_label, device)
            jaccard_score = Jaccard_score(target_label, predicted_label, num_classes)
            fscore = F1_score(target_label, predicted_label, num_classes)
            hd = Hausdorff_distance(target_label, predicted_label, num_classes)
            acc = accuracy(target_label, predicted_label, num_classes)
            senstvty = sensitivity(target_label, predicted_label, num_classes)
            prec = precision(target_label, predicted_label, num_classes)

            _, pred_labels = torch.max(predicted_label, 1)
            target_sum = (target_label1 == 1).sum().item()
            pred_sum_all = (pred_labels == 1).sum().item()

            if target_sum == 0:
                iou = 1.0 if pred_sum_all == 0 else 0.0
                dice_val = 1.0 if pred_sum_all == 0 else 0.0
            else:
                target_mask = (target_label1 == 1).float()
                pred_mask = (pred_labels == 1).float()
                intersection = (target_mask * pred_mask).sum()
                union = (target_mask + pred_mask).clamp(0, 1).sum()
                iou = (intersection / (union + 1e-6)).cpu().item()
                dice_val = (
                    ((2 * intersection) / (target_mask.sum() + pred_mask.sum() + 1e-6))
                    .cpu()
                    .item()
                )

            # Calculate False Positive Volume (FPV) & False Negative Volume (FNV)
            pred_pos = (pred_labels == 1).cpu().numpy().astype(np.uint8)
            gt_pos = (target_label1 == 1).cpu().numpy().astype(np.uint8)
            pred_pos_img = pred_pos[0]
            gt_pos_img = gt_pos[0]

            fpv_pixels = 0
            labeled_pred, n_pred_comp = measure.label(
                pred_pos_img, connectivity=1, return_num=True
            )
            for comp_id in range(1, n_pred_comp + 1):
                comp_mask = labeled_pred == comp_id
                if not (comp_mask & gt_pos_img.astype(bool)).any():
                    fpv_pixels += int(comp_mask.sum())

            fnv_pixels = 0
            labeled_gt, n_gt_comp = measure.label(
                gt_pos_img, connectivity=1, return_num=True
            )
            for comp_id in range(1, n_gt_comp + 1):
                comp_mask = labeled_gt == comp_id
                if not (comp_mask & pred_pos_img.astype(bool)).any():
                    fnv_pixels += int(comp_mask.sum())

            total_pred_pos = max(int(pred_pos_img.sum()), 1)
            total_gt_pos = max(int(gt_pos_img.sum()), 1)
            fpv_ratio = fpv_pixels / total_pred_pos
            fnv_ratio = fnv_pixels / total_gt_pos
            fpv_pix_sum += fpv_pixels
            fnv_pix_sum += fnv_pixels
            fpv_ratio_sum += fpv_ratio
            fnv_ratio_sum += fnv_ratio

            # Print and log FPV/FNV
            # print(f"FPV_pixels: {fpv_pixels} (ratio {fpv_ratio:.4f})")
            # print(f"FNV_pixels: {fnv_pixels} (ratio {fnv_ratio:.4f})")
            with open(info_txt_path, "a", encoding="utf-8") as fw:
                fw.write(f"FPV_pixels: {fpv_pixels} | FPV_ratio: {fpv_ratio:.4f}\n")
                fw.write(f"FNV_pixels: {fnv_pixels} | FNV_ratio: {fnv_ratio:.4f}\n")

            aji_scores += aji_score
            jaccard_scores += jaccard_score
            fscores += fscore
            hds += hd
            accuracies += acc
            senstivities += senstvty
            precisions += prec
            ious += iou
            dice_scores += dice_val
            cm += conf_matrix(target_label, predicted_label, num_classes)
            total_items += 1

            # True label mappings for visualization
            labels_t = np.zeros(
                (
                    target_label1.shape[0],
                    target_label1.shape[1],
                    target_label1.shape[2],
                    3,
                ),
                dtype=np.uint8,
            )
            labels_t[target_label1.cpu() == 1] = [255, 69, 0]
            labels_t[target_label1.cpu() == 2] = [127, 255, 0]
            labels_t[target_label1.cpu() == 3] = [135, 206, 250]

            # Predicted label mappings for visualization
            _, pred_labels = torch.max(predicted_label, 1)
            labels_p = np.zeros(
                (pred_labels.shape[0], pred_labels.shape[1], pred_labels.shape[2], 3),
                dtype=np.uint8,
            )
            labels_p[pred_labels.cpu() == 1] = [255, 69, 0]
            labels_p[pred_labels.cpu() == 2] = [127, 255, 0]
            labels_p[pred_labels.cpu() == 3] = [135, 206, 250]

            for i in range(target_label.shape[0]):
                image_label_t = labels_t[i]
                image_label_p = labels_p[i]

                out_label_path1 = os.path.join(
                    save_path1,
                    f"label_{indexing}_{np.round(np.mean(senstvty) / batch, 2)}_{np.round(np.mean(prec) / batch, 2)}.jpg",
                )
                cv2.imwrite(
                    out_label_path1, cv2.cvtColor(image_label_t, cv2.COLOR_RGB2BGR)
                )

                out_label_path2 = os.path.join(
                    save_path2,
                    f"label_{indexing}_{np.round(np.mean(senstvty) / batch, 2)}_{np.round(np.mean(prec) / batch, 2)}.jpg",
                )
                cv2.imwrite(
                    out_label_path2, cv2.cvtColor(image_label_p, cv2.COLOR_RGB2BGR)
                )

                # Save original raw image for comparison
                image = torch.permute(pet_img[i], (1, 2, 0)).cpu().numpy()
                image = np.uint8(255 * image)

                out_image_path = os.path.join(
                    save_path3,
                    f"image_{indexing}_{np.round(np.mean(senstvty) / batch, 2)}_{np.round(np.mean(prec) / batch, 2)}.jpg",
                )
                cv2.imwrite(out_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                indexing += 1

            p_bar.set_postfix(dice=np.round(dice_val, 4), iou=np.round(iou, 4))

        print(f"total items: {total_items}")
        aji_scores = aji_scores / total_items
        jaccard_scores = jaccard_scores / total_items
        fscores = fscores / total_items
        hds = hds / total_items
        accuracies = accuracies / total_items
        senstivities = senstivities / total_items
        precisions = precisions / total_items
        ious = ious / total_items
        dice_scores = dice_scores / total_items
        cm = np.round(cm / total_items).astype(int)

        with open(info_txt_path, "a", encoding="utf-8") as fw:
            fw.write("\n================ OVERALL AVERAGES ================\n")
            fw.write(f"Average AJI: {aji_scores}\n")
            fw.write(
                f"class jaccard scores: {jaccard_scores} | Average: {np.mean(jaccard_scores)}\n"
            )
            fw.write(f"class F1 scores: {fscores} | Average: {np.mean(fscores)}\n")
            fw.write(f"class HD95: {hds} | Average: {np.mean(hds)}\n")
            fw.write(
                f"class accuracies: {accuracies} | Average: {np.mean(accuracies)}\n"
            )
            fw.write(
                f"class sensitivities: {senstivities} | Average: {np.mean(senstivities)}\n"
            )
            fw.write(
                f"class precisions: {precisions} | Average: {np.mean(precisions)}\n"
            )
            fw.write(f"Class IoUs: {ious} | Mean: {np.mean(ious)}\n")
            fw.write(f"Class Dice: {dice_scores} | Mean: {np.mean(dice_scores)}\n")
            fw.write(
                f"FPV_pixels Mean: {fpv_pix_sum/total_items:.2f} | FPV_ratio Mean: {fpv_ratio_sum/total_items:.4f}\n"
            )
            fw.write(
                f"FNV_pixels Mean: {fnv_pix_sum/total_items:.2f} | FNV_ratio Mean: {fnv_ratio_sum/total_items:.4f}\n"
            )

        print("================ METRIC AVERAGES (SUMMARY) ================")
        print(f"AJI: {aji_scores:.4f}")
        print(
            f"Jaccard per-class: {jaccard_scores} | Mean: {np.mean(jaccard_scores):.4f}"
        )
        print(f"F1 per-class: {fscores} | Mean: {np.mean(fscores):.4f}")
        print(f"Hausdorff per-class: {hds} | Mean: {np.mean(hds):.4f}")
        print(f"Accuracy per-class: {accuracies} | Mean: {np.mean(accuracies):.4f}")
        print(
            f"Sensitivity per-class: {senstivities} | Mean: {np.mean(senstivities):.4f}"
        )
        print(f"Precision per-class: {precisions} | Mean: {np.mean(precisions):.4f}")
        print(f"IoU per-class: {ious} | Mean: {np.mean(ious):.4f}")
        print(f"Dice per-class: {dice_scores} | Mean: {np.mean(dice_scores):.4f}")
        print(
            f"FPV Mean pixels: {fpv_pix_sum/total_items:.2f} | FPV Mean ratio: {fpv_ratio_sum/total_items:.4f}"
        )
        print(
            f"FNV Mean pixels: {fnv_pix_sum/total_items:.2f} | FNV Mean ratio: {fnv_ratio_sum/total_items:.4f}"
        )

        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues", values_format="")
        plt.savefig(
            f"/path/to/project/results/cm_{SELECTED_CENTER}_{model_type}_{loss_type}_{BATCH_SIZE}.jpg",
            dpi=300,
        )


# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    img1_list = get_images_list(TEST_PET_DIR)
    label_file = loadmat(TEST_LABEL_PATH)
    train_labels = label_file["data"]

    test_dataset = Histo_Dataset(
        TEST_PET_DIR, TEST_CT_DIR, img1_list, train_labels, transform=transformations
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    path_train = (
        r"/path/to/project/snapshots_final/diff_fuse_SSFL/trained_downstream_model.pth"
    )
    try:
        # Explicitly turn off weights_only to allow loading older checkpoints containing optimizers
        snapshot = torch.load(path_train, map_location=DEVICE, weights_only=False)
    except pickle.UnpicklingError as e:
        print("UnpicklingError, attempting to add safe globals before loading:", e)
        from torch.serialization import add_safe_globals

        add_safe_globals([np.core.multiarray._reconstruct])
        snapshot = torch.load(path_train, map_location=DEVICE, weights_only=False)

    print(DEVICE)
    print(path_train)

    fusion_config = {"enabled": True, "fusion_channels": 16}

    # Ensure these paths are valid. These are used to instantiate the model structure.
    pet_chkpt_placeholder = r"/path/to/pretrained/pet_encoder_weights.pth"
    ct_chkpt_placeholder = r"/path/to/pretrained/ct_encoder_weights.pth"

    model = GuidedSegmentationModel(
        dim=64,
        channels=3,
        num_classes=NUM_CLASSES,
        pet_chkpt_path=pet_chkpt_placeholder,
        ct_chkpt_path=ct_chkpt_placeholder,
        fusion_config=fusion_config,
        enable_encoder_zero_layer=True,
    ).to(DEVICE)

    model.load_state_dict(snapshot["model_state_dict"], strict=True)
    eval_epoch(test_loader, model, NUM_CLASSES, DEVICE)


if __name__ == "__main__":
    LOG_FILE = f"/path/to/project/results/log_{SELECTED_CENTER}.txt"
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    class _TeeLogger(pyio.TextIOBase):
        """Captures stdout/stderr and writes to both console and log file."""

        def __init__(self, filepath, mode="w", encoding="utf-8"):
            self.file = open(filepath, mode, encoding=encoding)
            self.stdout = sys.__stdout__

        def write(self, data):
            if data:
                self.stdout.write(data)
                self.file.write(data)

        def flush(self):
            self.stdout.flush()
            self.file.flush()

        def close(self):
            try:
                self.file.close()
            except Exception:
                pass

    tee_logger = _TeeLogger(LOG_FILE)
    sys.stdout = tee_logger
    sys.stderr = tee_logger
    print(f"[Logger] Output mirrored to: {LOG_FILE}")

    main()
