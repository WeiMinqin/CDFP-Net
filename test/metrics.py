import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import math
import os
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    jaccard_score,
    confusion_matrix,
)
from scipy.spatial.distance import directed_hausdorff
from medpy.metric.binary import hd95

# Set to 0 if class:0 (background) is to be included as one class in the metric computation.
# Otherwise, set to 1 to compute metrics only for foreground classes.
start = 1


def dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    smooth = 1.0
    return ((2.0 * intersection) + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def dice_score(y_true, y_pred, num_classes):
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    dice = []
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        dice.append(dice_coef(true, pred))

    return dice


def conf_matrix(y_true, y_pred, num_classes):
    _, y_true = torch.max(y_true, 1)
    y_true = y_true.cpu().numpy()
    _, y_pred = torch.max(y_pred, 1)
    y_pred = y_pred.cpu().numpy()
    cm = confusion_matrix(
        y_true.reshape(-1), y_pred.reshape(-1), labels=np.arange(num_classes)
    )

    return cm


def precision(y_true, y_pred, num_classes):
    class_precisions = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = precision_score(true, pred, zero_division=1)
        class_precisions.append(value)

    return class_precisions


def sensitivity(y_true, y_pred, num_classes):
    class_sensitivities = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = recall_score(true, pred, zero_division=1)
        class_sensitivities.append(value)

    return class_sensitivities


def specificity(y_true, y_pred, num_classes):
    class_specifities = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = recall_score(true, pred, pos_label=0, zero_division=1)
        class_specifities.append(value)

    return class_specifities


def accuracy(y_true, y_pred, num_classes):
    """
    Foreground accuracy (only for foreground statistics): TP / (TP + FN)
    If the class does not exist in the Ground Truth (GT):
        - Returns 1.0 if prediction also lacks this class.
        - Returns 0.0 otherwise.
    Returns a list of foreground accuracies per class (length = num_classes - start).
    """
    class_fg_acc = []
    _, y_true = torch.max(y_true, 1)  # [B,H,W]
    _, y_pred = torch.max(y_pred, 1)  # [B,H,W]
    for i in range(start, num_classes):
        true_pos = y_true == i
        pred_pos = y_pred == i
        tp = (true_pos & pred_pos).sum().item()
        fn = (true_pos & (~pred_pos)).sum().item()
        gt_count = true_pos.sum().item()

        if gt_count == 0:
            value = 1.0 if pred_pos.sum().item() == 0 else 0.0
        else:
            value = tp / (gt_count + 1e-8)
        class_fg_acc.append(value)

    return class_fg_acc


def F1_score(y_true, y_pred, num_classes):
    class_f1_scores = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = f1_score(true, pred, zero_division=1)
        class_f1_scores.append(value)

    return class_f1_scores


def Jaccard_score(y_true, y_pred, num_classes):
    class_jaccard_scores = []
    _, y_true = torch.max(y_true, 1)
    _, y_pred = torch.max(y_pred, 1)
    for i in range(start, num_classes):
        true = (y_true == i).reshape(-1).cpu().numpy()
        pred = (y_pred == i).reshape(-1).cpu().numpy()
        value = jaccard_score(true, pred, zero_division=1)
        class_jaccard_scores.append(value)

    return class_jaccard_scores


def Hausdorff_distance(y_true, y_pred, num_classes):
    """
    Calculates 2D HD95 (in pixels) using medpy.metric.binary.hd95.
    Empty mask handling:
      - If both GT and Pred are empty -> 0.0
      - If one is empty and the other is not -> Set to diagonal length (maximum possible distance)
    """
    class_hd95 = []
    _, y_true = torch.max(y_true, 1)  # [B,H,W]
    _, y_pred = torch.max(y_pred, 1)  # [B,H,W]

    # Assumes batch size = 1 based on current evaluation settings.
    # Squeezes the batch dimension safely.
    for i in range(start, num_classes):
        true = (y_true == i).squeeze(0).cpu().numpy().astype(np.uint8)
        pred = (y_pred == i).squeeze(0).cpu().numpy().astype(np.uint8)

        gt_sum, pred_sum = int(true.sum()), int(pred.sum())
        if gt_sum == 0 and pred_sum == 0:
            hd_val = 0.0
        elif gt_sum == 0 or pred_sum == 0:
            H, W = true.shape
            hd_val = float((H**2 + W**2) ** 0.5)  # Diagonal length
        else:
            try:
                hd_val = float(hd95(pred, true))
            except Exception:
                H, W = true.shape
                hd_val = float((H**2 + W**2) ** 0.5)
        class_hd95.append(hd_val)

    return class_hd95


# This script is adopted from the official repository of AJI score
def Aggregated_jaccard_index(gt_map, predicted_map, gpu):
    _, gt_map = torch.max(gt_map, 1)
    _, predicted_map = torch.max(predicted_map, 1)

    gt_list = torch.unique(gt_map)
    pr_list = torch.unique(predicted_map)

    if start != 0:
        gt_list = gt_list[gt_list != 0]
        pr_list = pr_list[pr_list != 0]

    pr_list = torch.cat(
        (pr_list.view(-1, 1), torch.zeros(pr_list.size(0), 1).to(gpu)), dim=1
    )

    overall_correct_count = 0.0
    union_pixel_count = 0.0

    i = len(gt_list)

    while len(gt_list) > 0:
        gt = (gt_map == gt_list[i - 1]).float()
        predicted_match = gt * predicted_map.float()

        if predicted_match.sum() == 0:
            union_pixel_count += gt.sum()
            gt_list = gt_list[:-1]
            i = len(gt_list)
        else:
            predicted_nuc_index = torch.unique(predicted_match)
            if start != 0:
                predicted_nuc_index = predicted_nuc_index[predicted_nuc_index != 0]

            JI = 0
            best_match = None

            for j in range(len(predicted_nuc_index)):
                matched = (predicted_map == predicted_nuc_index[j]).float()
                nJI = matched.logical_and(gt).sum() / matched.logical_or(gt).sum()

                if nJI > JI:
                    best_match = predicted_nuc_index[j]
                    JI = nJI

            predicted_nuclei = (predicted_map == best_match).float()

            overall_correct_count += (gt.logical_and(predicted_nuclei)).sum()
            union_pixel_count += (gt.logical_or(predicted_nuclei)).sum()

            gt_list = gt_list[:-1]
            i = len(gt_list)

            best_match_idx = (pr_list[:, 0] == best_match).nonzero().item()
            pr_list[best_match_idx, 1] += 1

    unused_nuclei_list = (pr_list[:, 1] == 0).nonzero().view(-1)

    for k in range(len(unused_nuclei_list)):
        # print(pr_list[unused_nuclei_list[k], 0])
        unused_nuclei = (predicted_map == pr_list[unused_nuclei_list[k], 0]).float()
        union_pixel_count += unused_nuclei.sum()

    if overall_correct_count == 0 and union_pixel_count == 0:
        return 1.0

    aji = overall_correct_count / union_pixel_count
    return aji.cpu().numpy()
