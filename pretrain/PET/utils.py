import os
import re
import random
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CelebA
from scipy.ndimage import map_coordinates, gaussian_filter

IMG_SIZE = 256
SELECTED_CENTER = ""


# Top-level function to prevent DataLoader multiprocessing pickle errors
def repeat_if_gray(t):
    """Converts single-channel tensor to 3-channel: (1, H, W) -> (3, H, W)"""
    if t.dim() == 3 and t.shape[0] == 1:
        return t.repeat(3, 1, 1)
    return t


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, betas_schedule, device="cpu"):
    """
    Takes an image and a timestep as input and returns the noisy version of it.
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(
        betas_schedule["sqrt_alphas_cumprod"], t, x_0.shape
    )
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        betas_schedule["sqrt_one_minus_alphas_cumprod"], t, x_0.shape
    )

    return sqrt_alphas_cumprod_t.to(device) * x_0.to(
        device
    ) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def _my_normalization(x):
    return (x * 2) - 1


def get_images_list(path1, k=None, recursive_modalities=None):
    """
    Retrieves a list of image files.
    Supports both flat directory structures and nested modality subdirectories.
    Returns relative paths.
    """
    entries = os.listdir(path1)
    modalities = []

    # Check for subdirectories representing different modalities
    for e in entries:
        full = os.path.join(path1, e)
        if os.path.isdir(full):
            modalities.append(e)

    if modalities:
        # Default to a specific modality if not provided by user
        if recursive_modalities is None:
            recursive_modalities = [
                m for m in modalities if m.upper() == "PET"
            ] or modalities

        collected = []
        pattern = re.compile(r"image_(\d+)\.jpg$", re.IGNORECASE)

        for m in recursive_modalities:
            sub_dir = os.path.join(path1, m)
            if not os.path.isdir(sub_dir):
                continue
            for fname in os.listdir(sub_dir):
                if pattern.match(fname):
                    collected.append(os.path.join(m, fname).replace("\\", "/"))

        # Sort files by their numerical index
        collected = sorted(
            collected,
            key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".jpg")[0]),
        )
        if k is not None:
            collected = collected[:k]
        return np.array(collected)
    else:
        # Handle flat directory structure
        total_list1 = [f for f in entries if f.lower().endswith(".jpg")]
        total_list1 = sorted(
            total_list1, key=lambda x: int(x.split("_")[-1].split(".jpg")[0])
        )
        if k is None:
            return np.array(total_list1)
        else:
            return np.array(total_list1[:k])


class Histo_Dataset(Dataset):
    """Custom dataset loader for histological/medical images."""

    def __init__(self, image1_dir, image1_list, transform=None):
        self.image1_dir = image1_dir
        self.image1_list = image1_list
        self.transform = transform

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):
        img1_path = os.path.join(self.image1_dir, self.image1_list[index])
        image1 = io.imread(img1_path)

        if self.transform is not None:
            image1 = self.transform(image1)

        return image1


def load_transformed_dataset():
    """Builds transforms, splits data, and returns train/eval datasets."""
    data_transforms = [
        transforms.ToTensor(),
        transforms.Lambda(repeat_if_gray),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(_my_normalization),
    ]
    data_transform = transforms.Compose(data_transforms)

    data_size = None
    # Anonymized dataset path
    TRAIN_IMAGE_DIR = f"/path/to/your/dataset_directory/dataset{SELECTED_CENTER}/unlabelled_img_patches"

    img1_list = get_images_list(
        TRAIN_IMAGE_DIR, k=data_size, recursive_modalities=["PET"]
    )

    if img1_list.size == 0:
        raise RuntimeError(
            f"No valid images found in {TRAIN_IMAGE_DIR} (e.g., matching PET/image_#.jpg)"
        )

    ratio = 0.9
    idxs = np.random.RandomState(2023).permutation(img1_list.shape[0])
    split = int(img1_list.shape[0] * ratio)
    train_index = idxs[:split]
    valid_index = idxs[split:]

    train_dataset = Histo_Dataset(
        TRAIN_IMAGE_DIR, img1_list[train_index], transform=data_transform
    )
    eval_dataset = Histo_Dataset(
        TRAIN_IMAGE_DIR, img1_list[valid_index], transform=data_transform
    )

    return train_dataset, eval_dataset


def reverse_transforms_image(image):
    """Reverts image tensors back to PIL Images for visualization."""
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    # Take the first image if a batch is passed
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return reverse_transforms(image)


def get_beta_schedule(betas):
    """Computes and returns the alpha/beta schedules for the diffusion process."""
    schedule = {}
    schedule["alphas"] = 1.0 - betas
    schedule["alphas_cumprod"] = torch.cumprod(schedule["alphas"], dim=0)
    schedule["alphas_cumprod_prev"] = F.pad(
        schedule["alphas_cumprod"][:-1], (1, 0), value=1.0
    )
    schedule["sqrt_recip_alphas"] = torch.sqrt(1.0 / schedule["alphas"])
    schedule["sqrt_alphas_cumprod"] = torch.sqrt(schedule["alphas_cumprod"])
    schedule["sqrt_one_minus_alphas_cumprod"] = torch.sqrt(
        1.0 - schedule["alphas_cumprod"]
    )
    schedule["posterior_variance"] = (
        betas
        * (1.0 - schedule["alphas_cumprod_prev"])
        / (1.0 - schedule["alphas_cumprod"])
    )
    return schedule


def get_loss(noise, noise_pred, time_stamps, betas_schedule, gpu):
    """Computes the weighted MSE loss for the diffusion model."""
    t = time_stamps.cpu()
    snr = 1.0 / (1 - betas_schedule["alphas_cumprod"][t]) - 1
    k = 1.0
    gamma = 1.0
    lambda_t = 1.0 / ((k + snr) ** gamma)
    lambda_t = lambda_t.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(gpu)

    n = noise.shape[1] * noise.shape[2] * noise.shape[3]
    loss = torch.sum(lambda_t * F.mse_loss(noise, noise_pred, reduction="none")) / n
    return loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0, path="checkpoint.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, epoch=None, ddp=False):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, ddp)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, ddp)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, ddp):
        """Saves model when validation loss decreases. Compatible with single GPU and DDP."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        if epoch is not None:
            safe_loss = (
                float(val_loss) if not isinstance(val_loss, (float, int)) else val_loss
            )
            weight_path = (
                self.path[:-4] + "_" + str(epoch) + "_" + (f"{safe_loss:.5f}") + ".pth"
            )
        else:
            weight_path = self.path

        # Extract state_dict appropriately whether using DDP or not
        if ddp and hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        torch.save(
            {
                "epoch": epoch,
                "loss": float(val_loss),
                "model_state_dict": state_dict,
            },
            weight_path,
        )

        self.val_loss_min = val_loss
