import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from skimage import io
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

from losses import CELoss, Tversky_Loss, FLoss, SSLoss, PolyLogLoss
from model import GuidedSegmentationModel, LearnableNorm

# ==============================================================================
# Helper Functions for Data Processing and Logging
# ==============================================================================


def ensure_three_channel(t):
    """Ensures tensor has 3 channels. Repeats single channel 3 times if needed."""
    if t.dim() == 3 and t.shape[0] == 1:
        return t.repeat(3, 1, 1)
    return t


def normalize_m11(t):
    """Normalizes tensor to [-1, 1] range."""
    return t * 2 - 1


SELECTED_CENTER = ""
LOG_FILE = f"/path/to/project/logs/ct_fuse/log_{SELECTED_CENTER}.txt"


def _init_log_file_once():
    """Initializes the CSV log file header from the main process."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0 and not os.path.exists(LOG_FILE):
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("mode,epoch,batch,total_loss,ss_loss,focal_loss\n")


def _append_loss_line(mode, epoch, batch_idx, total_loss, ss_loss, focal_loss):
    """Appends a row of loss metrics to the log file from the main process."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"{mode},{epoch},{batch_idx},{total_loss:.6f},{ss_loss:.6f},{focal_loss:.6f}\n"
            )
    except Exception:
        pass  # Fail silently to avoid interrupting training


# ==============================================================================
# Environment & Hyperparameters
# ==============================================================================

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

seed = 66
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_CHANNELS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
HEAD_LR_MULT = 2.0
EPOCHS = 150
NUM_CLASSES = 2
LOSS_SS_LAMBDA = 1.0  # Weight for SSLoss within the combined SSFL Loss

# AMP and LR Scheduling configuration
WARMUP_EPOCHS = 5
WARMUP_STEPS = 1000
MAX_GRAD_NORM = 1.0
SCHEDULER = "cosine"  # Options: 'cosine' | 'plateau' | 'step' | 'none'
MIN_LR = 1e-7
STEP_SIZE = 40
STEP_GAMMA = 0.5
PLATEAU_PATIENCE = 3
PLATEAU_FACTOR = 0.5

# Anonymized dataset and label paths
TRAIN_PET_DIR = f"/path/to/dataset_{SELECTED_CENTER}/train/image_patches/PET"
TRAIN_CT_DIR = f"/path/to/dataset_{SELECTED_CENTER}/train/image_patches/CT"
TRAIN_LABEL_PATH = f"/path/to/dataset_{SELECTED_CENTER}/train/label_patches.mat"

MODEL_TYPE = "diff_fuse_SSFL"
LOSS_TYPE = f"SSFL_{BATCH_SIZE}"

transformations = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(ensure_three_channel),
        transforms.Lambda(normalize_m11),
    ]
)


def my_transforms(image1, image2, mask):
    """Applies synchronized random augmentations across multi-modal inputs and masks."""
    if random.random() > 0.5:
        image1 = TF.vflip(image1)
        image2 = TF.vflip(image2)
        mask = TF.vflip(mask)

    if random.random() > 0.5:
        image1 = TF.hflip(image1)
        image2 = TF.hflip(image2)
        mask = TF.hflip(mask)

    if random.random() > 0.7:
        # Apply identical blur to maintain spatial consistency across modalities
        blur_kernels = [3, 3]
        blur_sigma = [random.uniform(0.5, 1.5), random.uniform(0.5, 1.5)]
        image1 = TF.gaussian_blur(image1, blur_kernels, blur_sigma)
        image2 = TF.gaussian_blur(image2, blur_kernels, blur_sigma)

    return image1, image2, mask


# ==============================================================================
# Utility Classes (EarlyStopping & Dataset)
# ==============================================================================


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

    def __call__(
        self,
        val_loss,
        model,
        epoch=None,
        ddp=False,
        optimizer=None,
        scheduler=None,
        scaler=None,
        global_step=None,
    ):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(
                val_loss, model, epoch, ddp, optimizer, scheduler, scaler, global_step
            )
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(
                val_loss, model, epoch, ddp, optimizer, scheduler, scaler, global_step
            )
            self.counter = 0

    def save_checkpoint(
        self,
        val_loss,
        model,
        epoch,
        ddp,
        optimizer=None,
        scheduler=None,
        scaler=None,
        global_step=None,
    ):
        """Saves model and optimizer states when validation loss decreases."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )

        weight_path = self.path
        if epoch is not None:
            weight_path = f"{self.path[:-4]}_{epoch}_{str(val_loss.tolist())[:7]}.pth"

        state = {
            "epoch": epoch,
            "loss": float(val_loss),
            "model_state_dict": (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "global_step": int(global_step) if global_step is not None else 0,
            "early_stopping": {
                "best_score": (
                    float(self.best_score) if self.best_score is not None else None
                ),
                "counter": int(self.counter),
                "val_loss_min": (
                    float(self.val_loss_min)
                    if np.isfinite(self.val_loss_min)
                    else float("inf")
                ),
                "patience": int(self.patience),
                "delta": float(self.delta),
            },
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        torch.save(state, weight_path)
        self.val_loss_min = val_loss


def resume_if_needed(gpu, args, model, optimizer, scheduler, scaler):
    """Resumes training from a checkpoint if provided in args. Returns epoch, step, and ES state."""
    resume_path = args.get("resume_path")
    if not resume_path or not os.path.isfile(resume_path):
        return 0, 0, None

    map_loc = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(resume_path, map_location=map_loc)

    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model_state_dict"], strict=True)

    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("scheduler_state_dict") is not None and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ckpt.get("scaler_state_dict") is not None and scaler is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    rng = ckpt.get("rng_state", None)
    if rng is not None:
        try:
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        except Exception:
            pass

    start_epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    es_state = ckpt.get("early_stopping", None)

    if es_state is not None:
        print(
            f"[Resume] best_score={es_state.get('best_score')}, "
            f"val_loss_min={es_state.get('val_loss_min')}, "
            f"counter={es_state.get('counter')}"
        )

    print(
        f"[Resume] Loaded checkpoint from {resume_path} at epoch={start_epoch}, global_step={global_step}"
    )
    return start_epoch, global_step, es_state


def get_images_list(path1, k=None):
    total_list1 = os.listdir(path1)
    total_list1 = sorted(
        total_list1, key=lambda x: int(x.split("_")[-1].split(".jpg")[0])
    )
    return np.array(total_list1) if k is None else np.array(total_list1[:k])


from scipy.io import loadmat


class Histo_Dataset(Dataset):
    def __init__(
        self, pet_dir, ct_dir, image_list, label_list, transform=None, use_aug=True
    ):
        self.pet_dir = pet_dir
        self.ct_dir = ct_dir
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform
        self.use_aug = use_aug

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_name = self.image_list[index]
        pet_path = os.path.join(self.pet_dir, img_name)
        ct_path = os.path.join(self.ct_dir, img_name)

        image_pet = io.imread(pet_path)
        image_ct = io.imread(ct_path)
        mask = self.label_list[index]

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

        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.uint8)

        if self.transform is not None:
            image_pet = self.transform(image_pet)
            image_ct = self.transform(image_ct)
            mask = torch.from_numpy(mask).long()

        if self.use_aug:
            image_pet, image_ct, mask_t = my_transforms(
                image_pet, image_ct, mask.unsqueeze(0).float()
            )
            mask = mask_t.squeeze(0).long()

        return image_pet, image_ct, mask


# ==============================================================================
# Training & Evaluation Loops
# ==============================================================================


def class_weights(y_predict, y_true):
    fscores = np.zeros(NUM_CLASSES)
    for i in range(y_true.shape[0]):
        class_f1_scores = []
        _, y_true_idx = torch.max(y_true[i], 0)
        _, y_pred_idx = torch.max(y_predict[i], 0)
        for c in range(NUM_CLASSES):
            true_arr = (y_true_idx == c).reshape(-1).cpu().numpy()
            pred_arr = (y_pred_idx == c).reshape(-1).cpu().numpy()
            class_f1_scores.append(f1_score(true_arr, pred_arr, zero_division=1))
        fscores += class_f1_scores
    return fscores


def eval_epoch(eval_loader, model, gpu, epoch, early_stopping=None):
    with torch.no_grad():
        model.eval()
        val_loss = []
        p_bar = tqdm(eval_loader)
        _init_log_file_once()
        total_items = 0.0
        f_scores = np.zeros(NUM_CLASSES)

        for batch_idx, (h_pet, h_ct, true_label) in enumerate(p_bar):
            h_pet = h_pet.to(gpu, non_blocking=False)
            h_ct = h_ct.to(gpu, non_blocking=False)
            true_label = true_label.squeeze(1).to(gpu, non_blocking=False)
            target_label = (
                F.one_hot(true_label.long(), NUM_CLASSES).permute(0, 3, 1, 2).float()
            )

            with torch.amp.autocast("cuda", enabled=True):
                predicted_label = model(h_pet, h_ct)
                f_scores += class_weights(predicted_label, target_label)
                total_items += target_label.shape[0]

                ss_loss = SSLoss()(predicted_label, target_label)
                f_loss = FLoss(2.0)(predicted_label, target_label)
                loss = LOSS_SS_LAMBDA * ss_loss + f_loss

            val_loss.append(loss.item())
            _append_loss_line(
                "val", epoch, batch_idx, loss.item(), ss_loss.item(), f_loss.item()
            )

            p_bar.set_description(f"Epoch {epoch}")
            p_bar.set_postfix(loss=loss.item())

        f_scores = f_scores / total_items
        print("f_scores:", f_scores)
        print("mean f_scores:", np.mean(f_scores[1:]))
        print(f"Epoch: {epoch}\tval_loss {np.mean(val_loss):.4f}")
        return np.mean(val_loss)


def cleanup():
    dist.destroy_process_group()


# ==============================================================================
# Main Execution
# ==============================================================================


def main(gpu, args):
    torch.manual_seed(66)
    np.random.seed(66)
    random.seed(66)
    rank = args["nr"] * args["gpus"] + gpu

    # Note: Windows systems generally do not support NCCL natively. Use 'gloo' if required.
    dist.init_process_group("nccl", rank=rank, world_size=args["world_size"])
    torch.cuda.set_device(gpu)

    data_size = None
    backbone = "Unet"

    img1_list = get_images_list(TRAIN_PET_DIR, k=data_size)
    label_file = loadmat(TRAIN_LABEL_PATH)
    train_labels = label_file["data"]

    ratio = 0.9
    idxs = np.random.RandomState(2026).permutation(img1_list.shape[0])
    split = int(img1_list.shape[0] * ratio)
    train_index = idxs[:split]
    valid_index = idxs[split:]

    train_dataset = Histo_Dataset(
        TRAIN_PET_DIR,
        TRAIN_CT_DIR,
        img1_list[train_index],
        train_labels[train_index],
        transform=transformations,
        use_aug=True,
    )
    eval_dataset = Histo_Dataset(
        TRAIN_PET_DIR,
        TRAIN_CT_DIR,
        img1_list[valid_index],
        train_labels[valid_index],
        transform=transformations,
        use_aug=False,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args["world_size"], rank=rank
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )

    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset, num_replicas=args["world_size"], rank=rank, shuffle=False
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        sampler=eval_sampler,
    )

    fusion_config = {"enabled": True, "fusion_channels": 16}
    model = GuidedSegmentationModel(
        dim=64,
        channels=3,
        num_classes=2,
        pet_chkpt_path=args["load_from_pet_chkpt"],
        ct_chkpt_path=args["load_from_ct_chkpt"],
        time_step=50,
        fusion_config=fusion_config,
        enable_encoder_zero_layer=True,
    ).to(gpu)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params (decoder head only):", trainable_params)
    print(
        "Total params (incl. frozen encoders):",
        sum(p.numel() for p in model.parameters()),
    )

    model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    print(
        f"Optimizer configured for trainable decoder head params with LR={LEARNING_RATE}."
    )

    if SCHEDULER == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    elif SCHEDULER == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=PLATEAU_FACTOR,
            patience=PLATEAU_PATIENCE,
            min_lr=MIN_LR,
            verbose=True,
        )
    elif SCHEDULER == "step":
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA)
    else:
        scheduler = None

    checkpoint_path = args["checkpoints_path"]
    os.makedirs(checkpoint_path, exist_ok=True)

    if gpu == 0:
        early_stopping = EarlyStopping(
            patience=10,
            verbose=True,
            delta=0.0001,
            path=os.path.join(checkpoint_path, f"{BATCH_SIZE}_{LEARNING_RATE}.pth"),
        )
    else:
        early_stopping = None

    train_losses = []
    eval_losses = []
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    start_time = time.process_time()

    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    total_warmup_steps = min(WARMUP_STEPS, WARMUP_EPOCHS * steps_per_epoch)

    start_epoch, global_step, es_state = resume_if_needed(
        gpu, args, model, optimizer, scheduler, scaler
    )
    if gpu == 0 and es_state is not None:
        early_stopping.best_score = es_state.get("best_score", None)
        early_stopping.counter = es_state.get("counter", 0)
        early_stopping.val_loss_min = es_state.get("val_loss_min", np.inf)
    dist.barrier()

    for epoch in range(start_epoch, EPOCHS):
        print(f"epoch {epoch + 1}/{EPOCHS}")
        train_sampler.set_epoch(epoch)
        amp_enabled_epoch = epoch >= WARMUP_EPOCHS

        model.train()
        losses = []
        p_bar = tqdm(train_dataloader)
        _init_log_file_once()

        for batch_idx, (h_pet, h_ct, true_label) in enumerate(p_bar):
            h_pet = h_pet.to(gpu, non_blocking=False)
            h_ct = h_ct.to(gpu, non_blocking=False)
            true_label = true_label.squeeze(1).to(gpu, non_blocking=False)
            target_label = (
                F.one_hot(true_label.long(), NUM_CLASSES).permute(0, 3, 1, 2).float()
            )

            with torch.amp.autocast("cuda", enabled=amp_enabled_epoch):
                predicted_label = model(h_pet, h_ct)
                ss_loss = SSLoss()(predicted_label, target_label)
                f_loss = FLoss(2.0)(predicted_label, target_label)
                loss = LOSS_SS_LAMBDA * ss_loss + f_loss

            optimizer.zero_grad(set_to_none=True)
            if amp_enabled_epoch:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            losses.append(loss.item())
            _append_loss_line(
                "train",
                epoch + 1,
                batch_idx,
                loss.item(),
                ss_loss.item(),
                f_loss.item(),
            )

            # Linear warmup logic across parameter groups
            if global_step < total_warmup_steps:
                if global_step == 0:
                    for pg in optimizer.param_groups:
                        pg["initial_target_lr"] = pg["lr"]

                warmup_scale = (global_step + 1) / total_warmup_steps
                for pg in optimizer.param_groups:
                    target_lr = pg["initial_target_lr"]
                    pg["lr"] = target_lr * warmup_scale
                global_step += 1
            else:
                if global_step == total_warmup_steps:
                    for pg in optimizer.param_groups:
                        pg["lr"] = pg["initial_target_lr"]
                global_step += 1

            p_bar.set_description(f"Epoch {epoch + 1}")
            p_bar.set_postfix(loss=loss.item())

        train_loss = float(np.mean(losses))
        print(f"\nEpoch: {epoch + 1}\ttotal_loss {train_loss:.4f}")

        eval_loss = eval_epoch(eval_dataloader, model, gpu, epoch + 1, early_stopping)
        mean_eval_loss = torch.tensor(eval_loss / args["gpus"]).to(gpu)
        mean_train_loss = torch.tensor(train_loss / args["gpus"]).to(gpu)

        dist.barrier()
        dist.all_reduce(mean_eval_loss)
        dist.all_reduce(mean_train_loss)

        print(
            f"gpu {gpu} eval_loss:{eval_loss}, mean_loss:{mean_eval_loss.cpu().numpy()}"
        )

        if gpu == 0:
            early_stopping(
                mean_eval_loss.cpu().numpy(),
                model,
                epoch + 1,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
            )

        if early_stopping and early_stopping.early_stop:
            print("Early stop!")
            if scheduler is not None and gpu == 0:
                current_lr = (
                    scheduler.optimizer.param_groups[0]["lr"]
                    if SCHEDULER != "plateau"
                    else optimizer.param_groups[0]["lr"]
                )
                print(f"Final LR (before stop): {current_lr:.6e}")
            break

        # Schedulers step after warmup concludes to prevent interference
        if scheduler is not None and global_step > total_warmup_steps:
            if SCHEDULER == "plateau":
                scheduler.step(mean_eval_loss.cpu().item())
            else:
                scheduler.step()
            if gpu == 0:
                current_lr = scheduler.optimizer.param_groups[0]["lr"]
                print(f"LR after epoch {epoch+1}: {current_lr:.6e}")

        train_losses.append(mean_train_loss.cpu().numpy())
        eval_losses.append(mean_eval_loss.cpu().numpy())

    current_time = time.process_time()
    print(f"Total Time Elapsed={current_time - start_time:12.5f} seconds")

    plots_path = f"/path/to/project/plots_{SELECTED_CENTER}/{MODEL_TYPE}"
    os.makedirs(plots_path, exist_ok=True)

    actual_epochs = np.arange(len(train_losses))

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    axes.plot(actual_epochs, train_losses, "tab:blue", label="training loss")
    axes.plot(actual_epochs, eval_losses, "tab:orange", label="validation loss")
    axes.set_title(
        f"Training and Validation Loss (pretrained model = diffusion, loss = (SS + Focal) Loss, data size = {data_size})",
        weight="bold",
        fontsize=7,
    )
    axes.set_xlabel("Epochs", weight="bold", fontsize=9)
    axes.set_ylabel("Loss", weight="bold", fontsize=9)
    axes.legend(loc="best")
    plt.savefig(
        os.path.join(plots_path, f"{backbone}_{LOSS_TYPE}loss_{data_size}.jpg"), dpi=300
    )

    if gpu == 0:
        torch.save(
            {"model_state_dict": model.module.state_dict()},
            os.path.join(checkpoint_path, "final_seg_model.pth"),
        )

    cleanup()


if __name__ == "__main__":
    args = {}
    args["gpus"] = 1
    args["nr"] = 0
    args["world_size"] = args["gpus"]

    # Anonymized generic paths for checkpoints and logging
    args["checkpoints_path"] = (
        f"/path/to/project/snapshots_{SELECTED_CENTER}/{MODEL_TYPE}/"
    )
    args["load_from_pet_chkpt"] = r"/path/to/pretrained/pet_encoder_weights.pth"
    args["load_from_ct_chkpt"] = r"/path/to/pretrained/ct_encoder_weights.pth"
    args["resume_path"] = None

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29507"

    print(args["gpus"])
    mp.spawn(main, args=(args,), nprocs=args["gpus"])
