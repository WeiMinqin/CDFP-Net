import os
import shutil
import time
import math
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.io import savemat

# torchrun --nproc_per_node=4 SS_diff.py
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import utils
from model import DiffusionNet

# ========== Scheduling / Hyperparameters ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    torch.cuda.set_per_process_memory_fraction(0.9)
except Exception:
    pass

IMG_SIZE = 256
EPOCHS = 100
BATCH_SIZE = 1
BASE_LR = 0.0001  # Unified base learning rate
MIN_LR_RATIO = 0.1
MIN_LR = BASE_LR * MIN_LR_RATIO
WARMUP_EPOCHS = 1  # 1 epoch linear warmup
GRAD_CLIP = 1.0
T = 1000
DATA_TYPE = "diff_quadratic"
PRED_MODE = "v"  # 'v' or 'eps'
NUM_T_BINS = 10
SAMPLING_METHOD = "DDPM"
SELECTED_CENTER = ""

# Define betas and their derived schedules
betas = utils.quadratic_beta_schedule(timesteps=T)
betas_schedule = utils.get_beta_schedule(betas)


# ========== Cosine LR with Warmup ==========
def get_cosine_lr(
    epoch: int, total_epochs: int, base_lr: float, warmup_epochs: int, min_lr: float
):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def cleanup():
    # No need to destroy process group for a single GPU
    pass


@torch.no_grad()
def sample_timestep(model, x, t, sampling_method="DDIM"):
    """Generate images based on sampling method (DDPM or DDIM). Converts v-pred to eps."""
    betas_t = utils.get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = utils.get_index_from_list(
        betas_schedule["sqrt_one_minus_alphas_cumprod"], t, x.shape
    )
    sqrt_recip_alphas_t = utils.get_index_from_list(
        betas_schedule["sqrt_recip_alphas"], t, x.shape
    )
    sqrt_alphas_cumprod_t = utils.get_index_from_list(
        betas_schedule["sqrt_alphas_cumprod"], t, x.shape
    )

    # Model output -> unified to eps for subsequent inference
    raw_pred = model(x, t)
    if PRED_MODE == "v":
        # eps = v * sqrt(alpha_bar) + sqrt(1-alpha_bar) * x_t
        eps_pred = (
            raw_pred * sqrt_alphas_cumprod_t + sqrt_one_minus_alphas_cumprod_t * x
        )
    else:  # eps mode
        eps_pred = raw_pred

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * eps_pred / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = utils.get_index_from_list(
        betas_schedule["posterior_variance"], t, x.shape
    )

    if sampling_method == "DDIM":
        ddim_eta = 0.0
        ddim_noise = torch.randn_like(x) if ddim_eta > 0 else 0
        ddim_mean = (
            model_mean + ddim_eta * torch.sqrt(posterior_variance_t) * ddim_noise
        )
        return ddim_mean
    else:
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, gpu, epoch, sampling_method=None):
    """Uses the model's sampling method and avoids redundant figures."""
    if not is_main_process():  # Prevent duplicate saving in multi-GPU setup
        return

    # Auto-fetch sampling method (DDP wrapper compatible)
    if sampling_method is None:
        inner = model.module if hasattr(model, "module") else model
        sampling_method = getattr(inner, "sampling_method", "DDIM")

    img_size = IMG_SIZE
    torch.manual_seed(42)
    img = torch.randn((1, 3, img_size, img_size), device=gpu)
    num_images = 100
    stepsize = int(T / num_images)
    all_images = []

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=gpu, dtype=torch.long)
        img = sample_timestep(model, img, t, sampling_method=sampling_method)
        if i % stepsize == 0:
            all_images.append(img)

    fig, axs = plt.subplots(10, 10, figsize=(8, 8))
    x = 0
    for i in range(10):
        for j in range(10):
            out_img = utils.reverse_transforms_image(all_images[x].detach().cpu())
            axs[i, j].imshow(out_img)
            axs[i, j].axis("off")
            x += 1

    save_dir = f"/path/to/project/images_{SELECTED_CENTER}/{DATA_TYPE}"
    save_path = os.path.join(save_dir, f"image_{epoch}.jpg")

    try:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    except Exception as e:
        print(f"[WARN] Failed to save sampled image: {e}\nPath: {save_path}")
    finally:
        plt.close(fig)
        plt.close("all")


def initialize_weights(model):
    """Initializes weights according to a normal distribution."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight.data, 0.0, 0.01)


def train_epoch(train_dataloader, model, optimizer, gpu, epoch, args, scaler):
    model.train()
    losses = []
    bin_sums = torch.zeros(NUM_T_BINS, device=gpu)
    bin_counts = torch.zeros(NUM_T_BINS, device=gpu)

    # Create progress bar only on main process
    p_iter = (
        tqdm(train_dataloader, dynamic_ncols=True, desc=f"Train Epoch {epoch}")
        if is_main_process()
        else train_dataloader
    )

    for img_batch in p_iter:
        optimizer.zero_grad(set_to_none=True)
        img_batch = img_batch.to(gpu, non_blocking=False)
        t = (
            torch.randint(0, T, (img_batch.shape[0],))
            .long()
            .to(gpu, non_blocking=False)
        )
        x_noisy, noise = utils.forward_diffusion_sample(
            img_batch, t, betas_schedule, gpu
        )
        sqrt_alphas_cumprod_t = utils.get_index_from_list(
            betas_schedule["sqrt_alphas_cumprod"], t, x_noisy.shape
        )
        sqrt_one_minus_alphas_cumprod_t = utils.get_index_from_list(
            betas_schedule["sqrt_one_minus_alphas_cumprod"], t, x_noisy.shape
        )

        with torch.amp.autocast("cuda", enabled=True):
            pred = model(x_noisy, t)
            if PRED_MODE == "v":
                # v_target = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0
                v_target = (
                    sqrt_alphas_cumprod_t * noise
                    - sqrt_one_minus_alphas_cumprod_t * img_batch
                )
                per_pixel = (pred - v_target) ** 2
            else:  # eps mode
                per_pixel = (pred - noise) ** 2

            # SNR weighting
            snr = 1.0 / (1 - betas_schedule["alphas_cumprod"][t.cpu()]) - 1
            lambda_t = 1.0 / (1.0 + snr)
            lambda_t = lambda_t.to(gpu).view(-1, 1, 1, 1)
            per_pixel = lambda_t * per_pixel
            loss = per_pixel.mean()

        scaler.scale(loss).backward()
        if GRAD_CLIP and GRAD_CLIP > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # Statistics & time binning
        with torch.no_grad():
            losses.append(loss.item())
            bin_ids = (t.float() / T * NUM_T_BINS).clamp(max=NUM_T_BINS - 1).long()
            batch_losses = per_pixel.mean(dim=[1, 2, 3])
            for b_l, b_id in zip(batch_losses, bin_ids):
                bin_sums[b_id] += b_l
                bin_counts[b_id] += 1

        if is_main_process() and isinstance(p_iter, tqdm):
            p_iter.set_postfix(loss=loss.item())

    # Cross-process average before returning
    epoch_loss = sum(losses) / len(losses) if len(losses) > 0 else 0.0
    if dist.is_available() and dist.is_initialized():
        tensor_loss = torch.tensor(epoch_loss, device=gpu)
        dist.all_reduce(tensor_loss, op=dist.ReduceOp.SUM)
        epoch_loss = (tensor_loss / dist.get_world_size()).item()

    print(f"Epoch: {epoch}\ttotal_loss {epoch_loss:.4f}")

    with torch.no_grad():
        valid = bin_counts > 0
        bin_means = torch.zeros_like(bin_sums)
        bin_means[valid] = bin_sums[valid] / bin_counts[valid]
        bin_str = " | ".join(
            [f"{i}:{bin_means[i].item():.4f}" for i in range(NUM_T_BINS)]
        )
        print(f"[t-bin mean losses] {bin_str}")

    return epoch_loss


@torch.no_grad()
def eval_epoch(eval_dataloader, model, gpu, epoch, args, early_stopping=None):
    model.eval()
    losses = []
    bin_sums = torch.zeros(NUM_T_BINS, device=gpu)
    bin_counts = torch.zeros(NUM_T_BINS, device=gpu)
    p_iter = (
        tqdm(eval_dataloader, dynamic_ncols=True, desc=f"Eval Epoch {epoch}")
        if is_main_process()
        else eval_dataloader
    )

    for img_batch in p_iter:
        img_batch = img_batch.to(gpu, non_blocking=False)
        t = (
            torch.randint(0, T, (img_batch.shape[0],))
            .long()
            .to(gpu, non_blocking=False)
        )
        x_noisy, noise = utils.forward_diffusion_sample(
            img_batch, t, betas_schedule, gpu
        )
        sqrt_alphas_cumprod_t = utils.get_index_from_list(
            betas_schedule["sqrt_alphas_cumprod"], t, x_noisy.shape
        )
        sqrt_one_minus_alphas_cumprod_t = utils.get_index_from_list(
            betas_schedule["sqrt_one_minus_alphas_cumprod"], t, x_noisy.shape
        )

        pred = model(x_noisy, t)
        if PRED_MODE == "v":
            v_target = (
                sqrt_alphas_cumprod_t * noise
                - sqrt_one_minus_alphas_cumprod_t * img_batch
            )
            per_pixel = (pred - v_target) ** 2
        else:
            per_pixel = (pred - noise) ** 2

        snr = 1.0 / (1 - betas_schedule["alphas_cumprod"][t.cpu()]) - 1
        lambda_t = 1.0 / (1.0 + snr)
        lambda_t = lambda_t.to(gpu).view(-1, 1, 1, 1)
        per_pixel = lambda_t * per_pixel
        batch_loss = per_pixel.mean()
        losses.append(batch_loss.item())

        bin_ids = (t.float() / T * NUM_T_BINS).clamp(max=NUM_T_BINS - 1).long()
        sample_losses = per_pixel.mean(dim=[1, 2, 3])
        for s_l, b_id in zip(sample_losses, bin_ids):
            bin_sums[b_id] += s_l
            bin_counts[b_id] += 1

    mean_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    if dist.is_available() and dist.is_initialized():
        tensor_loss = torch.tensor(mean_loss, device=gpu)
        dist.all_reduce(tensor_loss, op=dist.ReduceOp.SUM)
        mean_loss = (tensor_loss / dist.get_world_size()).item()

    with torch.no_grad():
        valid = bin_counts > 0
        bin_means = torch.zeros_like(bin_sums)
        bin_means[valid] = bin_sums[valid] / bin_counts[valid]
        bin_str = " | ".join(
            [f"{i}:{bin_means[i].item():.4f}" for i in range(NUM_T_BINS)]
        )
        print(f"[Eval t-bin mean losses] {bin_str}")

    return mean_loss


# ========== Distributed Helper Functions ==========


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def is_main_process():
    return (
        (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    )


def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main(args):
    ddp_enabled, rank, world_size, local_rank = init_distributed()
    if ddp_enabled:
        gpu = torch.device(f"cuda:{local_rank}")
    else:
        gpu = DEVICE

    if is_main_process():
        print(f"DDP Enabled={ddp_enabled} | Rank={rank} | WorldSize={world_size}")

    data_size = None
    train_dataset, eval_dataset = utils.load_transformed_dataset()

    if ddp_enabled:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        sampler=eval_sampler,
    )

    model = DiffusionNet(
        dim=64,
        channels=3,
        sampling_method=SAMPLING_METHOD,
        enable_zero_layer=True,
    ).to(gpu)
    initialize_weights(model)

    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main_process():
        inner = model.module if hasattr(model, "module") else model
        print("Num params:", sum(p.numel() for p in inner.parameters()))

    optimizer = Adam(model.parameters(), lr=BASE_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    early_stopping = utils.EarlyStopping(
        patience=10,
        verbose=is_main_process(),
        delta=0.0001,
        path=os.path.join(args["checkpoints_path"], f"{BATCH_SIZE}_{BASE_LR}.pth"),
    )

    checkpoint_path = args["checkpoints_path"]
    if is_main_process():
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(
            f"/path/to/project/images_{SELECTED_CENTER}/{DATA_TYPE}",
            exist_ok=True,
        )

    # ========== Load Checkpoint (Resume optimizer/scaler) ==========
    if args["load_from_chkpt"] is not None:
        chkpt_file = args["load_from_chkpt"]
        if is_main_process():
            print("Loading checkpoint from:", chkpt_file)

        checkpoint = torch.load(chkpt_file, map_location=gpu)
        target_state = checkpoint.get("model_state_dict", checkpoint)

        if hasattr(model, "module"):
            model.module.load_state_dict(target_state, strict=False)
        else:
            model.load_state_dict(target_state, strict=False)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if is_main_process():
            print(f"[Resume] start from epoch {start_epoch}")
    else:
        start_epoch = 0

    train_losses, eval_losses, lrs = [], [], []
    start_time = time.process_time()

    for epoch in range(start_epoch, EPOCHS):
        if ddp_enabled:
            train_sampler.set_epoch(epoch)

        base_lr_now = get_cosine_lr(epoch, EPOCHS, BASE_LR, WARMUP_EPOCHS, MIN_LR)
        optimizer.param_groups[0]["lr"] = base_lr_now

        if is_main_process():
            print(f"[Epoch {epoch+1}/{EPOCHS}] lr={base_lr_now:.6e}")

        train_loss = train_epoch(
            train_dataloader, model, optimizer, gpu, epoch + 1, args, scaler
        )
        eval_loss = eval_epoch(eval_dataloader, model, gpu, epoch + 1, early_stopping)

        if is_main_process():
            print(f"eval_loss:{eval_loss:.6f}")
        lrs.append(base_lr_now)

        if is_main_process() and (epoch + 1) % 4 == 0:
            sample_plot_image(model, gpu, epoch + 1, sampling_method=SAMPLING_METHOD)

        if is_main_process():
            early_stopping(eval_loss, model, epoch, ddp=True)

        # Broadcast early stopping flag
        stop_flag = torch.tensor(1 if early_stopping.early_stop else 0, device=gpu)
        if ddp_enabled:
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            if is_main_process():
                print("Early stopping triggered at epoch", epoch)
            break

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

    current_time = time.process_time()
    if is_main_process():
        print(
            "Total Time Elapsed={:12.5} seconds".format(str(current_time - start_time))
        )
        np.save(os.path.join(checkpoint_path, "lrs.npy"), np.array(lrs))
        plots_path = f"/path/to/project/plots_{SELECTED_CENTER}/diff"
        os.makedirs(plots_path, exist_ok=True)

        actual_epochs = range(len(train_losses))
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))
        axes.plot(list(actual_epochs), train_losses, "tab:blue", label="Train Loss")
        axes.plot(list(actual_epochs), eval_losses, "tab:orange", label="Eval Loss")
        axes.set_title(f"Training and Validation Loss (data size={data_size})")
        axes.set_xlabel("Epochs")
        axes.set_ylabel("Loss")
        axes.legend()
        plt.savefig(f"{plots_path}/{DATA_TYPE}_loss_{data_size}.jpg", dpi=300)
        plt.close(fig)

    cleanup_ddp()


# ========== Top-level distributed worker (for mp.spawn) ==========
def _dist_worker(rank, world_size, base_args):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    main(base_args)


if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29502")

    args = {}
    args["checkpoints_path"] = (
        f"/path/to/project/snapshots_{SELECTED_CENTER}/{DATA_TYPE}/"
    )
    args["load_from_chkpt"] = None

    gpu_count = torch.cuda.device_count()

    # Top-level spawn function elevated, can be pickled normally
    if gpu_count > 1 and "WORLD_SIZE" not in os.environ:
        print(f"[Auto-DDP] Spawning {gpu_count} processes.")
        mp.spawn(_dist_worker, args=(gpu_count, args), nprocs=gpu_count, join=True)
    else:
        main(args)
