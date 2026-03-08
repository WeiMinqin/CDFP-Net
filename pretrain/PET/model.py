import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from inspect import isfunction
from einops import rearrange

# ========== Learnable Normalization & Zero Layer ==========


class LearnableNorm(nn.Module):
    """
    Learnable normalization (channel-wise).
    Uses softplus for eps to ensure numerical stability during training.
    """

    def __init__(self, num_channels, init_eps=1e-5):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.delta_param = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.raw_eps = nn.Parameter(torch.log(torch.exp(torch.tensor(init_eps)) - 1.0))
        self.min_eps = 1e-6

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        eps = torch.nn.functional.softplus(self.raw_eps) + self.min_eps
        var = torch.clamp(var, min=self.min_eps)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        return self.lambda_param * x_hat + self.delta_param


class ZeroLayerModality(nn.Module):
    """Modality Zero Layer: Conv -> SiLU -> LearnableNorm."""

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding)
        self.act = nn.SiLU(inplace=True)
        self.norm = LearnableNorm(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x


# ========== Minimal SE Module ==========


class SEModule(nn.Module):
    """Minimal Squeeze-and-Excitation Module."""

    def __init__(self, channels, reduction=4, scale_mode="sigmoid"):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )
        self.scale_mode = scale_mode

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        w = self.fc(y).view(b, c, 1, 1)

        if self.scale_mode == "offset":
            w = 1 + w  # amplify overall magnitude
        elif self.scale_mode == "double":
            w = 2 * w  # allow stronger weighting > 1

        return x * w


# ========== U-Net Core Components ==========


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        convnext_mult=2,
        encoder_only=False,
        enable_zero_layer=False,
    ):
        super().__init__()
        self.encoder_only = encoder_only
        self.channels = channels
        self.enable_zero_layer = enable_zero_layer

        if self.enable_zero_layer:
            self.modality_zero = ZeroLayerModality(channels)
            self.modality_se = SEModule(channels, reduction=3, scale_mode="sigmoid")
        else:
            self.modality_zero = nn.Identity()
            self.modality_se = nn.Identity()

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_se = SEModule(mid_dim, reduction=max(4, mid_dim // 32))

        if not self.encoder_only:
            for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
                is_last = ind >= (num_resolutions - 1)
                self.ups.append(
                    nn.ModuleList(
                        [
                            block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                            block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                            Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                            Upsample(dim_in) if not is_last else nn.Identity(),
                        ]
                    )
                )
            out_dim = default(out_dim, channels)
            self.final_conv = nn.Sequential(
                block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
            )

    def forward(self, x, time):
        # Input phase
        x = self.modality_zero(x)
        x = self.modality_se(x)
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        h = []

        # Downsampling
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.mid_se(x)

        # Upsampling
        if not self.encoder_only:
            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = block1(x, t)
                x = block2(x, t)
                x = attn(x)
                x = upsample(x)
            x = self.final_conv(x)

        return x


# ========== Top-Level Models ==========


class DiffusionNet(nn.Module):
    def __init__(self, dim, channels, sampling_method="DDPM", enable_zero_layer=False):
        super(DiffusionNet, self).__init__()
        self.net = Unet(
            dim=dim,
            channels=channels,
            dim_mults=(1, 2, 4, 8),
            enable_zero_layer=enable_zero_layer,
        )
        self.sampling_method = sampling_method

    def forward(self, x, time_stamps):
        e = self.net(x, time_stamps)
        return e


class SegNet(nn.Module):
    def __init__(self, dim, channels, num_classes, enable_zero_layer=False):
        super(SegNet, self).__init__()
        self.net = Unet(
            dim=dim,
            channels=channels,
            out_dim=num_classes,
            dim_mults=(1, 2, 4, 8),
            enable_zero_layer=enable_zero_layer,
        )

    def forward(self, x, time_stamps):
        e = self.net(x, time_stamps)
        return e
