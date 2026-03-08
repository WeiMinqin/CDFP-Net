import math
import torch
import torch.nn as nn
from functools import partial
from inspect import isfunction
from einops import rearrange

# ================== Common Utilities ==================


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# ================== LearnableNorm ==================


class LearnableNorm(nn.Module):
    """
    Aligns with pretrained version: y = \lambda * ( (x-\mu)/\sqrt{\sigma^2+eps} ) + \delta
    eps is kept positive via softplus and bounded by a minimum value.
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


# ================== ZeroLayer & SE Modules ==================


class ZeroLayerModality(nn.Module):
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


class SEModule(nn.Module):
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
            w = 1 + w
        elif self.scale_mode == "double":
            w = 2 * w
        return x * w


# ================== Base Components ==================


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
        half = self.dim // 2
        emb_factor = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb_factor)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


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
        return self.act(x)


# ================== Graph Fusion Modules ==================


class GraphTransfer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphTransfer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.q_proj = nn.Linear(in_features, in_features)
        self.k_proj = nn.Linear(in_features, in_features)
        self.v_proj = nn.Linear(in_features, out_features)
        self.out_proj = nn.Linear(out_features, out_features)
        self.act = nn.ReLU(inplace=True)
        self.norm = (
            nn.GroupNorm(8, out_features)
            if out_features % 8 == 0
            else nn.GroupNorm(1, out_features)
        )

    def forward(self, q_feat, k_feat):
        q = self.q_proj(q_feat)
        k = self.k_proj(k_feat)
        v = self.v_proj(k_feat)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context = torch.matmul(attention_probs, v)
        out = self.out_proj(context)
        out = self.act(self.norm(out.transpose(1, 2))).transpose(1, 2)
        return out


class GCN(nn.Module):
    def __init__(self, in_features, out_features, n_layer=2):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                nn.Linear(in_features if i == 0 else out_features, out_features)
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, feat, adj):
        for i, layer in enumerate(self.layers):
            feat = layer(torch.matmul(adj, feat))
            if i < len(self.layers) - 1:
                feat = self.act(feat)
        return feat


class GraphLayer(nn.Module):
    def __init__(self, mod1_dim, mod2_dim, out_dim):
        super(GraphLayer, self).__init__()
        self.mod1_gcn = GCN(mod1_dim, out_dim, n_layer=2)
        self.mod2_gcn = GCN(mod2_dim, out_dim, n_layer=2)
        self.transfer = GraphTransfer(out_dim, out_dim)

    def forward(self, mod1_feat, mod2_feat):
        B, C_mod1, H, W = mod1_feat.shape
        _, C_mod2, _, _ = mod2_feat.shape

        mod1_feat_nodes = mod1_feat.view(B, C_mod1, -1).transpose(1, 2)
        mod2_feat_nodes = mod2_feat.view(B, C_mod2, -1).transpose(1, 2)

        # Build adjacency matrix (simple fully connected graph)
        adj = torch.ones(H * W, H * W, device=mod1_feat.device) / (H * W)

        # Intra-modality graph convolution
        mod1_gcn_out = self.mod1_gcn(mod1_feat_nodes, adj)
        mod2_gcn_out = self.mod2_gcn(mod2_feat_nodes, adj)

        # Cross-modality graph information transfer
        fused_feat = self.transfer(mod1_gcn_out, mod2_gcn_out)

        out = fused_feat.transpose(1, 2).view(B, -1, H, W)
        return out


# ================== Attention & ResNet Blocks ==================


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
        hidden = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> b h d (x y)", h=self.heads),
            (q, k, v),
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
        hidden = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (hh d) x y -> b hh d (x y)", hh=self.heads),
            (q, k, v),
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
        return self.fn(self.norm(x))


# ================== U-Net ==================


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
        encoder_only=False,
        enable_encoder_zero_layer=False,
        fusion_config=None,
    ):
        super().__init__()
        self.encoder_only = encoder_only
        self.channels = channels
        self.fusion_config = fusion_config
        self.enable_encoder_zero_layer = enable_encoder_zero_layer

        if self.enable_encoder_zero_layer:
            self.mod1_zero = ZeroLayerModality(channels)
            self.mod1_se = SEModule(channels, reduction=3, scale_mode="sigmoid")
        else:
            self.mod1_zero = nn.Identity()
            self.mod1_se = nn.Identity()

        init_dim = default(init_dim, dim // 3 * 2)
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.mod1_init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        self.mod1_downs = nn.ModuleList([])

        if self.fusion_config and self.fusion_config.get("enabled"):
            if self.enable_encoder_zero_layer:
                self.mod2_zero = ZeroLayerModality(channels)
                self.mod2_se = SEModule(channels, reduction=3, scale_mode="sigmoid")
            else:
                self.mod2_zero = nn.Identity()
                self.mod2_se = nn.Identity()
            self.mod2_init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
            self.mod2_downs = nn.ModuleList([])

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

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            mod1_down_block = nn.ModuleList(
                [
                    block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else nn.Identity(),
                ]
            )
            self.mod1_downs.append(mod1_down_block)

            if self.fusion_config and self.fusion_config.get("enabled"):
                mod2_down_block = nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
                self.mod2_downs.append(mod2_down_block)

        mid_dim = dims[-1]

        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.graph_fusion = GraphLayer(
            mod1_dim=dim * 8, mod2_dim=dim * 8, out_dim=dim * 8
        )

        self.ups = nn.ModuleList([])
        if not self.encoder_only:
            for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
                is_last = ind >= (len(in_out) - 1)
                decoder_in_channels = (
                    dim_out * 3
                    if self.fusion_config and self.fusion_config.get("enabled")
                    else dim_out * 2
                )

                self.ups.append(
                    nn.ModuleList(
                        [
                            block_klass(
                                decoder_in_channels, dim_in, time_emb_dim=time_dim
                            ),
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
            self.softmax = nn.Softmax(dim=1)

    def encoder_forward(self, x, time, x_mod2=None):
        """Executes only the encoder portion and returns bottleneck features and all skip connections."""
        x_mod1 = self.mod1_zero(x)
        x_mod1 = self.mod1_se(x_mod1)

        if exists(x_mod2) and self.fusion_config and self.fusion_config.get("enabled"):
            x_mod2 = self.mod2_zero(x_mod2)
            x_mod2 = self.mod2_se(x_mod2)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        mod1_skips = []
        x_mod1 = self.mod1_init_conv(x_mod1)
        for block1, block2, attn, downsample in self.mod1_downs:
            x_mod1 = block1(x_mod1, t)
            x_mod1 = block2(x_mod1, t)
            x_mod1 = attn(x_mod1)
            mod1_skips.append(x_mod1)
            x_mod1 = downsample(x_mod1)

        mod2_skips = []
        if exists(x_mod2) and self.fusion_config and self.fusion_config.get("enabled"):
            x_mod2 = self.mod2_init_conv(x_mod2)
            for i, (block1, block2, attn, downsample) in enumerate(self.mod2_downs):
                x_mod2 = block1(x_mod2, t)
                x_mod2 = block2(x_mod2, t)
                x_mod2 = attn(x_mod2)
                mod2_skips.append(x_mod2)
                x_mod2 = downsample(x_mod2)

        return x_mod1, mod1_skips, x_mod2, mod2_skips

    def forward(self, x_mod1, time, x_mod2=None):
        x_mod1 = self.mod1_zero(x_mod1)
        x_mod1 = self.mod1_se(x_mod1)

        if exists(x_mod2) and self.fusion_config and self.fusion_config.get("enabled"):
            x_mod2 = self.mod2_zero(x_mod2)
            x_mod2 = self.mod2_se(x_mod2)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        mod1_skips = []
        x_mod1 = self.mod1_init_conv(x_mod1)
        for block1, block2, attn, downsample in self.mod1_downs:
            x_mod1 = block1(x_mod1, t)
            x_mod1 = block2(x_mod1, t)
            x_mod1 = attn(x_mod1)
            mod1_skips.append(x_mod1)
            x_mod1 = downsample(x_mod1)

        mod2_skips = []
        if exists(x_mod2) and self.fusion_config and self.fusion_config.get("enabled"):
            x_mod2 = self.mod2_init_conv(x_mod2)
            for i, (block1, block2, attn, downsample) in enumerate(self.mod2_downs):
                x_mod2 = block1(x_mod2, t)
                x_mod2 = block2(x_mod2, t)
                x_mod2 = attn(x_mod2)
                mod2_skips.append(x_mod2)
                x_mod2 = downsample(x_mod2)

        if exists(x_mod2) and self.fusion_config and self.fusion_config.get("enabled"):
            x = self.graph_fusion(x_mod1, x_mod2)
        else:
            x = x_mod1

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        if not self.encoder_only:
            for ind, (b1, b2, attn, up) in enumerate(self.ups):
                if (
                    exists(x_mod2)
                    and self.fusion_config
                    and self.fusion_config.get("enabled")
                ):
                    x = torch.cat(
                        (x, mod1_skips[-1 - ind], mod2_skips[-1 - ind]), dim=1
                    )
                else:
                    x = torch.cat((x, mod1_skips[-1 - ind]), dim=1)
                x = b1(x, t)
                x = b2(x, t)
                x = attn(x)
                x = up(x)
            x = self.final_conv(x)
            x = self.softmax(x)
        return x


# ================== Guided Segmentation Model ==================


class GuidedSegmentationModel(nn.Module):
    """
    Implementation Strategy: Uses frozen pretrained diffusion U-Nets as feature extractors,
    and only trains a newly constructed fusion decoder head.
    """

    def __init__(
        self,
        dim,
        channels,
        num_classes,
        mod1_chkpt_path,
        mod2_chkpt_path,
        time_step=50,
        fusion_config=None,
        enable_encoder_zero_layer=False,
    ):
        super().__init__()
        self.time_step = time_step
        self.fusion_config = fusion_config

        # 1. Load and freeze Modality 1 encoder
        mod1_unet = Unet(
            dim=dim,
            channels=channels,
            out_dim=channels,
            dim_mults=(1, 2, 4, 8),
            fusion_config=None,
            enable_encoder_zero_layer=enable_encoder_zero_layer,
        )
        mod1_chkpt = torch.load(mod1_chkpt_path, map_location="cpu")
        mod1_state_dict = mod1_chkpt.get("model_state_dict", mod1_chkpt)
        if "net.final_conv.1.weight" in mod1_state_dict:
            mod1_state_dict = {
                k.replace("net.", ""): v for k, v in mod1_state_dict.items()
            }
        mod1_unet.load_state_dict(mod1_state_dict, strict=False)
        self.mod1_encoder = mod1_unet.eval()
        for param in self.mod1_encoder.parameters():
            param.requires_grad = False
        print(f"Loaded and froze Modality 1 encoder from {mod1_chkpt_path}")

        # 2. Load and freeze Modality 2 encoder
        mod2_unet = Unet(
            dim=dim,
            channels=channels,
            out_dim=channels,
            dim_mults=(1, 2, 4, 8),
            fusion_config=None,
            enable_encoder_zero_layer=enable_encoder_zero_layer,
        )
        mod2_chkpt = torch.load(mod2_chkpt_path, map_location="cpu")
        mod2_state_dict = mod2_chkpt.get("model_state_dict", mod2_chkpt)
        if "net.final_conv.1.weight" in mod2_state_dict:
            mod2_state_dict = {
                k.replace("net.", ""): v for k, v in mod2_state_dict.items()
            }
        mod2_unet.load_state_dict(mod2_state_dict, strict=False)
        self.mod2_encoder = mod2_unet.eval()
        for param in self.mod2_encoder.parameters():
            param.requires_grad = False
        print(f"Loaded and froze Modality 2 encoder from {mod2_chkpt_path}")

        # 3. Create trainable decoder head
        block_klass = partial(ResnetBlock, groups=8)
        dims = [dim // 3 * 2, *map(lambda m: dim * m, (1, 2, 4, 8))]
        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]
        time_dim = dim * 4

        # Bottleneck fusion module
        self.graph_fusion = GraphLayer(
            mod1_dim=mid_dim, mod2_dim=mid_dim, out_dim=mid_dim
        )
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            decoder_in_channels = dim_out * 3
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(decoder_in_channels, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, num_classes, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_mod1, x_mod2, time=None):
        b, _, h, w = x_mod1.shape
        device = x_mod1.device

        # Use fixed time step
        t = torch.full((b,), self.time_step, device=device, dtype=torch.long)

        # 1. Extract features using frozen encoders
        with torch.no_grad():
            mod1_bottle, mod1_skips, _, _ = self.mod1_encoder.encoder_forward(x_mod1, t)
            mod2_bottle, mod2_skips, _, _ = self.mod2_encoder.encoder_forward(x_mod2, t)

        # 2. Process through trainable decoder head
        x = self.graph_fusion(mod1_bottle, mod2_bottle)

        time_emb = self.mod1_encoder.time_mlp(t)

        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        for ind, (b1, b2, attn, up) in enumerate(self.ups):
            x = torch.cat((x, mod1_skips[-1 - ind], mod2_skips[-1 - ind]), dim=1)
            x = b1(x, time_emb)
            x = b2(x, time_emb)
            x = attn(x)
            x = up(x)

        x = self.final_conv(x)
        x = self.softmax(x)
        return x
