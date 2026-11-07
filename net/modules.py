import torch.nn as nn
try:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    class DropPath(nn.Identity):
        pass
    def to_2tuple(x):
        return (x, x) if not isinstance(x, tuple) else x
    def trunc_normal_(tensor, mean=0., std=1.):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std)
import torch
from bisect import bisect
import torch.nn.functional as F
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SQAWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.

    SQA (spatial query attention), adapted from SNR-EQ-JSCC:
    - SNR embedding: x <- x + g_snr(mu) before qkv projection.
    - CAQ: q <- a_mu * q + b_mu before qk attention score calculation.

    The additions are disabled unless use_snreq=True, so the original SwinJSCC
    paths keep the same behaviour.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., use_snreq=False, use_caq=True,
                 use_snr_embedding=True, snr_scale=20.0, snr_eq_hidden_dim=128,
                 snr_emb_scale=0.1):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.use_snreq = bool(use_snreq)
        self.use_caq = bool(use_snreq and use_caq)
        self.use_snr_embedding = bool(use_snreq and use_snr_embedding)
        self.snr_scale = float(snr_scale)
        self.snr_eq_hidden_dim = int(snr_eq_hidden_dim)
        self.snr_emb_scale = float(snr_emb_scale)
        self.last_caq_a = None
        self.last_caq_b = None
        self.last_caq_snr = None

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable SNR embedding g_thetaSNR(mu), broadcast to all tokens.
        if self.use_snr_embedding:
            hidden = max(4, int(self.snr_eq_hidden_dim))
            self.snr_embed = nn.Sequential(
                nn.Linear(1, hidden),
                nn.GELU(),
                nn.Linear(hidden, dim),
            )
            nn.init.zeros_(self.snr_embed[-1].weight)
            nn.init.zeros_(self.snr_embed[-1].bias)

        # Channel-adaptive query generators g_thetaq,1(mu), g_thetaq,2(mu).
        # Per-head scalar a_mu and b_mu are used; a_mu is constrained positive.
        if self.use_caq:
            hidden = max(4, int(self.snr_eq_hidden_dim))
            self.caq_a = nn.Sequential(
                nn.Linear(1, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_heads),
            )
            self.caq_b = nn.Sequential(
                nn.Linear(1, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_heads),
            )
            nn.init.zeros_(self.caq_a[-1].weight)
            nn.init.constant_(self.caq_a[-1].bias, 1.0)
            nn.init.zeros_(self.caq_b[-1].weight)
            nn.init.zeros_(self.caq_b[-1].bias)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def reset_sqa_identity(self):
        if self.use_snr_embedding:
            nn.init.zeros_(self.snr_embed[-1].weight)
            nn.init.zeros_(self.snr_embed[-1].bias)
        if self.use_caq:
            nn.init.zeros_(self.caq_a[-1].weight)
            nn.init.constant_(self.caq_a[-1].bias, 1.0)
            nn.init.zeros_(self.caq_b[-1].weight)
            nn.init.zeros_(self.caq_b[-1].bias)

    # Historical method name is retained for callers of the earlier code.
    reset_snreq_identity = reset_sqa_identity

    def _snr_to_window_tensor(self, snr, batch_windows, device, dtype):
        if snr is None:
            return None
        if torch.is_tensor(snr):
            snr_tensor = snr.to(device=device, dtype=dtype).flatten()
        else:
            snr_tensor = torch.tensor(snr, device=device, dtype=dtype).flatten()
        if snr_tensor.numel() == 0:
            return None
        if snr_tensor.numel() == 1:
            snr_tensor = snr_tensor.view(1, 1).expand(batch_windows, 1)
        elif snr_tensor.numel() == batch_windows:
            snr_tensor = snr_tensor.view(batch_windows, 1)
        elif batch_windows % snr_tensor.numel() == 0:
            repeat = batch_windows // snr_tensor.numel()
            snr_tensor = snr_tensor.view(-1, 1).repeat_interleave(repeat, dim=0)
        else:
            # Fallback for unusual input: use the batch mean SNR.
            snr_tensor = snr_tensor.mean().view(1, 1).expand(batch_windows, 1)
        return snr_tensor / self.snr_scale

    def forward(self, x, add_token=True, token_num=0, mask=None, snr=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            snr: scalar dB value or a tensor of shape [B].
        """
        self.last_caq_a = None
        self.last_caq_b = None
        self.last_caq_snr = None
        B_, N, C = x.shape
        snr_window = self._snr_to_window_tensor(snr, B_, x.device, x.dtype)

        if self.use_snr_embedding and snr_window is not None:
            x = x + self.snr_emb_scale * self.snr_embed(snr_window).unsqueeze(1)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_caq and snr_window is not None:
            # a_mu > 0; reset_sqa_identity initializes it close to 1.
            a_mu = F.relu(self.caq_a(snr_window)) + 1e-6
            b_mu = self.caq_b(snr_window)
            self.last_caq_a = a_mu
            self.last_caq_b = b_mu
            self.last_caq_snr = snr_window.detach()
            q = q * a_mu.view(B_, self.num_heads, 1, 1) + b_mu.view(B_, self.num_heads, 1, 1)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        if add_token:
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            if add_token:
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, use_sqa={self.use_snreq}, use_caq={self.use_caq}, use_snr_embedding={self.use_snr_embedding}, snr_emb_scale={self.snr_emb_scale}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


# Import alias only: no additional registered module or checkpoint key.
WindowAttention = SQAWindowAttention


class SFRC(nn.Module):
    r"""
    Spatial-frequency residual correction (SFRC) for JSCC feature refinement.

    Core principle:
        frequency branch is internal only:
            token feature -> FFT -> SNR-conditioned frequency selection -> iFFT -> residual fusion.
        It does not create an extra transmitted feature, does not change the latent size,
        and therefore does not increase CBR/channel uses.

    The block is initialized as an exact residual no-op: projection is zero-initialized and
    gamma is 1.0, so x + SFRC(x) initially equals x while gradients still flow into proj.
    """
    def __init__(self, dim, input_resolution, reduction=4, snr_min=0.0, snr_max=20.0,
                 init_gamma=1.0, max_modulation=0.5):
        super().__init__()
        self.dim = int(dim)
        self.input_resolution = tuple(input_resolution)
        self.reduction = int(reduction)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.init_gamma = float(init_gamma)
        self.max_modulation = float(max_modulation)

        hidden = max(self.dim // max(self.reduction, 1), 8)
        self.snr_gate = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.dim * 2),
        )
        self.proj = nn.Linear(self.dim, self.dim)
        self.gamma = nn.Parameter(torch.ones(1) * self.init_gamma)
        self.reset_sfrc_identity()

    def reset_sfrc_identity(self):
        # Neutral low/high scales at startup: tanh(0)=0 => scale=1.
        nn.init.zeros_(self.snr_gate[-1].weight)
        nn.init.zeros_(self.snr_gate[-1].bias)
        # Exact residual no-op while still allowing gradients to flow into proj.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.gamma.fill_(self.init_gamma)

    # Historical method name is retained for old initialization code.
    reset_sfra_identity = reset_sfrc_identity

    def set_input_resolution(self, input_resolution):
        self.input_resolution = tuple(input_resolution)

    def extra_repr(self):
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"snr_range=({self.snr_min},{self.snr_max}), reduction={self.reduction}")

    def _normalize_snr(self, snr, B, device, dtype):
        if snr is None:
            # Use middle value when SFRC is called without an SNR.
            return torch.full((B, 1), 0.5, device=device, dtype=dtype)
        if not torch.is_tensor(snr):
            snr = torch.tensor(snr, device=device, dtype=dtype)
        snr = snr.to(device=device, dtype=dtype)
        if snr.dim() == 0:
            snr = snr.view(1).expand(B)
        elif snr.dim() > 1:
            # Accept [B, 1] or [B, M] instantaneous SNRs; frequency trust uses image-level mean.
            snr = snr.view(snr.shape[0], -1).mean(dim=1)
        else:
            snr = snr.flatten()
        if snr.numel() == 1 and B > 1:
            snr = snr.expand(B)
        if snr.numel() != B:
            if snr.numel() > B:
                snr = snr[:B]
            else:
                snr = snr.mean().expand(B)
        snr = snr.view(B, 1)
        snr = (snr - self.snr_min) / max(self.snr_max - self.snr_min, 1e-6)
        return snr.clamp(0.0, 1.0)

    @staticmethod
    def _radial_low_mask(H, W, device, dtype):
        # rfft2 output size is [H, W//2 + 1]. Smooth radial low-pass mask.
        fy = torch.fft.fftfreq(H, d=1.0, device=device).view(H, 1)
        fx = torch.fft.rfftfreq(W, d=1.0, device=device).view(1, W // 2 + 1)
        radius = torch.sqrt(fy * fy + fx * fx)
        radius = radius / (radius.max() + 1e-6)
        low = torch.sigmoid((0.45 - radius) * 18.0).to(dtype=dtype)
        return low.view(1, 1, H, W // 2 + 1)

    def forward(self, x, snr=None):
        """
        Args:
            x: token feature [B, L, C].
            snr: scalar, [B], [B,1], or [B,M]. It conditions frequency trust only.
        Returns:
            residual token feature [B, L, C] to be added to x outside this module.
        """
        B, L, C = x.shape
        H, W = self.input_resolution
        if C != self.dim or L != H * W:
            # Safe fallback for unexpected dynamic shapes.
            return torch.zeros_like(x)

        out_dtype = x.dtype
        feat = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous().float()
        freq = torch.fft.rfft2(feat, dim=(-2, -1), norm='ortho')

        low_mask = self._radial_low_mask(H, W, feat.device, freq.real.dtype)
        high_mask = 1.0 - low_mask

        snr_norm = self._normalize_snr(snr, B, feat.device, feat.dtype)
        gate = self.snr_gate(snr_norm).view(B, 2, C, 1, 1)
        low_scale = 1.0 + self.max_modulation * torch.tanh(gate[:, 0])
        high_scale = 1.0 + self.max_modulation * torch.tanh(gate[:, 1])

        freq = freq * (low_mask * low_scale + high_mask * high_scale)
        feat = torch.fft.irfft2(freq, s=(H, W), dim=(-2, -1), norm='ortho')
        feat = feat.permute(0, 2, 3, 1).contiguous().view(B, L, C).to(out_dtype)
        return self.gamma.to(out_dtype) * self.proj(feat)


# Legacy import name is retained for older scripts and serialized full models.
SNRFrequencyResidualAdapter = SFRC


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)
        # self.proj = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        # self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        # print(x.shape)
        # print(self.input_resolution)
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H*W//4, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchMerging4x(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()
        H, W = input_resolution
        self.patch_merging1 = PatchMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_merging2 = PatchMerging((H // 2, W // 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)

    def forward(self, x, H=None, W=None):
        if H is None:
            H, W = self.input_resolution
        x = self.patch_merging1(x, H, W)
        x = self.patch_merging2(x, H//2, W//2)
        return x


class PatchReverseMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)
        # self.proj = nn.ConvTranspose2d(dim // 4, 3, 3, stride=1, padding=1)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.norm(x)
        x = self.increment(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = nn.PixelShuffle(2)(x)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        # print(x.shape)
        x = x.flatten(2).permute(0, 2, 1)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops


class PatchReverseMerging4x(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()
        self.use_conv = use_conv
        self.input_resolution = input_resolution
        self.dim = dim
        H, W = input_resolution
        self.patch_reverse_merging1 = PatchReverseMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_reverse_merging2 = PatchReverseMerging((H * 2, W * 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)

    def forward(self, x, H=None, W=None):
        if H is None:
            H, W = self.input_resolution
        x = self.patch_reverse_merging1(x, H, W)
        x = self.patch_reverse_merging2(x, H*2, W*2)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
