import math
import sys
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

# from denoising_diffusion_pytorch.version import __version__
__version__ = "1.0.0"
import inspect

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# data

class Dataset1D(Dataset):
    """
    Dataset wrapper for 1D signals with optional per-sample condition vectors.

    - signals: Tensor of shape (N, C, L)
    - conditions: Tensor of shape (N, cond_dim) or (N, cond_dim, L) or None

    __getitem__ returns (signal, cond) where cond is either a Tensor (possibly with time dim)
    or an empty tensor of shape (0,) when no conditions are provided.
    """
    def __init__(self, signals: Tensor, conditions: Tensor | None = None):
        super().__init__()
        assert isinstance(signals, torch.Tensor), 'signals must be a torch.Tensor'
        # store signals as float32 copy
        self.signals = signals.clone().detach().float()

        n = len(self.signals)

        if exists(conditions):
            assert isinstance(conditions, torch.Tensor), 'conditions must be a torch.Tensor or None'
            assert len(conditions) == n, 'conditions must match signals length'
            # keep a float32 detached copy
            self.conditions = conditions.clone().detach().float()
        else:
            # default to an empty condition tensor (will be interpreted as None downstream)
            self.conditions = torch.zeros((n, 0), dtype=torch.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx].clone()
        if self.conditions.numel() != 0:
            cond = self.conditions[idx].clone()
        else:
            cond = torch.zeros(0, dtype=torch.float32)
        return sig, cond

# small helper modules

class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model

class Unet1D(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        dropout = 0.,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        """
        原始的一维 U-Net，用于无条件扩散建模。
        """
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        # 下采样(encoder)部分
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # U-Net 结构
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        # 下采样(encoder)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        # 中间层（标准 Attention 捕获全局依赖）
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        # 上采样(decoder)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        """
        标准无条件去噪过程。
        """
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class UFourierLayer(Module):
    """
    U-Wavelet Layer（基于连续小波变换 CWT 的实现），对应论文中 Figure 4 上半部分的
    物理驱动分解模块 [file:///c%3A/Users/User/Documents/GitHub/diffusion_try/diffusionTry/DiffPhysiNet.pdf]。

    主要功能：
    - 对输入特征在时间维做连续小波变换（CWT）；
    - 依据小波系数能量，使用累积能量占比法（Cumulative Energy Ratio）动态选取显著分量（式 (5)(6)）；
    - 仅保留累积能量达到阈值的分量并做逆小波变换（ICWT）得到物理驱动故障分量 P_{i,t}(x)（式 (7)）；
    - 通过时间 / 条件嵌入产生的缩放向量，对输入进行调制，近似式 (4) 中基于 Timestep 和 WCE 的注意力。
    """

    def __init__(self, channels: int, time_dim: int, energy_threshold: float = 0.9,
                 wavelet: str = 'morl', num_scales: int = 32):
        """
        参数
        - channels: 输入特征通道数
        - time_dim: 时间嵌入维度
        - energy_threshold: 累积能量占比阈值，默认 0.9（保留 90% 能量的分量）
        - wavelet: 小波基函数类型，默认 'morl'（Morlet 小波）
        - num_scales: 小波变换的尺度数量，默认 32
        """
        super().__init__()
        self.channels = channels
        self.energy_threshold = energy_threshold
        self.wavelet = wavelet
        self.num_scales = num_scales

        # 利用时间嵌入（已经包含 WCE 信息）生成通道尺度因子，近似 Q,K,V 的调制效果
        self.time_to_scale = nn.Linear(time_dim, channels)

        # 预计算 Morlet 小波滤波器（可微分实现）
        # 注册为 buffer，不参与梯度计算但会随模型移动到正确设备
        self._wavelet_filters = None
        self._filter_length = None

    def _get_morlet_wavelet(self, length: int, scales: Tensor, omega0: float = 6.0) -> Tensor:
        """
        生成 Morlet 小波滤波器组（频域实现，支持 GPU 和自动微分）

        参数
        - length: 信号长度
        - scales: 尺度参数张量 (num_scales,)
        - omega0: Morlet 小波中心频率，默认 6.0

        返回
        - filters: (num_scales, length) 频域小波滤波器
        """
        device = scales.device
        dtype = scales.dtype

        # 频率轴（归一化频率）
        freqs = torch.fft.rfftfreq(length, d=1.0, device=device).to(dtype)  # (length//2+1,)

        # 对每个尺度生成频域 Morlet 小波
        # Morlet 小波的频域表示: exp(-0.5 * (s * omega - omega0)^2)
        scales_expanded = scales.view(-1, 1)  # (num_scales, 1)
        freqs_expanded = freqs.view(1, -1) * 2 * math.pi * length  # (1, length//2+1)

        # 计算频域 Morlet 小波
        omega_scaled = scales_expanded * freqs_expanded  # (num_scales, length//2+1)
        filters = torch.exp(-0.5 * (omega_scaled - omega0) ** 2)  # (num_scales, length//2+1)

        # 归一化
        filters = filters / (filters.sum(dim=-1, keepdim=True) + 1e-8)

        return filters

    def _cwt(self, x: Tensor) -> Tensor:
        """
        连续小波变换（CWT）的可微分实现

        参数
        - x: (B, C, L) 输入信号

        返回
        - coeffs: (B, C, num_scales, L) 小波系数
        """
        b, c, n = x.shape
        device = x.device
        dtype = x.dtype

        # 生成尺度参数（对数间隔，覆盖不同频率范围）
        scales = torch.logspace(0, math.log10(n / 4 + 1), self.num_scales, device=device, dtype=dtype)

        # 获取频域小波滤波器
        wavelet_filters = self._get_morlet_wavelet(n, scales)  # (num_scales, n//2+1)

        # 对输入信号做 FFT
        x_fft = torch.fft.rfft(x, dim=-1)  # (B, C, n//2+1)

        # 在频域应用小波滤波器（卷积定理）
        # x_fft: (B, C, n//2+1) -> (B, C, 1, n//2+1)
        # wavelet_filters: (num_scales, n//2+1) -> (1, 1, num_scales, n//2+1)
        x_fft_expanded = x_fft.unsqueeze(2)  # (B, C, 1, n//2+1)
        filters_expanded = wavelet_filters.unsqueeze(0).unsqueeze(0)  # (1, 1, num_scales, n//2+1)

        # 频域相乘
        coeffs_fft = x_fft_expanded * filters_expanded  # (B, C, num_scales, n//2+1)

        # IFFT 得到小波系数
        coeffs = torch.fft.irfft(coeffs_fft, n=n, dim=-1)  # (B, C, num_scales, L)

        return coeffs

    def _icwt(self, coeffs: Tensor, mask: Tensor) -> Tensor:
        """
        逆连续小波变换（ICWT）的可微分实现

        参数
        - coeffs: (B, C, num_scales, L) 小波系数
        - mask: (B, C, num_scales) 能量掩码，标记保留的尺度分量

        返回
        - reconstructed: (B, C, L) 重构信号
        """
        # 应用掩码到小波系数
        # mask: (B, C, num_scales) -> (B, C, num_scales, 1)
        mask_expanded = mask.unsqueeze(-1)  # (B, C, num_scales, 1)
        masked_coeffs = coeffs * mask_expanded  # (B, C, num_scales, L)

        # 沿尺度维度求和重构信号（简化的 ICWT）
        # 使用加权求和，权重可以根据尺度调整
        reconstructed = masked_coeffs.sum(dim=2)  # (B, C, L)

        return reconstructed

    def _compute_energy_mask(self, coeffs: Tensor) -> Tensor:
        """
        使用累积能量占比法计算掩码

        参数
        - coeffs: (B, C, num_scales, L) 小波系数

        返回
        - mask: (B, C, num_scales) 掩码张量，保留的分量为 1，其余为 0
        """
        b, c, num_scales, n = coeffs.shape

        # 计算每个尺度的能量（沿时间维求和）
        energy = (coeffs ** 2).sum(dim=-1)  # (B, C, num_scales)

        # 对每个 (batch, channel) 独立处理
        # 将能量降序排列
        sorted_energy, sorted_idx = torch.sort(energy, dim=-1, descending=True)  # (B, C, num_scales)

        # 计算总能量
        total_energy = sorted_energy.sum(dim=-1, keepdim=True) + 1e-8  # (B, C, 1)

        # 计算累积能量比例
        cumsum_energy = torch.cumsum(sorted_energy, dim=-1)  # (B, C, num_scales)
        cumsum_ratio = cumsum_energy / total_energy  # (B, C, num_scales)

        # 找到累积能量首次达到阈值的位置
        # 创建阈值掩码：cumsum_ratio <= threshold 的位置保留
        # 但要确保至少保留一个分量，且在达到阈值后的第一个分量也保留
        threshold_mask = cumsum_ratio <= self.energy_threshold  # (B, C, num_scales)

        # 为了确保达到阈值，需要包含刚好超过阈值的那个分量
        # 使用 roll 来实现：如果当前位置为 False 但前一个位置为 True，则当前位置也应该为 True
        shifted_mask = torch.roll(threshold_mask, shifts=1, dims=-1)
        shifted_mask[..., 0] = True  # 第一个位置始终保留

        # 最终的排序空间掩码
        sorted_mask = threshold_mask | (shifted_mask & ~threshold_mask)
        # 确保至少保留第一个分量
        sorted_mask[..., 0] = True

        # 将掩码映射回原始尺度顺序
        # 创建用于 scatter 的索引
        mask = torch.zeros_like(energy)
        mask.scatter_(-1, sorted_idx, sorted_mask.float())

        return mask

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        """
        利用连续小波变换提取最显著的故障分量特征

        输入
        - x: (B, C, L) 时间域特征 x_t 或其 lifted 表示；
        - time_emb: (B, time_dim) 已融合 WCE 的时间步嵌入。

        输出
        - p_it: (B, C, L) 物理驱动故障分量 P_{i,t}(x)。
        """
        b, c, n = x.shape
        input_dtype = x.dtype

        # ① 使用时间 / 条件嵌入进行通道重标定，近似 Attention(Q,K,V) 中的调制(figure4中左上角的Transformer部分)
        # 为避免在 AMP 下产生问题，这里全部在 float32 上完成计算
        scale = self.time_to_scale(time_emb.float()).view(b, c, 1)  # (B, C, 1), float32
        x_mod = x.float() * (1.0 + torch.tanh(scale))               # 调制后的 x_t，float32(生成包含噪音，工况和时间步的融合特征)

        # ② 连续小波变换（CWT）
        coeffs = self._cwt(x_mod)  # (B, C, num_scales, L)

        # ③ 使用累积能量占比法计算掩码
        mask = self._compute_energy_mask(coeffs)  # (B, C, num_scales)

        # ④ 逆小波变换（ICWT）重构物理驱动故障分量
        p_it = self._icwt(coeffs, mask)  # (B, C, L)

        # 与输入实数张量保持同一 dtype（便于与 AMP / 其余网络兼容）
        return p_it.to(input_dtype)


class SinusoidalPositionalEncoding1D(Module):
    """
    1D 正弦位置编码（Sinusoidal Positional Encoding）。

    为序列中的每个位置生成固定的正弦/余弦位置编码向量，
    支持最大长度 max_len 的序列。不含可学习参数。

    输入 : (B, L, D)
    输出 : (B, L, D)  —— 叠加位置编码后的序列
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        return x + self.pe[:, :x.size(1), :]


class PhysiNet(Module):
    """
    基于 DiffPhysiNet 结构的一维 Physi-UNet（简化实现）。

    结构对应论文 Figure 3 与 Figure 4：
    - U-Fourier Layer：利用 Top-K 频率做物理驱动分解，得到 P_{i,t}(x)；
    - U-Net：对残差 v0(x) - P_{i,t}(x) 建模，得到域特征 D_{i,t}(x)；
    - 通过 reweight & 激活，将物理分量与域特征融合得到 C_{i,t}(x)，再做投影得到 x_{t-1}。

    条件注入采用"基于 Class Token 的异构条件联合嵌入网络"：
    - 将工况条件映射为一个 Condition Token（Class Token），前置拼接到 lifted 特征序列；
    - 经 Transformer 编码器自注意力运算，Class Token 自动吸附序列中关键的时频物理特征；
    - 最终 Class Token 映射到 time_dim 后与时间嵌入相加，完成条件注入。
    """

    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        cond_dim: int = 0,
        dropout = 0.,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        cond_transformer_layers: int = 2,
        cond_transformer_heads: int = 4,
    ):
        """
        参数
        - dim, init_dim, out_dim, dim_mults, dropout, self_condition, ...:
          与 `Unet1D` 含义相同；
        - channels: 原始振动信号的通道数（通常为 1）；
        - cond_dim: 条件向量维度（例如工况/物理参数个数），0 表示退化为无条件 U-Net；
        - cond_transformer_layers: 条件 Transformer 编码器层数，默认 2；
        - cond_transformer_heads: 条件 Transformer 编码器注意力头数，默认 4。
        """
        super().__init__()

        self.base_channels = channels
        self.cond_dim = cond_dim
        self.self_condition = self_condition

        # 这里保持与原始信号通道一致，条件信息不直接作为额外通道拼接
        # 而是通过基于 Class Token 的异构条件联合嵌入网络在"时间嵌入"空间中进行融合。
        self.channels = channels

        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_dim = init_dim  # 保存以供 forward 使用
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        # 下采样(encoder)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # ========== 基于 Class Token 的异构条件联合嵌入网络 ==========
        if self.cond_dim > 0:
            # (1) Condition Fusion MLP：将离散工况特征（RPM、Load 等）融合为
            #     维度为 init_dim 的单一 Condition Token（Class Token）
            self.cond_fusion_mlp = nn.Sequential(
                nn.Linear(cond_dim, init_dim),
                nn.GELU(),
                nn.Linear(init_dim, init_dim),
            )

            # (2) 1D 正弦位置编码（最大长度 2048，覆盖 L=1024 + 1 个 Class Token）
            self.cond_pos_emb = SinusoidalPositionalEncoding1D(d_model=init_dim, max_len=2048)

            # (3) Transformer 编码器：对 [Class Token; 特征序列] 做自注意力
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=init_dim,
                nhead=cond_transformer_heads,
                dim_feedforward=init_dim * 4,
                batch_first=True,
                dropout=dropout,
            )
            self.cond_transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=cond_transformer_layers,
            )

            # (4) 特征对齐层：将 Transformer 输出的 Class Token (init_dim) 映射到 time_dim
            self.cls_to_time = nn.Linear(init_dim, time_dim)
        else:
            self.cond_fusion_mlp = None
            self.cond_pos_emb = None
            self.cond_transformer = None
            self.cls_to_time = None

        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # U-Net 结构
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        # 下采样(encoder)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        # 中间层（标准 Attention 捕获全局依赖）
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        # 上采样(decoder)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        # 输出维度与输入通道一致（或 2 倍用于 learned_variance）
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        # 物理分量重加权 W(·)（式 (9) 中的 W），以及两层激活 σ1, σ2
        self.reweight = nn.Conv1d(init_dim, init_dim, 1)
        self.act_sigma1 = nn.SiLU()
        self.act_sigma2 = nn.SiLU()

        # 投影层 Q(·)，将高维特征映射回 1D 振动信号空间（式 (10)）
        self.proj_Q = nn.Conv1d(init_dim, self.out_dim, 1)

        # U-Fourier Layer：在 lifted 特征空间上做物理分量抽取 P_{i,t}(x)
        # 使用基于连续小波变换（CWT）和累积能量占比法的实现
        self.u_fourier = UFourierLayer(channels = init_dim, time_dim = time_dim, energy_threshold = 0.9)

        # 工况回归头（TSTR 思路）：从全局池化特征预测条件向量
        if self.cond_dim > 0:
            self.cond_predictor = nn.Linear(init_dim, cond_dim)
        else:
            self.cond_predictor = None

    def forward(self, x, time, cond: Tensor = None, x_self_cond = None):
        """
        条件去噪前向过程（基于 Class Token 的异构条件联合嵌入）。

        输入
        - x: (B, C, L) 原始信号；
        - time: (B,) 离散时间步；
        - cond: (B, cond_dim) 或 (B, cond_dim, L) 条件张量 / WCE；
        - x_self_cond: 自条件（与 `Unet1D` 一致，默认关闭）。
        """
        # 1) 时间嵌入
        t = self.time_mlp(time)  # (B, time_dim)

        # 2) Self-condition 拼接（如果启用）
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        # 3) Lifting：将输入提前提升到高维特征 v0(x)（在条件嵌入之前完成）
        v0 = self.init_conv(x)  # (B, init_dim, L)

        # 4) 基于 Class Token 的异构条件联合嵌入
        if self.cond_dim > 0 and cond is not None and exists(self.cond_fusion_mlp):
            # 兼容 (B, cond_dim) 或 (B, cond_dim, L) 的输入形状：
            if cond.dim() == 3:
                cond_vec = cond.mean(dim=-1)       # (B, cond_dim)
            elif cond.dim() == 2:
                cond_vec = cond
            else:
                raise ValueError(
                    f"cond must have shape (B, cond_dim) or (B, cond_dim, L), got {tuple(cond.shape)}"
                )

            # 第二步：生成 Class Token — 将工况条件映射为 init_dim 维的 Condition Token
            z_cond = self.cond_fusion_mlp(cond_vec)   # (B, init_dim)
            z_cond = z_cond.unsqueeze(1)               # (B, 1, init_dim)

            # 第三步：维度转换与级联拼接
            v0_seq = v0.transpose(1, 2)                # (B, L, init_dim)

            # 将 Class Token 前置拼接到特征序列
            x_joint = torch.cat([z_cond, v0_seq], dim=1)  # (B, L+1, init_dim)

            # 第四步：添加位置编码并送入 Transformer 编码器做自注意力运算
            # 在 float32 下运行 Transformer 以保持数值稳定性（AMP 兼容）
            x_joint_f32 = x_joint.float()
            x_joint_f32 = self.cond_pos_emb(x_joint_f32)       # (B, L+1, init_dim)
            x_joint_f32 = self.cond_transformer(x_joint_f32)    # (B, L+1, init_dim)

            # 第五步：提取更新后的 Class Token（第 0 个位置）
            cls_token = x_joint_f32[:, 0, :]             # (B, init_dim)

            # 第六步：映射到 time_dim 并注入时间嵌入
            cond_emb = self.cls_to_time(cls_token)       # (B, time_dim)
            t = t + cond_emb.to(t.dtype)                 # 条件注入

        # 5) U-Fourier Layer：得到物理驱动故障分量 P_{i,t}(x)
        p_it = self.u_fourier(v0, t)  # (B, init_dim, L)

        # 6) U-Net：在残差 v0(x) - P_{i,t}(x) 上建模域特征 D_{i,t}(x)
        x_u = v0 - p_it
        h = []

        for block1, block2, attn, downsample in self.downs:
            x_u = block1(x_u, t)
            h.append(x_u)

            x_u = block2(x_u, t)
            x_u = attn(x_u)
            h.append(x_u)

            x_u = downsample(x_u)

        x_u = self.mid_block1(x_u, t)
        x_u = self.mid_attn(x_u)
        x_u = self.mid_block2(x_u, t)

        for block1, block2, attn, upsample in self.ups:
            x_u = torch.cat((x_u, h.pop()), dim=1)
            x_u = block1(x_u, t)

            x_u = torch.cat((x_u, h.pop()), dim=1)
            x_u = block2(x_u, t)
            x_u = attn(x_u)

            x_u = upsample(x_u)

        # 7) 物理分量重加权并与域特征融合（式 (9)）
        p_weighted = self.act_sigma1(self.reweight(p_it))
        c_it = self.act_sigma2(x_u + p_weighted)

        # 8) 投影得到 x_{t-1}（式 (10)），与 DDPM-Backbone 接口兼容
        x_out = self.proj_Q(c_it)

        # 9) 工况回归分支（TSTR 思路）：对 c_it 全局平均池化后预测条件向量
        if self.cond_predictor is not None:
            global_feat = c_it.mean(dim=-1)          # (B, init_dim)
            pred_cond = self.cond_predictor(global_feat)  # (B, cond_dim)
        else:
            pred_cond = None

        return x_out, pred_cond

# -----------------------------------------------------------------------------
# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# -----------------------------------------------------------------------------
# UCFilter: Unsupervised Clustering Filter (simplified implementation)
# -----------------------------------------------------------------------------

def ucfilter_kl_scores(
    samples: Tensor,
    *,
    sigma: float = 1.0,
    embed_dim: int = 2,
) -> Tensor:
    """
    计算每个样本对应的 KL 代价 C_d，用于 UCFilter 质量评估。

    实现思想基于论文中的式 (11)(12) 与 KL(R||T) 定义：
    - 在高维空间（原始信号）上，利用 Gaussian 核得到对称联合分布 R_ij；
      r_ij = (r_i|j + r_j|i) / (2n)
    - 在低维空间上，利用 Student-t 分布得到联合分布 T_ij；
      t_ij ∝ (1 + ||y_i - y_j||^2)^(-1)
    - C_d(i) = Σ_j r_ij (log r_ij - log t_ij)

    参数
    - samples: Tensor, 形状 (N, L) 或 (N, C, L)，生成的振动信号；
    - sigma: 高维 Gaussian 核带宽；
    - embed_dim: 低维嵌入维度（默认 2，对应 Figure 5 中的可视化）。

    返回
    - scores: Tensor, 形状 (N,), 每个样本的 KL 代价 C_d，越小表示与整体分布越一致。
    """
    assert samples.dim() in (2, 3), 'samples must be (N, L) or (N, C, L)'

    if samples.dim() == 3:
        n, c, l = samples.shape
        x = samples.reshape(n, c * l)
    else:
        n, l = samples.shape
        x = samples

    device = x.device
    x = x.float()

    # 去中心化
    x_centered = x - x.mean(dim = 0, keepdim = True)

    # 低维嵌入：使用 PCA 近似 t-SNE 的低维空间（更高效且无需额外依赖）
    # y: (N, embed_dim)
    try:
        # torch>=1.9 提供 pca_lowrank
        u, s, v = torch.pca_lowrank(x_centered, q = embed_dim)
        y = x_centered @ v[:, :embed_dim]
    except Exception:
        # 回退到简单的随机线性投影
        proj = torch.randn(x_centered.size(1), embed_dim, device = device)
        y = x_centered @ proj

    eps = 1e-12

    # ------------------------------
    # 高维联合分布 R_ij
    # ------------------------------
    # pairwise squared distances in high-dim
    d2_x = torch.cdist(x_centered, x_centered, p = 2) ** 2  # (N, N)

    # 条件概率 r_j|i ∝ exp(-d2 / (2 sigma^2))
    mask_offdiag = ~torch.eye(n, dtype = torch.bool, device = device)
    p_cond = torch.exp(-d2_x / (2 * sigma ** 2))
    p_cond = p_cond * mask_offdiag  # 对角为 0
    p_cond = p_cond / (p_cond.sum(dim = 1, keepdim = True) + eps)

    # 对称联合分布 R_ij = (r_i|j + r_j|i) / (2n)
    R = (p_cond + p_cond.t()) / (2.0 * n)
    R = R * mask_offdiag

    # ------------------------------
    # 低维联合分布 T_ij（Student-t）
    # ------------------------------
    d2_y = torch.cdist(y, y, p = 2) ** 2  # (N, N)

    num = 1.0 / (1.0 + d2_y)             # (1 + ||y_i - y_j||^2)^(-1)
    num = num * mask_offdiag

    denom = num.sum()
    T = num / (denom + eps)
    T = T * mask_offdiag

    # ------------------------------
    # KL 代价：C_d(i) = Σ_j r_ij (log r_ij - log t_ij)
    # ------------------------------
    log_R = torch.log(R + eps)
    log_T = torch.log(T + eps)
    cd_matrix = R * (log_R - log_T)

    scores = cd_matrix.sum(dim = 1)  # 每个样本一个分数
    return scores


def ucfilter_select_indices(
    samples: Tensor,
    *,
    k_ratio: float = 0.9,
    n_select: int | None = None,
    sigma: float = 1.0,
    embed_dim: int = 2,
) -> tuple[Tensor, Tensor]:
    """
    UCFilter 选择高质量样本的索引。

    参数
    - samples: Tensor, 形状 (N, L) 或 (N, C, L)，生成信号；
    - k_ratio: 选取样本占总数的比例 k = n / N，对应论文中的边界决策；
    - n_select: 若指定，则精确选 n_select 个样本；否则按 k_ratio * N 取整；
    - sigma, embed_dim: 传递给 ucfilter_kl_scores 的高维带宽与低维维度。

    返回
    - selected_indices: Tensor[int], 形状 (n_select,)，被 UCFilter 保留的样本索引；
    - scores: Tensor[float], 形状 (N,)，所有样本的 KL 代价（越小越好）。
    """
    scores = ucfilter_kl_scores(samples, sigma = sigma, embed_dim = embed_dim)
    n = scores.shape[0]

    if n_select is None:
        n_select = max(1, int(k_ratio * n))

    # KL 越小越好，因此取 scores 升序的前 n_select 个
    sorted_scores, sorted_idx = torch.sort(scores)  # 升序
    selected_indices = sorted_idx[:n_select]

    return selected_indices, scores


def ucfilter_kmeans_select_indices(
    samples: Tensor,
    *,
    num_clusters: int = 3,
    k_ratio: float = 0.9,
    sigma: float = 1.0,
    embed_dim: int = 2,
    max_iters: int = 50,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    更贴近论文 Figure 5 的 UCFilter：
    先在低维特征空间上进行 K-means 聚类，再在每个簇内按 KL 代价排序并按比例截取。

    参数
    - samples: (N, L) 或 (N, C, L)，生成信号；
    - num_clusters: K-means 聚类的簇数；
    - k_ratio: 在每个簇内保留的样本比例（例如 0.9 表示保留该簇中 KL 代价最小的 90% 样本）；
    - sigma, embed_dim: 与 ucfilter_kl_scores 一致；
    - max_iters: K-means 迭代次数上限。

    返回
    - selected_indices: (M,), M 为最终保留的样本数（各簇拼接）；
    - scores: (N,), 每个样本的 KL 代价；
    - labels: (N,), 每个样本的簇标签。
    """
    assert samples.dim() in (2, 3), 'samples must be (N, L) or (N, C, L)'

    if samples.dim() == 3:
        n, c, l = samples.shape
        x = samples.reshape(n, c * l)
    else:
        n, l = samples.shape
        x = samples

    device = x.device
    x = x.float()
    x_centered = x - x.mean(dim = 0, keepdim = True)

    # 低维特征 y（用于 K-means 聚类）
    try:
        _, _, v = torch.pca_lowrank(x_centered, q = embed_dim)
        y = x_centered @ v[:, :embed_dim]
    except Exception:
        proj = torch.randn(x_centered.size(1), embed_dim, device = device)
        y = x_centered @ proj

    # KL 代价（与全局 UCFilter 一致）
    scores = ucfilter_kl_scores(samples, sigma = sigma, embed_dim = embed_dim)

    # ------------------------------
    # 在 y 上执行 K-means 聚类
    # ------------------------------
    y_for_cluster = y
    try:
        out = torch.kmeans(
            y_for_cluster,
            num_clusters,
            dist = 'euclidean',
            niter = max_iters,
            device = device,
        )
        labels = out.labels
    except Exception:
        # 手写简单 K-means（Lloyd 算法）
        idx = torch.randperm(n, device = device)[:num_clusters]
        centers = y_for_cluster[idx]  # (K, D)

        for _ in range(max_iters):
            dists = torch.cdist(y_for_cluster, centers, p = 2)  # (N, K)
            labels = dists.argmin(dim = 1)                      # (N,)

            new_centers = []
            for c_idx in range(num_clusters):
                mask = labels == c_idx
                if mask.any():
                    new_centers.append(y_for_cluster[mask].mean(dim = 0))
                else:
                    # 若某个簇为空，则保持原中心
                    new_centers.append(centers[c_idx])
            new_centers = torch.stack(new_centers, dim = 0)

            if torch.allclose(new_centers, centers):
                break
            centers = new_centers

    # ------------------------------
    # 在每个簇内按 KL 代价排序并截取前 k_ratio 部分
    # ------------------------------
    selected_indices_list = []

    for c_idx in range(num_clusters):
        mask = labels == c_idx
        cluster_indices = torch.nonzero(mask, as_tuple = False).flatten()
        if cluster_indices.numel() == 0:
            continue

        n_c = cluster_indices.numel()
        n_keep_c = max(1, int(k_ratio * n_c))

        cluster_scores = scores[cluster_indices]
        _, sorted_local_idx = torch.sort(cluster_scores)  # 升序
        keep_local_idx = sorted_local_idx[:n_keep_c]
        keep_global_idx = cluster_indices[keep_local_idx]

        selected_indices_list.append(keep_global_idx)

    if len(selected_indices_list) == 0:
        # 极端情况：如果没有选中任何样本，则退回全局 KL 选择策略
        selected_indices, scores_global = ucfilter_select_indices(
            samples,
            k_ratio = k_ratio,
            sigma = sigma,
            embed_dim = embed_dim,
        )
        dummy_labels = torch.zeros(scores_global.shape[0], dtype = torch.long, device = device)
        return selected_indices, scores_global, dummy_labels

    selected_indices = torch.cat(selected_indices_list, dim = 0)

    return selected_indices, scores, labels


def batch_kl_loss(real_x0: torch.Tensor, pred_x0: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    流形对齐散度损失（J-FTSD 思路）。

    将真实 x0 和预测 x0（均为 3D 张量）展平为 2D，分别构建联合概率矩阵，
    然后计算 KL(R_real || T_pred)。

    参数
    - real_x0 : (B, C, L)  真实 x0 张量（来自训练样本）；
    - pred_x0 : (B, C, L)  模型预测的 x0 张量；
    - sigma    : 高维 Gaussian 核带宽，默认 1.0。

    返回
    - kl_val  : 标量 Tensor，KL 散度之和。
    """
    eps = 1e-12

    # 展平为 2D: (B, C*L)
    b = real_x0.shape[0]
    real_flat = real_x0.reshape(b, -1).float()
    pred_flat = pred_x0.reshape(b, -1).float()

    # ---- 构建 R_real：基于高斯核 ----
    d2_real = torch.cdist(real_flat, real_flat, p=2) ** 2          # (B, B)
    mask_od = ~torch.eye(b, dtype=torch.bool, device=real_x0.device)
    p_cond_real = torch.exp(-d2_real / (2.0 * sigma ** 2))
    p_cond_real = p_cond_real * mask_od
    p_cond_real = p_cond_real / (p_cond_real.sum(dim=1, keepdim=True) + eps)
    R_real = (p_cond_real + p_cond_real.t()) / (2.0 * b)
    R_real = R_real * mask_od
    # 归一化确保 R_real 是合法概率分布
    R_real = R_real / (R_real.sum() + eps)

    # ---- 构建 T_pred：基于 Student-t 分布 ----
    d2_pred = torch.cdist(pred_flat, pred_flat, p=2) ** 2          # (B, B)
    num_t = 1.0 / (1.0 + d2_pred)
    num_t = num_t * mask_od
    T_pred = num_t / (num_t.sum() + eps)
    T_pred = T_pred * mask_od

    # ---- KL(R_real || T_pred) ----
    kl_val = (R_real * (torch.log(R_real + eps) - torch.log(T_pred + eps))).sum()
    return kl_val


class GaussianDiffusion1D(Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        channels = None,
        self_condition = None,
        channel_first = True
    ):
        super().__init__()
        self.model = model
        self.channels = default(channels, lambda: self.model.channels)
        self.self_condition = default(self_condition, lambda: self.model.self_condition)

        self.channel_first = channel_first
        self.seq_index = -2 if not channel_first else -1

        self.seq_length = seq_length

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, model_forward_kwargs: dict = dict(), **kwargs):
        """Call the underlying model to produce either predicted noise, x0 or v depending on objective.
        Filters model_forward_kwargs to only arguments accepted by model.forward to avoid TypeError when
        passing 'cond' to models that don't accept it.
        """
        # merge any kwargs (e.g., if caller did model_predictions(x,t, **model_forward_kwargs))
        if len(kwargs) > 0:
            model_forward_kwargs = {**model_forward_kwargs, **kwargs}

        if exists(x_self_cond):
            model_forward_kwargs = {**model_forward_kwargs, 'self_cond': x_self_cond}

        # filter model_forward_kwargs to only those accepted by the underlying model
        try:
            sig = inspect.signature(self.model.forward)
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not accepts_var_kw:
                allowed = set(sig.parameters.keys()) - {'self'}
                model_forward_kwargs = {k: v for k, v in model_forward_kwargs.items() if k in allowed}
        except Exception:
            # if introspection fails, fall back to passing kwargs as-is
            pass

        model_output = self.model(x, t, **model_forward_kwargs)

        # PhysiNet returns (signal_output, pred_cond) tuple; unwrap if so
        if isinstance(model_output, tuple):
            model_output, _ = model_output

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        else:
            raise ValueError(f'unknown objective {self.objective}')

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True, model_forward_kwargs: dict = dict()):

        if exists(x_self_cond):
            model_forward_kwargs = {**model_forward_kwargs, 'self_cond': x_self_cond}

        # pass model_forward_kwargs as the named parameter so it gets routed inside model_predictions
        preds = self.model_predictions(x, t, model_forward_kwargs = model_forward_kwargs)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True, model_forward_kwargs: dict = dict()):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised, model_forward_kwargs = model_forward_kwargs) #计算后验分布的均值和方差，p_mean_variance调用Unet预测模型并生成预测信号的均值和方差
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start # pred_img是预测后的结果，x_start是模型信号

    @torch.no_grad()
    def p_sample_loop(self, shape, return_noise = False, model_forward_kwargs: dict = dict()):
        batch, device = shape[0], self.betas.device

        noise = torch.randn(shape, device=device) #初始化纯噪声
        img = noise 

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):  #从第1000个时间步到第0个时间步
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond, model_forward_kwargs = model_forward_kwargs)

        img = self.unnormalize(img)

        if not return_noise:
            return img

        return img, noise

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True, model_forward_kwargs: dict = dict(), return_noise = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        noise = torch.randn(shape, device = device)
        img = noise

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised, model_forward_kwargs = model_forward_kwargs)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.unnormalize(img)

        if not return_noise:
            return img

        return img, noise

    @torch.no_grad()
    def sample(self, batch_size = 16, return_noise = False, model_forward_kwargs: dict = dict()):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        shape = (batch_size, channels, seq_length) if self.channel_first else (batch_size, seq_length, channels)
        return sample_fn(shape, return_noise = return_noise, model_forward_kwargs = model_forward_kwargs)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, model_forward_kwargs: dict = dict(), return_reduced_loss = True):
        b = x_start.shape[0]
        n = x_start.shape[self.seq_index]

        noise = default(noise, lambda: torch.randn_like(x_start)) #产生与x_start相同形状的噪声

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise) #向前扩散（加噪）

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:  #自条件预测(提升模型的预测能力)
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

            model_forward_kwargs = {**model_forward_kwargs, 'self_cond': x_self_cond}

        # model kwargs

        # predict and take gradient step

        # filter model_forward_kwargs similarly before calling the model
        try:  #防止参数错误传递
            sig = inspect.signature(self.model.forward)
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not accepts_var_kw:
                allowed = set(sig.parameters.keys()) - {'self'}
                model_forward_kwargs = {k: v for k, v in model_forward_kwargs.items() if k in allowed}
        except Exception:
            pass

        model_out = self.model(x, t, **model_forward_kwargs)

        # Unpack (signal_output, pred_cond) tuple returned by PhysiNet
        pred_cond = None
        if isinstance(model_out, tuple):
            model_out, pred_cond = model_out

        #训练过程预测不同目标
        if self.objective == 'pred_noise':
            target = noise #预测纯噪音
        elif self.objective == 'pred_x0':
            target = x_start #预测干净的信号
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v  #预测 v 参数（速度参数）
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        loss_diff = loss  # 基础扩散 MSE 损失 (B,)

        # ---- 工况回归损失 loss_reg（TSTR 思路） ----
        loss_reg = torch.tensor(0.0, device=x_start.device)
        if pred_cond is not None and 'cond' in model_forward_kwargs:
            cond_target = model_forward_kwargs['cond']          # (B, cond_dim) 或 (B, cond_dim, L)
            if cond_target.dim() == 3:
                cond_target = cond_target.mean(dim=-1)          # -> (B, cond_dim)
            loss_reg = F.mse_loss(pred_cond, cond_target.to(pred_cond.dtype))

        # ---- 流形对齐散度损失 loss_kl（J-FTSD 思路） ----
        # 从带梯度的 model_out 还原 pred_x0（保留完整梯度流）
        if self.objective == 'pred_noise':
            pred_x0 = self.predict_start_from_noise(x, t, model_out)
        elif self.objective == 'pred_x0':
            pred_x0 = model_out
        elif self.objective == 'pred_v':
            pred_x0 = self.predict_start_from_v(x, t, model_out)
        else:
            pred_x0 = x_start
        loss_kl = batch_kl_loss(x_start, pred_x0)

        # ---- 融合损失 ----
        total_loss = loss_diff.mean() + 0.1 * loss_reg + 0.01 * loss_kl

        if not return_reduced_loss:
            # 仅在需要非缩减模式时返回原始 per-sample 损失（忽略辅助项）
            return loss_diff

        return total_loss

    def forward(self, img, *args, **kwargs):
        b, n, device, seq_length, = img.shape[0], img.shape[self.seq_index], img.device, self.seq_length

        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()  #随机采样时间步

        img = self.normalize(img)

        # Extract optional condition and model_forward_kwargs, merge and pass into p_losses
        #条件参数传递逻辑:将cond传递给更底层的PhysiNet网络
        cond = kwargs.pop('cond', None)
        model_forward_kwargs = kwargs.pop('model_forward_kwargs', {})
        if cond is not None:
            # pass condition to the underlying model (PhysiNet expects 'cond' kwarg)
            model_forward_kwargs = {**model_forward_kwargs, 'cond': cond}

        noise = kwargs.pop('noise', None)
        return_reduced = kwargs.pop('return_reduced_loss', True)

        return self.p_losses(img, t, noise = noise, model_forward_kwargs = model_forward_kwargs, return_reduced_loss = return_reduced)

# trainer class

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,  #梯度累积步数，batch_size=8, gradient_accumulate_every=4 → 等效批大小 32
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000, #每 N 步保存一次模型并生成样本
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        denorm_min: float | None = None,
        denorm_max: float | None = None,
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        # optional denormalization range (used to convert model outputs back to physical units for plotting)
        self.denorm_min = denorm_min
        self.denorm_max = denorm_max

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader
        # On Windows, use num_workers=0 to avoid multiprocessing issues
        num_workers = 0 if sys.platform == 'win32' else cpu_count()
        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = num_workers)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__,
            # 保存归一化参数，用于推理时反归一化
            'normalization_params': {
                'signal_min': self.denorm_min,
                'signal_max': self.denorm_max,
            }
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()  #调用它会将 self.model及其所有子模块的内部状态标志（training 属性）设置为 True

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every): #gradient_accumulate_every是等效批，即等效批次 = batchsize * gradient_accumulate_every,即循环gradient_accumulate_every次之后再进行梯度更新
                    batch = next(self.dl)  # 获取一个batch(48个样本)的数据
                    # dataloader may return (data, cond) or just data
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        data, cond = batch
                    else:
                        data = batch
                        cond = None

                    # move tensors to device (safe even if already on device)
                    data = data.to(device)
                    if isinstance(cond, torch.Tensor) and cond.numel() != 0:
                        cond = cond.to(device)
                    else:
                        cond = None

                    with self.accelerator.autocast():  # 模型训练
                        # pass condition through the diffusion wrapper; GaussianDiffusion1D.forward will forward cond
                        loss = self.model(data, cond=cond)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)  #反向传播，计算梯度

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()  #优化器优化，更新参数
                self.opt.zero_grad() #梯度清零

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every  #计算当前采样/保存的批次编号
                            batches = num_to_groups(self.num_samples, self.batch_size) #将总样本数 num_samples 分成多个批次
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches)) #生成样本

                        all_samples = torch.cat(all_samples_list, dim = 0)

                        try:
                            # 转换到 CPU 并转为 numpy，形状通常是 (Batch, Channels, Length)
                            samples_np = all_samples.detach().cpu().numpy()

                            # 如果设置了 denorm 范围（signal_min/signal_max），先把 samples 从模型范围（[-1,1]）反归一化到物理量级
                            if getattr(self, 'denorm_min', None) is not None and getattr(self, 'denorm_max', None) is not None:
                                try:
                                    den_min = float(self.denorm_min)
                                    den_max = float(self.denorm_max)
                                    samples_np = (samples_np + 1.0) * 0.5 * (den_max - den_min) + den_min
                                except Exception:
                                    # 若发生任何错误，保留原始 samples_np
                                    pass

                            plt.figure(figsize=(12, 6))
                            # 只画前 4 个样本，避免图太乱
                            num_plots = min(4, samples_np.shape[0])

                            for i in range(num_plots):
                                # 假设是单通道，取第 0 个通道
                                plt.subplot(num_plots, 1, i + 1)
                                plt.plot(samples_np[i, 0, :], linewidth=1)
                                plt.title(f'Sample {i} at step {self.step}')
                                plt.grid(True, alpha=0.3)

                            plt.tight_layout()
                            # 保存为真正的 PNG 图片
                            plt.savefig(str(self.results_folder / f'sample-{milestone}.png'))
                            plt.close()  # 关闭画布释放内存

                        except Exception as e:
                            print(f"Error plotting samples: {e}")

                            # --- 修改结束 ---

                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

    def sample_with_condition(self, batch_size = 16, cond: torch.Tensor | None = None, return_noise = False, model_forward_kwargs: dict = None):
        """
        使用 EMA 模型进行条件采样（如果可用）。

        输入
        - batch_size: 批次大小；
        - return_noise: 是否返回噪声；
        - model_forward_kwargs: 传递给模型前向传播的其他参数。

        输出
        - 生成的样本。
        """
        if not self.accelerator.is_main_process:
            return None

        if not hasattr(self, 'ema') or self.ema is None:
            print("EMA model is not available, falling back to standard sampling.")
            return self.sample(batch_size = batch_size, return_noise = return_noise, model_forward_kwargs = model_forward_kwargs or {})

        device = self.device

        with torch.no_grad():
            self.ema.ema_model.eval()

            # determine cond_dim from underlying model (diffusion wrapper may hold model at .model)
            cond_dim = None
            if hasattr(self.model, 'cond_dim'):
                cond_dim = getattr(self.model, 'cond_dim')
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'cond_dim'):
                cond_dim = getattr(self.model.model, 'cond_dim')

            # use provided cond if given; otherwise sample a random cond only if we know cond_dim
            if cond is None:
                if cond_dim is None or cond_dim == 0:
                    # no conditioning supported; leave cond as None
                    cond = None
                else:
                    cond = torch.randn(batch_size, cond_dim, device = device)
            else:
                # ensure on device and right dtype
                cond = cond.to(device)

            mfw = dict(model_forward_kwargs or {})
            mfw['cond'] = cond

            # 生成样本 using EMA model
            img = self.ema.ema_model.sample(batch_size = batch_size, return_noise = return_noise, model_forward_kwargs = mfw)

        return img
