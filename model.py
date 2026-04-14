# -*- coding: utf-8 -*-
"""
轴承故障诊断对比实验模型库（v3）。

包含 7 种 PyTorch 模型：
  BearingCNN1D       — 4 层标准 1D-CNN（基线，旧 checkpoint 可直接加载）
  BearingTransformer — 4 层 Patch-ViT Transformer Encoder（全局注意力）
  BearingTCN         — 4 层扩张残差 TCN（dilation=4^i，感受野≈1021）
  BearingMobileNet1D — 2 块无扩张 DSConv（超轻量，推理算子数与 CNN 持平）
  BearingResNet1D    — 4 块残差网络（Skip Connection + BN，深层表达力强）
  BearingShuffleNet1D— 3 阶段 ShuffleNet V2（分组卷积+通道洗牌，HIL 低延迟）
  BearingConformer   — CNN×2 + Transformer×4 混合（局部降噪→全局周期识别）
  BearingRFWrapper   — 随机森林（sklearn），兼容 PyTorch 推理接口

所有 PyTorch 模型统一接受 (B, 1, L=1024)，输出 (B, num_classes) logits。
离线 / 在线 t-SNE 对比时，分类器 Linear 前的向量由 ``extract_classifier_input_features`` 统一提取。

架构切换：修改顶部的 ``SelectModel`` 变量，无需改动训练脚本。

典型用法::
    from model import build_model, SelectModel
    model = build_model(SelectModel, num_classes=10)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# ★ 全局模型选择变量 ★
# 修改此处切换架构；train_bearing_cnn1d.py 会自动读取。
# =============================================================================

SelectModel: str = "cnn"
"""全局架构选择，可选值：
  "cnn"         — BearingCNN1D（标准 1D-CNN，推荐基线）
  "transformer" — BearingTransformer（Patch-ViT Encoder）
  "tcn"         — BearingTCN（扩张残差 TCN）
  "mobilenet"   — BearingMobileNet1D（深度可分离卷积，轻量化）
  "resnet"      — BearingResNet1D（残差网络，深层表达力强）
  "shufflenet"  — BearingShuffleNet1D（ShuffleNet V2，HIL 低延迟优先）
  "conformer"   — BearingConformer（CNN+Transformer 混合，推荐高精度）
"""

# =============================================================================
# 架构标识常量（checkpoint["architecture"] 字段，保证版本兼容）
# =============================================================================

N_FEATURE_LAYERS: int = 4   # 多数模型的主干层数参考值

ARCH_CNN         = "cnn1d_bearing_v1"
ARCH_TRANSFORMER = "transformer_bearing_v1"
ARCH_TCN         = "tcn_bearing_v1"
ARCH_MOBILENET   = "mobilenet1d_bearing_v1"
ARCH_RESNET      = "resnet1d_bearing_v1"
ARCH_SHUFFLENET  = "shufflenet1d_bearing_v1"
ARCH_CONFORMER   = "conformer_bearing_v1"
ARCH_RF          = "rf_bearing_v1"

_ARCH_ALIASES: dict[str, str] = {
    # CNN
    "cnn":                  ARCH_CNN,
    ARCH_CNN:               ARCH_CNN,
    # Transformer
    "transformer":          ARCH_TRANSFORMER,
    "tf":                   ARCH_TRANSFORMER,
    ARCH_TRANSFORMER:       ARCH_TRANSFORMER,
    # TCN
    "tcn":                  ARCH_TCN,
    ARCH_TCN:               ARCH_TCN,
    # MobileNet
    "mobilenet":            ARCH_MOBILENET,
    "mobilenet1d":          ARCH_MOBILENET,
    "mobile":               ARCH_MOBILENET,
    ARCH_MOBILENET:         ARCH_MOBILENET,
    # ResNet
    "resnet":               ARCH_RESNET,
    "resnet1d":             ARCH_RESNET,
    ARCH_RESNET:            ARCH_RESNET,
    # ShuffleNet
    "shufflenet":           ARCH_SHUFFLENET,
    "shufflenet1d":         ARCH_SHUFFLENET,
    "shuffle":              ARCH_SHUFFLENET,
    ARCH_SHUFFLENET:        ARCH_SHUFFLENET,
    # Conformer
    "conformer":            ARCH_CONFORMER,
    "cnn_transformer":      ARCH_CONFORMER,
    "hybrid":               ARCH_CONFORMER,
    ARCH_CONFORMER:         ARCH_CONFORMER,
    # RF
    "rf":                   ARCH_RF,
    "forest":               ARCH_RF,
    ARCH_RF:                ARCH_RF,
}


def arch_name(model_type: str) -> str:
    """将短别名或正式常量统一转换为架构字符串（用于写入 checkpoint）。"""
    key = _ARCH_ALIASES.get(model_type.lower())
    if key is None:
        raise ValueError(
            f"未知模型类型 {model_type!r}，支持: {sorted(set(_ARCH_ALIASES.values()))}"
        )
    return key


# =============================================================================
# 模型 1：BearingCNN1D — 标准 1D-CNN 基线
# =============================================================================

class BearingCNN1D(nn.Module):
    """
    轻量 1D-CNN，4 层 Conv1d 特征提取 + MaxPool 下采样 + AdaptiveAvgPool + 分类头。
    作为可解释的基线模型，旧版 checkpoint 可直接加载。

    输入:  (B, 1, L=1024)
    输出:  (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # Layer 1 — 1024 → 256
            nn.Conv1d(1,   32,  kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(32),  nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            # Layer 2 — 256 → 64
            nn.Conv1d(32,  64,  kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),  nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            # Layer 3 — 64 → 16
            nn.Conv1d(64,  128, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            # Layer 4 — 16 → 1
            nn.Conv1d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).squeeze(-1))


# =============================================================================
# 模型 2：BearingTransformer — Patch-ViT 风格 Transformer
# =============================================================================

class BearingTransformer(nn.Module):
    """
    Patch-based 1D Transformer，4 层 Pre-LN TransformerEncoderLayer。
    将 L=1024 切成 64 个 patch（每 patch 16 点）→ 线性投影 + 位置编码
    → 4 层 Encoder → 全局平均池化 → 分类头。

    输入:  (B, 1, L=1024)
    输出:  (B, num_classes) logits
    """

    D_MODEL    : int   = 64
    PATCH_SIZE : int   = 16       # 1024 / 16 = 64 patches
    N_HEAD     : int   = 4
    DIM_FF     : int   = 256
    DROPOUT    : float = 0.20     # 较强正则化

    def __init__(self, num_classes: int = 10, num_layers: int = N_FEATURE_LAYERS):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size  = self.PATCH_SIZE
        n_patches        = 1024 // self.PATCH_SIZE   # 64

        self.patch_embed = nn.Linear(self.PATCH_SIZE, self.D_MODEL)
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, self.D_MODEL))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL, nhead=self.N_HEAD,
            dim_feedforward=self.DIM_FF, dropout=self.DROPOUT,
            batch_first=True, norm_first=True,   # Pre-LN，训练更稳定
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm       = nn.LayerNorm(self.D_MODEL)
        self.classifier = nn.Linear(self.D_MODEL, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, L = x.shape
        x = x.squeeze(1).reshape(B, L // self.patch_size, self.patch_size)
        x = self.patch_embed(x) + self.pos_embed       # (B, n_patches, D)
        x = self.norm(self.encoder(x).mean(dim=1))     # GlobalAvgPool → (B, D)
        return self.classifier(x)


# =============================================================================
# 模型 3：BearingTCN — 时间卷积网络（扩张残差 TCN）
# =============================================================================

class _TCNResBlock(nn.Module):
    """
    TCN 残差块：两层扩张 Conv1d + BN + GELU + Dropout + 残差连接。
    padding = (k−1)×d//2，输出序列长度与输入相同（same padding）。
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                      padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation,
                      padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = (nn.Conv1d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.skip(x))


class BearingTCN(nn.Module):
    """
    时间卷积网络（TCN），4 层扩张残差块。
    dilation = 4^i → {1, 4, 16, 64}，感受野约 1021 点，覆盖完整 1024 点窗口。

    输入:  (B, 1, L=1024)
    输出:  (B, num_classes) logits
    """

    _CHANNELS      = [32, 64, 128, 256]
    _KERNEL_SIZE   = 7
    _DILATION_BASE = 4      # dilation = 4^i
    _DROPOUT       = 0.1

    def __init__(self, num_classes: int = 10, num_layers: int = N_FEATURE_LAYERS):
        super().__init__()
        self.num_classes = num_classes
        channels = self._CHANNELS[:num_layers]
        in_ch    = 1
        blocks: list[nn.Module] = []
        for i, out_ch in enumerate(channels):
            blocks.append(
                _TCNResBlock(in_ch, out_ch, self._KERNEL_SIZE,
                             dilation=self._DILATION_BASE ** i,
                             dropout=self._DROPOUT)
            )
            in_ch = out_ch
        self.network    = nn.Sequential(*blocks)
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.network(x)).squeeze(-1))


# =============================================================================
# 模型 4：BearingMobileNet1D — 超轻量深度可分离卷积网络（v3 极简版）
# =============================================================================

class _DSConvBlock1D(nn.Module):
    """
    无扩张深度可分离卷积块（1D）。

    结构：DW(k, s) → BN → ReLU6 → PW(1×1) → BN → ReLU6
    共 2 Conv + 2 BN，比反转残差块（3 Conv + 3 BN）少 1/3 串行算子，
    中间张量无需扩张，内存带宽开销更小，推理延迟更低。
    """

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, dw_kernel: int = 3):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, dw_kernel, stride=stride,
                      padding=dw_kernel // 2, groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch), nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class BearingMobileNet1D(nn.Module):
    """
    超极简 MobileNet-1D v3（无扩张深度可分离卷积），专为 HIL 在线诊断优化。

    v3 相对 v2（反转残差版）的核心改进：
      ① 去掉 PW 升维步骤，每块从 3×Conv 降至 2×Conv，减少 ~33% 串行算子；
      ② 从 3 块缩为 2 块，进一步缩短推理路径；
      ③ Block1 步长升至 4（256→64），减少后续层的序列长度开销；
      ④ DW 核从 5 降至 3，DW 计算量减少 40%；
      ⑤ 总串行算子：5 Conv + 5 BN，与 CNN（4 Conv + 4 BN）基本持平，
         但 DW 卷积实际 MAC 远低于标准卷积，综合延迟应低于 CNN。

    结构：
      Stem  — Conv(k=9, s=4)  → BN → ReLU6   [1024→256, ch:1→24]
      Block1 — DSConv(24→48, s=4, dk=3)        [256→64]
      Block2 — DSConv(48→64, s=2, dk=3)        [64→32]
      GAP + Dropout(0.1) + Linear(64→C)

    参数量 ≈ 5.5K（v2 ≈ 21K，CNN ≈ 258K）
    注意：v3 架构与 v2 不兼容，旧 checkpoint 需重新训练。

    输入:  (B, 1, L=1024)
    输出:  (B, num_classes) logits
    """

    _CHANNELS    = [24, 48, 64]   # Stem + 2 块的输出通道
    _DW_KERNEL   = 3              # DW 核（k=3 比 k=5 快 40%，感受野已足够）
    _STEM_KERNEL = 9              # Stem 大核，保证初始时频感受野
    _STEM_STRIDE = 4              # Stem 大步长，快速将 1024 压缩至 256
    _DROPOUT     = 0.10           # 分类头前 Dropout，防小模型过拟合

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        c = self._CHANNELS   # [24, 48, 64]

        self.stem = nn.Sequential(
            nn.Conv1d(1, c[0], kernel_size=self._STEM_KERNEL,
                      stride=self._STEM_STRIDE,
                      padding=self._STEM_KERNEL // 2, bias=False),
            nn.BatchNorm1d(c[0]), nn.ReLU6(inplace=True),
        )
        # 2 个 DSConv 块：1024→256(stem)→64(s=4)→32(s=2)
        self.blocks = nn.Sequential(
            _DSConvBlock1D(c[0], c[1], stride=4, dw_kernel=self._DW_KERNEL),
            _DSConvBlock1D(c[1], c[2], stride=2, dw_kernel=self._DW_KERNEL),
        )
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.drop       = nn.Dropout(self._DROPOUT)
        self.classifier = nn.Linear(c[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)    # (B, 24, 256)
        x = self.blocks(x)  # (B, 64,  32)
        return self.classifier(self.drop(self.pool(x).squeeze(-1)))


# =============================================================================
# 模型 5：BearingResNet1D — 深层残差网络
# =============================================================================

class _ResBlock1D(nn.Module):
    """
    1D 残差块（BasicBlock）：
      主路径：Conv(k=7) → BN → ReLU → Conv(k=7) → BN
      Skip 路径：Identity（同维）或 1×1 投影卷积（升维/下采样）
      融合：主路径 + Skip → ReLU
    """

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            # 第一层卷积（可含 stride 下采样）
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            # 第二层卷积（stride=1 保持尺寸）
            nn.Conv1d(out_ch, out_ch, kernel_size,
                      stride=1, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        # Skip Connection：维度不同时用 1×1 投影对齐
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.skip(x))


class BearingResNet1D(nn.Module):
    """
    1D ResNet，4 个残差块（对应 ResNet-8 骨干），专为振动信号设计。

    结构：
      Stem  — Conv(k=7, s=4) → BN → ReLU  [1024 → 256, ch:1→64]
      Block1 — 64→64,   stride=1           [256 → 256]
      Block2 — 64→128,  stride=2           [256 → 128]
      Block3 — 128→256, stride=2           [128 →  64]
      Block4 — 256→512, stride=2           [ 64 →  32]
      GAP + Linear(512, num_classes)

    参数量约 2.5M（是 MobileNet-1D 的 25 倍，但精度通常更高）。

    输入:  (B, 1, L=1024)
    输出:  (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Stem：大步长快速压缩，1024 → 256
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        # 4 个残差块：逐步升通道 + 降分辨率
        self.blocks = nn.Sequential(
            _ResBlock1D(64,  64,  stride=1),  # Block 1
            _ResBlock1D(64,  128, stride=2),  # Block 2
            _ResBlock1D(128, 256, stride=2),  # Block 3
            _ResBlock1D(256, 512, stride=2),  # Block 4
        )
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                              # (B, 64,  256)
        x = self.blocks(x)                            # (B, 512,  32)
        return self.classifier(self.pool(x).squeeze(-1))


# =============================================================================
# 模型 6：BearingShuffleNet1D — 轻量化 ShuffleNet V2（HIL 低延迟优先）
# =============================================================================

def _channel_shuffle_1d(x: torch.Tensor) -> torch.Tensor:
    """
    ShuffleNet V2 通道洗牌（2 组）：
    将 C 通道按 2 分组后交织重排，使两条分支的特征充分融合。
    (B, C, L) → reshape(B,2,C//2,L) → transpose → reshape(B,C,L)
    """
    B, C, L = x.shape
    # 等价于在 group 维和 channel-per-group 维之间做转置
    x = x.view(B, 2, C // 2, L).transpose(1, 2).contiguous()
    return x.view(B, C, L)


class _ShuffleBlock1D(nn.Module):
    """
    ShuffleNet V2 基本单元（1D 版本）。

    stride=1 模式（通道分裂）：
      输入沿通道维劈成两半 x1 / x2
      x1  ─────────────────────────────────────►  concat → ChannelShuffle
      x2  → PW(1×1) → DW(k=3,s=1) → PW(1×1) ►
      输出通道数等于输入（无变化）

    stride=2 模式（空间下采样 + 通道翻倍）：
      x ──→ DW(k=3,s=2) → PW(1×1) → BN → ReLU ──►  concat → ChannelShuffle
      x ──→ PW(1×1) → DW(k=3,s=2) → PW(1×1)    ►
      输出通道数 = 2 × (out_ch//2) = out_ch
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        mid_ch = out_ch // 2    # 每条分支的输出通道数

        if stride == 1:
            # stride=1：in_ch 必须等于 out_ch（通道数不变）
            assert in_ch == out_ch, \
                f"ShuffleBlock stride=1 要求 in_ch==out_ch，得到 {in_ch} vs {out_ch}"
            branch_in = in_ch // 2
            # 主分支（处理后半段通道）
            self.branch_main = nn.Sequential(
                nn.Conv1d(branch_in, branch_in, 1, bias=False),
                nn.BatchNorm1d(branch_in), nn.ReLU(inplace=True),
                nn.Conv1d(branch_in, branch_in, 3, stride=1, padding=1,
                          groups=branch_in, bias=False),            # 深度卷积
                nn.BatchNorm1d(branch_in),
                nn.Conv1d(branch_in, branch_in, 1, bias=False),
                nn.BatchNorm1d(branch_in), nn.ReLU(inplace=True),
            )
        else:
            # stride=2：skip 分支用 DW(s=2)+PW；main 分支用 PW+DW(s=2)+PW
            self.branch_skip = nn.Sequential(
                nn.Conv1d(in_ch, in_ch, 3, stride=stride, padding=1,
                          groups=in_ch, bias=False),                # 深度卷积下采样
                nn.BatchNorm1d(in_ch),
                nn.Conv1d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm1d(mid_ch), nn.ReLU(inplace=True),
            )
            self.branch_main = nn.Sequential(
                nn.Conv1d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm1d(mid_ch), nn.ReLU(inplace=True),
                nn.Conv1d(mid_ch, mid_ch, 3, stride=stride, padding=1,
                          groups=mid_ch, bias=False),               # 深度卷积下采样
                nn.BatchNorm1d(mid_ch),
                nn.Conv1d(mid_ch, mid_ch, 1, bias=False),
                nn.BatchNorm1d(mid_ch), nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)               # 通道分裂
            out = torch.cat([x1, self.branch_main(x2)], dim=1)
        else:
            out = torch.cat([self.branch_skip(x), self.branch_main(x)], dim=1)
        return _channel_shuffle_1d(out)               # 通道洗牌


class BearingShuffleNet1D(nn.Module):
    """
    ShuffleNet V2 1D，专为 HIL 实时推理设计。

    使用分组卷积（Group Conv）+ 通道洗牌（Channel Shuffle），推理速度
    比标准 MobileNet-1D 更快，且精度相当。

    结构：
      Stem  — Conv(k=7,s=4) → BN → ReLU  [1024→256, ch:1→24]
      Stage1 — 2 个 ShuffleBlock,  s=2/1  [256→128, ch:24→48]
      Stage2 — 3 个 ShuffleBlock,  s=2/1  [128→ 64, ch:48→96]
      Stage3 — 4 个 ShuffleBlock,  s=2/1  [ 64→ 32, ch:96→192]
      GAP + Linear(192, num_classes)

    总参数量约 300 K（远低于 ResNet 的 2.5 M）。

    输入:  (B, 1, L=1024)
    输出:  (B, num_classes) logits
    """

    # 各阶段输出通道（遵循 ShuffleNetV2-0.5x 比例）
    _CHANNELS = [24, 48, 96, 192]

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        c = self._CHANNELS   # [24, 48, 96, 192]

        # Stem：大步长预压缩 1024 → 256
        self.stem = nn.Sequential(
            nn.Conv1d(1, c[0], kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(c[0]), nn.ReLU(inplace=True),
        )
        # 3 个 Stage，逐步提取多尺度特征
        self.stage1 = self._make_stage(c[0], c[1], num_blocks=2)  # 2 blocks
        self.stage2 = self._make_stage(c[1], c[2], num_blocks=3)  # 3 blocks
        self.stage3 = self._make_stage(c[2], c[3], num_blocks=4)  # 4 blocks

        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(c[3], num_classes)

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, num_blocks: int) -> nn.Sequential:
        """首个 block 做 stride=2 下采样+通道升维，后续 block 维持尺寸。"""
        layers: list[nn.Module] = [_ShuffleBlock1D(in_ch, out_ch, stride=2)]
        for _ in range(1, num_blocks):
            layers.append(_ShuffleBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)      # (B, 24, 256)
        x = self.stage1(x)    # (B, 48, 128)
        x = self.stage2(x)    # (B, 96,  64)
        x = self.stage3(x)    # (B, 192,  32)
        return self.classifier(self.pool(x).squeeze(-1))


# =============================================================================
# 模型 7：BearingConformer — CNN × 2 + Transformer × 4（混合架构）
# =============================================================================

class BearingConformer(nn.Module):
    """
    CNN-Transformer 混合架构（Conformer），专为轴承故障诊断设计。

    ─ 阶段一：CNN 局部特征提取（降噪 + 空间压缩） ──────────────────────
      2 层 1D-CNN（k=9, stride=4）
        (B, 1, 1024) → (B, 32, 256) → (B, 128, 64)
      物理意义：提取轴承冲击脉冲等高频局部特征，同时压缩冗余序列长度。

    ─ 阶段二：Transformer 全局周期判别 ─────────────────────────────────
      64 个特征帧（tokens）× 128 维 → 4 层 Pre-LN Encoder
        (B, 64, 128) → Transformer → GlobalAvgPool → (B, 128)
      物理意义：利用自注意力捕捉故障频率谐波、旋转周期等长程依赖。

    CNN 与 Transformer 的职责分离使两者各司其职，协同提升诊断精度。

    输入:  (B, 1, L=1024)
    输出:  (B, num_classes) logits
    """

    # ── CNN 阶段参数 ──────────────────────────────────────────────────────
    CNN_CHANNELS: list[int] = [32, 128]   # 逐层输出通道
    CNN_KERNEL  : int       = 9           # 大卷积核增大局部感受野
    CNN_STRIDE  : int       = 4           # 激进降采样：1024 → 256 → 64

    # ── Transformer 阶段参数 ─────────────────────────────────────────────
    # D_MODEL 必须等于 CNN_CHANNELS[-1]，无需额外投影层
    D_MODEL : int   = 128    # = CNN_CHANNELS[-1]，确保维度对齐
    N_HEAD  : int   = 4
    DIM_FF  : int   = 512    # FFN 容量为 D_MODEL 的 4 倍
    N_TF_LAYERS: int = 4
    TF_DROPOUT: float = 0.15

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # ── 阶段一：CNN 局部降噪与特征压缩 ──────────────────────────────
        self.cnn_stem = nn.Sequential(
            # CNN Layer 1: 1024 → 256，通道 1→32
            nn.Conv1d(
                1, self.CNN_CHANNELS[0],
                kernel_size=self.CNN_KERNEL,
                stride=self.CNN_STRIDE,
                padding=self.CNN_KERNEL // 2,
                bias=False,
            ),
            nn.BatchNorm1d(self.CNN_CHANNELS[0]),
            nn.GELU(),
            # CNN Layer 2: 256 → 64，通道 32→128
            nn.Conv1d(
                self.CNN_CHANNELS[0], self.CNN_CHANNELS[1],
                kernel_size=self.CNN_KERNEL,
                stride=self.CNN_STRIDE,
                padding=self.CNN_KERNEL // 2,
                bias=False,
            ),
            nn.BatchNorm1d(self.CNN_CHANNELS[1]),
            nn.GELU(),
        )

        # 动态计算 CNN 压缩后的序列长度（用于位置编码维度）
        # L=1024, k=9, s=4, p=4: 每层输出 = (L + 2p - k) // s + 1
        seq_len = 1024
        for _ in range(len(self.CNN_CHANNELS)):
            seq_len = (seq_len + 2 * (self.CNN_KERNEL // 2) - self.CNN_KERNEL) \
                      // self.CNN_STRIDE + 1
        # seq_len = 64，与 D_MODEL=128 共同定义位置编码形状

        # ── 阶段二：Transformer 全局周期判别 ────────────────────────────
        self.pos_embed = nn.Parameter(
            torch.zeros(1, seq_len, self.D_MODEL)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL, nhead=self.N_HEAD,
            dim_feedforward=self.DIM_FF, dropout=self.TF_DROPOUT,
            batch_first=True, norm_first=True,   # Pre-LN，更稳定
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=self.N_TF_LAYERS)
        self.norm       = nn.LayerNorm(self.D_MODEL)
        self.classifier = nn.Linear(self.D_MODEL, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 阶段一：CNN 提取局部特征并压缩序列
        x = self.cnn_stem(x)                  # (B, 128, 64)
        x = x.permute(0, 2, 1)               # (B, 64, 128) — tokens × features

        # 阶段二：Transformer 捕捉全局周期依赖
        x = x + self.pos_embed               # 注入位置信息
        x = self.encoder(x)                  # (B, 64, 128)
        x = self.norm(x.mean(dim=1))         # GlobalAvgPool → (B, 128)
        return self.classifier(x)


# =============================================================================
# 随机森林（手工特征 + sklearn RF），兼容 PyTorch 推理接口
# =============================================================================

def _skewness(x: np.ndarray) -> float:
    mu    = float(np.mean(x))
    sigma = float(np.std(x)) + 1e-9
    return float(np.mean(((x - mu) / sigma) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    mu    = float(np.mean(x))
    sigma = float(np.std(x)) + 1e-9
    return float(np.mean(((x - mu) / sigma) ** 4))


def extract_rf_features(window: np.ndarray) -> np.ndarray:
    """
    从 1024 点振动窗提取 26 维手工特征：
      时域 10 维：均值、标准差、最大、最小、峰峰值、RMS、MAV、波峰因子、偏度、峭度
      频域 16 维：8 个等宽频带 × (频段均值 + 频段最大值)
    """
    w   = window.astype(np.float64).ravel()
    rms = float(np.sqrt(np.mean(w ** 2)))
    feats: list[float] = [
        float(np.mean(w)), float(np.std(w)),
        float(np.max(w)),  float(np.min(w)),
        float(np.max(w) - np.min(w)),
        rms,
        float(np.mean(np.abs(w))),
        float(np.max(np.abs(w))) / (rms + 1e-9),
        _skewness(w),
        _kurtosis(w),
    ]
    fft_mag = np.abs(np.fft.rfft(w))
    n_bins  = len(fft_mag)
    band_sz = max(1, n_bins // 8)
    for i in range(8):
        band = fft_mag[i * band_sz: (i + 1) * band_sz]
        feats.append(float(np.mean(band)))
        feats.append(float(np.max(band)))
    return np.array(feats, dtype=np.float64)


class BearingRFWrapper:
    """
    将 sklearn RandomForestClassifier 包装成与 PyTorch 推理循环兼容的接口。
    __call__(x: Tensor[B,1,L]) → Tensor[B,C]（概率）；eval() / to() 均为空操作。
    """

    def __init__(self, rf_estimator, num_classes: int):
        self.rf          = rf_estimator
        self.num_classes = num_classes

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_np  = x.squeeze(1).cpu().numpy()
        feats = np.stack([extract_rf_features(xi) for xi in x_np])
        proba = self.rf.predict_proba(feats).astype(np.float32)
        return torch.from_numpy(proba)

    def eval(self):          return self
    def to(self, *a, **kw): return self

    def state_dict(self):
        raise NotImplementedError("RF 不支持 state_dict，请通过 ckpt['rf_wrapper'] 整体保存。")

    def load_state_dict(self, *a, **kw):
        raise NotImplementedError("RF 不支持 load_state_dict。")


def extract_classifier_input_features(
    model: nn.Module | BearingRFWrapper,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    最后一层分类 Linear **之前** 的表示 (B, D)，用于 t-SNE / 特征分析。

    与各模型 ``forward`` 中传入 ``self.classifier``（或等价路径）的向量一致；
    ``BearingRFWrapper`` 对应 ``extract_rf_features`` 拼成的手工特征 (B, 26)。

    Parameters
    ----------
    model : nn.Module | BearingRFWrapper
    x : (B, 1, L)，通常 L=1024

    Returns
    -------
    (B, D) tensor；RF 为 float32，其余与 ``x`` 的 device/dtype 一致。
    """
    if isinstance(model, BearingRFWrapper):
        x_np = x.squeeze(1).detach().cpu().numpy()
        arr = np.stack([extract_rf_features(xi) for xi in x_np]).astype(np.float32)
        return torch.from_numpy(arr).to(device=x.device, dtype=torch.float32)

    if isinstance(model, BearingCNN1D):
        return model.features(x).squeeze(-1)

    if isinstance(model, BearingTransformer):
        B, _, L = x.shape
        ps = model.patch_size
        t = x.squeeze(1).reshape(B, L // ps, ps)
        t = model.patch_embed(t) + model.pos_embed
        return model.norm(model.encoder(t).mean(dim=1))

    if isinstance(model, BearingTCN):
        return model.pool(model.network(x)).squeeze(-1)

    if isinstance(model, BearingMobileNet1D):
        t = model.stem(x)
        t = model.blocks(t)
        return model.drop(model.pool(t).squeeze(-1))

    if isinstance(model, BearingResNet1D):
        t = model.stem(x)
        t = model.blocks(t)
        return model.pool(t).squeeze(-1)

    if isinstance(model, BearingShuffleNet1D):
        t = model.stem(x)
        t = model.stage1(t)
        t = model.stage2(t)
        t = model.stage3(t)
        return model.pool(t).squeeze(-1)

    if isinstance(model, BearingConformer):
        t = model.cnn_stem(x)
        t = t.permute(0, 2, 1)
        t = t + model.pos_embed
        t = model.encoder(t)
        return model.norm(t.mean(dim=1))

    # receive_udp_hil 占位模型：仅 nn.Linear(1024, C)，无主干
    fc = getattr(model, "fc", None)
    if isinstance(fc, nn.Linear) and not hasattr(model, "features"):
        if x.dim() == 3:
            return x.squeeze(1)
        return x

    # 旧版 / 自定义：若有标准 ``features`` + ``classifier``(Linear)
    feats = getattr(model, "features", None)
    clf = getattr(model, "classifier", None)
    if feats is not None and isinstance(clf, nn.Linear):
        t = feats(x)
        if t.dim() == 3:
            t = t.squeeze(-1)
        return t

    raise TypeError(
        f"无法提取分类器输入特征，未知模型类型: {type(model).__name__}。"
        "请为该机型在 model.extract_classifier_input_features 中补充分支。"
    )


# =============================================================================
# 工厂函数
# =============================================================================

def build_model(arch_or_type: str, num_classes: int) -> nn.Module:
    """
    根据架构名或简写别名返回已初始化的 PyTorch 模型（不含 RF）。

    支持：
      "cnn"        / "cnn1d_bearing_v1"
      "transformer"/ "transformer_bearing_v1" / "tf"
      "tcn"        / "tcn_bearing_v1"
      "mobilenet"  / "mobilenet1d" / "mobilenet1d_bearing_v1"
      "resnet"     / "resnet1d"    / "resnet1d_bearing_v1"
      "shufflenet" / "shufflenet1d"/ "shufflenet1d_bearing_v1" / "shuffle"
      "conformer"  / "cnn_transformer" / "hybrid" / "conformer_bearing_v1"
    """
    key = _ARCH_ALIASES.get(arch_or_type.lower(), arch_or_type)
    if key == ARCH_CNN:         return BearingCNN1D(num_classes=num_classes)
    if key == ARCH_TRANSFORMER: return BearingTransformer(num_classes=num_classes)
    if key == ARCH_TCN:         return BearingTCN(num_classes=num_classes)
    if key == ARCH_MOBILENET:   return BearingMobileNet1D(num_classes=num_classes)
    if key == ARCH_RESNET:      return BearingResNet1D(num_classes=num_classes)
    if key == ARCH_SHUFFLENET:  return BearingShuffleNet1D(num_classes=num_classes)
    if key == ARCH_CONFORMER:   return BearingConformer(num_classes=num_classes)
    if key == ARCH_RF:
        raise ValueError(
            "RF 模型请直接使用 BearingRFWrapper 封装 sklearn RF，不通过 build_model 构建。"
        )
    raise ValueError(
        f"未知模型架构: {arch_or_type!r}。支持: {sorted(set(_ARCH_ALIASES.values()))}"
    )


# =============================================================================
# 各架构推荐训练超参数（由 train_bearing_cnn1d.py 中 get_hparams() 读取）
# =============================================================================

_HPARAMS: dict[str, dict] = {
    # ── CNN：收敛快稳定，适度 wd 防过拟合 ────────────────────────────────────
    ARCH_CNN: dict(
        lr=1e-3,    weight_decay=5e-4,   batch_size=64,
        epochs=200, patience=15,
        warmup_epochs=0,  grad_clip=0.0,  label_smoothing=0.05,
    ),
    # ── Transformer：对 lr 极敏感；warmup + clip 必须；更多 epoch ─────────────
    ARCH_TRANSFORMER: dict(
        lr=2.5e-4,  weight_decay=0.04,   batch_size=64,
        epochs=300, patience=25,
        warmup_epochs=10, grad_clip=1.0,  label_smoothing=0.10,
    ),
    # ── TCN：扩张卷积梯度较大，保留 clip ──────────────────────────────────────
    ARCH_TCN: dict(
        lr=6e-4,    weight_decay=1e-3,   batch_size=64,
        epochs=250, patience=20,
        warmup_epochs=5,  grad_clip=1.0,  label_smoothing=0.05,
    ),
    # ── MobileNet-1D v3（无扩张 DSConv，参数量 ~5.5K）────────────────────────
    # 模型极小，需适当提高 lr 防欠拟合；patience=25 给足收敛时间；无需 warmup
    ARCH_MOBILENET: dict(
        lr=1.5e-3,  weight_decay=1e-4,   batch_size=64,
        epochs=200, patience=25,
        warmup_epochs=0,  grad_clip=0.0,  label_smoothing=0.05,
    ),
    # ── ResNet-1D：比 CNN 更深，warmup + clip 有助稳定梯度流 ──────────────────
    ARCH_RESNET: dict(
        lr=8e-4,    weight_decay=2e-4,   batch_size=64,
        epochs=200, patience=15,
        warmup_epochs=5,  grad_clip=1.0,  label_smoothing=0.05,
    ),
    # ── ShuffleNet-1D：轻量快速，类 MobileNet 训练策略 ────────────────────────
    ARCH_SHUFFLENET: dict(
        lr=1e-3,    weight_decay=1e-4,   batch_size=64,
        epochs=200, patience=15,
        warmup_epochs=0,  grad_clip=0.0,  label_smoothing=0.05,
    ),
    # ── Conformer：Transformer 主导训练节奏；warmup + clip 必须 ─────────────────
    # CNN 部分先收敛，Transformer 部分需要较低 lr 和更多 epoch 才能充分学习
    ARCH_CONFORMER: dict(
        lr=3e-4,    weight_decay=0.01,   batch_size=64,
        epochs=250, patience=20,
        warmup_epochs=10, grad_clip=1.0,  label_smoothing=0.10,
    ),
}


def get_hparams(model_type: str) -> dict:
    """
    返回指定架构的推荐训练超参数字典（副本，修改不影响原始值）。

    字段说明
    --------
    lr              : 初始学习率（AdamW）
    weight_decay    : L2 正则化系数
    batch_size      : mini-batch 大小
    epochs          : 最大训练 epoch 上限
    patience        : 早停耐心值（验证集连续无提升的最大 epoch 数）
    warmup_epochs   : 线性 warmup 的 epoch 数（0 = 不使用）
    grad_clip       : 梯度裁剪最大范数（0.0 = 不裁剪）
    label_smoothing : 交叉熵标签平滑系数（0.0 = 标准交叉熵）

    未知架构自动回退到 CNN 超参（向后兼容）。
    """
    key = _ARCH_ALIASES.get(model_type.lower(), model_type)
    return dict(_HPARAMS.get(key, _HPARAMS[ARCH_CNN]))


# =============================================================================
# 工具函数
# =============================================================================

def count_feature_layers(model: nn.Module) -> int:
    """
    返回模型主干的特征提取深度（用于日志打印）：
      CNN1D       → Conv1d 层数（4）
      Transformer → Encoder 层数（4）
      TCN         → 扩张残差块数（4）
      MobileNet   → DSConv 块数（2）
      ResNet1D    → 残差块数（4）
      ShuffleNet  → 全部 ShuffleBlock 总数（stage1+2+3 = 2+3+4 = 9）
      Conformer   → CNN 层数 + Transformer Encoder 层数（2+4 = 6）
    """
    if isinstance(model, BearingCNN1D):
        return sum(1 for m in model.features.children() if isinstance(m, nn.Conv1d))
    if isinstance(model, BearingTransformer):
        return len(model.encoder.layers)
    if isinstance(model, BearingTCN):
        return len(model.network)
    if isinstance(model, BearingMobileNet1D):
        return len(model.blocks)
    if isinstance(model, BearingResNet1D):
        return len(model.blocks)
    if isinstance(model, BearingShuffleNet1D):
        # 统计三个 Stage 内的 ShuffleBlock 总数
        return len(model.stage1) + len(model.stage2) + len(model.stage3)
    if isinstance(model, BearingConformer):
        # CNN 层数 + Transformer Encoder 层数
        n_cnn = sum(1 for m in model.cnn_stem.children() if isinstance(m, nn.Conv1d))
        return n_cnn + len(model.encoder.layers)
    # 回退：统计所有 Conv1d
    return sum(1 for m in model.modules() if isinstance(m, nn.Conv1d))
