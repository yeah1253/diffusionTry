# -*- coding: utf-8 -*-
"""
轴承故障诊断对比实验模型库。

包含 4 种模型，**特征提取层数统一为 N_FEATURE_LAYERS = 4**：
  BearingCNN1D       — 4 层 Conv1d（1D 卷积）
  BearingLSTM        — 4 层堆叠双向 LSTM
  BearingTransformer — 4 层 Transformer Encoder（Patch-ViT 风格）
  BearingRFWrapper   — 随机森林（sklearn），不含 PyTorch 参数

前三者均接受输入 (B, 1, L=1024)，输出 (B, num_classes) logits。
BearingRFWrapper.__call__ 同样接受 (B, 1, L) torch.Tensor，返回 (B, C) 概率张量，
与 PyTorch 推理循环接口兼容。

典型用法（训练端）：
    from model import build_model, arch_name
    model = build_model("lstm", num_classes=10)
    ckpt_arch = arch_name("lstm")   # → "lstm_bearing_v1"

典型用法（推理端，依据 checkpoint 还原）：
    arch = ckpt["architecture"]
    model = build_model(arch, num_classes)
    model.load_state_dict(ckpt["state_dict"])
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# 架构标识常量
# =============================================================================

# 特征提取层数（三种神经网络模型统一）
N_FEATURE_LAYERS: int = 4

# checkpoint["architecture"] 字段值；
# ARCH_CNN 与 bearing_models.BEARING_CNN_ARCH 取相同字符串，保证旧 checkpoint 兼容
ARCH_CNN         = "cnn1d_bearing_v1"
ARCH_LSTM        = "lstm_bearing_v1"
ARCH_TRANSFORMER = "transformer_bearing_v1"
ARCH_RF          = "rf_bearing_v1"

# 短别名 → 正式 arch 常量
_ARCH_ALIASES: dict[str, str] = {
    "cnn":              ARCH_CNN,
    ARCH_CNN:           ARCH_CNN,
    "lstm":             ARCH_LSTM,
    ARCH_LSTM:          ARCH_LSTM,
    "transformer":      ARCH_TRANSFORMER,
    "tf":               ARCH_TRANSFORMER,
    ARCH_TRANSFORMER:   ARCH_TRANSFORMER,
    "rf":               ARCH_RF,
    "forest":           ARCH_RF,
    ARCH_RF:            ARCH_RF,
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
# 模型 1：BearingCNN1D（与 bearing_models.py 结构完全一致，旧 checkpoint 可直接加载）
# =============================================================================

class BearingCNN1D(nn.Module):
    """
    轻量 1D CNN，**4 个 Conv1d 特征提取层** + AdaptiveAvgPool + 分类头。

    输入:  (B, 1, L=1024)
    输出:  (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv1d(1,   32,  kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),          # 1024 → 256
            # Layer 2
            nn.Conv1d(32,  64,  kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),          # 256 → 64
            # Layer 3
            nn.Conv1d(64,  128, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),          # 64 → 16
            # Layer 4
            nn.Conv1d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),                        # → (B, 256, 1)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.squeeze(-1)           # (B, 256)
        return self.classifier(h)


# =============================================================================
# 模型 2：BearingLSTM
# =============================================================================

class BearingLSTM(nn.Module):
    """
    **4 层堆叠双向 LSTM**，以最后时刻的拼接隐状态接 LayerNorm + 分类头。

    输入:  (B, 1, L=1024)  → 内部视序列 (B, L, 1)
    输出:  (B, num_classes) logits
    """

    HIDDEN_SIZE: int = 128

    def __init__(self, num_classes: int = 10, num_layers: int = N_FEATURE_LAYERS):
        super().__init__()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            input_size   = 1,
            hidden_size  = self.HIDDEN_SIZE,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = 0.2 if num_layers > 1 else 0.0,
        )
        self.norm       = nn.LayerNorm(self.HIDDEN_SIZE * 2)
        self.classifier = nn.Linear(self.HIDDEN_SIZE * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L) → (B, L, 1)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)       # (B, L, H*2)
        h = out[:, -1, :]           # 取最后时刻 (B, H*2)
        return self.classifier(self.norm(h))


# =============================================================================
# 模型 3：BearingTransformer（Patch-ViT 风格）
# =============================================================================

class BearingTransformer(nn.Module):
    """
    Patch-based 1D Transformer，**4 层 Pre-LN TransformerEncoderLayer**。

    将 L=1024 的信号切成 n_patches=64 个大小为 patch_size=16 的块，
    线性投影 → 可学习位置编码 → 4 层 Encoder → 全局平均池化 → 分类头。

    输入:  (B, 1, L=1024)
    输出:  (B, num_classes) logits
    """

    D_MODEL    : int = 64
    PATCH_SIZE : int = 16       # 1024 / 16 = 64 patches
    N_HEAD     : int = 4
    DIM_FF     : int = 256
    DROPOUT    : float = 0.1

    def __init__(self, num_classes: int = 10, num_layers: int = N_FEATURE_LAYERS):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size  = self.PATCH_SIZE
        n_patches        = 1024 // self.PATCH_SIZE  # 64

        self.patch_embed = nn.Linear(self.PATCH_SIZE, self.D_MODEL)
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, self.D_MODEL))

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = self.D_MODEL,
            nhead           = self.N_HEAD,
            dim_feedforward = self.DIM_FF,
            dropout         = self.DROPOUT,
            batch_first     = True,
            norm_first      = True,     # Pre-LN，训练更稳定
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm       = nn.LayerNorm(self.D_MODEL)
        self.classifier = nn.Linear(self.D_MODEL, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, L = x.shape
        x = x.squeeze(1)                                # (B, L)
        n = L // self.patch_size
        x = x.reshape(B, n, self.patch_size)            # (B, n, P)
        x = self.patch_embed(x) + self.pos_embed        # (B, n, D)
        x = self.encoder(x)                             # (B, n, D)
        x = self.norm(x.mean(dim=1))                    # (B, D)  全局平均池化
        return self.classifier(x)


# =============================================================================
# 模型 4：随机森林（手工特征 + sklearn RF）
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
    从 1024 点振动窗提取 **26 维** 手工特征：
      时域 10 维：均值、标准差、最大、最小、峰峰值、RMS、MAV、波峰因子、偏度、峭度
      频域 16 维：8 个等宽频带 × (频段均值 + 频段最大值)

    返回 float64 一维数组，长度固定 26。
    """
    w    = window.astype(np.float64).ravel()
    rms  = float(np.sqrt(np.mean(w ** 2)))

    feats: list[float] = [
        float(np.mean(w)),
        float(np.std(w)),
        float(np.max(w)),
        float(np.min(w)),
        float(np.max(w) - np.min(w)),           # peak-to-peak
        rms,
        float(np.mean(np.abs(w))),               # MAV
        float(np.max(np.abs(w))) / (rms + 1e-9), # crest factor
        _skewness(w),
        _kurtosis(w),
    ]

    # 频域：8 频带 × (均值 + 最大值) = 16 维
    fft_mag = np.abs(np.fft.rfft(w))
    n_bins  = len(fft_mag)
    n_bands = 8
    band_sz = max(1, n_bins // n_bands)
    for i in range(n_bands):
        band = fft_mag[i * band_sz: (i + 1) * band_sz]
        feats.append(float(np.mean(band)))
        feats.append(float(np.max(band)))

    return np.array(feats, dtype=np.float64)


class BearingRFWrapper:
    """
    将 sklearn RandomForestClassifier 包装成与 PyTorch 模型兼容的可调用接口。

    __call__(x) 接受 (B, 1, L) torch.Tensor → 返回 (B, num_classes) 概率张量；
    eval() / to() 均为空操作，保持与推理循环接口一致。
    checkpoint 中以 "rf_wrapper" 键整体（pickle）保存，无 state_dict。
    """

    def __init__(self, rf_estimator, num_classes: int):
        self.rf          = rf_estimator
        self.num_classes = num_classes

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_np  = x.squeeze(1).cpu().numpy()                          # (B, L)
        feats = np.stack([extract_rf_features(xi) for xi in x_np]) # (B, F)
        proba = self.rf.predict_proba(feats).astype(np.float32)     # (B, C)
        return torch.from_numpy(proba)

    # ── 兼容 PyTorch 接口（空操作）──
    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def state_dict(self):
        raise NotImplementedError(
            "RF 不支持 state_dict，请通过 ckpt['rf_wrapper'] 整体保存/加载。"
        )

    def load_state_dict(self, *a, **kw):
        raise NotImplementedError("RF 不支持 load_state_dict。")


# =============================================================================
# 工厂函数 & 工具
# =============================================================================

def build_model(arch_or_type: str, num_classes: int) -> "nn.Module":
    """
    根据架构名或简写别名返回已初始化的 PyTorch 模型（不含 RF）。

    支持的 arch_or_type:
      "cnn" / "cnn1d_bearing_v1"
      "lstm" / "lstm_bearing_v1"
      "transformer" / "transformer_bearing_v1" / "tf"

    RF 不通过此函数构建（需先 fit sklearn RF，再用 BearingRFWrapper 封装）。
    """
    key = _ARCH_ALIASES.get(arch_or_type.lower(), arch_or_type)
    if key == ARCH_CNN:
        return BearingCNN1D(num_classes=num_classes)
    if key == ARCH_LSTM:
        return BearingLSTM(num_classes=num_classes)
    if key == ARCH_TRANSFORMER:
        return BearingTransformer(num_classes=num_classes)
    if key == ARCH_RF:
        raise ValueError(
            "RF 模型请使用 BearingRFWrapper 直接封装 sklearn RF，不通过 build_model 构建。"
        )
    raise ValueError(
        f"未知模型架构: {arch_or_type!r}。支持: {sorted(set(_ARCH_ALIASES.values()))}"
    )


def count_feature_layers(model: "nn.Module") -> int:
    """
    返回模型中特征提取层数（用于日志打印）：
      CNN         → nn.Conv1d 层数
      LSTM        → LSTM 堆叠层数
      Transformer → TransformerEncoderLayer 层数
    """
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv1d))
    if n_conv:
        return n_conv
    if isinstance(model, BearingLSTM):
        return model.lstm.num_layers
    if isinstance(model, BearingTransformer):
        return len(model.encoder.layers)
    return 0
