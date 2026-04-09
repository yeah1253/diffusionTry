# -*- coding: utf-8 -*-
"""
轴承故障诊断对比实验模型库（v2）。

包含 5 种 PyTorch 模型（特征提取层数统一 N_FEATURE_LAYERS = 4）：
  BearingCNN1D       — 4 层 Conv1d（标准 1D 卷积）
  BearingTransformer — 4 层 Pre-LN Transformer Encoder（Patch-ViT 风格）
  BearingTCN         — 4 层扩张残差 TCN，dilation=4^i，感受野≈1021（覆盖 1024 点序列）
  BearingMobileNet1D — 4 个深度可分离卷积块（轻量化边缘部署，参数量约 CNN 的 1/8）
  BearingLSTM        — 4 层堆叠双向 LSTM
  BearingRFWrapper   — 随机森林（sklearn），兼容 PyTorch 推理接口

所有 PyTorch 模型统一接受 (B, 1, L=1024)，输出 (B, num_classes) logits。

架构切换：修改本文件顶部的 ``SelectModel`` 变量，无需改动训练脚本。

典型用法（训练端）::
    from model import build_model, SelectModel
    model = build_model(SelectModel, num_classes=10)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# ★ 全局模型选择变量 ★
# 修改此处切换架构；训练脚本（train_bearing_cnn1d.py）会自动读取。
# =============================================================================

SelectModel: str = "lstm"
"""全局架构选择，可选值：
  "cnn"         — BearingCNN1D（标准 1D 卷积，推荐默认）
  "transformer" — BearingTransformer（Patch-ViT Encoder）
  "tcn"         — BearingTCN（扩张残差 TCN，dilation=4^i）
  "mobilenet"   — BearingMobileNet1D（深度可分离卷积，轻量化）
  "lstm"        — BearingLSTM（双向 LSTM）
"""

# =============================================================================
# 架构标识常量（用于 checkpoint["architecture"] 字段，保证版本兼容）
# =============================================================================

N_FEATURE_LAYERS: int = 4

ARCH_CNN         = "cnn1d_bearing_v1"
ARCH_LSTM        = "lstm_bearing_v1"
ARCH_TRANSFORMER = "transformer_bearing_v1"
ARCH_TCN         = "tcn_bearing_v1"
ARCH_MOBILENET   = "mobilenet1d_bearing_v1"
ARCH_RF          = "rf_bearing_v1"

_ARCH_ALIASES: dict[str, str] = {
    # CNN
    "cnn":              ARCH_CNN,
    ARCH_CNN:           ARCH_CNN,
    # LSTM
    "lstm":             ARCH_LSTM,
    ARCH_LSTM:          ARCH_LSTM,
    # Transformer
    "transformer":      ARCH_TRANSFORMER,
    "tf":               ARCH_TRANSFORMER,
    ARCH_TRANSFORMER:   ARCH_TRANSFORMER,
    # TCN
    "tcn":              ARCH_TCN,
    ARCH_TCN:           ARCH_TCN,
    # MobileNet
    "mobilenet":        ARCH_MOBILENET,
    "mobilenet1d":      ARCH_MOBILENET,
    "mobile":           ARCH_MOBILENET,
    ARCH_MOBILENET:     ARCH_MOBILENET,
    # RF
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
# 模型 1：BearingCNN1D — 标准 1D 卷积（旧版 checkpoint 可直接加载）
# =============================================================================

class BearingCNN1D(nn.Module):
    """
    轻量 1D CNN，4 个 Conv1d 特征提取层 + AdaptiveAvgPool + 分类头。

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
# 模型 2：BearingLSTM — 堆叠双向 LSTM
# =============================================================================

class BearingLSTM(nn.Module):
    """
    4 层堆叠双向 LSTM，取最后时刻拼接隐状态 → LayerNorm → 分类头。

    输入:  (B, 1, L=1024)
    输出:  (B, num_classes) logits
    """

    HIDDEN_SIZE: int = 128

    def __init__(self, num_classes: int = 10, num_layers: int = N_FEATURE_LAYERS):
        super().__init__()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.norm       = nn.LayerNorm(self.HIDDEN_SIZE * 2)
        self.classifier = nn.Linear(self.HIDDEN_SIZE * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.permute(0, 2, 1))   # (B, L, 1) → (B, L, H*2)
        return self.classifier(self.norm(out[:, -1, :]))


# =============================================================================
# 模型 3：BearingTransformer — Patch-ViT 风格 Transformer
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
    PATCH_SIZE : int   = 16      # 1024 / 16 = 64 patches
    N_HEAD     : int   = 4
    DIM_FF     : int   = 256
    DROPOUT    : float = 0.20    # 增强正则化（原 0.1 → 0.2）

    def __init__(self, num_classes: int = 10, num_layers: int = N_FEATURE_LAYERS):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size  = self.PATCH_SIZE
        n_patches        = 1024 // self.PATCH_SIZE

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
# 模型 4：BearingTCN — 时间卷积网络（扩张残差 TCN）
# =============================================================================

class _TCNResBlock(nn.Module):
    """
    TCN 残差块：两层扩张 Conv1d + BatchNorm + GELU + 残差连接。
    padding = (k−1)×d//2，保持序列长度不变（same padding）。
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                      padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
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

    扩张系数 dilation = 4^i → {1, 4, 16, 64}，对应感受野约 1021 点，
    可完整覆盖长度为 1024 的振动窗口。

    输入:  (B, 1, L=1024)
    输出:  (B, num_classes) logits
    """

    _CHANNELS      = [32, 64, 128, 256]
    _KERNEL_SIZE   = 7
    _DILATION_BASE = 4      # dilation = 4^i → {1,4,16,64}，RF≈1021
    _DROPOUT       = 0.1

    def __init__(self, num_classes: int = 10, num_layers: int = N_FEATURE_LAYERS):
        super().__init__()
        self.num_classes = num_classes
        channels = self._CHANNELS[:num_layers]
        in_ch    = 1
        blocks: list[nn.Module] = []
        for i, out_ch in enumerate(channels):
            dilation = self._DILATION_BASE ** i
            blocks.append(
                _TCNResBlock(in_ch, out_ch, self._KERNEL_SIZE,
                             dilation=dilation, dropout=self._DROPOUT)
            )
            in_ch = out_ch
        self.network    = nn.Sequential(*blocks)
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.network(x)).squeeze(-1))


# =============================================================================
# 模型 5：BearingMobileNet1D — 深度可分离卷积（轻量化）
# =============================================================================

class _DSConv1d(nn.Module):
    """深度可分离卷积块（Depthwise + Pointwise + BN + ReLU6）。"""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.dw  = nn.Conv1d(in_ch, in_ch, kernel_size, stride=stride,
                              padding=pad, groups=in_ch, bias=False)
        self.pw  = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class BearingMobileNet1D(nn.Module):
    """
    轻量化 MobileNet-1D，4 个深度可分离卷积块（DS-Conv），
    参数量约为标准 CNN 的 1/8，适合边缘部署场景。

    结构：Stem Conv(stride=2) → DS×4 → AdaptiveAvgPool → 分类头。

    输入:  (B, 1, L=1024)
    输出:  (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        # 标准 Conv 做初始通道扩展（不计入 N_FEATURE_LAYERS）
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU6(inplace=True),
        )
        # 4 个 DS-Conv 块（stride=2 下采样 × 3 次，1024/2^4=64）
        self.blocks = nn.Sequential(
            _DSConv1d(32,  64,  stride=2),   # Layer 1
            _DSConv1d(64,  128, stride=2),   # Layer 2
            _DSConv1d(128, 128, stride=1),   # Layer 3
            _DSConv1d(128, 256, stride=2),   # Layer 4
        )
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.blocks(self.stem(x))).squeeze(-1))


# =============================================================================
# 模型 6：随机森林（手工特征 + sklearn RF）
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

    def eval(self):     return self
    def to(self, *a, **kw): return self

    def state_dict(self):
        raise NotImplementedError("RF 不支持 state_dict，请通过 ckpt['rf_wrapper'] 整体保存。")

    def load_state_dict(self, *a, **kw):
        raise NotImplementedError("RF 不支持 load_state_dict。")


# =============================================================================
# 工厂函数 & 工具函数
# =============================================================================

def build_model(arch_or_type: str, num_classes: int) -> nn.Module:
    """
    根据架构名或简写别名返回已初始化的 PyTorch 模型（不含 RF）。

    支持的 arch_or_type：
      "cnn"  / "cnn1d_bearing_v1"
      "lstm" / "lstm_bearing_v1"
      "transformer" / "transformer_bearing_v1" / "tf"
      "tcn"  / "tcn_bearing_v1"
      "mobilenet" / "mobilenet1d" / "mobilenet1d_bearing_v1"
    """
    key = _ARCH_ALIASES.get(arch_or_type.lower(), arch_or_type)
    if key == ARCH_CNN:         return BearingCNN1D(num_classes=num_classes)
    if key == ARCH_LSTM:        return BearingLSTM(num_classes=num_classes)
    if key == ARCH_TRANSFORMER: return BearingTransformer(num_classes=num_classes)
    if key == ARCH_TCN:         return BearingTCN(num_classes=num_classes)
    if key == ARCH_MOBILENET:   return BearingMobileNet1D(num_classes=num_classes)
    if key == ARCH_RF:
        raise ValueError(
            "RF 模型请直接使用 BearingRFWrapper 封装 sklearn RF，不通过 build_model 构建。"
        )
    raise ValueError(
        f"未知模型架构: {arch_or_type!r}。支持: {sorted(set(_ARCH_ALIASES.values()))}"
    )


# =============================================================================
# 各架构推荐训练超参数
# =============================================================================

_HPARAMS: dict[str, dict] = {
    # ── CNN：收敛快、稳定，适度增大 batch 和 weight_decay 防止过拟合 ──────────
    ARCH_CNN: dict(
        lr=1e-3,    weight_decay=5e-4,  batch_size=64,
        epochs=200, patience=15,
        warmup_epochs=0,  grad_clip=0.0,  label_smoothing=0.05,
    ),
    # ── Transformer：对 lr 极敏感，必须 warmup + clip；需更多 epoch ────────────
    # 参考：早期成功配置 lr=2.5e-4 / wd=0.04 / warmup=8 / clip=1.0 / patience=22
    ARCH_TRANSFORMER: dict(
        lr=2.5e-4,  weight_decay=0.04,  batch_size=64,
        epochs=300, patience=25,
        warmup_epochs=10, grad_clip=1.0,  label_smoothing=0.10,
    ),
    # ── TCN：扩张卷积梯度较大，保留 clip；介于 CNN 与 Transformer 之间 ────────
    ARCH_TCN: dict(
        lr=6e-4,    weight_decay=1e-3,  batch_size=64,
        epochs=250, patience=20,
        warmup_epochs=5,  grad_clip=1.0,  label_smoothing=0.05,
    ),
    # ── MobileNet-1D：轻量模型，收敛速度接近 CNN ─────────────────────────────
    ARCH_MOBILENET: dict(
        lr=8e-4,    weight_decay=4e-4,  batch_size=64,
        epochs=200, patience=15,
        warmup_epochs=0,  grad_clip=0.0,  label_smoothing=0.05,
    ),
    # ── LSTM：RNN 梯度易爆炸，保留 clip ──────────────────────────────────────
    ARCH_LSTM: dict(
        lr=5e-4,    weight_decay=1e-4,  batch_size=32,
        epochs=200, patience=20,
        warmup_epochs=0,  grad_clip=1.0,  label_smoothing=0.0,
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
    warmup_epochs   : 线性 warmup 的 epoch 数（0 = 不使用；Transformer 推荐 10）
    grad_clip       : 梯度裁剪最大范数（0.0 = 不裁剪；Transformer/TCN/LSTM 推荐 1.0）
    label_smoothing : 交叉熵标签平滑系数（0.0 = 标准交叉熵）

    未知架构自动回退到 CNN 超参（向后兼容）。
    """
    key = _ARCH_ALIASES.get(model_type.lower(), model_type)
    return dict(_HPARAMS.get(key, _HPARAMS[ARCH_CNN]))


def count_feature_layers(model: nn.Module) -> int:
    """
    返回模型特征提取层数（用于日志打印）：
      CNN         → nn.Conv1d 层数（4）
      LSTM        → LSTM 堆叠层数
      Transformer → TransformerEncoderLayer 层数
      TCN         → TCN 残差块数（4）
      MobileNet   → DS-Conv 块数（4，不含 stem）
    """
    if isinstance(model, BearingCNN1D):
        return sum(1 for m in model.features.children() if isinstance(m, nn.Conv1d))
    if isinstance(model, BearingLSTM):
        return model.lstm.num_layers
    if isinstance(model, BearingTransformer):
        return len(model.encoder.layers)
    if isinstance(model, BearingTCN):
        return len(model.network)
    if isinstance(model, BearingMobileNet1D):
        return len(model.blocks)
    return sum(1 for m in model.modules() if isinstance(m, nn.Conv1d))
