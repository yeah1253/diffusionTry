"""
models.py — 下游故障诊断 1D-CNN 分类器

提供一个轻量级 1D CNN 分类器，通道逐层递增 (16→32→64→128)，
用于 TRTR / TSTR 对比实验。

依赖：torch
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class BearingCNN1D(nn.Module):
    """
    1D-CNN 轴承故障诊断分类器。

    网络结构：
        Conv1d(in_ch, 16) → BN → ReLU → MaxPool
        Conv1d(16, 32)    → BN → ReLU → MaxPool
        Conv1d(32, 64)    → BN → ReLU → MaxPool
        Conv1d(64, 128)   → BN → ReLU → AdaptiveAvgPool
        Flatten → Dropout → Linear(128, num_classes)

    输入 : (B, 1, L)   — 单通道 1D 振动信号
    输出 : (B, num_classes) — 分类 logits（未经 softmax）
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        seq_length: int = 1024,
        dropout: float = 0.3,
    ) -> None:
        """
        参数
        ----
        num_classes : int
            故障类别数量。
        in_channels : int
            输入通道数（默认 1）。
        seq_length : int
            输入序列长度（默认 1024），仅供文档记录，网络通过
            AdaptiveAvgPool1d 自动适应不同长度。
        dropout : float
            分类头中 Dropout 的概率。
        """
        super().__init__()
        self.num_classes = num_classes

        # ---------- 特征提取器 ----------
        self.features = nn.Sequential(
            # Block 1: 1 → 16
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L → L/2

            # Block 2: 16 → 32
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L/2 → L/4

            # Block 3: 32 → 64
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L/4 → L/8

            # Block 4: 64 → 128
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),                # L/8 → 1
        )

        # ---------- 分类头 ----------
        self.classifier = nn.Sequential(
            nn.Flatten(),               # (B, 128, 1) → (B, 128)
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        # 参数初始化
        self._init_weights()

    def _init_weights(self) -> None:
        """使用 Kaiming 初始化卷积层，Xavier 初始化全连接层。"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播。

        参数
        ----
        x : Tensor, shape (B, 1, L)

        返回
        ----
        logits : Tensor, shape (B, num_classes)
        """
        h = self.features(x)       # (B, 128, 1)
        logits = self.classifier(h) # (B, num_classes)
        return logits

    def predict(self, x: Tensor) -> Tensor:
        """返回预测类别索引。"""
        logits = self.forward(x)
        return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# 辅助：快速构建模型并移到指定设备
# ---------------------------------------------------------------------------

def build_cnn(
    num_classes: int,
    device: Optional[torch.device] = None,
    **kwargs,
) -> BearingCNN1D:
    """
    快速构建 BearingCNN1D 并转移到指定设备。

    参数
    ----
    num_classes : int
        故障类别数量。
    device : torch.device | None
        目标设备，默认使用 CUDA（如可用）。
    **kwargs
        其他传给 BearingCNN1D 的参数。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BearingCNN1D(num_classes=num_classes, **kwargs)
    return model.to(device)

