# -*- coding: utf-8 -*-
"""
轴承 1D CNN 诊断模型定义：与训练脚本 train_bearing_cnn1d.py、推理 receive_udp_hil.py 共用。

输入:  (batch, 1, 1024)  — 单通道、长度 1024 的振动片段
输出:  (batch, num_classes)  — 未归一化的 logits（CrossEntropyLoss 前向）
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 与 checkpoint 中 architecture 字段一致，用于 receive 端按 state_dict 重建网络
BEARING_CNN_ARCH = "cnn1d_bearing_v1"


def count_conv1d_layers(module: nn.Module) -> int:
    """统计模块树中 nn.Conv1d 个数（用于实时性评估与日志）。"""
    return sum(1 for m in module.modules() if isinstance(m, nn.Conv1d))


class BearingCNN1D(nn.Module):
    """
    轻量 1D CNN，适合 1024 点轴承振动片段的十分类（或其它 K 类，由 num_classes 指定）。

    卷积特征提取部分含 **4 个 Conv1d 层**（其后为 BN/ReLU/池化，最后 AdaptiveAvgPool1d + Linear）。
    修改实时性时可减少 Conv1d 数量或通道数。
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 1024 -> 256
            nn.Conv1d(32, 64, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 256 -> 64
            nn.Conv1d(64, 128, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 64 -> 16
            nn.Conv1d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # [B, 256, L] -> [B, 256, 1]
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"期望 x 形状 [B,1,L]，当前 {tuple(x.shape)}")
        h = self.features(x)
        h = h.squeeze(-1)  # [B, 256]
        return self.classifier(h)


def build_bearing_cnn(num_classes: int) -> BearingCNN1D:
    return BearingCNN1D(num_classes=num_classes)
