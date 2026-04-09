"""
models.py — 下游故障诊断深层 1D-ResNet 分类器

提供面向轴承振动信号的残差网络：
  - 大卷积核输入层用于捕获宽频带物理模式
  - 残差块 + BatchNorm 提升梯度稳定性与跨工况鲁棒性
  - GAP + Dropout + FC 完成分类

依赖：torch
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ResidualBlock1D(nn.Module):
    """基础残差块：Conv-BN-ReLU-Conv-BN + Skip。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.relu(out)


class BearingResNet1D(nn.Module):
    """
    1D-ResNet 轴承故障诊断分类器。

    物理与算法设计要点：
      1) 输入层使用大卷积核近似带通感受野，适配 1024 点宽频振动序列。
      2) 残差块通过跳连缓解深层训练退化，并配合 BN 抑制幅值尺度波动。
      3) GAP 输出全局语义嵌入（默认 256 维），供分类与 t-SNE 共用。
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        seq_length: int = 1024,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seq_length = seq_length

        # 输入层：大卷积核+下采样，扩大有效感受野并降低高频噪声抖动
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=64, stride=4, padding=32, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
        )

        # 主体：4 个残差块，通道阶梯 64 -> 128 -> 256
        self.layer1 = ResidualBlock1D(64, 64, stride=1, kernel_size=7)
        self.layer2 = ResidualBlock1D(64, 128, stride=2, kernel_size=7)
        self.layer3 = ResidualBlock1D(128, 256, stride=2, kernel_size=5)
        self.layer4 = ResidualBlock1D(256, 256, stride=1, kernel_size=5)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, num_classes)

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

    def forward_features(self, x: Tensor) -> Tensor:
        """提取全连接层前的深层物理语义嵌入，输出 shape=(B, 256)。"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1)
        return x

    def features(self, x: Tensor) -> Tensor:
        """兼容旧调用：保留 `model.features(signals)` 接口用于 t-SNE 提取。"""
        return self.forward_features(x)

    def forward(self, x: Tensor) -> Tensor:
        emb = self.forward_features(x)
        emb = self.dropout(emb)
        return self.fc(emb)

    def predict(self, x: Tensor) -> Tensor:
        """返回预测类别索引。"""
        logits = self.forward(x)
        return logits.argmax(dim=-1)


class BearingCNN1D(BearingResNet1D):
    """兼容旧命名：内部实现已升级为 BearingResNet1D。"""


# ---------------------------------------------------------------------------
# 辅助：快速构建模型并移到指定设备
# ---------------------------------------------------------------------------

def build_cnn(
    num_classes: int,
    device: Optional[torch.device] = None,
    **kwargs,
) -> BearingResNet1D:
    """
    快速构建 BearingResNet1D 并转移到指定设备。

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
    model = BearingResNet1D(num_classes=num_classes, **kwargs)
    return model.to(device)

