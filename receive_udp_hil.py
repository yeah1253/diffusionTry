#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边缘实时诊断节点：从 Speedgoat / Simulink 经 UDP 接收 1D 振动信号（每包 128×double），
凑满 1024 点后做 PyTorch 推理，并对连续推理结果做多数投票后输出「最终确诊」。

Simulink 参考：UDP Receive width=1024，Byte Unpack [128] double。

在 IDE 中直接点击 Run 即可运行；所有参数在下方「全局配置区」修改。
"""

from __future__ import annotations

import errno
import socket
import struct
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

# =============================================================================
# 全局配置区（直接修改此处，无需命令行）
# =============================================================================

BIND_IP = "0.0.0.0"
UDP_PORT = 10001
POINTS_PER_INFERENCE = 1024
DOUBLES_PER_PACKET = 128
LITTLE_ENDIAN = True

# 相对路径相对于本脚本所在目录解析
MODEL_PATH = "best_model.pth"
# 分类类别数（与 train_bearing_cnn1d 训练一致；占位模型也会用此维度）
NUM_CLASSES = 10

# 须与训练脚本 train_bearing_cnn1d.py 中 NORMALIZE_PER_WINDOW 一致
NORMALIZE_PER_WINDOW = True

VOTE_WINDOW = 3  # 最近多少次单次推理结果参与多数投票

# 推理设备：边缘节点一般用 cpu
DEVICE_STR = "cpu"

# =============================================================================
# 工具函数
# =============================================================================


def unpack_packet(data: bytes, n_double: int, little_endian: bool) -> tuple[float, ...]:
    if len(data) != n_double * 8:
        raise ValueError(f"expected {n_double * 8} bytes, got {len(data)}")
    fmt = "<" if little_endian else ">"
    return struct.unpack(fmt + f"{n_double}d", data)


def resolve_model_path(model_path: str) -> Path:
    p = Path(model_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def _make_dummy_classifier(num_classes: int):
    """占位 1D 分类器：输入 [B,1,L]，输出 [B, num_classes]。"""
    import torch
    import torch.nn as nn

    class _Dummy(nn.Module):
        def __init__(self, n_cls: int):
            super().__init__()
            self.fc = nn.Linear(POINTS_PER_INFERENCE, n_cls)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, 1, L] -> 去掉通道维 -> [B, L]，与常见 1D CNN 前 Flatten 类似
            if x.dim() != 3:
                raise ValueError(f"期望输入 3D [B,1,L]，当前 dim={x.dim()}, shape={tuple(x.shape)}")
            x = x.squeeze(1)
            return self.fc(x)

    return _Dummy(num_classes)


def DummyDiagnosticModel(num_classes: int = 2):
    """无权重或加载失败时使用，保证 UDP 主循环可运行。"""
    return _make_dummy_classifier(num_classes)


def load_diagnostic_model(model_path: str, num_classes: int):
    """
    加载预训练诊断模型（函数桩 + 可运行占位实现）。

    期望前向输入张量形状: (batch, 1, 1024)，即 [N, C, L]，C=1 为单通道 1D 序列。

    支持情况：
    - train_bearing_cnn1d 保存的 checkpoint（含 architecture=cnn1d_bearing_v1 与 state_dict）
    - 完整 nn.Module 或 dict['model'] 为 nn.Module
    - 失败或文件不存在：DummyDiagnosticModel
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[错误] 未安装 PyTorch，请执行: pip install torch", file=sys.stderr)
        return None

    try:
        from bearing_models import BEARING_CNN_ARCH, BearingCNN1D
    except ImportError:
        BEARING_CNN_ARCH = None  # type: ignore
        BearingCNN1D = None  # type: ignore

    path = resolve_model_path(model_path)
    if not path.is_file():
        print(f"[警告] 模型文件不存在: {path}，使用占位模型（随机初始化权重，仅用于通路测试）。")
        m = DummyDiagnosticModel(num_classes=num_classes)
        m.eval()
        return m

    try:
        # PyTorch 2.4+ 新增 weights_only；旧版无此参数
        try:
            obj = torch.load(str(path), map_location=DEVICE_STR, weights_only=False)
        except TypeError:
            obj = torch.load(str(path), map_location=DEVICE_STR)
    except Exception as e:
        print(f"[警告] torch.load 失败: {e}，改用占位模型。")
        m = DummyDiagnosticModel(num_classes=num_classes)
        m.eval()
        return m

    if isinstance(obj, nn.Module):
        obj.to(DEVICE_STR)
        obj.eval()
        print(f"[信息] 已加载 nn.Module: {path}")
        return obj

    # 轴承训练脚本保存: architecture + state_dict + num_classes
    if isinstance(obj, dict):
        arch = obj.get("architecture")
        if (
            BearingCNN1D is not None
            and BEARING_CNN_ARCH is not None
            and arch == BEARING_CNN_ARCH
            and "state_dict" in obj
        ):
            nc = int(obj.get("num_classes", num_classes))
            m = BearingCNN1D(num_classes=nc)
            m.load_state_dict(obj["state_dict"], strict=True)
            m.to(DEVICE_STR)
            m.eval()
            names = obj.get("class_names")
            print(f"[信息] 已加载轴承 CNN1D checkpoint: {path} | num_classes={nc}")
            if names:
                print(f"[信息] 类别顺序: {names}")
            return m

        if "model" in obj and isinstance(obj["model"], nn.Module):
            m = obj["model"]
            m.to(DEVICE_STR)
            m.eval()
            print(f"[信息] 已从 checkpoint 字段 'model' 加载: {path}")
            return m
        print(
            "[警告] checkpoint 为 dict 但无法识别（非 cnn1d_bearing_v1 / 无 model）。"
            "请检查 train_bearing_cnn1d 输出或自行扩展 load_diagnostic_model。"
        )
    else:
        print(f"[警告] 不支持的加载结果类型: {type(obj)}，改用占位模型。")

    m = DummyDiagnosticModel(num_classes=num_classes)
    m.eval()
    return m


def run_single_inference(model, chunk: list[float], device_str: str) -> int:
    """
    将 1024 个 float 转为张量，形状变换后前向推理，返回类别 ID（0..NUM_CLASSES-1）。

    Shape 流程：
      list[1024] -> ndarray float32（可选按窗标准化，与训练一致）
      -> Tensor[1, 1, 1024]
    """
    import numpy as np
    import torch

    if len(chunk) != POINTS_PER_INFERENCE:
        raise ValueError(f"chunk 长度应为 {POINTS_PER_INFERENCE}，实际 {len(chunk)}")

    arr = np.asarray(chunk, dtype=np.float32)
    if NORMALIZE_PER_WINDOW:
        # 与 train_bearing_cnn1d.BearingMatWindowDataset 中逐窗 z-score 一致
        arr = (arr - float(arr.mean())) / (float(arr.std()) + 1e-6)

    x = torch.from_numpy(arr).to(device=device_str)
    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 1024]

    with torch.no_grad():
        out = model(x)

    if out.dim() == 0:
        return int(out.item())
    # [B, num_classes] 取 argmax
    if out.dim() >= 2:
        return int(out.argmax(dim=-1).item())
    return int(out.item())


def majority_vote(
    recent_preds: list[int],
    tie_fallback: int | None,
) -> tuple[int, bool]:
    """
    多数投票。若最高票并列：优先沿用 tie_fallback（上一次【最终确诊】）；若其为 None，取并列类中 ID 最小者。
    返回: (最终类别, 是否发生平局)
    """
    if not recent_preds:
        raise ValueError("recent_preds 为空")

    cnt = Counter(recent_preds)
    max_v = max(cnt.values())
    candidates = sorted(k for k, v in cnt.items() if v == max_v)
    if len(candidates) == 1:
        return candidates[0], False

    # 平局：保留上一次确诊（若存在），否则取最小类 ID
    if tie_fallback is not None:
        return tie_fallback, True
    return candidates[0], True


def bind_udp_or_fallback(sock: socket.socket, host: str, port: int) -> str:
    """绑定 UDP；若指定 IP 非本机（如误填 Speedgoat 地址），回退到 0.0.0.0。"""
    h = (host or "").strip() or "0.0.0.0"

    def _not_avail(err: OSError) -> bool:
        if err.errno == errno.EADDRNOTAVAIL:
            return True
        return getattr(err, "winerror", None) == 10049

    try:
        sock.bind((h, port))
        return h
    except OSError as e:
        if h != "0.0.0.0" and _not_avail(e):
            print(
                f"[警告] 无法绑定 {h}:{port}（该地址可能不属于本机）。改用 0.0.0.0。",
                file=sys.stderr,
            )
            sock.bind(("0.0.0.0", port))
            return "0.0.0.0"
        raise


def main() -> int:
    pkt_size = DOUBLES_PER_PACKET * 8
    script_dir = Path(__file__).resolve().parent

    # ----- 模型加载（先于 UDP） -----
    try:
        model = load_diagnostic_model(MODEL_PATH, NUM_CLASSES)
    except Exception as e:
        print(f"[错误] 模型初始化异常: {e}", file=sys.stderr)
        return 1

    if model is None:
        return 1

    device_str = DEVICE_STR
    try:
        import torch

        if device_str == "cuda" and not torch.cuda.is_available():
            print("[警告] CUDA 不可用，改用 CPU。", file=sys.stderr)
            device_str = "cpu"
    except ImportError:
        pass

    # ----- UDP -----
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        bound_ip = bind_udp_or_fallback(sock, BIND_IP, UDP_PORT)
    except OSError as e:
        print(f"[错误] UDP 绑定失败 {BIND_IP}:{UDP_PORT} — {e}", file=sys.stderr)
        return 1

    print(
        f"[启动] {datetime.now():%Y-%m-%d %H:%M:%S} | "
        f"监听 udp://{bound_ip}:{UDP_PORT} | "
        f"每包 {DOUBLES_PER_PACKET} doubles ({pkt_size} B) | "
        f"每 {POINTS_PER_INFERENCE} 点推理一次 | 投票窗口 {VOTE_WINDOW}"
    )

    buf: list[float] = []
    vote_q: deque[int] = deque(maxlen=VOTE_WINDOW)
    last_final: int | None = None

    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except OSError as e:
                print(f"[错误] recvfrom: {e}", file=sys.stderr)
                continue

            if len(data) != pkt_size:
                print(f"[警告] 来自 {addr} 的包长 {len(data)} != {pkt_size}，跳过。", file=sys.stderr)
                continue

            try:
                samples = unpack_packet(data, DOUBLES_PER_PACKET, LITTLE_ENDIAN)
            except ValueError as e:
                print(f"[警告] 解析失败 {addr}: {e}", file=sys.stderr)
                continue

            buf.extend(samples)

            while len(buf) >= POINTS_PER_INFERENCE:
                chunk = buf[:POINTS_PER_INFERENCE]
                del buf[:POINTS_PER_INFERENCE]

                try:
                    pred = run_single_inference(model, chunk, device_str)
                    if pred < 0 or pred >= NUM_CLASSES:
                        print(f"[警告] 推理类别越界 pred={pred}，已钳制到 [0,{NUM_CLASSES-1}]", file=sys.stderr)
                        pred = max(0, min(NUM_CLASSES - 1, pred))
                except Exception as e:
                    print(f"[警告] 推理失败（跳过本窗）: {e}", file=sys.stderr)
                    continue

                vote_q.append(pred)
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if len(vote_q) < VOTE_WINDOW:
                    print(
                        f"[{now_str}] 单次推理={pred} | 投票未满 ({len(vote_q)}/{VOTE_WINDOW}) | 最终确诊=—"
                    )
                    continue

                final_id, tied = majority_vote(list(vote_q), last_final)
                if tied:
                    print(f"[{now_str}] 【平局】按规则沿用上次确诊或取较小类 -> 最终={final_id}", file=sys.stderr)
                last_final = final_id

                print(
                    f"[{now_str}] 单次推理={pred} | 窗口={list(vote_q)} | 【最终确诊】class={final_id}"
                )

    except KeyboardInterrupt:
        print("\n[信息] 用户中断，退出。", file=sys.stderr)
    finally:
        sock.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
