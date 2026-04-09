#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边缘实时诊断节点（双进程）：从 Speedgoat / Simulink 经 UDP 接收带标签的 1D 振动报文。

说明：离线 val/test 高而 UDP 准确率低，常见原因包括 (1) 目标机侧随机增强使分布偏离训练集；
(2) 换类时推理端滑窗仍拼接旧类样本。请与 interpolate_hil.m 的 USE_RANDOM_AUGMENT_FOR_UDP 与本脚本
GT 切换清空缓冲配合使用。
由独立接收进程极速收包并打点，推理进程维护 1024 点缓冲、PyTorch 推理与多数投票，
支持端到端延迟测评、准确率统计与 CSV 持久化。

报文格式（每包）：
  第 1 个 double：Ground Truth 标签（解析为 int）
  后 128 个 double：振动数据
  共 DOUBLES_PER_PACKET = 129 个 double → 129×8 字节。

Simulink 侧需与 Byte Unpack [129] double 对齐。

在 IDE 中直接点击 Run 即可；所有参数在下方「全局配置区」修改（无 argparse）。

空闲退出：UDP_IDLE_TIMEOUT_SEC 秒内未收到「长度正确且解析成功并入队」的包时，接收进程停止并触发汇总，
主进程打印平均诊断时间（端到端 ms）与平均诊断准确率后退出。
"""

from __future__ import annotations

import csv
import errno
import queue
import socket
import struct
import sys
import time
from collections import Counter, deque
from datetime import datetime
from multiprocessing import Event, Process, Queue
from pathlib import Path

# =============================================================================
# 全局配置区（直接修改此处，无需命令行）
# =============================================================================

BIND_IP = "0.0.0.0"
UDP_PORT = 10001
POINTS_PER_INFERENCE = 1024
# 1 个标签(double 转 int) + 128 个振动采样
DOUBLES_PER_PACKET = 129
SIGNAL_DOUBLES_PER_PACKET = 128  # 每包中振动点数（= DOUBLES_PER_PACKET - 1）
LITTLE_ENDIAN = True

# 接收 → 推理 之间的有界队列；满时丢弃最旧条目。略小可降低排队导致的 E2E 尖峰与过时样本
QUEUE_MAXSIZE = 24

# 模型选择：从 train_bearing_cnn1d.py 双模型训练生成的两个文件中选一个
#   "model_mixed.pth"     — 模型1：真实+生成数据混合（核心区 1:1，边缘区 1:9）
#   "model_real_only.pth" — 模型2：纯真实数据（数量与混合模型中真实数据相同）
#   "best_model.pth"      — 原有单模型训练输出（--single 模式）
# 修改下方 MODEL_PATH 切换模型。
MODEL_PATH = "model_real_only_cnn.pth"
NUM_CLASSES = 10

NORMALIZE_PER_WINDOW = True
VOTE_WINDOW = 3
DEVICE_STR = "cpu"

# 测评结果写入 CSV（由推理进程追加）
SAVE_METRICS = True
METRICS_FILE = "evaluation_results.csv"

# 接收空闲超时：连续若干秒未收到「有效」UDP 包（长度正确且解析成功并入队）则结束运行并输出汇总
UDP_IDLE_TIMEOUT_SEC = 5.0

# =============================================================================
# 工具函数
# =============================================================================


def fetch_stats_from_queue(stats_queue: Queue, retries: int = 40, sleep_s: float = 0.05) -> dict | None:
    """推理进程退出后会 put 一条汇总；短暂轮询避免竞态。"""
    for _ in range(retries):
        try:
            return stats_queue.get_nowait()
        except queue.Empty:
            time.sleep(sleep_s)
    return None


def print_session_summary(
    stats: dict | None,
    metrics_path: Path,
    *,
    reason: str,
) -> None:
    """
    打印会话汇总：突出「平均诊断时间（端到端 ms）」与「平均诊断准确率」
    （投票后的最终确诊准确率；若无投票记录则回退为单次窗准确率）。
    """
    print("\n" + "=" * 60)
    print(f"[测评汇总] {reason}")
    print("=" * 60)
    if not stats or not stats.get("ok"):
        err = stats.get("error", stats) if isinstance(stats, dict) else stats
        print(f"  未能获取完整统计: {err}")
        print("=" * 60 + "\n")
        return

    ti = int(stats["total_inferences"])
    cs = int(stats["correct_single"])
    nf = int(stats["n_final"])
    cf = int(stats["correct_final"])
    mean_lat = float(stats["mean_latency_ms"])

    # 平均诊断准确率：优先「最终确诊」；尚无投票输出时用语义明确的单次窗准确率
    if nf > 0:
        diag_acc = cf / nf
        diag_note = f"最终确诊 {cf}/{nf}"
    else:
        diag_acc = (cs / ti) if ti > 0 else 0.0
        diag_note = f"单次推理 {cs}/{ti}（投票未满 {VOTE_WINDOW} 次，无最终确诊统计）"

    print(f"  【平均诊断时间】   {mean_lat:.3f} ms（各次推理端到端延迟算术平均，共 {ti} 次）")
    print(f"  【平均诊断准确率】 {diag_acc:.4f}（{diag_note}）")
    print("  —— 明细 ——")
    print(f"  总推理次数:        {ti}")
    print(f"  单次预测准确率:    {cs}/{ti} = {(cs / ti if ti else 0):.4f}")
    print(f"  最终确诊次数:      {nf}")
    print(f"  最终确诊准确率:    {cf}/{nf} = {(cf / nf if nf else 0):.4f}")
    print(f"  平均端到端延迟:    {mean_lat:.3f} ms")
    mn, mx = stats.get("min_latency_ms"), stats.get("max_latency_ms")
    if ti > 0 and mn is not None and mx is not None:
        print(f"  最小端到端延迟:    {mn:.3f} ms")
        print(f"  最大端到端延迟:    {mx:.3f} ms")
    elif ti == 0:
        print("  最小/最大延迟:     无推理记录")
    if SAVE_METRICS:
        print(f"  指标已写入:        {metrics_path.resolve()}")
    print("=" * 60 + "\n")


def unpack_labeled_packet(
    data: bytes, n_double: int, little_endian: bool
) -> tuple[int, tuple[float, ...]]:
    """
    解析 UDP 载荷：第 1 个 double → 真实标签（转 int），其余为振动数据。
    返回 (label_int, (128 个 float 信号))
    """
    if len(data) != n_double * 8:
        raise ValueError(f"expected {n_double * 8} bytes, got {len(data)}")
    fmt = "<" if little_endian else ">"
    vals = struct.unpack(fmt + f"{n_double}d", data)
    raw_label = vals[0]
    label_int = int(round(raw_label))
    signal = vals[1:]
    if len(signal) != SIGNAL_DOUBLES_PER_PACKET:
        raise ValueError(
            f"信号长度应为 {SIGNAL_DOUBLES_PER_PACKET}，实际 {len(signal)}（请检查 DOUBLES_PER_PACKET）"
        )
    return label_int, signal


def resolve_model_path(model_path: str) -> Path:
    p = Path(model_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def _make_dummy_classifier(num_classes: int):
    import torch
    import torch.nn as nn

    class _Dummy(nn.Module):
        def __init__(self, n_cls: int):
            super().__init__()
            self.fc = nn.Linear(POINTS_PER_INFERENCE, n_cls)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 3:
                raise ValueError(f"期望输入 3D [B,1,L]，当前 dim={x.dim()}, shape={tuple(x.shape)}")
            x = x.squeeze(1)
            return self.fc(x)

    return _Dummy(num_classes)


def DummyDiagnosticModel(num_classes: int = 2):
    return _make_dummy_classifier(num_classes)


def num_classes_from_model(model) -> int | None:
    """从已加载模块推断分类数，避免与 checkpoint 不一致。"""
    try:
        clf = getattr(model, "classifier", None)
        if clf is not None and hasattr(clf, "out_features"):
            return int(clf.out_features)
        fc = getattr(model, "fc", None)
        if fc is not None and hasattr(fc, "out_features"):
            return int(fc.out_features)
        nc = getattr(model, "num_classes", None)
        if nc is not None:
            return int(nc)
    except (TypeError, ValueError):
        pass
    return None


def load_diagnostic_model(model_path: str, num_classes: int):
    """
    在**当前进程**内加载模型（子进程须各自调用，勿跨进程传递 nn.Module）。

    支持从 model.py 导出的全部架构（CNN / LSTM / Transformer / RF），
    也兼容仅用 bearing_models.py 保存的旧版 CNN checkpoint。
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[错误] 未安装 PyTorch，请执行: pip install torch", file=sys.stderr)
        return None

    # 尝试导入新架构模块（model.py）
    try:
        from model import (
            ARCH_CNN, ARCH_LSTM, ARCH_TRANSFORMER, ARCH_RF,
            build_model, BearingRFWrapper,
        )
        _model_ok = True
    except ImportError:
        _model_ok = False
        ARCH_CNN = ARCH_LSTM = ARCH_TRANSFORMER = ARCH_RF = None  # type: ignore

    # 保留旧版 bearing_models.py 兼容（BEARING_CNN_ARCH = "cnn1d_bearing_v1"）
    try:
        from bearing_models import BEARING_CNN_ARCH, BearingCNN1D as _LegacyCNN
    except ImportError:
        BEARING_CNN_ARCH = None  # type: ignore
        _LegacyCNN = None       # type: ignore

    path = resolve_model_path(model_path)
    if not path.is_file():
        print(f"[警告] 模型文件不存在: {path}，使用占位模型。")
        m = DummyDiagnosticModel(num_classes=num_classes)
        m.eval()
        return m

    try:
        try:
            obj = torch.load(str(path), map_location=DEVICE_STR, weights_only=False)
        except TypeError:
            obj = torch.load(str(path), map_location=DEVICE_STR)
    except Exception as e:
        print(f"[警告] torch.load 失败: {e}，改用占位模型。")
        m = DummyDiagnosticModel(num_classes=num_classes)
        m.eval()
        return m

    # ── 直接存储的 nn.Module ──
    if isinstance(obj, nn.Module):
        obj.to(DEVICE_STR)
        obj.eval()
        print(f"[信息] 已加载 nn.Module: {path}")
        return obj

    if isinstance(obj, dict):
        arch = obj.get("architecture")
        nc   = int(obj.get("num_classes", num_classes))
        names = obj.get("class_names")

        # ── 随机森林 ──
        if _model_ok and arch == ARCH_RF and "rf_wrapper" in obj:
            wrapper = obj["rf_wrapper"]
            print(f"[信息] 已加载 RandomForest checkpoint: {path} | num_classes={nc}")
            if names:
                print(f"[信息] 类别顺序: {names}")
            return wrapper

        # ── PyTorch 神经网络（CNN / LSTM / Transformer）──
        if "state_dict" in obj:
            try:
                if _model_ok and arch in (ARCH_CNN, ARCH_LSTM, ARCH_TRANSFORMER):
                    m = build_model(arch, nc)
                elif _LegacyCNN is not None and arch == BEARING_CNN_ARCH:
                    # 兼容旧版 bearing_models.py 保存的 CNN checkpoint
                    m = _LegacyCNN(num_classes=nc)
                else:
                    print(
                        f"[警告] 未知架构 {arch!r}，尝试退回 CNN。",
                        file=sys.stderr,
                    )
                    if _model_ok:
                        m = build_model("cnn", nc)
                    else:
                        m = DummyDiagnosticModel(num_classes=nc)
                        m.eval()
                        return m

                m.load_state_dict(obj["state_dict"], strict=True)
                m.to(DEVICE_STR)
                m.eval()
                print(f"[信息] 已加载 {arch} checkpoint: {path} | num_classes={nc}")
                if names:
                    print(f"[信息] 类别顺序: {names}")
                return m
            except Exception as e:
                print(f"[警告] 加载 state_dict 失败: {e}，改用占位模型。", file=sys.stderr)

        if "model" in obj and isinstance(obj["model"], nn.Module):
            m = obj["model"]
            m.to(DEVICE_STR)
            m.eval()
            print(f"[信息] 已从 checkpoint 字段 'model' 加载: {path}")
            return m

        print("[警告] checkpoint dict 无法识别，改用占位模型。", file=sys.stderr)

    m = DummyDiagnosticModel(num_classes=num_classes)
    m.eval()
    return m


def run_single_inference(model, chunk: list[float], device_str: str) -> int:
    import numpy as np
    import torch
    import torch.nn as nn

    if len(chunk) != POINTS_PER_INFERENCE:
        raise ValueError(f"chunk 长度应为 {POINTS_PER_INFERENCE}，实际 {len(chunk)}")

    arr = np.asarray(chunk, dtype=np.float32)
    if NORMALIZE_PER_WINDOW:
        arr = (arr - float(arr.mean())) / (float(arr.std()) + 1e-6)

    x = torch.from_numpy(arr).to(device=device_str)
    x = x.unsqueeze(0).unsqueeze(0)   # (1, 1, L)

    if isinstance(model, nn.Module):
        # PyTorch 神经网络（CNN / LSTM / Transformer）
        with torch.no_grad():
            out = model(x)
    else:
        # 非 nn.Module 可调用对象（如 BearingRFWrapper），不需要 no_grad 上下文
        out = model(x)

    if isinstance(out, torch.Tensor):
        if out.dim() == 0:
            return int(out.item())
        return int(out.argmax(dim=-1).item())
    return int(out)


def majority_vote(recent_preds: list[int], tie_fallback: int | None) -> tuple[int, bool]:
    if not recent_preds:
        raise ValueError("recent_preds 为空")

    cnt = Counter(recent_preds)
    max_v = max(cnt.values())
    candidates = sorted(k for k, v in cnt.items() if v == max_v)
    if len(candidates) == 1:
        return candidates[0], False

    if tie_fallback is not None:
        return tie_fallback, True
    return candidates[0], True


def bind_udp_or_fallback(sock: socket.socket, host: str, port: int) -> str:
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


def put_queue_drop_oldest(q: Queue, item: object, max_attempts: int = 200) -> None:
    """
    有界队列写入：若已满则丢弃队列中最旧的一条再重试，
    保证保留最新 UDP 数据，避免推理过慢时无限积压旧包。
    """
    for _ in range(max_attempts):
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
    print("[警告] put_queue_drop_oldest 重试过多，丢弃本条新数据。", file=sys.stderr)


def drain_queue_completely(q: Queue) -> int:
    """
    清空有界队列中所有已入队、尚未被推理进程取走的 UDP 条目。
    用于「当前包真实标签与上一包不一致」时丢弃混类积压，避免跨故障拼窗。
    """
    n = 0
    while True:
        try:
            q.get_nowait()
            n += 1
        except queue.Empty:
            break
    return n


# =============================================================================
# 进程 A：UDP 接收（仅收包 + T_recv 打点 + 入队）
# =============================================================================


def receiver_main(
    pkt_queue: Queue,
    stop_event: Event,
    bind_ip: str,
    port: int,
    pkt_size: int,
    idle_timeout_sec: float,
) -> None:
    """
    在刚执行完 socket.recvfrom 返回后的第一行记录 T_recv（perf_counter），
    再解析报文，将 (T_recv, label, signal_tuple) 送入队列。
    若 idle_timeout_sec > 0：启动后或上一包「有效入队」后，连续超过该秒数无有效包则置 stop_event 并退出。
    """
    sock: socket.socket | None = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bound_ip = bind_udp_or_fallback(sock, bind_ip, port)
        print(
            f"[接收进程] 监听 udp://{bound_ip}:{port} | 每包 {pkt_size} B "
            f"({DOUBLES_PER_PACKET} doubles)",
            flush=True,
        )
        if idle_timeout_sec > 0:
            print(
                f"[接收进程] 空闲超时: 连续 {idle_timeout_sec:.1f}s 无有效 UDP 则自动停止。",
                flush=True,
            )

        # 上一包成功解析后的 GT；若本包 GT 与其不同则先清空 Queue（单生产者下 get_nowait 可排空）
        last_gt_label: int | None = None
        recv_started_mono = time.monotonic()
        last_valid_rx_mono: float | None = None

        def _check_idle_exit() -> bool:
            """若已超时返回 True（并已 set stop_event）。"""
            if idle_timeout_sec <= 0:
                return False
            now = time.monotonic()
            if last_valid_rx_mono is None:
                if now - recv_started_mono >= idle_timeout_sec:
                    print(
                        f"[接收进程] 已超过 {idle_timeout_sec:.1f}s 未收到任何有效 UDP，停止接收。",
                        flush=True,
                    )
                    stop_event.set()
                    return True
            else:
                if now - last_valid_rx_mono >= idle_timeout_sec:
                    print(
                        f"[接收进程] 已连续 {idle_timeout_sec:.1f}s 未收到有效 UDP，停止接收。",
                        flush=True,
                    )
                    stop_event.set()
                    return True
            return False

        while not stop_event.is_set():
            try:
                sock.settimeout(0.3)
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                if _check_idle_exit():
                    break
                continue
            except OSError as e:
                if stop_event.is_set():
                    break
                print(f"[接收进程] recvfrom 错误: {e}", file=sys.stderr)
                continue

            # —— 端到端延迟起点：数据到达本机的时刻（recvfrom 返回后立即打点）——
            t_recv = time.perf_counter()

            if len(data) != pkt_size:
                print(
                    f"[接收进程] 来自 {addr} 包长 {len(data)} != {pkt_size}，跳过。",
                    file=sys.stderr,
                )
                if _check_idle_exit():
                    break
                continue

            try:
                label_int, signal = unpack_labeled_packet(data, DOUBLES_PER_PACKET, LITTLE_ENDIAN)
            except ValueError as e:
                print(f"[接收进程] 解析失败 {addr}: {e}", file=sys.stderr)
                if _check_idle_exit():
                    break
                continue

            # 若与上一 UDP 包的真实标签不一致，丢弃 Queue 内全部待处理数据，再入队本包
            if last_gt_label is not None and label_int != last_gt_label:
                dropped = drain_queue_completely(pkt_queue)
                if dropped > 0:
                    print(
                        f"[接收进程] 真实标签 {last_gt_label} -> {label_int}，"
                        f"已清空有界队列（移除 {dropped} 条）。",
                        flush=True,
                    )
            last_gt_label = label_int

            # 打包为可 pickle 的轻量元组，供推理进程消费
            item = (t_recv, label_int, signal)
            try:
                put_queue_drop_oldest(pkt_queue, item)
                last_valid_rx_mono = time.monotonic()
            except Exception as e:
                print(f"[接收进程] 入队异常: {e}", file=sys.stderr)
                if _check_idle_exit():
                    break

    except Exception as e:
        print(f"[接收进程] 未捕获异常: {e}", file=sys.stderr)
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        print("[接收进程] 已退出。", flush=True)


# =============================================================================
# 进程 B：推理与测评（滑动缓冲 + T_done + 准确率/延迟统计）
# =============================================================================


def _inference_worker_with_stats(
    pkt_queue: Queue,
    stop_event: Event,
    stats_queue: Queue,
    metrics_path: Path,
    save_metrics: bool,
    num_classes: int,
    model_path: str,
    device_str: str,
) -> None:
    """
    从队列取 (T_recv, label, 128 点振动)；deque 缓冲样本流；
    每满 1024 点推理一次，T_done 与触发该次推理的数据包 T_recv 之差为端到端延迟。
    退出时经 stats_queue 向主进程发送汇总（供 Ctrl+C 后打印报告）。
    """

    try:
        model = load_diagnostic_model(model_path, num_classes)
    except Exception as e:
        print(f"[推理进程] 模型加载失败: {e}", file=sys.stderr)
        stats_queue.put(
            {
                "ok": False,
                "error": str(e),
            }
        )
        return

    if model is None:
        stats_queue.put({"ok": False, "error": "model is None"})
        return

    inferred_nc = num_classes_from_model(model)
    if inferred_nc is not None and inferred_nc != num_classes:
        print(
            f"[推理进程] 分类数以模型为准: num_classes={inferred_nc}（全局配置为 {num_classes}）",
            flush=True,
        )
        num_classes = inferred_nc
    elif inferred_nc is not None:
        num_classes = inferred_nc

    try:
        import torch

        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
    except ImportError:
        pass

    sample_buf: deque[float] = deque()
    vote_q: deque[int] = deque(maxlen=VOTE_WINDOW)
    last_final: int | None = None
    last_gt_consumer: int | None = None

    total_inferences = 0
    correct_single = 0
    n_final = 0
    correct_final = 0
    sum_latency_ms = 0.0
    min_lat_ms: float | None = None
    max_lat_ms: float | None = None

    csv_file = None
    csv_writer = None
    if save_metrics:
        try:
            new_file = not metrics_path.is_file()
            csv_file = open(metrics_path, "a", newline="", encoding="utf-8-sig")
            csv_writer = csv.writer(csv_file)
            if new_file:
                csv_writer.writerow(
                    [
                        "wall_time",
                        "true_label",
                        "single_pred",
                        "final_diagnosis",
                        "e2e_latency_ms",
                    ]
                )
                csv_file.flush()
        except OSError as e:
            print(f"[推理进程] 无法打开指标文件: {e}", file=sys.stderr)
            csv_file = None
            csv_writer = None

    print(
        f"[推理进程] 就绪 | 窗长 {POINTS_PER_INFERENCE} | 投票 {VOTE_WINDOW} | CSV={'开' if csv_writer else '关'}",
        flush=True,
    )

    def record_latency_ms(dt_s: float) -> None:
        nonlocal sum_latency_ms, min_lat_ms, max_lat_ms
        ms = dt_s * 1000.0
        sum_latency_ms += ms
        min_lat_ms = ms if min_lat_ms is None else min(min_lat_ms, ms)
        max_lat_ms = ms if max_lat_ms is None else max(max_lat_ms, ms)

    try:
        while True:
            if stop_event.is_set() and pkt_queue.empty():
                break

            try:
                t_recv, gt_label, signal = pkt_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break

            try:
                # GT 变化时清空滑窗与投票：否则 1024 点会跨故障拼接，与训练「单类连续采样」不一致
                gt_clamped = max(0, min(num_classes - 1, int(gt_label)))
                if last_gt_consumer is not None and gt_clamped != last_gt_consumer:
                    sample_buf.clear()
                    vote_q.clear()
                    last_final = None
                    print(
                        f"[推理进程] GT {last_gt_consumer} -> {gt_clamped}，已清空样本滑窗与投票状态。",
                        flush=True,
                    )
                last_gt_consumer = gt_clamped

                if gt_label != gt_clamped:
                    print(
                        f"[推理进程] 标签 {gt_label} 越界，已钳制为 {gt_clamped}。",
                        file=sys.stderr,
                    )

                # 当前队列元素对应「本 UDP 包」的 T_recv / 标签；128 点并入缓冲。
                # 当本次 extend 后首次满足 len>=1024 时，视为该包「触发」本次推理，端到端起点取本包 T_recv。
                sample_buf.extend(signal)

                while len(sample_buf) >= POINTS_PER_INFERENCE:
                    chunk = [sample_buf.popleft() for _ in range(POINTS_PER_INFERENCE)]

                    pred = run_single_inference(model, chunk, device_str)
                    if pred < 0 or pred >= num_classes:
                        pred = max(0, min(num_classes - 1, pred))

                    # 端到端终点：本窗 forward 结束时刻（与上方该次 recv 对应的 t_recv 同用 perf_counter 时钟域）
                    t_done = time.perf_counter()
                    e2e_ms = (t_done - t_recv) * 1000.0
                    record_latency_ms(t_done - t_recv)

                    total_inferences += 1
                    if pred == gt_clamped:
                        correct_single += 1

                    vote_q.append(pred)
                    now_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    final_str = ""
                    final_id: int | None = None
                    if len(vote_q) < VOTE_WINDOW:
                        print(
                            f"[{now_wall}] GT={gt_clamped} | 单次={pred} | 投票未满 ({len(vote_q)}/{VOTE_WINDOW}) | "
                            f"最终=— | E2E={e2e_ms:.3f} ms | 单次准确率={correct_single}/{total_inferences}"
                        )
                    else:
                        final_id, tied = majority_vote(list(vote_q), last_final)
                        if tied:
                            print(
                                f"[{now_wall}] 【平局】沿用规则 -> 最终={final_id}",
                                file=sys.stderr,
                            )
                        last_final = final_id
                        final_str = str(final_id)
                        n_final += 1
                        if final_id == gt_clamped:
                            correct_final += 1

                        acc_s = correct_single / total_inferences
                        acc_f = correct_final / n_final if n_final else 0.0
                        mean_lat = sum_latency_ms / total_inferences
                        print(
                            f"[{now_wall}] GT={gt_clamped} | 单次={pred} | 窗口={list(vote_q)} | "
                            f"【最终确诊】={final_id} | E2E={e2e_ms:.3f} ms | "
                            f"单次准确率={acc_s:.4f} | 最终准确率={acc_f:.4f} | 平均E2E={mean_lat:.3f} ms"
                        )

                    if csv_writer is not None:
                        try:
                            csv_writer.writerow(
                                [
                                    now_wall,
                                    gt_clamped,
                                    pred,
                                    final_str if final_str else "NA",
                                    f"{e2e_ms:.6f}",
                                ]
                            )
                            csv_file.flush()
                        except OSError as e:
                            print(f"[推理进程] 写 CSV 失败: {e}", file=sys.stderr)

            except Exception as e:
                print(f"[推理进程] 处理包异常（跳过）: {e}", file=sys.stderr)

    except Exception as e:
        print(f"[推理进程] 未捕获异常: {e}", file=sys.stderr)
    finally:
        if csv_file is not None:
            try:
                csv_file.close()
            except OSError:
                pass
        print("[推理进程] 已退出。", flush=True)

    mean_all = sum_latency_ms / total_inferences if total_inferences else 0.0
    stats_queue.put(
        {
            "ok": True,
            "total_inferences": total_inferences,
            "correct_single": correct_single,
            "n_final": n_final,
            "correct_final": correct_final,
            "mean_latency_ms": mean_all,
            "min_latency_ms": min_lat_ms,
            "max_latency_ms": max_lat_ms,
        }
    )


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    pkt_size = DOUBLES_PER_PACKET * 8
    metrics_path = script_dir / METRICS_FILE if not Path(METRICS_FILE).is_absolute() else Path(METRICS_FILE)

    pkt_queue: Queue = Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = Event()
    stats_queue: Queue = Queue()

    recv_proc = Process(
        target=receiver_main,
        args=(pkt_queue, stop_event, BIND_IP, UDP_PORT, pkt_size, UDP_IDLE_TIMEOUT_SEC),
        name="UDPReceiver",
        daemon=False,
    )
    inf_proc = Process(
        target=_inference_worker_with_stats,
        args=(
            pkt_queue,
            stop_event,
            stats_queue,
            metrics_path,
            SAVE_METRICS,
            NUM_CLASSES,
            MODEL_PATH,
            DEVICE_STR,
        ),
        name="InferenceWorker",
        daemon=False,
    )

    print(
        f"[主进程] {datetime.now():%Y-%m-%d %H:%M:%S} 启动闭环测评 | "
        f"队列容量={QUEUE_MAXSIZE}（满则丢最旧）| "
        f"每包 {DOUBLES_PER_PACKET} doubles（1 标签 + {SIGNAL_DOUBLES_PER_PACKET} 信号）| "
        f"空闲≥{UDP_IDLE_TIMEOUT_SEC:.1f}s 无有效 UDP 则自动退出"
    )

    user_interrupt = False
    try:
        recv_proc.start()
        inf_proc.start()

        while recv_proc.is_alive() or inf_proc.is_alive():
            recv_proc.join(timeout=0.5)
            inf_proc.join(timeout=0.5)

    except KeyboardInterrupt:
        user_interrupt = True
        print("\n[主进程] KeyboardInterrupt，正在停止子进程…", file=sys.stderr)
        stop_event.set()
        recv_proc.join(timeout=3.0)
        inf_proc.join(timeout=10.0)
        if recv_proc.is_alive():
            recv_proc.terminate()
        if inf_proc.is_alive():
            inf_proc.terminate()
        recv_proc.join(timeout=2.0)
        inf_proc.join(timeout=2.0)

    except Exception as e:
        print(f"[主进程] 异常: {e}", file=sys.stderr)
        stop_event.set()
        try:
            recv_proc.terminate()
            inf_proc.terminate()
            recv_proc.join(timeout=2.0)
            inf_proc.join(timeout=2.0)
        except Exception:
            pass
        stats_err = fetch_stats_from_queue(stats_queue)
        if stats_err is None:
            stats_err = {"ok": False, "error": str(e)}
        print_session_summary(stats_err, metrics_path, reason="主进程异常退出")
        return 1

    if user_interrupt:
        summary_reason = "用户中断 (KeyboardInterrupt)"
    else:
        summary_reason = (
            f"已连续 ≥{UDP_IDLE_TIMEOUT_SEC:.1f}s 未收到有效 UDP，接收进程已停止（或子进程已正常结束）"
        )

    stats = fetch_stats_from_queue(stats_queue)
    if stats is None:
        stats = {"ok": False, "error": "无统计（进程可能被强制结束或推理尚未写入汇总）"}
    print_session_summary(stats, metrics_path, reason=summary_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
