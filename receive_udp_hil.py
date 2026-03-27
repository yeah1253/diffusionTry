#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边缘实时诊断节点（双进程）：从 Speedgoat / Simulink 经 UDP 接收带标签的 1D 振动报文，
由独立接收进程极速收包并打点，推理进程维护 1024 点缓冲、PyTorch 推理与多数投票，
支持端到端延迟测评、准确率统计与 CSV 持久化。

报文格式（每包）：
  第 1 个 double：Ground Truth 标签（解析为 int）
  后 128 个 double：振动数据
  共 DOUBLES_PER_PACKET = 129 个 double → 129×8 字节。

Simulink 侧需与 Byte Unpack [129] double 对齐。

在 IDE 中直接点击 Run 即可；所有参数在下方「全局配置区」修改（无 argparse）。
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

# 接收 → 推理 之间的有界队列；满时丢弃最旧条目，优先保留最新数据（降低积压滞后）
QUEUE_MAXSIZE = 50

# 相对路径相对于本脚本所在目录解析
MODEL_PATH = "best_model.pth"
NUM_CLASSES = 10

NORMALIZE_PER_WINDOW = True
VOTE_WINDOW = 3
DEVICE_STR = "cpu"

# 测评结果写入 CSV（由推理进程追加）
SAVE_METRICS = True
METRICS_FILE = "evaluation_results.csv"

# =============================================================================
# 工具函数
# =============================================================================


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


def load_diagnostic_model(model_path: str, num_classes: int):
    """在**当前进程**内加载模型（子进程须各自调用，勿跨进程传递 nn.Module）。"""
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

    if isinstance(obj, nn.Module):
        obj.to(DEVICE_STR)
        obj.eval()
        print(f"[信息] 已加载 nn.Module: {path}")
        return obj

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
        print("[警告] checkpoint dict 无法识别，改用占位模型。", file=sys.stderr)

    m = DummyDiagnosticModel(num_classes=num_classes)
    m.eval()
    return m


def run_single_inference(model, chunk: list[float], device_str: str) -> int:
    import numpy as np
    import torch

    if len(chunk) != POINTS_PER_INFERENCE:
        raise ValueError(f"chunk 长度应为 {POINTS_PER_INFERENCE}，实际 {len(chunk)}")

    arr = np.asarray(chunk, dtype=np.float32)
    if NORMALIZE_PER_WINDOW:
        arr = (arr - float(arr.mean())) / (float(arr.std()) + 1e-6)

    x = torch.from_numpy(arr).to(device=device_str)
    x = x.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out = model(x)

    if out.dim() == 0:
        return int(out.item())
    if out.dim() >= 2:
        return int(out.argmax(dim=-1).item())
    return int(out.item())


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
) -> None:
    """
    在刚执行完 socket.recvfrom 返回后的第一行记录 T_recv（perf_counter），
    再解析报文，将 (T_recv, label, signal_tuple) 送入队列。
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

        # 上一包成功解析后的 GT；若本包 GT 与其不同则先清空 Queue（单生产者下 get_nowait 可排空）
        last_gt_label: int | None = None

        while not stop_event.is_set():
            try:
                sock.settimeout(0.3)
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
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
                continue

            try:
                label_int, signal = unpack_labeled_packet(data, DOUBLES_PER_PACKET, LITTLE_ENDIAN)
            except ValueError as e:
                print(f"[接收进程] 解析失败 {addr}: {e}", file=sys.stderr)
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
            except Exception as e:
                print(f"[接收进程] 入队异常: {e}", file=sys.stderr)

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

    try:
        import torch

        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
    except ImportError:
        pass

    sample_buf: deque[float] = deque()
    vote_q: deque[int] = deque(maxlen=VOTE_WINDOW)
    last_final: int | None = None

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
                    gt_clamped = max(0, min(num_classes - 1, gt_label))
                    if gt_label != gt_clamped:
                        print(
                            f"[推理进程] 标签 {gt_label} 越界，已钳制为 {gt_clamped}。",
                            file=sys.stderr,
                        )
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
        args=(pkt_queue, stop_event, BIND_IP, UDP_PORT, pkt_size),
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
        f"每包 {DOUBLES_PER_PACKET} doubles（1 标签 + {SIGNAL_DOUBLES_PER_PACKET} 信号）"
    )

    try:
        recv_proc.start()
        inf_proc.start()

        # 主进程阻塞等待，Ctrl+C 触发 KeyboardInterrupt
        while recv_proc.is_alive() or inf_proc.is_alive():
            recv_proc.join(timeout=0.5)
            inf_proc.join(timeout=0.5)

    except KeyboardInterrupt:
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

        # 汇总报告（推理进程退出前已 put 一份 stats；若被 terminate 可能无数据）
        stats = None
        for _ in range(20):
            try:
                stats = stats_queue.get_nowait()
                break
            except queue.Empty:
                time.sleep(0.05)
        if stats is None:
            stats = {"ok": False, "error": "无统计（进程可能被强制结束或尚未写入）"}

        print("\n" + "=" * 60)
        print("[测评汇总] 会话结束")
        print("=" * 60)
        if stats.get("ok"):
            ti = int(stats["total_inferences"])
            cs = int(stats["correct_single"])
            nf = int(stats["n_final"])
            cf = int(stats["correct_final"])
            print(f"  总推理次数:        {ti}")
            print(f"  单次预测准确率:    {cs}/{ti} = {(cs / ti if ti else 0):.4f}")
            print(f"  最终确诊次数:      {nf}")
            print(f"  最终确诊准确率:    {cf}/{nf} = {(cf / nf if nf else 0):.4f}")
            print(f"  平均端到端延迟:    {stats['mean_latency_ms']:.3f} ms")
            mn, mx = stats.get("min_latency_ms"), stats.get("max_latency_ms")
            if ti > 0 and mn is not None and mx is not None:
                print(f"  最小端到端延迟:    {mn:.3f} ms")
                print(f"  最大端到端延迟:    {mx:.3f} ms")
            elif ti == 0:
                print("  最小/最大延迟:     无推理记录")
            if SAVE_METRICS:
                print(f"  指标已写入:        {metrics_path.resolve()}")
        else:
            print(f"  未能获取完整统计: {stats.get('error', stats)}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"[主进程] 异常: {e}", file=sys.stderr)
        stop_event.set()
        recv_proc.terminate()
        inf_proc.terminate()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
