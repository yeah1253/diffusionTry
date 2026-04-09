#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景二：故障突发与诊断延迟（Fault Injection Step）

目的：在转速、负载恒定前提下，Simulink 某时刻将 fault_sel 从「正常」阶跃到「内圈故障」，
测量从 PC 端**首次收到新 GT 的 UDP 包**到**首次给出与 GT 一致的最终确诊（多数投票）**的时间差。

前提与 receive_udp_hil.py 一致：
  - 每 UDP 包 129×double（1 个 GT + 128 点振动），Little-endian；
  - 1024 点凑满一次单次推理；最终确诊 = 最近 VOTE_WINDOW 次单次预测的多数票。

Simulink 侧建议：
  - target_load / target_rpm 恒定；
  - 先用 fault_sel 对应 NC（正常），运行稳定后阶跃到内圈类（如 IF0.2 → 标签 0）；
  - MATLAB Function 采样时间与 Digital Clock 一致（参见 interpolate_hil.m 头注释）。

在脚本顶部修改 GT 类别索引、UDP、模型路径等；直接 Run 即可。
"""

from __future__ import annotations

import csv
import json
import queue
import sys
import time
from collections import Counter, deque
from datetime import datetime
from multiprocessing import Event, Process, Queue
from pathlib import Path

# =============================================================================
# 配置（按训练 class_names 字典序：0=IF0.2, 1=IF0.4, …, 3=NC）
# =============================================================================

BIND_IP = "0.0.0.0"
UDP_PORT = 10001
POINTS_PER_INFERENCE = 1024
DOUBLES_PER_PACKET = 129
SIGNAL_DOUBLES_PER_PACKET = 128
LITTLE_ENDIAN = True
QUEUE_MAXSIZE = 24

MODEL_PATH = "best_model.pth"
NUM_CLASSES = 10
NORMALIZE_PER_WINDOW = True
VOTE_WINDOW = 3
DEVICE_STR = "cpu"

# 阶跃：从「正常」到「内圈故障」（可按 checkpoint 修改）
GT_LABEL_NORMAL = 3  # NC
GT_LABEL_INNER_FAULT = 0  # IF0.2

# 空闲退出（秒）：长时间无有效 UDP 则结束（与 receive_udp_hil 一致）
UDP_IDLE_TIMEOUT_SEC = 30.0

# 记录满 N 次有效「NC→内圈」延迟后主动结束（0=不限制，仅靠空闲超时）
STOP_AFTER_N_EVENTS = 1

# 结果 JSON/CSV（相对脚本目录）
SAVE_RESULTS = True
RESULTS_JSON = "fault_step_latency_results.json"
RESULTS_CSV = "fault_step_latency_runs.csv"

# =============================================================================
# 复用 receive_udp_hil
# =============================================================================

import receive_udp_hil as ruh  # noqa: E402


def load_class_names(model_path: str) -> list[str] | None:
    try:
        import torch
    except ImportError:
        return None
    path = ruh.resolve_model_path(model_path)
    if not path.is_file():
        return None
    try:
        try:
            obj = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(str(path), map_location="cpu")
    except Exception:
        return None
    if isinstance(obj, dict):
        names = obj.get("class_names")
        if names is not None:
            return [str(x) for x in names]
    return None


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


def fault_step_inference_worker(
    pkt_queue: Queue,
    stop_event: Event,
    results_queue: Queue,
    num_classes: int,
    model_path: str,
    device_str: str,
    normalize: bool,
    vote_window: int,
    label_normal: int,
    label_fault: int,
    stop_after_n: int,
) -> None:
    """
    检测 GT 从 label_normal → label_fault 的跃迁：
    - t_fault：该跃迁**首包**的 t_recv（perf_counter，与推理结束时刻同域）
    - t_diag：首次「最终确诊 == label_fault」且当前 GT 仍为 label_fault 时的 t_done
    """
    ruh.DEVICE_STR = device_str
    ruh.NORMALIZE_PER_WINDOW = normalize

    try:
        model = ruh.load_diagnostic_model(model_path, num_classes)
    except Exception as e:
        results_queue.put({"ok": False, "error": f"模型加载失败: {e}"})
        return

    if model is None:
        results_queue.put({"ok": False, "error": "model is None"})
        return

    inferred_nc = ruh.num_classes_from_model(model)
    if inferred_nc is not None:
        num_classes = inferred_nc

    label_normal = max(0, min(num_classes - 1, label_normal))
    label_fault = max(0, min(num_classes - 1, label_fault))

    try:
        import torch

        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
    except ImportError:
        pass

    class_names = load_class_names(model_path)

    sample_buf: deque[float] = deque()
    vote_q: deque[int] = deque(maxlen=vote_window)
    last_final: int | None = None
    last_gt_consumer: int | None = None

    # wait_normal: 等待曾见到正常；armed: 已见过正常，等待 NC→fault 沿
    phase = "wait_normal"
    t_fault_recv: float | None = None
    events: list[dict] = []

    print(
        f"[故障阶跃] 推理就绪 | 正常类索引={label_normal}"
        f"{f' ({class_names[label_normal]})' if class_names and label_normal < len(class_names) else ''}"
        f" | 内圈类索引={label_fault}"
        f"{f' ({class_names[label_fault]})' if class_names and label_fault < len(class_names) else ''}"
        f" | 投票窗={vote_window}",
        flush=True,
    )

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
                gt_clamped = max(0, min(num_classes - 1, int(gt_label)))
                prev_gt = last_gt_consumer

                if last_gt_consumer is not None and gt_clamped != last_gt_consumer:
                    sample_buf.clear()
                    vote_q.clear()
                    last_final = None
                    print(
                        f"[故障阶跃] GT {last_gt_consumer} -> {gt_clamped}，已清空滑窗与投票。",
                        flush=True,
                    )
                last_gt_consumer = gt_clamped

                # 阶跃检测：须在「已见过正常」之后，识别 normal -> fault
                if phase == "wait_normal" and gt_clamped == label_normal:
                    phase = "armed"
                    print("[故障阶跃] 已观察到正常类，等待阶跃到内圈故障…", flush=True)

                if (
                    phase == "armed"
                    and prev_gt == label_normal
                    and gt_clamped == label_fault
                ):
                    t_fault_recv = t_recv
                    phase = "wait_diagnosis"
                    print(
                        f"[故障阶跃] ★ 检测到 NC→内圈（GT {label_normal}→{label_fault}），"
                        f"t_fault 取本包 t_recv（perf_counter 域）。",
                        flush=True,
                    )

                sample_buf.extend(signal)

                stop_worker = False
                while len(sample_buf) >= POINTS_PER_INFERENCE:
                    chunk = [sample_buf.popleft() for _ in range(POINTS_PER_INFERENCE)]

                    pred = ruh.run_single_inference(model, chunk, device_str)
                    if pred < 0 or pred >= num_classes:
                        pred = max(0, min(num_classes - 1, pred))

                    t_done = time.perf_counter()

                    vote_q.append(pred)

                    if len(vote_q) < vote_window:
                        continue

                    final_id, tied = majority_vote(list(vote_q), last_final)
                    if tied:
                        print(
                            f"[故障阶跃] 【平局】沿用规则 -> 最终={final_id}",
                            file=sys.stderr,
                        )
                    last_final = final_id

                    now_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"[{now_wall}] GT={gt_clamped} | 单次={pred} | 窗口={list(vote_q)} | "
                        f"【最终确诊】={final_id}",
                        flush=True,
                    )

                    if (
                        phase == "wait_diagnosis"
                        and t_fault_recv is not None
                        and gt_clamped == label_fault
                        and final_id == label_fault
                    ):
                        latency_s = t_done - t_fault_recv
                        latency_ms = latency_s * 1000.0
                        rec = {
                            "wall_time": now_wall,
                            "t_fault_recv_perf": t_fault_recv,
                            "t_diagnosis_perf": t_done,
                            "latency_ms": latency_ms,
                            "vote_window": list(vote_q),
                            "final_diagnosis": final_id,
                            "gt": gt_clamped,
                        }
                        events.append(rec)
                        print(
                            "\n" + "=" * 60
                            + f"\n[故障阶跃] ★ 确诊延迟: {latency_ms:.3f} ms\n"
                            + f"  （自首包新 GT 的 t_recv 至本次最终投票正确的 t_done）\n"
                            + "=" * 60
                            + "\n",
                            flush=True,
                        )
                        phase = "wait_normal"
                        t_fault_recv = None
                        vote_q.clear()
                        last_final = None

                        if stop_after_n > 0 and len(events) >= stop_after_n:
                            print(
                                f"[故障阶跃] 已记录 {len(events)} 次，停止推理循环。",
                                flush=True,
                            )
                            stop_event.set()
                            stop_worker = True
                            break

                if stop_worker:
                    break

            except Exception as e:
                print(f"[故障阶跃] 处理包异常（跳过）: {e}", file=sys.stderr)

    except Exception as e:
        results_queue.put({"ok": False, "error": str(e)})
        return

    results_queue.put(
        {
            "ok": True,
            "events": events,
            "class_names": class_names,
            "label_normal": label_normal,
            "label_fault": label_fault,
        }
    )


def save_results(
    script_dir: Path,
    payload: dict,
) -> None:
    if not SAVE_RESULTS or not payload.get("ok"):
        return
    events = payload.get("events") or []
    json_path = script_dir / RESULTS_JSON
    csv_path = script_dir / RESULTS_CSV
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "saved_at": datetime.now().isoformat(),
                    "label_normal": payload.get("label_normal"),
                    "label_fault": payload.get("label_fault"),
                    "class_names": payload.get("class_names"),
                    "runs": events,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[故障阶跃] 已写入 {json_path.resolve()}", flush=True)
    except OSError as e:
        print(f"[故障阶跃] 写 JSON 失败: {e}", file=sys.stderr)

    try:
        new_file = not csv_path.is_file()
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(
                    [
                        "wall_time",
                        "latency_ms",
                        "gt",
                        "final_diagnosis",
                        "vote_window",
                    ]
                )
            for ev in events:
                w.writerow(
                    [
                        ev.get("wall_time", ""),
                        f'{ev.get("latency_ms", 0):.6f}',
                        ev.get("gt", ""),
                        ev.get("final_diagnosis", ""),
                        json.dumps(ev.get("vote_window", []), ensure_ascii=False),
                    ]
                )
        print(f"[故障阶跃] 已追加 {csv_path.resolve()}", flush=True)
    except OSError as e:
        print(f"[故障阶跃] 写 CSV 失败: {e}", file=sys.stderr)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    pkt_size = DOUBLES_PER_PACKET * 8

    pkt_queue: Queue = Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = Event()
    results_queue: Queue = Queue()

    recv_proc = Process(
        target=ruh.receiver_main,
        args=(
            pkt_queue,
            stop_event,
            BIND_IP,
            UDP_PORT,
            pkt_size,
            UDP_IDLE_TIMEOUT_SEC,
        ),
        name="UDPReceiver",
        daemon=False,
    )
    inf_proc = Process(
        target=fault_step_inference_worker,
        args=(
            pkt_queue,
            stop_event,
            results_queue,
            NUM_CLASSES,
            MODEL_PATH,
            DEVICE_STR,
            NORMALIZE_PER_WINDOW,
            VOTE_WINDOW,
            GT_LABEL_NORMAL,
            GT_LABEL_INNER_FAULT,
            STOP_AFTER_N_EVENTS,
        ),
        name="FaultStepInference",
        daemon=False,
    )

    print(
        f"[主进程] {datetime.now():%Y-%m-%d %H:%M:%S} 场景二：故障阶跃延迟 | "
        f"UDP {BIND_IP}:{UDP_PORT} | 空闲≥{UDP_IDLE_TIMEOUT_SEC:.1f}s 无包则退出 | "
        f"记录 {STOP_AFTER_N_EVENTS or '∞'} 次后停止"
    )

    try:
        recv_proc.start()
        inf_proc.start()

        while recv_proc.is_alive() or inf_proc.is_alive():
            recv_proc.join(timeout=0.5)
            inf_proc.join(timeout=0.5)

    except KeyboardInterrupt:
        print("\n[主进程] KeyboardInterrupt，正在停止…", file=sys.stderr)
        stop_event.set()
        recv_proc.join(timeout=3.0)
        inf_proc.join(timeout=10.0)
        if recv_proc.is_alive():
            recv_proc.terminate()
        if inf_proc.is_alive():
            inf_proc.terminate()
        recv_proc.join(timeout=2.0)
        inf_proc.join(timeout=2.0)

    payload = None
    for _ in range(40):
        try:
            payload = results_queue.get_nowait()
            break
        except queue.Empty:
            time.sleep(0.05)

    if payload is None:
        payload = {"ok": False, "error": "未收到推理进程结果"}

    if payload.get("ok"):
        events = payload.get("events") or []
        if events:
            latencies = [float(e["latency_ms"]) for e in events]
            print(
                f"\n[汇总] 共 {len(events)} 次阶跃测量 | "
                f"平均延迟 {sum(latencies)/len(latencies):.3f} ms | "
                f"最小 {min(latencies):.3f} ms | 最大 {max(latencies):.3f} ms\n"
            )
        else:
            print(
                "\n[汇总] 未记录到完整事件（需先稳定输出正常类，再阶跃到内圈；"
                "或检查 GT 索引与 Simulink fault_sel 是否一致）。\n"
            )
        save_results(script_dir, payload)
    else:
        print(f"\n[汇总] 失败: {payload.get('error', payload)}\n", file=sys.stderr)

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
