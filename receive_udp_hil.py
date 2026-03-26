#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Receive UDP packets from Speedgoat / Simulink (Byte Unpack: 128 x double = 1024 bytes per packet).

Simulink 侧参考: UDP Receive Receive width=1024, Byte Unpack dimensions [128], type double.

本脚本将收到的 double 依次追加到缓冲区；当累计达到 1024 个标量时，写入一个文件（默认 .npy），
并打印文件路径。可多次触发保存（每满 1024 点一次）。

默认端口已为 10001；在 Cursor/VS Code 中可用「运行和调试」选配置 UDP receive，或点右上角运行当前文件。

用法示例:
  python receive_udp_hil.py
  python receive_udp_hil.py --bind 0.0.0.0 --out-dir ./captures
"""

from __future__ import annotations

import argparse
import csv
import socket
import struct
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UDP receiver: 128 doubles per datagram, save every 1024 samples.")
    p.add_argument(
        "--bind",
        default="0.0.0.0",
        help="本机绑定地址。Speedgoat 若发到 PC 的某网卡 IP，可用 0.0.0.0 监听所有接口。",
    )
    p.add_argument("--port", type=int, default=10001, help="UDP 端口（与发送端目标端口一致）。")
    p.add_argument(
        "--points",
        type=int,
        default=1024,
        help="累计多少个 double 后写入一次文件。",
    )
    p.add_argument(
        "--doubles-per-packet",
        type=int,
        default=128,
        help="每个 UDP 报文包含的 double 个数（与 Byte Unpack 维度一致）。",
    )
    p.add_argument(
        "--endian",
        choices=("little", "big"),
        default="little",
        help="字节序。x86/Speedgoat 常见为 little（对应 '<128d'）。",
    )
    p.add_argument(
        "--format",
        choices=("npy", "csv", "bin"),
        default="npy",
        help="输出格式: npy(需 numpy)、csv、或原始 float64 二进制 bin。",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="保存目录；省略则使用脚本同目录下的 udp_captures（不依赖当前工作目录）。",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="最多保存多少个文件后退出；0 表示不限制（Ctrl+C 结束）。",
    )
    return p.parse_args()


def unpack_packet(data: bytes, n_double: int, little_endian: bool) -> tuple[float, ...]:
    if len(data) != n_double * 8:
        raise ValueError(f"expected {n_double * 8} bytes, got {len(data)}")
    fmt = "<" if little_endian else ">"
    return struct.unpack(fmt + f"{n_double}d", data)


def save_chunk(chunk: list[float], out_dir: Path, fmt: str, index: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"hil_samples_{ts}_{index:05d}"

    if fmt == "npy":
        import numpy as np

        path = base.with_suffix(".npy")
        np.save(path, np.asarray(chunk, dtype=np.float64))
        return path

    if fmt == "csv":
        path = base.with_suffix(".csv")
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "value"])
            for i, v in enumerate(chunk):
                w.writerow([i, f"{v:.17g}"])
        return path

    # bin: raw float64 little-endian
    path = base.with_suffix(".f64.bin")
    with path.open("wb") as f:
        f.write(struct.pack(f"<{len(chunk)}d", *chunk))
    return path


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    if args.out_dir is None:
        args.out_dir = script_dir / "udp_captures"
    elif not args.out_dir.is_absolute():
        args.out_dir = (Path.cwd() / args.out_dir).resolve()

    n_d = args.doubles_per_packet
    pkt_size = n_d * 8
    little = args.endian == "little"

    if args.format == "npy":
        try:
            import numpy as np  # noqa: F401
        except ImportError:
            print("npy 格式需要安装 numpy: pip install numpy", file=sys.stderr)
            return 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.bind, args.port))
    except OSError as e:
        print(f"绑定失败 {args.bind}:{args.port} — {e}", file=sys.stderr)
        return 1

    print(
        f"监听 udp://{args.bind}:{args.port} ，每包 {n_d} doubles ({pkt_size} bytes)，"
        f"累计 {args.points} 个标量后写入 --format {args.format} -> {args.out_dir.resolve()}"
    )

    buf: list[float] = []
    saved = 0
    file_index = 0

    try:
        while args.max_files == 0 or saved < args.max_files:
            data, addr = sock.recvfrom(65535)
            if len(data) != pkt_size:
                print(f"警告: 来自 {addr} 的包长 {len(data)} != 期望 {pkt_size}，已跳过。", file=sys.stderr)
                continue
            try:
                samples = unpack_packet(data, n_d, little)
            except ValueError as e:
                print(f"解析失败 {addr}: {e}", file=sys.stderr)
                continue

            buf.extend(samples)

            while len(buf) >= args.points:
                chunk = buf[: args.points]
                del buf[: args.points]
                path = save_chunk(chunk, args.out_dir, args.format, file_index)
                file_index += 1
                saved += 1
                print(f"[{time.strftime('%H:%M:%S')}] 已保存 ({len(chunk)} 点) -> {path.resolve()}")
                if args.max_files > 0 and saved >= args.max_files:
                    break

    except KeyboardInterrupt:
        print("\n已停止。", file=sys.stderr)
    finally:
        sock.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
