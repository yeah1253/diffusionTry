#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对照训练数据根目录与 HIL 打包类别顺序（字典序文件夹名 ↔ pack_hil_for_coder fault_list）。

运行: python check_hil_class_alignment.py
依赖 train_bearing_cnn1d.DATA_ROOT 与 collect_class_folders。
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_bearing_cnn1d import DATA_ROOT, collect_class_folders  # noqa: E402

EXPECTED = (
    "IF0.2",
    "IF0.4",
    "IF0.6",
    "NC",
    "OF0.2",
    "OF0.4",
    "OF0.6",
    "RF0.2",
    "RF0.4",
    "RF0.6",
)


def folder_to_mat_var(name: str) -> str:
    return name.replace(".", "_")


def main() -> int:
    root = Path(DATA_ROOT)
    print("标准十类（字典序，与 receive_udp_hil.EXPECTED_CLASS_ORDER_FOR_HIL 一致）:")
    print(" ", list(EXPECTED))
    print()

    if not root.is_dir():
        print(f"[跳过] DATA_ROOT 不存在: {root}")
        print("请修改 train_bearing_cnn1d.DATA_ROOT 后重试。")
        return 1

    pairs = collect_class_folders(root)
    names = [n for n, _ in pairs]
    print(f"DATA_ROOT = {root}")
    print(f"实际一级子目录（排序后）: {names}")
    print()

    if names != list(EXPECTED):
        print(
            "[★ 不一致] 训练用类别顺序与标准十类列表不同。\n"
            "  → checkpoint class_names 将按**你的文件夹**排序；\n"
            "  → HIL 的 pack_hil_for_coder fault_list 必须按**同一顺序**对应 .mat 变量，\n"
            "     不能再用脚本里的默认十类顺序硬套。"
        )
    else:
        print("[OK] 文件夹名字典序与标准十类一致。")

    print("\nMATLAB fault_list（须与 HIL_data.mat 变量名一致，可复制到 pack_hil_for_coder）:")
    print("fault_list = {" + ", ".join(f"'{folder_to_mat_var(n)}'" for n in names) + "};")

    print("\n标签对照（Simulink fault_sel = 索引+1，UDP GT = 索引）:")
    for name, idx in pairs:
        print(f"  {idx:2d}  fault_sel={idx+1:2d}  {name:<8}  .mat 建议 '{folder_to_mat_var(name)}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
