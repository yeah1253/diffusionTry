#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对特征矩阵做 t-SNE 降维并绘图，散点颜色按 **GT** 区分。

单文件模式（--npz）:
  必需: X (N, D), y (N,)

离线与在线同图:
  --offline-npz  由 train_bearing_cnn1d 导出的双路头特征（X_mixed / X_real_only + 共用 y）
  --offline-source mixed | real_only  选用哪一套离线特征（默认 mixed = 混合训练权重）
  --npz          在线 UDP 等保存的分类器头特征（与 receive_udp_hil 一致）

  同一 t-SNE 空间：纵向拼接 [离线特征; 在线特征] 后一次 fit_transform。
  标记：离线 × (cross)、在线 ● (circle)。

依赖:
  pip install numpy matplotlib scikit-learn

用法:
  python plot_udp_windows_tsne.py
  python plot_udp_windows_tsne.py --npz hil_udp_classifier_head.npz --out tsne.png
  python plot_udp_windows_tsne.py \\
      --offline-npz offline_classifier_heads_cnn.npz --offline-source mixed \\
      --npz hil_udp_classifier_head.npz --out tsne_on_off.png
  python plot_udp_windows_tsne.py --offline-npz offline_classifier_heads_cnn.npz --offline-source real_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CLASS_NAMES = (
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

# -----------------------------------------------------------------------------
# 运行配置（不想写命令行时，直接改这里）
# -----------------------------------------------------------------------------
USE_CLI_ARGS = False
RUN_CFG = {
    # 在线（或其它单源）特征 npz：含 X, y
    "npz": str(SCRIPT_DIR / "hil_udp_classifier_head.npz"),
    # 离线双模型头特征 npz（含 X_mixed、X_real_only、y）；留空表示不使用离线
    "offline_npz": str(SCRIPT_DIR / "offline_classifier_heads_cnn.npz"),
    # "mixed"=混合训练权重特征（叉号），"real_only"=纯真实权重特征（叉号）
    "offline_source": "mixed",
    "out": str(SCRIPT_DIR / "tsne_mixed_vs_udp.png"),
    "names": "",
    "perplexity": 30.0,
    "seed": 42,
    "pca": 50,
    "max_samples": 8000,
}


def _load_xy_from_npz(data) -> tuple:
    import numpy as np

    if "X" not in data.files:
        raise KeyError("npz 中缺少数组 'X'")
    X = np.asarray(data["X"], dtype=np.float64)
    if "y" in data.files:
        y_gt = np.asarray(data["y"], dtype=np.int64).ravel()
    elif "labels" in data.files:
        y_gt = np.asarray(data["labels"], dtype=np.int64).ravel()
    elif "gt" in data.files:
        y_gt = np.asarray(data["gt"], dtype=np.int64).ravel()
    else:
        raise KeyError("npz 中缺少 GT 数组，请提供 'y'、'labels' 或 'gt' 之一")
    return X, y_gt


def _load_offline_branch(data, source: str) -> tuple:
    """从 train_bearing_cnn1d 导出的 npz 读取 X_mixed 或 X_real_only 与 y。"""
    import numpy as np

    key = "X_mixed" if source.strip().lower() == "mixed" else "X_real_only"
    if key not in data.files:
        raise KeyError(f"npz 中缺少 '{key}'，请确认为 train_bearing_cnn1d 导出的离线头特征文件")
    X = np.asarray(data[key], dtype=np.float64)
    if "y" not in data.files:
        raise KeyError("离线 npz 缺少 'y'")
    y_gt = np.asarray(data["y"], dtype=np.int64).ravel()
    return X, y_gt


def _legend_label(cid: int, names: list[str] | None) -> str:
    if names is not None and 0 <= int(cid) < len(names):
        return f"{int(cid)}:{names[int(cid)]}"
    return str(int(cid))


def _args_from_cfg() -> argparse.Namespace:
    """将脚本内 RUN_CFG 映射为与 argparse 一致的命名空间。"""
    return argparse.Namespace(
        npz=str(RUN_CFG.get("npz", str(SCRIPT_DIR / "hil_udp_classifier_head.npz"))),
        offline_npz=str(RUN_CFG.get("offline_npz", "")),
        offline_source=str(RUN_CFG.get("offline_source", "mixed")),
        out=str(RUN_CFG.get("out", str(SCRIPT_DIR / "hil_udp_tsne_gt.png"))),
        names=str(RUN_CFG.get("names", "")),
        perplexity=float(RUN_CFG.get("perplexity", 30.0)),
        seed=int(RUN_CFG.get("seed", 42)),
        pca=int(RUN_CFG.get("pca", 50)),
        max_samples=int(RUN_CFG.get("max_samples", 8000)),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="特征 t-SNE：按 GT 着色；可选离线与在线同图（× / ●）",
    )
    p.add_argument(
        "--npz",
        type=str,
        default=str(SCRIPT_DIR / "hil_udp_classifier_head.npz"),
        help="在线（或其它单源）特征 npz：含 X, y",
    )
    p.add_argument(
        "--offline-npz",
        type=str,
        default="",
        help="离线双模型头特征 npz（含 X_mixed、X_real_only、y）；与 --npz 可二选一或同用",
    )
    p.add_argument(
        "--offline-source",
        type=str,
        choices=("mixed", "real_only"),
        default="mixed",
        help="同图时选用离线矩阵：mixed=混合训练权重，real_only=纯真实训练权重",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(SCRIPT_DIR / "hil_udp_tsne_gt.png"),
        help="输出 PNG 路径",
    )
    p.add_argument(
        "--names",
        type=str,
        default="",
        help="图例用类别名，逗号分隔；留空则用内置十类名",
    )
    p.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument(
        "--pca",
        type=int,
        default=50,
        metavar="D",
        help="若 D>0 且特征维>D，则先做 PCA 再 t-SNE；0 表示不做 PCA",
    )
    p.add_argument("--max-samples", type=int, default=8000, help="合并后总样本上限（0=不限制）")
    args = p.parse_args() if USE_CLI_ARGS else _args_from_cfg()

    try:
        import numpy as np
    except ImportError:
        print("[错误] 需要 numpy", file=sys.stderr)
        return 1

    path_on = Path(args.npz)
    path_off = Path(args.offline_npz.strip()) if args.offline_npz.strip() else None

    X_on = y_on = None
    X_off = y_off = None

    if path_off is not None and path_off.is_file():
        d0 = np.load(str(path_off), allow_pickle=True)
        try:
            X_off, y_off = _load_offline_branch(d0, args.offline_source)
        except KeyError as e:
            print(f"[错误] {e}", file=sys.stderr)
            d0.close()
            return 1
        d0.close()
        if X_off.ndim != 2 or y_off.shape[0] != X_off.shape[0]:
            print("[错误] 离线 X / y 形状异常", file=sys.stderr)
            return 1

    if path_on.is_file():
        d1 = np.load(str(path_on), allow_pickle=True)
        try:
            X_on, y_on = _load_xy_from_npz(d1)
        except KeyError as e:
            print(f"[错误] {e}", file=sys.stderr)
            d1.close()
            return 1
        d1.close()
        if X_on.ndim != 2 or y_on.shape[0] != X_on.shape[0]:
            print("[错误] 在线 X / y 形状异常", file=sys.stderr)
            return 1

    if X_off is None and X_on is None:
        print(
            f"[错误] 未找到可用 npz：在线 {path_on.resolve()} 与离线 "
            f"{(path_off.resolve() if path_off else '（未指定）')} 均不可用。",
            file=sys.stderr,
        )
        return 1

    mode_dual = X_off is not None and X_on is not None
    if mode_dual and X_off.shape[1] != X_on.shape[1]:
        print(
            f"[错误] 离线与在线特征维不一致 D_off={X_off.shape[1]} D_on={X_on.shape[1]}，"
            f"需同架构权重与同一前向定义。",
            file=sys.stderr,
        )
        return 1

    if X_off is not None and X_on is None:
        X = X_off
        labels = y_off.astype(np.int64, copy=False)
        is_online = np.zeros(X.shape[0], dtype=bool)
        source_note = f"离线 only | branch={args.offline_source}"
    elif X_off is None and X_on is not None:
        X = X_on
        labels = y_on.astype(np.int64, copy=False)
        is_online = np.ones(X.shape[0], dtype=bool)
        source_note = "在线 only"
    else:
        n0, n1 = X_off.shape[0], X_on.shape[0]
        X = np.vstack([X_off.astype(np.float64), X_on.astype(np.float64)])
        labels = np.concatenate(
            [y_off.astype(np.int64, copy=False), y_on.astype(np.int64, copy=False)]
        )
        is_online = np.concatenate(
            [np.zeros(n0, dtype=bool), np.ones(n1, dtype=bool)]
        )
        source_note = (
            f"离线({args.offline_source}) n={n0} + 在线 n={n1} | ×离线 ●在线"
        )

    n, d = X.shape
    class_names: list[str] | None
    if args.names.strip():
        class_names = [s.strip() for s in args.names.split(",") if s.strip()]
    else:
        class_names = list(DEFAULT_CLASS_NAMES)

    rng = np.random.default_rng(args.seed)
    if args.max_samples and n > args.max_samples:
        idx = rng.choice(n, size=args.max_samples, replace=False)
        X = X[idx]
        labels = labels[idx]
        is_online = is_online[idx]
        n = X.shape[0]
        print(f"[信息] 子采样后总 n={n}")

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError:
        print("[错误] 需要 scikit-learn: pip install scikit-learn", file=sys.stderr)
        return 1

    pca_dim = min(args.pca, max(2, n - 1), d) if args.pca > 0 else 0
    if args.pca > 0 and d > pca_dim and pca_dim >= 2:
        X_use = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(X)
        print(f"[信息] PCA: {d} -> {pca_dim}")
    else:
        X_use = X

    perplexity = float(min(args.perplexity, max(5.0, (n - 1) / 3.0)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200.0,
        init="pca",
        random_state=args.seed,
        verbose=1,
    )
    Z = tsne.fit_transform(X_use)
    print(f"[信息] t-SNE 完成 Z.shape={Z.shape}, perplexity={perplexity:.1f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[错误] 需要 matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    uniq = np.unique(labels)
    n_u = int(len(uniq))
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap_use = plt.get_cmap("tab20" if n_u > 10 else "tab10")

    for j, c in enumerate(uniq):
        cid = int(c)
        t = j / max(n_u - 1, 1) if n_u > 1 else 0.5
        rgba = cmap_use(t)
        base = _legend_label(cid, class_names)

        m_off = (labels == c) & (~is_online)
        m_on = (labels == c) & is_online

        if np.any(m_off):
            ax.scatter(
                Z[m_off, 0],
                Z[m_off, 1],
                s=36,
                marker="x",
                linewidths=1.0,
                color=rgba,
                alpha=0.85,
                label=f"{base} · 离线",
            )
        if np.any(m_on):
            ax.scatter(
                Z[m_on, 0],
                Z[m_on, 1],
                s=14,
                marker="o",
                color=rgba,
                alpha=0.78,
                edgecolors="none",
                label=f"{base} · 在线",
            )

    ax.set_title(
        f"t-SNE（n={n}，D={d}）| 颜色=GT | {source_note}",
        fontsize=11,
    )
    ax.set_xlabel("t-SNE 维度 1")
    ax.set_ylabel("t-SNE 维度 2")
    ax.legend(
        title="类别 · 数据源",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
    )
    fig.tight_layout()
    out_path = Path(args.out)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成] 已写入 {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
