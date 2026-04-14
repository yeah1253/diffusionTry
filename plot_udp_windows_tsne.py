#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对保存的二维样本矩阵做 t-SNE 降维并绘图，散点颜色按每窗的 **GT（真实类别）** 区分。

支持的 npz（与 receive_udp_hil 写入格式一致，或由其它脚本生成）:
  - 必需: X (N, L), y (N,)  — y 为每窗 GT 类别索引
  - 可选: pred (N,) — 不参与着色，仅兼容加载

依赖:
  pip install numpy matplotlib scikit-learn

用法:
  python plot_udp_windows_tsne.py
  python plot_udp_windows_tsne.py --npz my_generated_windows.npz --out tsne_gt.png
  python plot_udp_windows_tsne.py --names IF0.2,IF0.4,IF0.6,NC,OF0.2,OF0.4,OF0.6,RF0.2,RF0.4,RF0.6
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


def _load_xy(data) -> tuple:
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


def _legend_label(cid: int, names: list[str] | None) -> str:
    if names is not None and 0 <= int(cid) < len(names):
        return f"{int(cid)}:{names[int(cid)]}"
    return str(int(cid))


def main() -> int:
    p = argparse.ArgumentParser(
        description="窗数据 t-SNE 可视化（颜色 = 每窗 GT）",
    )
    p.add_argument(
        "--npz",
        type=str,
        default=str(SCRIPT_DIR / "hil_udp_inference_windows.npz"),
        help="含 X 与 y(GT) 的 npz 路径",
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
        help="图例用类别名，逗号分隔，顺序与类别索引 0,1,… 对齐；留空则用内置十类名（仅前 10 类）",
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
    p.add_argument("--max-samples", type=int, default=8000, help="最多随机子采样条数（0=不限制）")
    args = p.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.is_file():
        print(f"[错误] 找不到文件: {npz_path.resolve()}", file=sys.stderr)
        return 1

    try:
        import numpy as np
    except ImportError:
        print("[错误] 需要 numpy", file=sys.stderr)
        return 1

    data = np.load(str(npz_path), allow_pickle=True)
    try:
        X, y_gt = _load_xy(data)
    except KeyError as e:
        print(f"[错误] {e}", file=sys.stderr)
        data.close()
        return 1
    data.close()

    if X.ndim != 2:
        print(f"[错误] X 应为二维，当前 shape={X.shape}", file=sys.stderr)
        return 1
    n, d = X.shape
    if y_gt.shape[0] != n:
        print("[错误] GT 向量长度与 X 行数不一致", file=sys.stderr)
        return 1

    labels = y_gt.astype(np.int64, copy=False)
    class_names: list[str] | None = None
    if args.names.strip():
        class_names = [s.strip() for s in args.names.split(",") if s.strip()]
    else:
        class_names = list(DEFAULT_CLASS_NAMES)

    rng = np.random.default_rng(args.seed)
    if args.max_samples and n > args.max_samples:
        idx = rng.choice(n, size=args.max_samples, replace=False)
        X = X[idx]
        labels = labels[idx]
        n = X.shape[0]
        print(f"[信息] 子采样至 n={n} 条")

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
    print(f"[信息] t-SNE 完成, Z.shape={Z.shape}, perplexity={perplexity:.1f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[错误] 需要 matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    uniq = np.unique(labels)
    n_u = int(len(uniq))
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap_use = plt.get_cmap("tab20" if n_u > 10 else "tab10")
    for j, c in enumerate(uniq):
        m = labels == c
        cid = int(c)
        t = j / max(n_u - 1, 1) if n_u > 1 else 0.5
        rgba = cmap_use(t)
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            s=10,
            alpha=0.78,
            color=rgba,
            label=_legend_label(cid, class_names),
        )
    ax.set_title(f"t-SNE（n={n}，特征维={d}）| 颜色 = 每窗 Ground Truth (GT)")
    ax.set_xlabel("t-SNE 维度 1")
    ax.set_ylabel("t-SNE 维度 2")
    ax.legend(title="GT 类别", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = Path(args.out)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成] 已写入 {out_path.resolve()}（按 GT 着色）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
