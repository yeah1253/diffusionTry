"""
train_diagnosis_simple.py — 下游故障诊断（简单分类器版）

使用手工特征 + Random Forest / LogisticRegression，无需复杂 CNN。
适用于验证生成数据质量的 TRTR / TSTR 对比实验。

依赖：numpy, scikit-learn
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple

import numpy as np

from .dataset import BearingSignalDataset


# ---------------------------------------------------------------------------
# 特征提取（与 visualize_tsne 的 extract_raw_features 一致）
# ---------------------------------------------------------------------------

def extract_raw_features(signals: np.ndarray) -> np.ndarray:
    """
    从原始信号提取 23 维手工特征：时域 7 维 + FFT 前 16 分量。

    参数
    ----
    signals : np.ndarray, shape (N, 1, L) 或 (N, L)

    返回
    ----
    features : np.ndarray, shape (N, 23)
    """
    if signals.ndim == 3:
        signals = signals.squeeze(1)

    feats_list = []
    for sig in signals:
        mean = np.mean(sig)
        std = np.std(sig)
        rms = np.sqrt(np.mean(sig ** 2))
        peak = np.max(np.abs(sig))
        pp = np.max(sig) - np.min(sig)
        skew = float(np.mean(((sig - mean) / (std + 1e-8)) ** 3))
        kurt = float(np.mean(((sig - mean) / (std + 1e-8)) ** 4))
        fft_amp = np.abs(np.fft.rfft(sig))[:16]
        feat = np.concatenate([[mean, std, rms, peak, pp, skew, kurt], fft_amp])
        feats_list.append(feat)
    return np.array(feats_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# TRTR / TSTR（简单分类器版）
# ---------------------------------------------------------------------------

def run_trtr_tstr_simple(
    real_train: BearingSignalDataset,
    gen_train: BearingSignalDataset,
    real_test: BearingSignalDataset,
    num_classes: int,
    class_names: Optional[list] = None,
    classifier: str = "rf",
    max_per_class_warn: int = 10000,
) -> Dict[str, float]:
    """
    运行 TRTR / TSTR 对比实验（手工特征 + sklearn 分类器）。

    - 实验 A (TRTR)：真实数据训练 → 真实数据测试
    - 实验 B (TSTR)：生成数据训练 → 真实数据测试

    参数
    ----
    real_train, gen_train, real_test : BearingSignalDataset
    num_classes : int
    class_names : list | None
        用于打印 classification_report。
    classifier : str
        "rf" (RandomForest) 或 "lr" (LogisticRegression)。
    max_per_class_warn : int
        若某类样本数超过此值会打印提示（可选）。

    返回
    ----
    {"TRTR_acc": float, "TSTR_acc": float}
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("=" * 60)
    print("  模块二：下游故障诊断 — TRTR / TSTR（简单分类器）")
    print("=" * 60)

    # 提取特征
    X_real_train = extract_raw_features(real_train.signals.numpy())
    y_real_train = real_train.labels.numpy()
    X_gen_train = extract_raw_features(gen_train.signals.numpy())
    y_gen_train = gen_train.labels.numpy()
    X_real_test = extract_raw_features(real_test.signals.numpy())
    y_real_test = real_test.labels.numpy()

    print(f"  特征维度: {X_real_train.shape[1]}")
    print(f"  真实训练集: {len(real_train)} 条")
    print(f"  生成训练集: {len(gen_train)} 条")
    print(f"  真实测试集: {len(real_test)} 条")
    print(f"  分类器: {classifier}")

    target_names = class_names if class_names else [str(i) for i in range(num_classes)]

    def _train_and_eval(X_tr, y_tr, X_te, y_te, tag: str) -> Tuple[object, float]:
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        if classifier == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(X_tr_scaled, y_tr)
        acc = clf.score(X_te_scaled, y_te)
        return clf, acc

    # ---------- TRTR ----------
    print(f"\n{'─'*50}")
    print("  实验 A (TRTR): 真实数据训练 → 真实数据测试")
    print(f"{'─'*50}")
    _, acc_a = _train_and_eval(X_real_train, y_real_train, X_real_test, y_real_test, "TRTR")
    print(f"  TRTR 准确率: {acc_a:.4f} ({acc_a*100:.2f}%)")

    # ---------- TSTR ----------
    print(f"\n{'─'*50}")
    print("  实验 B (TSTR): 生成数据训练 → 真实数据测试")
    print(f"{'─'*50}")
    _, acc_b = _train_and_eval(X_gen_train, y_gen_train, X_real_test, y_real_test, "TSTR")
    print(f"  TSTR 准确率: {acc_b:.4f} ({acc_b*100:.2f}%)")

    # ---------- 汇总 ----------
    print(f"\n{'═'*60}")
    print("  TRTR / TSTR 对比结果")
    print(f"{'═'*60}")
    print(f"  实验 A (TRTR) 测试准确率: {acc_a:.4f}  ({acc_a*100:.2f}%)")
    print(f"  实验 B (TSTR) 测试准确率: {acc_b:.4f}  ({acc_b*100:.2f}%)")
    diff = acc_a - acc_b
    print(f"  差距 (TRTR - TSTR): {diff:+.4f}")
    if abs(diff) < 0.05:
        print("  → 生成数据质量优秀，TSTR 接近 TRTR 基准！")
    elif diff > 0:
        print("  → 生成数据质量有提升空间，TSTR 略低于 TRTR。")
    else:
        print("  → TSTR 超过 TRTR，可能存在数据泄露或过拟合，需注意。")
    print()

    try:
        from sklearn.metrics import classification_report
        scaler_a = StandardScaler()
        X_tr_a = scaler_a.fit_transform(X_real_train)
        X_te_a = scaler_a.transform(X_real_test)
        clf_a = RandomForestClassifier(n_estimators=100, random_state=42) if classifier == "rf" else LogisticRegression(max_iter=1000, random_state=42)
        clf_a.fit(X_tr_a, y_real_train)
        pred_a = clf_a.predict(X_te_a)

        scaler_b = StandardScaler()
        X_tr_b = scaler_b.fit_transform(X_gen_train)
        X_te_b = scaler_b.transform(X_real_test)
        clf_b = RandomForestClassifier(n_estimators=100, random_state=42) if classifier == "rf" else LogisticRegression(max_iter=1000, random_state=42)
        clf_b.fit(X_tr_b, y_gen_train)
        pred_b = clf_b.predict(X_te_b)

        print("  [TRTR] 详细分类报告:")
        print(classification_report(y_real_test, pred_a, target_names=target_names, digits=4))
        print("  [TSTR] 详细分类报告:")
        print(classification_report(y_real_test, pred_b, target_names=target_names, digits=4))
    except ImportError:
        pass

    return {"TRTR_acc": acc_a, "TSTR_acc": acc_b}
