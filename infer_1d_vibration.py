import os
import glob
import numpy as np
import torch
from scipy.io import loadmat

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import (
    PhysiNet,
    GaussianDiffusion1D,
    ucfilter_kmeans_select_indices,
)


def find_latest_checkpoint(results_folder: str) -> str:
    """
    在给定目录中查找最新的 model-*.pt 权重文件。
    """
    pattern = os.path.join(results_folder, "model-*.pt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No checkpoint files found in {results_folder}")

    # filename: model-{milestone}.pt
    def get_milestone(p: str) -> int:
        base = os.path.basename(p)
        name, _ = os.path.splitext(base)
        parts = name.split("-")
        try:
            return int(parts[-1])
        except Exception:
            return -1

    paths = sorted(paths, key=get_milestone)
    return paths[-1]


def build_model_and_diffusion(seq_length: int, channels: int) -> GaussianDiffusion1D:
    """
    构建与训练阶段一致的 PhysiNet + GaussianDiffusion1D 结构。
    需要与 train_1d_vibration.py 中的配置保持完全一致。
    """
    model = PhysiNet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=channels,
        cond_dim=0,  # 目前仍然是无条件生成
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=seq_length,
        timesteps=1000,
        objective="pred_v",
        auto_normalize=False,
    )

    return diffusion


def load_trained_diffusion_from_checkpoint(
    ckpt_path: str,
    diffusion: GaussianDiffusion1D,
) -> GaussianDiffusion1D:
    """
    从训练保存的 model-*.pt 中加载 GaussianDiffusion1D 权重。
    注意：这里直接使用训练好的扩散模型进行采样，
    与 train_1d_vibration.py 结束时调用 diffusion.sample 的行为保持一致。
    """
    device = next(diffusion.parameters()).device

    data = torch.load(ckpt_path, map_location=device, weights_only=True)

    diffusion.load_state_dict(data["model"])
    diffusion.eval()
    return diffusion


def compute_signal_min_max(mat_file_path: str, cutoff_points: int = 5000) -> tuple[float, float]:
    """
    复用训练脚本的数据预处理逻辑，计算原始振动信号的全局 min / max，
    以便对生成样本做反归一化，使频谱分析与训练阶段保持一致。
    """
    mat_data = loadmat(mat_file_path)
    bearing_acc_x = mat_data["Bearing_Acc_X"]

    if bearing_acc_x.shape[0] == 1:
        signal = bearing_acc_x[0]
    else:
        signal = bearing_acc_x.flatten()

    if len(signal) > cutoff_points:
        signal = signal[cutoff_points:]
    else:
        raise ValueError(
            f"Signal length ({len(signal)}) is shorter than cutoff points ({cutoff_points})!"
        )

    signal_min = float(signal.min())
    signal_max = float(signal.max())
    return signal_min, signal_max


def main():
    # 与训练脚本保持一致的超参数和数据路径
    SEQ_LENGTH = 1024
    CHANNELS = 1
    RESULTS_FOLDER = "./results_vibration"
    MAT_FILE_PATH = r"D:\haoran\数据集\simple_bearing\simple_bearing\ball\9005k.mat"
    CUTOFF_POINTS = 5000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {device}")

    # 1) 找到最新的 checkpoint
    ckpt_path = find_latest_checkpoint(RESULTS_FOLDER)
    print(f"Loading checkpoint: {ckpt_path}")

    # 2) 构建模型与扩散对象
    diffusion = build_model_and_diffusion(SEQ_LENGTH, CHANNELS)
    diffusion.to(device)

    # 3) 加载训练好的扩散模型权重（不重新训练）
    diffusion = load_trained_diffusion_from_checkpoint(ckpt_path, diffusion)

    # 4) 计算原始信号的 min/max，用于反归一化
    signal_min, signal_max = compute_signal_min_max(MAT_FILE_PATH, cutoff_points=CUTOFF_POINTS)
    print(f"Loaded signal stats: min={signal_min:.4f}, max={signal_max:.4f}")

    # 5) 使用训练好的扩散模型进行采样生成信号
    num_samples = 64
    print(f"Sampling {num_samples} sequences from trained diffusion model...")

    with torch.no_grad():
        sampled = diffusion.sample(batch_size=num_samples)  # (N, C, L)

    # 保存原始生成样本（反归一化）到 generated_samples_infer_raw
    sampled_np = sampled.squeeze(1).cpu().numpy()  # (N, L)
    # denormalize from [-1, 1] back to original physical range
    sampled_denorm = (sampled_np + 1.0) * 0.5 * (signal_max - signal_min) + signal_min

    raw_folder = './generated_samples_infer_raw'
    os.makedirs(raw_folder, exist_ok=True)
    np.save(os.path.join(raw_folder, 'all_generated.npy'), sampled_denorm)
    for i, sig in enumerate(sampled_denorm):
        np.save(os.path.join(raw_folder, f'raw_infer_signal_{i}.npy'), sig)

    # 6) 使用 UCFilter（K-means + KL 边界）筛选高质量样本
    print("Applying UCFilter (K-means + KL) to sampled sequences...")
    with torch.no_grad():
        selected_idx, kl_scores, cluster_labels = ucfilter_kmeans_select_indices(
            sampled.detach().cpu(),
            num_clusters=3,
            k_ratio=0.9,
            sigma=1.0,
            embed_dim=2,
        )

    selected_idx_np = selected_idx.numpy()
    print(f"Total sampled: {sampled.shape[0]}, selected by UCFilter: {len(selected_idx_np)}")

    # 7) 反归一化到原始物理量级（与训练脚本一致）
    sampled_denorm = sampled_denorm
    sampled_filtered = sampled_denorm[selected_idx_np]

    # 8) 保存生成且通过 UCFilter 筛选的样本到 ./generated_samples_infer
    save_folder = "./generated_samples_infer"
    os.makedirs(save_folder, exist_ok=True)

    # 保存索引与 KL 分数
    np.save(os.path.join(save_folder, 'selected_idx.npy'), selected_idx_np)
    try:
        kl_np = kl_scores.numpy()
    except Exception:
        kl_np = np.array(kl_scores)
    np.save(os.path.join(save_folder, 'kl_scores.npy'), kl_np)

    for i, (idx, sig) in enumerate(zip(selected_idx_np, sampled_filtered)):
        out_path = os.path.join(save_folder, f'infer_signal_{i}.npy')
        np.save(out_path, sig)
        print(f"Saved UCFilter-selected generated signal #{i} (orig idx={idx}) to {out_path}")

    print("Inference completed.")


if __name__ == "__main__":
    main()

