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


def build_model_and_diffusion(seq_length: int, channels: int, cond_dim: int = 2) -> GaussianDiffusion1D:
    """
    构建与训练阶段一致的 PhysiNet + GaussianDiffusion1D 结构。
    需要与 train_1d_vibration.py 中的配置保持完全一致。
    
    参数:
    - seq_length: 序列长度
    - channels: 通道数
    - cond_dim: 条件维度，2 表示 RPM 和 Load，0 表示无条件生成
    """
    model = PhysiNet(
        dim=128,                      # 与训练一致 (64 -> 128)
        dim_mults=(1, 2, 4, 8),
        channels=channels,
        cond_dim=cond_dim,            # 支持条件生成：RPM 和 Load
        dropout=0.1,                  # 与训练一致，eval()模式下自动禁用
        attn_dim_head=64,             # 与训练一致 (32 -> 64)
        attn_heads=8,                 # 与训练一致 (4 -> 8)
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

    data = torch.load(ckpt_path, map_location=device, weights_only=False)

    diffusion.load_state_dict(data["model"])
    diffusion.eval()
    return diffusion


def load_normalization_params_from_checkpoint(ckpt_path: str) -> tuple[float, float]:
    """
    尝试从检查点文件中加载归一化参数。
    如果检查点中没有保存，则返回 None。
    """
    try:
        data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'normalization_params' in data:
            params = data['normalization_params']
            signal_min = params.get('signal_min')
            signal_max = params.get('signal_max')
            if signal_min is not None and signal_max is not None:
                print(f"Loaded normalization params from checkpoint: min={signal_min:.4f}, max={signal_max:.4f}")
                return signal_min, signal_max
    except Exception as e:
        print(f"Warning: Could not load normalization params from checkpoint: {e}")
    return None, None


# def compute_signal_min_max_from_dataset(data_path: str) -> tuple[float, float]:
#     """
#     从训练数据集路径加载数据，计算原始振动信号的全局 min / max，
#     以便对生成样本做反归一化，使频谱分析与训练阶段保持一致。
#
#     参数:
#     - data_path: 训练数据集的路径（包含 .mat 文件的目录）
#     """
#
#     mat_files = glob.glob(os.path.join(data_path, '*.mat'))
#     if len(mat_files) == 0:
#         raise ValueError(f"No .mat files found in {data_path}")
#
#     all_signals = []
#     for mat_file in sorted(mat_files):
#         try:
#             data = loadmat(mat_file)
#             # 提取 y_values 的第一列（与训练代码一致）
#             signal = data['Signal']['y_values'][0, 0]['values'].item()[:, 0]
#             all_signals.append(signal)
#         except Exception as e:
#             print(f"Warning: Failed to load {mat_file}: {e}")
#             continue
#
#     if len(all_signals) == 0:
#         raise ValueError("No valid signals loaded from .mat files")
#
#     # 计算全局 min/max
#     all_signals_array = np.concatenate(all_signals)
#     signal_min = float(all_signals_array.min())
#     signal_max = float(all_signals_array.max())
#
#     return signal_min, signal_max


def main():
    # 与训练脚本保持一致的超参数和数据路径
    SEQ_LENGTH = 1024
    CHANNELS = 1
    COND_DIM = 2  # RPM 和 Load 两个条件
    RESULTS_FOLDER = "./results_vibration"
    # # 训练数据集路径，用于获取归一化参数（如果检查点中没有保存）
    # TRAIN_DATA_PATH = r'D:\speedLoad'
    
    # 条件生成的目标值（可以修改）
    TARGET_RPM = 2000.0
    TARGET_LOAD = 40.0  # 可选值: 0, 20, 40, 60

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {device}")

    # 1) 找到最新的 checkpoint
    ckpt_path = find_latest_checkpoint(RESULTS_FOLDER)
    print(f"Loading checkpoint: {ckpt_path}")

    # 2) 构建模型与扩散对象（支持条件生成）
    diffusion = build_model_and_diffusion(SEQ_LENGTH, CHANNELS, cond_dim=COND_DIM)
    diffusion.to(device)

    # 3) 加载训练好的扩散模型权重（不重新训练）
    diffusion = load_trained_diffusion_from_checkpoint(ckpt_path, diffusion)

    # 4) 获取归一化参数，用于反归一化
    # 首先尝试从检查点加载，如果失败则从训练数据集计算
    signal_min, signal_max = load_normalization_params_from_checkpoint(ckpt_path)
    # if signal_min is None or signal_max is None:
    #     print("Normalization params not found in checkpoint, computing from training dataset...")
    #     signal_min, signal_max = compute_signal_min_max_from_dataset(TRAIN_DATA_PATH)
    print(f"Signal normalization params: min={signal_min:.4f}, max={signal_max:.4f}")

    # 5) 使用训练好的扩散模型进行条件采样生成信号
    num_samples = 64
    print(f"Sampling {num_samples} sequences from trained diffusion model...")
    print(f"Conditional generation: RPM={TARGET_RPM}, Load={TARGET_LOAD}")
    
    # 归一化条件（与训练代码一致）
    target_norm_rpm = (TARGET_RPM - 1000.0) / 2000.0
    target_norm_load = TARGET_LOAD / 60.0
    
    # 构造条件批次张量 (batch, cond_dim)
    cond_batch = torch.tensor(
        [[target_norm_rpm, target_norm_load]] * num_samples,
        dtype=torch.float32,
        device=device,
    )
    print(f"Normalized conditions: RPM={target_norm_rpm:.4f}, Load={target_norm_load:.4f}")

    with torch.no_grad():
        # 条件采样
        sampled = diffusion.sample(
            batch_size=num_samples,
            model_forward_kwargs={"cond": cond_batch}
        )  # (N, C, L)

    # 保存原始生成样本（反归一化）到 generated_samples_infer_raw
    sampled_np = sampled.squeeze(1).cpu().numpy()  # (N, L)
    # 反归一化公式：x_original = (x_norm + 1) / 2 * (max - min) + min
    signal_range = signal_max - signal_min + 1e-8
    sampled_denorm = (sampled_np + 1.0) / 2.0 * signal_range + signal_min

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

