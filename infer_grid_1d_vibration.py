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
    pattern = os.path.join(results_folder, "bear_digtal_model.pt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No checkpoint files found in {results_folder}")

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
    """
    model = PhysiNet(
        dim=128,
        dim_mults=(1, 2, 4, 8),
        channels=channels,
        cond_dim=cond_dim,
        dropout=0.1,
        attn_dim_head=64,
        attn_heads=8,
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
    从 checkpoint 加载扩散模型权重并设置为 eval()。
    """
    device = next(diffusion.parameters()).device
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    diffusion.load_state_dict(data["model"])
    diffusion.eval()
    return diffusion


def load_normalization_params_from_checkpoint(ckpt_path: str) -> tuple[float, float]:
    """
    从 checkpoint 中提取 signal_min 和 signal_max。
    如果不存在则返回 (None, None)。
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


def main():
    # 与训练脚本保持一致的超参数
    SEQ_LENGTH = 1024
    CHANNELS = 1
    COND_DIM = 2  # 使用 RPM + Load 条件
    RESULTS_FOLDER = "./results_vibration"

    # 网格设置：负载 0..60 以 2 步长，速度 1000..3000 以 50 步长
    loads = np.arange(0, 60 + 1, 2, dtype=np.float32)
    rpms = np.arange(1000, 3000 + 1, 50, dtype=np.float32)

    # 每个组合采样数量，可根据需要调整
    num_samples_per_combo = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for grid inference: {device}")

    ckpt_path = find_latest_checkpoint(RESULTS_FOLDER)
    print(f"Loading checkpoint: {ckpt_path}")

    diffusion = build_model_and_diffusion(SEQ_LENGTH, CHANNELS, cond_dim=COND_DIM)
    diffusion.to(device)
    diffusion = load_trained_diffusion_from_checkpoint(ckpt_path, diffusion)

    signal_min, signal_max = load_normalization_params_from_checkpoint(ckpt_path)
    if signal_min is None or signal_max is None:
        raise RuntimeError("Normalization parameters not found in checkpoint.")
    signal_range = signal_max - signal_min + 1e-8

    out_root = './generated_grid'
    os.makedirs(out_root, exist_ok=True)

    for load in loads:
        for rpm in rpms:
            print(f"Sampling for Load={load:.1f}, RPM={rpm:.1f}")
            norm_load = load / 60.0
            norm_rpm = (rpm - 1000.0) / 2000.0

            cond_batch = torch.tensor(
                [[norm_rpm, norm_load]] * num_samples_per_combo,
                dtype=torch.float32,
                device=device,
            )

            with torch.no_grad():
                sampled = diffusion.sample(
                    batch_size=num_samples_per_combo,
                    model_forward_kwargs={"cond": cond_batch},
                )

            sampled_np = sampled.squeeze(1).cpu().numpy()
            sampled_denorm = (sampled_np + 1.0) / 2.0 * signal_range + signal_min

            # 运行 UCFilter 选取高质量子集
            max_keep = 10
            try:
                with torch.no_grad():
                    sel_idx, kl_scores, _ = ucfilter_kmeans_select_indices(
                        sampled.detach().cpu(),
                        num_clusters=3,
                        k_ratio=0.9,
                        sigma=1.0,
                        embed_dim=2,
                    )
                sel_idx = sel_idx.numpy()
                try:
                    kl_np = kl_scores.numpy()
                except Exception:
                    kl_np = np.array(kl_scores)
            except Exception as e:
                print(f"  Warning: UCFilter failed for load={load}, rpm={rpm}: {e}. Falling back to all samples.")
                sel_idx = np.arange(len(sampled_denorm))
                kl_np = np.zeros(len(sampled_denorm))

            # 如果过滤后为空，回退到全部样本
            if len(sel_idx) == 0:
                print(f"  Warning: UCFilter returned empty for load={load}, rpm={rpm}. Using all samples.")
                sel_idx = np.arange(len(sampled_denorm))
                kl_np = np.zeros(len(sampled_denorm))

            filtered = sampled_denorm[sel_idx]

            # 截断到 max_keep；若不足 max_keep 则随机重采样补足（有放回）
            if len(filtered) >= max_keep:
                sel_idx = sel_idx[:max_keep]
                filtered = filtered[:max_keep]
                kl_np = kl_np[:max_keep]
            else:
                n_have = len(filtered)
                n_need = max_keep - n_have
                extra_idx = np.random.choice(n_have, size=n_need, replace=True)
                sel_idx = np.concatenate([sel_idx, sel_idx[extra_idx]])
                filtered = np.concatenate([filtered, filtered[extra_idx]], axis=0)
                kl_np = np.concatenate([kl_np, kl_np[extra_idx]])
                print(f"  Info: Only {n_have} samples after UCFilter for load={load}, rpm={rpm}. "
                      f"Resampled {n_need} extras to reach {max_keep}.")

            combo_folder = os.path.join(out_root, f"load_{int(load)}", f"rpm_{int(rpm)}")
            os.makedirs(combo_folder, exist_ok=True)

            np.save(os.path.join(combo_folder, 'all_generated.npy'), sampled_denorm)
            np.save(os.path.join(combo_folder, 'selected_idx.npy'), sel_idx)
            np.save(os.path.join(combo_folder, 'kl_scores.npy'), kl_np)

            for i, sig in enumerate(filtered):
                np.save(os.path.join(combo_folder, f'filtered_{i}.npy'), sig)

    print("Grid inference finished.")


if __name__ == "__main__":
    main()
