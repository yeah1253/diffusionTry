import os
import sys
import glob
import numpy as np
import torch
from collections import OrderedDict

# 添加项目根以便导入 eval.dataset
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import (
    PhysiNet,
    GaussianDiffusion1D,
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


def _strip_module_prefix(state_dict: dict) -> dict:
    """若 checkpoint 由 accelerate/DDP 保存，去除 'module.' 前缀。"""
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_dict[name] = v
    return new_dict


def load_trained_diffusion_from_checkpoint(
    ckpt_path: str,
    diffusion: GaussianDiffusion1D,
) -> GaussianDiffusion1D:
    """
    从训练保存的 model-*.pt 中加载 GaussianDiffusion1D 权重。
    支持 accelerate 保存的 checkpoint（可能带 module. 前缀）。
    """
    device = next(diffusion.parameters()).device

    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = data["model"]

    # 兼容 accelerate  wrapped model
    state_dict = _strip_module_prefix(state_dict)

    missing, unexpected = diffusion.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Warning: unexpected keys when loading: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

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
    COND_DIM = 2  # RPM、Load；与 train_1d_vibration.py 中 use_condition=True 时一致
    RESULTS_FOLDER = "./results_vibration"
    # 可选：直接指定 checkpoint 路径，若为 None 则从 RESULTS_FOLDER 中查找最新的 model-*.pt
    CKPT_PATH_OVERRIDE = None  # 例如: r"./results_vibration/model-20.pt"
    
    # 故障类型 key，用于生成 phys_signal 物理先验；与训练数据目录名一致（如 IF0.2）
    # 设为 None 或 "NC" 则不使用 phys_signal
    FAULT_KEY = "IF0.2"

    # 多工况生成：24 个 (RPM, Load) 组合，每工况 40 个样本
    # RPM: 1000, 1500, 1800, 2000, 2500, 3000
    # Load: 0, 20, 40, 60
    RPM_LIST = [1000, 1500, 1800, 2000, 2500, 3000]
    LOAD_LIST = [0, 20, 40, 60]
    SAMPLES_PER_CONDITION = 40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {device}")

    # 1) 找到或指定 checkpoint
    if CKPT_PATH_OVERRIDE:
        if not os.path.isfile(CKPT_PATH_OVERRIDE):
            raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH_OVERRIDE}")
        ckpt_path = CKPT_PATH_OVERRIDE
    else:
        ckpt_path = find_latest_checkpoint(RESULTS_FOLDER)
    print(f"Loading checkpoint: {ckpt_path}")

    # 2) 构建模型与扩散对象（支持条件生成）
    diffusion = build_model_and_diffusion(SEQ_LENGTH, CHANNELS, cond_dim=COND_DIM)
    diffusion.to(device)

    # 3) 加载训练好的扩散模型权重（不重新训练）
    diffusion = load_trained_diffusion_from_checkpoint(ckpt_path, diffusion)

    # 4) 获取归一化参数，用于反归一化
    signal_min, signal_max = load_normalization_params_from_checkpoint(ckpt_path)
    if signal_min is None or signal_max is None:
        signal_min, signal_max = -1.0, 1.0
        print("Normalization params not in checkpoint, using identity denorm (output stays in [-1,1]).")
    else:
        print(f"Signal normalization params: min={signal_min:.4f}, max={signal_max:.4f}")

    # 5) 构建工况列表：(RPM, Load)
    conditions = [(rpm, load) for rpm in RPM_LIST for load in LOAD_LIST]
    total_samples = len(conditions) * SAMPLES_PER_CONDITION
    print(f"Generating {SAMPLES_PER_CONDITION} samples × {len(conditions)} conditions = {total_samples} total")
    print(f"Conditions: {conditions[:4]}... (first 4 of {len(conditions)})")

    signal_range = signal_max - signal_min + 1e-8
    save_root = os.path.join(_PROJECT_ROOT, "generated_samples_infer")
    fault_dir = FAULT_KEY if FAULT_KEY else "unknown"
    fault_path = os.path.join(save_root, fault_dir)
    all_denorm = []  # 用于 raw 汇总

    # 6) 逐工况生成并保存
    for cond_idx, (rpm, load) in enumerate(conditions):
        cond_folder = f"{rpm} {load}"  # 目录名，如 "1000 0"
        save_folder = os.path.join(fault_path, cond_folder)
        os.makedirs(save_folder, exist_ok=True)

        # 归一化条件（与训练代码一致）
        norm_rpm = (rpm - 1000.0) / 2000.0
        norm_load = load / 60.0
        cond_batch = torch.tensor(
            [[norm_rpm, norm_load]] * SAMPLES_PER_CONDITION,
            dtype=torch.float32,
            device=device,
        )

        model_kwargs = {}
        if COND_DIM > 0:
            model_kwargs["cond"] = cond_batch
        if COND_DIM > 0 and FAULT_KEY and FAULT_KEY != "NC":
            try:
                import Bearing
                if FAULT_KEY in Bearing.FAULT_TYPE_MAP:
                    phys_at_target = Bearing.main(
                        fault_key=FAULT_KEY, rpm=float(rpm), no_plot=True, target_len=SEQ_LENGTH
                    )
                    phys_t = torch.from_numpy(phys_at_target).float().to(device).unsqueeze(0).unsqueeze(0)
                    phys_t = phys_t.expand(SAMPLES_PER_CONDITION, -1, -1)
                    model_kwargs["phys_signal"] = phys_t
            except Exception as e:
                if cond_idx == 0:
                    print(f"Warning: Could not generate phys_signal ({e}), sampling without phys_signal.")

        with torch.no_grad():
            sampled = diffusion.sample(
                batch_size=SAMPLES_PER_CONDITION,
                model_forward_kwargs=model_kwargs,
            )  # (N, C, L)

        sampled_np = sampled.squeeze(1).cpu().numpy()
        sampled_denorm = (sampled_np + 1.0) / 2.0 * signal_range + signal_min
        all_denorm.append(sampled_denorm)

        for i, sig in enumerate(sampled_denorm):
            out_path = os.path.join(save_folder, f"infer_signal_{i}.npy")
            np.save(out_path, sig)

        print(f"  [{cond_idx+1}/{len(conditions)}] RPM={rpm} Load={load} -> {save_folder} ({SAMPLES_PER_CONDITION} samples)")

    # 7) 可选：保存汇总到 raw 目录
    raw_folder = os.path.join(_PROJECT_ROOT, "generated_samples_infer_raw")
    os.makedirs(raw_folder, exist_ok=True)
    all_arr = np.concatenate(all_denorm, axis=0)
    np.save(os.path.join(raw_folder, "all_generated.npy"), all_arr)

    print(f"Inference completed. Saved {total_samples} samples to {fault_path}/{{RPM Load}}/ (24 conditions × 40)")


if __name__ == "__main__":
    main()

