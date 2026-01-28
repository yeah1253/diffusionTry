import torch
import numpy as np
import os
from scipy.io import loadmat
import glob
import re

# 直接从一维扩散子模块中导入 PhysiNet / GaussianDiffusion1D 等，
# 避免依赖外部已安装的 denoising_diffusion_pytorch 包版本中可能不存在 PhysiNet 的问题。
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import (
    PhysiNet,
    GaussianDiffusion1D,
    Trainer1D,
    Dataset1D,
    ucfilter_kmeans_select_indices,
)

import matplotlib.pyplot as plt

# Real SDUST bearing dataset loader
class RealSDUSTDataset(torch.utils.data.Dataset):
    """
    从真实 SDUST 轴承数据集中加载数据。
    
    从指定路径下的所有 .mat 文件中加载 Signal.y_values 的第一列数据。
    将长序列分割成固定长度的样本用于训练。
    
    参数:
    - data_path: .mat 文件所在目录
    - seq_length: 每个样本的序列长度
    - overlap: 滑动窗口的重叠比例 (0-1)
    - use_condition: 是否使用条件（从文件名提取 RPM 和 Load），如果 False 则 cond_dim=0
    """
    def __init__(self, data_path, seq_length=1024, overlap=0.5, use_condition=False):
        super().__init__()
        self.seq_length = int(seq_length)
        self.overlap = float(overlap)
        self.use_condition = use_condition
        
        # 查找所有 .mat 文件
        mat_files = glob.glob(os.path.join(data_path, '*.mat'))
        if len(mat_files) == 0:
            raise ValueError(f"No .mat files found in {data_path}")
        
        print(f"Found {len(mat_files)} .mat files in {data_path}")
        
        # 加载所有数据
        all_signals = []
        all_conditions = []
        
        for mat_file in sorted(mat_files):
            print(f"Loading {os.path.basename(mat_file)}...")
            try:
                data = loadmat(mat_file)
                
                # 提取 y_values 的第一列
                signal = data['Signal']['y_values'][0, 0]['values'].item()[:, 0]
                
                # 从文件名提取 RPM 和 Load（如果启用条件）
                # 文件名格式: "NC 1800 0.mat" -> RPM=1800, Load=0
                rpm = None
                load = None
                if self.use_condition:
                    # 提取文件名中的所有数字
                    numbers = re.findall(r'(\d+)', os.path.basename(mat_file))
                    if len(numbers) >= 2:
                        rpm = float(numbers[0])   # 第一个数字是 RPM
                        load = float(numbers[1])  # 第二个数字是 Load
                    elif len(numbers) == 1:
                        rpm = float(numbers[0])
                        load = 0.0  # 默认负载为 0
                    else:
                        rpm = 2000.0  # 默认 RPM
                        load = 0.0    # 默认负载
                
                # 将长序列分割成多个样本
                samples = create_samples(signal, self.seq_length, self.overlap)
                
                for sample in samples:
                    all_signals.append(sample.astype(np.float32))
                    if self.use_condition and rpm is not None and load is not None:
                        # 归一化 RPM: 假设范围 1000-3000
                        norm_rpm = (rpm - 1000.0) / 2000.0
                        # 归一化 Load: 假设范围 0-60
                        norm_load = load / 60.0
                        all_conditions.append(np.array([norm_rpm, norm_load], dtype=np.float32))
                
                print(f"  Extracted {len(samples)} samples from {len(signal)} data points (RPM={rpm}, Load={load})")
                
            except Exception as e:
                print(f"  Warning: Failed to load {mat_file}: {e}")
                continue
        
        if len(all_signals) == 0:
            raise ValueError("No valid samples extracted from .mat files")
        
        # 堆叠所有样本
        self.signals = np.stack(all_signals, axis=0)  # (N, L)
        
        # 归一化信号到 [-1, 1]，并保存归一化参数用于后续反归一化
        self.signal_min = self.signals.min()
        self.signal_max = self.signals.max()
        signal_range = self.signal_max - self.signal_min + 1e-8
        self.signals = 2.0 * (self.signals - self.signal_min) / signal_range - 1.0
        
        print(f"Signal normalization: min={self.signal_min:.4f}, max={self.signal_max:.4f}")
        print(f"Normalized range: [{self.signals.min():.4f}, {self.signals.max():.4f}]")
        
        # 条件信息
        if self.use_condition and len(all_conditions) > 0:
            self.conditions = np.stack(all_conditions, axis=0)  # (N, 2) - [RPM, Load]
            self.cond_dim = 2
        else:
            self.conditions = None
            self.cond_dim = 0
        
        print(f"Total dataset size: {len(self.signals)} samples")
        print(f"Condition dimension: {self.cond_dim} (RPM, Load)")
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        sig = torch.from_numpy(self.signals[idx]).unsqueeze(0)  # (1, L)
        
        if self.use_condition and self.conditions is not None:
            cond = torch.from_numpy(self.conditions[idx])  # (1,)
            return sig, cond
        else:
            return sig


# Synthetic SDUST-like dataset implementation (保留作为参考)
class SyntheticSDUSTDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset mimicking SDUST bearing signals controlled by RPM and Load.

    - RPM range: 1000 .. 3000 (continuous random)
    - Load range: 0 .. 60 (continuous random)
    - fs: 25600 Hz
    - seq_length: 1024
    Returns (signal, cond) where
      - signal: Tensor (1, L) float32 normalized to [-1, 1]
      - cond:   Tensor (2,) normalized [norm_rpm, norm_load]
    """
    def __init__(self, n_samples=2048, seq_length=1024, fs=25600.0, seed=None):
        super().__init__()
        self.n_samples = int(n_samples)
        self.seq_length = int(seq_length)
        self.fs = float(fs)
        self.t = np.arange(self.seq_length) / float(self.fs)

        self.rng = np.random.default_rng(seed)

        # sample parameters
        self.rpms = self.rng.uniform(1000.0, 3000.0, size=self.n_samples).astype(np.float32)
        self.loads = self.rng.uniform(0.0, 60.0, size=self.n_samples).astype(np.float32)

        signals = []
        conds = []
        for rpm, load in zip(self.rpms, self.loads):
            sig = self._synthesize_once(rpm, load)
            signals.append(sig.astype(np.float32))
            conds.append(self._normalize_cond(rpm, load).astype(np.float32))

        self.signals = np.stack(signals, axis=0)  # (N, L)
        self.conds = np.stack(conds, axis=0)      # (N, 2)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sig = torch.from_numpy(self.signals[idx]).unsqueeze(0)  # (1, L)
        cond = torch.from_numpy(self.conds[idx])                # (2,)
        return sig, cond

    def _normalize_cond(self, rpm, load):
        norm_rpm = (rpm - 1000.0) / 2000.0
        norm_load = load / 60.0
        return np.array([norm_rpm, norm_load], dtype=np.float32)

    def _synthesize_once(self, rpm, load):
        # f_fault locked to RPM, outer-race-like approx
        f_fault = (rpm / 60.0) * 3.5
        A = 0.5 + 0.5 * (load / 60.0)
        signal = A * np.sin(2.0 * np.pi * f_fault * self.t)

        sigma = 0.06 * A
        noise = self.rng.normal(0.0, sigma, size=self.t.shape)
        signal = signal + noise

        # small second harmonic to resemble more realistic spectrum
        signal += 0.05 * (load / 60.0) * np.sin(2.0 * np.pi * 2.0 * f_fault * self.t)

        max_abs = np.max(np.abs(signal)) + 1e-8
        signal = signal / max_abs
        signal = np.clip(signal, -1.0, 1.0)
        return signal


# 将长序列分割成多个样本（保留给参考，但现在使用合成数据）
def create_samples(signal, seq_length, overlap=0.5):
    step = int(seq_length * (1 - overlap))
    samples = []
    for i in range(0, len(signal) - seq_length + 1, step):
        sample = signal[i:i + seq_length]
        samples.append(sample)
    return np.array(samples)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # ---------- dataset configuration (real SDUST data) ----------
    SEQ_LENGTH = 1024
    DATA_PATH = r'D:\speedLoad'
    OVERLAP = 0.5  # 滑动窗口重叠比例
    USE_CONDITION = True  # 是否使用条件（从文件名提取 RPM），False 表示无条件生成

    print(f"Loading real SDUST dataset from {DATA_PATH}...")
    dataset = RealSDUSTDataset(
        data_path=DATA_PATH,
        seq_length=SEQ_LENGTH,
        overlap=OVERLAP,
        use_condition=USE_CONDITION
    )
    print(f"Dataset size: {len(dataset)}, seq_length={SEQ_LENGTH}")

    # ---------- model and diffusion ----------
    CHANNELS = 1
    COND_DIM = dataset.cond_dim  # 根据数据集自动设置

    print(f"Building PhysiNet (cond_dim={COND_DIM})...")
    model = PhysiNet(
        dim = 128,                    # 增加基础维度 (64 -> 128) 以提升模型容量
        dim_mults = (1, 2, 4, 8),     # 保持4层深度结构
        channels = CHANNELS,
        cond_dim = COND_DIM,
        dropout = 0.1,                # 增加 dropout (0.0 -> 0.1) 防止过拟合
        attn_dim_head = 64,           # 增加注意力头维度 (32 -> 64)
        attn_heads = 8,               # 增加注意力头数 (4 -> 8)
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = SEQ_LENGTH,
        timesteps = 1000,
        objective = 'pred_v',
        auto_normalize = False,
    )

    # ---------- trainer (adjusted for 23 files with RPM+Load conditions) ----------
    # 数据量估算: 23个文件 × ~1000样本/文件 ≈ 23000样本
    # 有效batch: 48 × 2 = 96, 每epoch约240步, 30000步约125个epoch
    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 48,        # 调整为48，避免显存不足 (dim=128模型更大)
        train_lr = 4e-5,              # 降低学习率提升训练稳定性
        train_num_steps = 30000,      # 约125个epoch，足够收敛
        gradient_accumulate_every = 2, # 梯度累积，有效batch=96
        ema_decay = 0.9995,           # 较高的EMA衰减率
        amp = True,                   # 启用AMP加速训练
        save_and_sample_every = 1500, # 每1500步保存一次，共20个checkpoint
        num_samples = 16,
        results_folder = './results_vibration',
    )

    print("Starting training on real SDUST dataset...")
    trainer.train()
    print("Training finished.")

    # ---------- generate larger batch (optionally conditional on RPM) and run UCFilter ----------
    print("Generating bulk samples and running UCFilter...")

    # 根据是否有条件决定采样方式
    if COND_DIM > 0:
        # 有条件模型：显式指定目标 RPM 和 Load
        TARGET_RPM = 2000.0
        TARGET_LOAD = 40.0  # 可以修改为其他负载值，如 0, 20, 40, 60
        
        # 与数据集中的归一化方式保持一致
        target_norm_rpm = (TARGET_RPM - 1000.0) / 2000.0
        target_norm_load = TARGET_LOAD / 60.0

        print(f"Conditional generation enabled. Target RPM = {TARGET_RPM} (norm = {target_norm_rpm:.4f}), Load = {TARGET_LOAD} (norm = {target_norm_load:.4f})")

        # 构造条件批次张量 (batch, cond_dim)
        batch_size = 64
        if COND_DIM == 2:
            # 两个条件：RPM 和 Load
            cond_batch = torch.tensor(
                [[target_norm_rpm, target_norm_load]] * batch_size,
                dtype=torch.float32,
                device=device,
            )
        else:
            # 单个条件：RPM（向后兼容）
            cond_batch = torch.full(
                (batch_size, COND_DIM),
                fill_value=target_norm_rpm,
                dtype=torch.float32,
                device=device,
            )

        # 优先使用 Trainer1D 的 EMA 条件采样接口
        try:
            print("Sampling with Trainer1D.sample_with_condition (EMA model)...")
            sampled_seqs = trainer.sample_with_condition(
                batch_size=batch_size,
                cond=cond_batch,
            )
            if sampled_seqs is None:
                raise RuntimeError("trainer.sample_with_condition returned None")
        except Exception as e:
            print(f"Trainer1D.sample_with_condition failed, fallback to diffusion.sample with cond. Error: {e}")
            sampled_seqs = diffusion.sample(
                batch_size=batch_size,
                model_forward_kwargs={"cond": cond_batch},
            )
    else:
        # 无条件模型：保持原有无条件采样
        sampled_seqs = diffusion.sample(batch_size=64)
    sampled_seqs_np = sampled_seqs.squeeze(1).cpu().numpy()

    raw_folder = './generated_samples_raw'
    os.makedirs(raw_folder, exist_ok=True)
    np.save(os.path.join(raw_folder, 'all_generated.npy'), sampled_seqs_np)

    with torch.no_grad():
        selected_idx, kl_scores, cluster_labels = ucfilter_kmeans_select_indices(
            sampled_seqs.detach().cpu(),
            num_clusters = 3,
            k_ratio = 0.9,
            sigma = 1.0,
            embed_dim = 2,
        )

    selected_idx_np = selected_idx.numpy()
    sampled_filtered = sampled_seqs_np[selected_idx_np]

    # 反归一化生成的信号到原始尺度
    # 从 [-1, 1] 反归一化到 [signal_min, signal_max]
    signal_min = dataset.signal_min
    signal_max = dataset.signal_max
    signal_range = signal_max - signal_min + 1e-8
    
    # 反归一化公式：x_original = (x_norm + 1) / 2 * (max - min) + min
    sampled_filtered_denorm = (sampled_filtered + 1.0) / 2.0 * signal_range + signal_min
    
    print(f"\n反归一化参数:")
    print(f"  signal_min: {signal_min:.4f}")
    print(f"  signal_max: {signal_max:.4f}")
    print(f"  生成信号范围 (归一化): [{sampled_filtered.min():.4f}, {sampled_filtered.max():.4f}]")
    print(f"  生成信号范围 (反归一化): [{sampled_filtered_denorm.min():.4f}, {sampled_filtered_denorm.max():.4f}]")
    
    # 幅值修正：基于训练数据的统计特性
    # 计算训练数据的标准差和RMS
    train_signals_original = dataset.signals  # 已经是归一化后的
    # 反归一化训练数据用于计算统计量
    train_signals_denorm = (train_signals_original + 1.0) / 2.0 * signal_range + signal_min
    train_std = np.std(train_signals_denorm.flatten())
    train_rms = np.sqrt(np.mean(train_signals_denorm.flatten()**2))
    
    # 计算生成信号的统计量
    gen_std = np.std(sampled_filtered_denorm.flatten())
    gen_rms = np.sqrt(np.mean(sampled_filtered_denorm.flatten()**2))
    
    # 计算修正系数（使用标准差，更稳定）
    if gen_std > 1e-8:
        amplitude_correction_factor = train_std / gen_std
        print(f"\n幅值修正分析:")
        print(f"  训练数据 std: {train_std:.4f}, RMS: {train_rms:.4f}")
        print(f"  生成数据 std: {gen_std:.4f}, RMS: {gen_rms:.4f}")
        print(f"  修正系数: {amplitude_correction_factor:.4f}")
        
        # 应用修正：保持均值不变，缩放波动部分
        gen_mean = np.mean(sampled_filtered_denorm, axis=1, keepdims=True)
        gen_centered = sampled_filtered_denorm - gen_mean
        sampled_filtered_denorm = gen_mean + gen_centered * amplitude_correction_factor
        
        print(f"  修正后范围: [{sampled_filtered_denorm.min():.4f}, {sampled_filtered_denorm.max():.4f}]")
        print(f"  修正后 std: {np.std(sampled_filtered_denorm.flatten()):.4f}")
    else:
        amplitude_correction_factor = 1.0
        print(f"\n警告: 生成信号标准差过小，跳过幅值修正")

    filtered_folder = './generated_samples'
    os.makedirs(filtered_folder, exist_ok=True)
    np.save(os.path.join(filtered_folder, 'selected_idx.npy'), selected_idx_np)
    try:
        kl_np = kl_scores.numpy()
    except Exception:
        kl_np = np.array(kl_scores)
    np.save(os.path.join(filtered_folder, 'kl_scores.npy'), kl_np)
    
    # 保存归一化参数和修正参数
    np.save(os.path.join(filtered_folder, 'normalization_params.npy'), {
        'signal_min': signal_min,
        'signal_max': signal_max,
        'amplitude_correction_factor': amplitude_correction_factor,
        'train_std': train_std,
        'train_rms': train_rms
    })

    for i, (idx, gen_signal) in enumerate(zip(selected_idx_np, sampled_filtered_denorm)):
        np.save(os.path.join(filtered_folder, f'generated_signal_{i}.npy'), gen_signal)
        print(f"Saved UCFilter-selected generated signal {i} (orig idx={idx}) to {os.path.join(filtered_folder, f'generated_signal_{i}.npy')}")

    print("\nScript completed.")
