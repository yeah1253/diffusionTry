import torch
import numpy as np
import os
from scipy.io import loadmat

# 直接从一维扩散子模块中导入 PhysiNet / GaussianDiffusion1D 等，
# 避免依赖外部已安装的 denoising_diffusion_pytorch 包版本中可能不存在 PhysiNet 的问题。
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import (
    PhysiNet,
    GaussianDiffusion1D,
    Trainer1D,
    Dataset1D,
    ucfilter_kmeans_select_indices,
)

# 将长序列分割成多个样本
def create_samples(signal, seq_length, overlap=0.5):
    """将长序列分割成多个样本"""
    step = int(seq_length * (1 - overlap))
    samples = []
    
    for i in range(0, len(signal) - seq_length + 1, step):
        sample = signal[i:i + seq_length]
        samples.append(sample)
    
    return np.array(samples)

if __name__ == '__main__':
    # 加载 .mat 文件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    mat_file_path = r'D:\PostGraduate\YearOne\HIL\data\simple_bearing\ball\9005k.mat'
    print(f"Loading data from {mat_file_path}...")
    mat_data = loadmat(mat_file_path)

    # 提取 Bearing_Acc_X 信号
    bearing_acc_x = mat_data['Bearing_Acc_X']  # 应该是 1x50001 的数组

    # 转换为 numpy 数组并展平
    if bearing_acc_x.shape[0] == 1:
        signal = bearing_acc_x[0]  # 如果是 1xN，取第一行
    else:
        signal = bearing_acc_x.flatten()

    #截取前5000个采样点
    CUTOFF_POINTS = 5000
    print(f"Original signal length: {len(signal)}")

    if len(signal) > CUTOFF_POINTS:
        signal = signal[CUTOFF_POINTS:]
        print(f"Signal length after cropping start (steady state): {len(signal)}")
    else:
        raise ValueError(f"Error: Signal length ({len(signal)}) is shorter than cutoff points ({CUTOFF_POINTS})!")

    print(f"Signal shape: {signal.shape}")
    print(f"Signal length: {len(signal)}")
    print(f"Signal range: [{signal.min():.4f}, {signal.max():.4f}]")

    # 参数设置
    SEQ_LENGTH = 1024  # 序列长度，可以根据需要调整 (128, 256, 512, 1024 等)
    CHANNELS = 1  # 单通道信号
    OVERLAP = 0.5  # 重叠比例，用于生成更多训练样本

    # 创建样本
    samples = create_samples(signal, SEQ_LENGTH, OVERLAP)
    print(f"Created {len(samples)} samples of length {SEQ_LENGTH}")

    # 归一化数据到 [0, 1] 范围
    signal_min = samples.min()
    signal_max = samples.max()
    samples_normalized = (samples - signal_min) / (signal_max - signal_min + 1e-8)

    print(f"Normalized range: [{samples_normalized.min():.4f}, {samples_normalized.max():.4f}]")

    # 转换为 torch tensor，格式: (batch, channels, seq_length)
    samples_tensor = torch.from_numpy(samples_normalized).float()
    samples_tensor = samples_tensor.unsqueeze(1)  # 添加通道维度: (batch, 1, seq_length)

    print(f"Training data shape: {samples_tensor.shape}")

    # 创建数据集
    dataset = Dataset1D(samples_tensor)

    # 创建条件扩散模型的骨干网络（Physi-UNet）
    # 当前示例未显式使用条件（cond_dim=0），
    # 后续可以根据 DiffPhysiNet.pdf 中的物理/健康状态设计，将 cond_dim 设置为对应维度，
    # 并在训练与采样时通过 GaussianDiffusion1D 的 model_forward_kwargs 传入 cond。
    model = PhysiNet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = CHANNELS,
        cond_dim = 0  # 0 表示当前退化为无条件模型，占好“条件生成”接口
    )

    # 创建扩散模型
    diffusion = GaussianDiffusion1D(
        model,
        seq_length = SEQ_LENGTH,
        timesteps = 1000,
        objective = 'pred_v'
    )

    # 创建训练器
    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 16,
        train_lr = 8e-5,
        train_num_steps = 10000,  # 训练步数，可以根据需要调整
        gradient_accumulate_every = 2,
        ema_decay = 0.995,
        amp = True,  # 混合精度训练
        save_and_sample_every = 1000,  # 每1000步保存一次
        results_folder = './results_vibration'
    )

    # 开始训练
    print("Starting training...")
    trainer.train()

    # 训练完成后生成样本
    print("\nGenerating samples...")
    sampled_seqs = diffusion.sample(batch_size = 64)  # 先生成一定数量的样本
    print(f"Generated samples shape: {sampled_seqs.shape}")  # (N, 1, L)

    # 使用 UCFilter 选择高质量样本（基于 KL 代价）
    print("Applying UCFilter to select high-quality generated samples...")
    with torch.no_grad():
        # UCFilter 期望输入 (N, C, L) 的 tensor
        selected_idx, kl_scores, cluster_labels = ucfilter_kmeans_select_indices(
            sampled_seqs.detach().cpu(),
            num_clusters = 3,  # 可根据数据分布调整聚类簇数
            k_ratio = 0.9,     # 在每个簇内保留 KL 代价较小的前 90% 样本
            sigma = 1.0,
            embed_dim = 2,
        )

    selected_idx_np = selected_idx.numpy()
    print(f"Total generated: {sampled_seqs.shape[0]}, selected by UCFilter: {len(selected_idx_np)}")

    # 仅对通过 UCFilter 的样本做反归一化与保存
    sampled_seqs_np = sampled_seqs.squeeze(1).cpu().numpy()   # (N, L)
    sampled_seqs_denorm = sampled_seqs_np * (signal_max - signal_min) + signal_min
    sampled_filtered = sampled_seqs_denorm[selected_idx_np]

    # 保存生成且通过 UCFilter 筛选的样本
    os.makedirs('./generated_samples', exist_ok=True)
    for i, (idx, gen_signal) in enumerate(zip(selected_idx_np, sampled_filtered)):
        np.save(f'./generated_samples/generated_signal_{i}.npy', gen_signal)
        print(f"Saved UCFilter-selected generated signal {i} (orig idx={idx}) to ./generated_samples/generated_signal_{i}.npy")

    print("\nTraining completed!")
