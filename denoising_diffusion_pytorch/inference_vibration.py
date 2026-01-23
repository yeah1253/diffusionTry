import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# 导入模型定义 (确保 denoising_diffusion_pytorch_1d.py 在同级目录下或 Python 路径中)
from denoising_diffusion_pytorch_1d import (
    PhysiNet,
    GaussianDiffusion1D,
    Trainer1D
)


def norm_cond(rpm, load):
    """
    将物理工况归一化为模型可接受的条件向量。
    逻辑必须与训练代码 (SyntheticSDUSTDataset) 严格一致。
    来源: train_1d_vibration.txt [1, 2]
    """
    # RPM 归一化: (rpm - 1000) / 2000 -> 映射 1000~3000 到 0~1 (近似)
    norm_rpm = (rpm - 1000.0) / 2000.0
    # Load 归一化: load / 60 -> 映射 0~60 到 0~1
    norm_load = load / 60.0
    return torch.tensor([norm_rpm, norm_load], dtype=torch.float32)


def load_model(model_path, device):
    """
    加载模型架构并载入权重 (修复版)
    """
    print(f"正在加载模型: {model_path} ...")

    # 1. 实例化模型
    model = PhysiNet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        cond_dim=2,
        dropout=0.0,
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=1024,
        timesteps=1000,
        objective='pred_v',
        auto_normalize=False,
    )

    diffusion.to(device)

    # 2. 加载权重
    checkpoint = torch.load(model_path, map_location=device)

    # --- 关键修改开始 ---
    # 方案 A: 强制加载标准权重 'model' (最稳妥，肯定能跑)
    # 因为 Trainer 保存 'model' 时使用的是 accelerator.get_state_dict(self.model)，
    # 这通常与我们这里实例化的结构直接匹配。

    if 'model' in checkpoint:
        print("正在加载标准模型权重 ('model')...")
        state_dict = checkpoint['model']
    elif 'ema' in checkpoint:
        # 如果只有 EMA，我们需要去掉键名前面的 "ema_model." 前缀
        print("正在加载 EMA 权重并修复键名...")
        state_dict = checkpoint['ema']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('ema_model.'):
                # 去掉 'ema_model.' 前缀
                new_state_dict[k.replace('ema_model.', '')] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    else:
        state_dict = checkpoint

    # 加载处理后的权重
    # 依然使用 strict=False 以避开 'betas' 等缓冲区缺失的问题，
    # 但现在核心权重应该能匹配上了。
    msg = diffusion.load_state_dict(state_dict, strict=False)
    print(f"权重加载结果: {msg}")
    # 如果看到 missing_keys 只有 betas/alphas 等，说明加载成功了。
    # 如果 missing_keys 包含 'model.init_conv.weight' 等，说明还是没加载上。

    # --- 关键修改结束 ---

    diffusion.eval()
    return diffusion


def compute_fft(sig, fs=25600.0, n_fft=8192):
    """
    计算 FFT，包含补0操作以提高频谱显示分辨率
    """
    fft_vals = np.fft.rfft(sig, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mag = np.abs(fft_vals)
    return freqs, mag


def main():
    # --- 参数设置 ---
    parser = argparse.ArgumentParser(description="DiffPhysiNet 轴承振动信号推理")
    parser.add_argument('--rpm', type=float, default=3000, help='输入转速 (RPM), 例如 1000-3000')
    parser.add_argument('--load', type=float, default=30, help='输入负载 (N), 例如 0-60')
    parser.add_argument('--milestone', type=int, default=10,
                        help='要加载的模型 checkpoint 编号 (例如 model-1.pt 则填 1)')
    parser.add_argument('--samples', type=int, default=1, help='生成的样本数量')
    parser.add_argument('--save_dir', type=str, default='./inference_results', help='结果保存路径')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 1. 准备路径 ---
    # 使用您提供的绝对路径 (注意：Windows路径建议在字符串前加 r 或将 \ 改为 /)
    model_path = f'D:/haoran/diffusionTry/results_vibration/model-{args.milestone}.pt'
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请检查路径或确认 'milestone' 参数是否正确。")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    # --- 2. 加载模型 ---
    diffusion_model = load_model(model_path, device)

    # --- 3. 构造条件 ---
    # 根据输入的 RPM 和 Load 生成归一化的条件向量
    cond_single = norm_cond(args.rpm, args.load)  # shape (2,)
    # 扩展为 Batch 维度: (batch_size, 2)
    cond_batch = cond_single.unsqueeze(0).repeat(args.samples, 1).to(device)

    print(f"正在生成信号... \n条件: RPM={args.rpm}, Load={args.load}")
    print(f"归一化条件向量: {cond_single.numpy()}")

    # --- 4. 采样 (推理) ---
    with torch.no_grad():
        # 调用 sample 函数并传入 cond 参数
        # 注意：源码中 sample 接受 model_forward_kwargs
        generated_data = diffusion_model.sample(
            batch_size=args.samples,
            model_forward_kwargs={'cond': cond_batch}  # 关键：注入物理条件 [6, 7]
        )

    # 转换到 CPU numpy
    # generated_data shape: (Batch, Channels, Length) -> (N, 1, 1024)
    signals = generated_data.detach().cpu().numpy().squeeze(1)

    # --- 5. 保存与可视化 ---
    for i, sig in enumerate(signals):
        # 5.1 计算理论故障频率 (用于验证)
        # 公式来源: f_fault = (rpm / 60.0) * 3.5 [8]
        theoretical_fault_freq = (args.rpm / 60.0) * 3.5
        print(f"样本 {i + 1}: 理论故障特征频率 = {theoretical_fault_freq:.2f} Hz")

        # 5.2 FFT 分析 (补0到 8192 点以获得更高分辨率)
        freqs, mag = compute_fft(sig, fs=25600.0, n_fft=8192)

        # 找到主峰
        peak_idx = np.argmax(mag[freqs > 5])  # 忽略直流分量
        peak_freq = freqs[freqs > 5][peak_idx]
        print(f"样本 {i + 1}: 实际生成主频 = {peak_freq:.2f} Hz")

        # 5.3 绘图
        plt.figure(figsize=(12, 8))

        # 时域图
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(sig)) / 25600.0, sig)
        plt.title(f'Generated Signal (Time Domain) - RPM: {args.rpm}, Load: {args.load}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

        # 频域图
        plt.subplot(2, 1, 2)
        plt.plot(freqs, mag)
        plt.xlim(0, 500)  # 重点关注低频段 (0-500Hz)

        # 标记理论频率
        plt.axvline(x=theoretical_fault_freq, color='r', linestyle='--', alpha=0.6,
                    label=f'Theoretical Fault: {theoretical_fault_freq:.1f}Hz')
        plt.legend()

        plt.title(f'FFT Spectrum (Resolution enhanced)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(args.save_dir, f'inference_rpm{int(args.rpm)}_load{int(args.load)}_sample{i}.png')
        plt.savefig(save_path)
        plt.close()

        # 5.4 保存原始数据
        data_save_path = os.path.join(args.save_dir, f'inference_rpm{int(args.rpm)}_load{int(args.load)}_sample{i}.npy')
        np.save(data_save_path, sig)

    print(f"推理完成！结果已保存至 {args.save_dir}")


if __name__ == '__main__':
    main()