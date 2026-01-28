"""
分析生成信号幅值偏小的原因并提供修正方案
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from scipy.fft import fft

# ===== 配置参数 =====
REAL_DATA_PATH = r'D:\山东科技大学轴承齿轮数据集\轴承数据集\NC\NC 2000 0.mat'
GENERATED_DIR = './generated_samples'
NORMALIZATION_PARAMS_FILE = os.path.join(GENERATED_DIR, 'normalization_params.npy')

def load_real_signal():
    """加载真实信号"""
    print("加载真实信号...")
    data = loadmat(REAL_DATA_PATH)
    
    # 从 Signal.y_values 提取第一列
    if 'Signal' in data:
        signal_data = data['Signal']
        if hasattr(signal_data, 'dtype') and signal_data.dtype.names and 'y_values' in signal_data.dtype.names:
            y_values = signal_data['y_values'][0, 0]
            if 'values' in y_values.dtype.names:
                signal = y_values['values'].item()[:, 0]
                print(f"  成功提取，长度: {len(signal)}")
                return signal.flatten().astype(np.float64)
    
    raise ValueError("无法从真实数据中提取信号")

def load_generated_signals():
    """加载生成的信号"""
    print("加载生成的信号...")
    signals = []
    indices = []
    
    for i in range(20):  # 检查更多信号
        file_path = os.path.join(GENERATED_DIR, f'generated_signal_{i}.npy')
        if os.path.exists(file_path):
            signal = np.load(file_path).astype(np.float64)
            signals.append(signal)
            indices.append(i)
            print(f"  加载 Signal {i}: shape={signal.shape}, range=[{signal.min():.4f}, {signal.max():.4f}]")
    
    return signals, indices

def check_normalization():
    """检查归一化参数"""
    print("\n检查归一化参数...")
    if os.path.exists(NORMALIZATION_PARAMS_FILE):
        params = np.load(NORMALIZATION_PARAMS_FILE, allow_pickle=True).item()
        print(f"  找到归一化参数文件:")
        print(f"    signal_min: {params['signal_min']:.4f}")
        print(f"    signal_max: {params['signal_max']:.4f}")
        return params
    else:
        print("  警告: 未找到归一化参数文件，可能需要重新生成信号")
        return None

def analyze_amplitude_statistics(real_signal, gen_signals):
    """分析幅值统计信息"""
    print("\n" + "="*70)
    print("幅值统计分析")
    print("="*70)
    
    # 真实信号统计
    real_stats = {
        'mean': np.mean(real_signal),
        'std': np.std(real_signal),
        'min': np.min(real_signal),
        'max': np.max(real_signal),
        'rms': np.sqrt(np.mean(real_signal**2)),
        'peak_to_peak': np.max(real_signal) - np.min(real_signal)
    }
    
    print("\n真实信号统计:")
    for key, value in real_stats.items():
        print(f"  {key:12s}: {value:12.4f}")
    
    # 生成信号统计（平均）
    gen_means = [np.mean(s) for s in gen_signals]
    gen_stds = [np.std(s) for s in gen_signals]
    gen_rms = [np.sqrt(np.mean(s**2)) for s in gen_signals]
    gen_peak_to_peak = [np.max(s) - np.min(s) for s in gen_signals]
    
    gen_stats = {
        'mean': np.mean(gen_means),
        'std': np.mean(gen_stds),
        'min': np.min([np.min(s) for s in gen_signals]),
        'max': np.max([np.max(s) for s in gen_signals]),
        'rms': np.mean(gen_rms),
        'peak_to_peak': np.mean(gen_peak_to_peak)
    }
    
    print("\n生成信号统计 (平均值):")
    for key, value in gen_stats.items():
        print(f"  {key:12s}: {value:12.4f}")
    
    # 计算比例
    print("\n幅值比例 (生成/真实):")
    ratios = {}
    for key in real_stats.keys():
        if real_stats[key] != 0:
            ratios[key] = gen_stats[key] / real_stats[key]
            print(f"  {key:12s}: {ratios[key]:12.4f} ({ratios[key]*100:.2f}%)")
        else:
            ratios[key] = 0
            print(f"  {key:12s}: N/A")
    
    return real_stats, gen_stats, ratios

def analyze_frequency_amplitude(real_signal, gen_signals, fs=5000, n_fft_multiplier=4):
    """分析频域幅值"""
    print("\n" + "="*70)
    print("频域幅值分析")
    print("="*70)
    
    # 计算FFT
    signal_len = len(real_signal)
    base_fft_size = 2 ** int(np.ceil(np.log2(signal_len)))
    N = base_fft_size * n_fft_multiplier
    
    # 真实信号FFT
    Y_real = fft(real_signal, N)
    f_vec = fs * np.arange(N // 2 + 1) / N
    P_real = np.abs(Y_real[:N // 2 + 1]) / signal_len
    P_real[0] = 0  # 去除直流分量
    
    # 找到主要峰值
    peak_indices = []
    peak_freqs = []
    peak_mags_real = []
    
    # 简单峰值检测：找到前10个最大值
    sorted_indices = np.argsort(P_real)[::-1][:10]
    for idx in sorted(sorted_indices):
        if P_real[idx] > np.max(P_real) * 0.05:  # 至少是最大值的5%
            peak_indices.append(idx)
            peak_freqs.append(f_vec[idx])
            peak_mags_real.append(P_real[idx])
    
    print(f"\n真实信号主要频率峰值 (前{len(peak_freqs)}个):")
    for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags_real)):
        print(f"  峰值 {i+1}: {freq:8.2f} Hz, 幅值: {mag:.6f}")
    
    # 生成信号在相同频率处的幅值
    print(f"\n生成信号在相同频率处的幅值 (平均值):")
    gen_peak_mags = []
    for freq, idx in zip(peak_freqs, peak_indices):
        gen_mags_at_freq = []
        for gen_signal in gen_signals:
            gen_len = len(gen_signal)
            gen_base_fft = 2 ** int(np.ceil(np.log2(gen_len)))
            gen_N = gen_base_fft * n_fft_multiplier
            Y_gen = fft(gen_signal, gen_N)
            gen_f_vec = fs * np.arange(gen_N // 2 + 1) / gen_N
            P_gen = np.abs(Y_gen[:gen_N // 2 + 1]) / gen_len
            P_gen[0] = 0
            
            # 找到最接近的频率
            closest_idx = np.argmin(np.abs(gen_f_vec - freq))
            gen_mags_at_freq.append(P_gen[closest_idx])
        
        avg_gen_mag = np.mean(gen_mags_at_freq)
        gen_peak_mags.append(avg_gen_mag)
        ratio = avg_gen_mag / peak_mags_real[peak_freqs.index(freq)] if peak_mags_real[peak_freqs.index(freq)] > 0 else 0
        print(f"  {freq:8.2f} Hz: 真实={peak_mags_real[peak_freqs.index(freq)]:.6f}, "
              f"生成={avg_gen_mag:.6f}, 比例={ratio:.4f} ({ratio*100:.2f}%)")
    
    return peak_freqs, peak_mags_real, gen_peak_mags

def visualize_comparison(real_signal, gen_signals, indices, real_stats, gen_stats):
    """可视化对比"""
    print("\n生成可视化图表...")
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Amplitude Analysis: Real vs Generated Signals', fontsize=16, fontweight='bold')
    
    # 子图1: 时域信号对比
    ax1 = plt.subplot(3, 2, 1)
    # 只显示前1000个点以便观察
    n_points = min(1000, len(real_signal))
    ax1.plot(real_signal[:n_points], 'b-', linewidth=1.5, label='Real Signal', alpha=0.8)
    for i, (gen_sig, idx) in enumerate(zip(gen_signals[:3], indices[:3])):
        ax1.plot(gen_sig[:min(n_points, len(gen_sig))], 'r--', linewidth=1, 
                alpha=0.6, label=f'Generated {idx}' if i == 0 else '')
    ax1.set_title('Time Domain Comparison (First 1000 points)', fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 幅值分布直方图
    ax2 = plt.subplot(3, 2, 2)
    ax2.hist(real_signal, bins=50, alpha=0.6, label='Real', color='blue', density=True)
    for gen_sig in gen_signals[:5]:
        ax2.hist(gen_sig, bins=50, alpha=0.2, color='red', density=True)
    ax2.hist(gen_signals[0], bins=50, alpha=0.4, label='Generated (sample)', color='red', density=True)
    ax2.set_title('Amplitude Distribution', fontweight='bold')
    ax2.set_xlabel('Amplitude')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 统计量对比
    ax3 = plt.subplot(3, 2, 3)
    stats_names = ['mean', 'std', 'rms', 'peak_to_peak']
    real_vals = [real_stats[s] for s in stats_names]
    gen_vals = [gen_stats[s] for s in stats_names]
    x = np.arange(len(stats_names))
    width = 0.35
    ax3.bar(x - width/2, real_vals, width, label='Real', alpha=0.8, color='blue')
    ax3.bar(x + width/2, gen_vals, width, label='Generated', alpha=0.8, color='red')
    ax3.set_xticks(x)
    ax3.set_xticklabels(stats_names)
    ax3.set_title('Statistical Comparison', fontweight='bold')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 子图4: 频域对比
    ax4 = plt.subplot(3, 2, 4)
    signal_len = len(real_signal)
    base_fft_size = 2 ** int(np.ceil(np.log2(signal_len)))
    N = base_fft_size * 4
    Y_real = fft(real_signal, N)
    fs = 25600  # 采样频率
    f_vec = fs * np.arange(N // 2 + 1) / N
    P_real = np.abs(Y_real[:N // 2 + 1]) / signal_len
    P_real[0] = 0
    ax4.plot(f_vec, P_real, 'b-', linewidth=1.5, label='Real', alpha=0.8)
    
    # 生成信号平均频谱
    P_gen_avg = None
    for gen_sig in gen_signals:
        gen_len = len(gen_sig)
        gen_base_fft = 2 ** int(np.ceil(np.log2(gen_len)))
        gen_N = gen_base_fft * 4
        Y_gen = fft(gen_sig, gen_N)
        gen_f_vec = fs * np.arange(gen_N // 2 + 1) / gen_N
        P_gen = np.abs(Y_gen[:gen_N // 2 + 1]) / gen_len
        P_gen[0] = 0
        
        if P_gen_avg is None:
            P_gen_avg = np.zeros_like(P_gen)
            gen_f_vec_ref = gen_f_vec
        
        # 插值到相同的频率向量
        from scipy.interpolate import interp1d
        if len(gen_f_vec) == len(P_gen):
            interp_func = interp1d(gen_f_vec, P_gen, kind='linear', 
                                  bounds_error=False, fill_value=0)
            P_gen_interp = interp_func(f_vec[:len(P_gen_avg)])
            P_gen_avg[:len(P_gen_interp)] += P_gen_interp
    
    if P_gen_avg is not None:
        P_gen_avg /= len(gen_signals)
        ax4.plot(f_vec, P_gen_avg, 'r--', linewidth=1.5, label='Generated (avg)', alpha=0.8)
    
    ax4.set_title('Frequency Domain Comparison', fontweight='bold')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude')
    ax4.set_xlim([0, min(2000, fs/2)])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 幅值比例
    ax5 = plt.subplot(3, 2, 5)
    ratios = {}
    for key in stats_names:
        if real_stats[key] != 0:
            ratios[key] = gen_stats[key] / real_stats[key]
    ax5.bar(ratios.keys(), ratios.values(), alpha=0.7, color='orange')
    ax5.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Ideal (1.0)')
    ax5.set_title('Amplitude Ratio (Generated/Real)', fontweight='bold')
    ax5.set_ylabel('Ratio')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 子图6: 建议修正方案
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    recommendations = []
    
    avg_ratio = np.mean(list(ratios.values()))
    if avg_ratio < 0.5:
        recommendations.append("1. 生成信号幅值明显偏小")
        recommendations.append("2. 检查是否已正确反归一化")
        recommendations.append("3. 考虑在训练时使用幅值加权损失")
        recommendations.append("4. 检查模型输出层是否需要调整")
    elif avg_ratio < 0.8:
        recommendations.append("1. 生成信号幅值略小")
        recommendations.append("2. 可能需要微调模型")
        recommendations.append("3. 考虑增加训练步数")
    else:
        recommendations.append("1. 幅值基本正常")
        recommendations.append("2. 可以进一步优化")
    
    recommendations.append(f"\n当前平均幅值比例: {avg_ratio:.2%}")
    recommendations.append(f"建议修正系数: {1.0/avg_ratio:.4f}")
    
    ax6.text(0.1, 0.5, '\n'.join(recommendations), 
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('Recommendations', fontweight='bold')
    
    plt.tight_layout()
    output_path = './amplitude_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到: {output_path}")
    plt.close()
    
    return avg_ratio

if __name__ == '__main__':
    print('=' * 70)
    print('生成信号幅值问题诊断')
    print('=' * 70)
    
    # 检查归一化参数
    norm_params = check_normalization()
    
    # 加载数据
    real_signal = load_real_signal()
    gen_signals, indices = load_generated_signals()
    
    if len(gen_signals) == 0:
        raise ValueError("未找到生成的信号文件！")
    
    # 截取相同长度
    min_len = min(len(real_signal), min(len(s) for s in gen_signals))
    real_signal = real_signal[:min_len]
    gen_signals = [s[:min_len] for s in gen_signals]
    
    # 分析
    real_stats, gen_stats, ratios = analyze_amplitude_statistics(real_signal, gen_signals)
    peak_freqs, peak_mags_real, gen_peak_mags = analyze_frequency_amplitude(
        real_signal, gen_signals, fs=25600
    )
    
    # 可视化
    avg_ratio = visualize_comparison(real_signal, gen_signals, indices, real_stats, gen_stats)
    
    # 输出建议
    print("\n" + "="*70)
    print("修正建议")
    print("="*70)
    print(f"\n平均幅值比例: {avg_ratio:.4f} ({avg_ratio*100:.2f}%)")
    print(f"建议修正系数: {1.0/avg_ratio:.4f}")
    print("\n如果生成信号幅值偏小，可以:")
    print("1. 确保生成后已正确反归一化（已修复）")
    print("2. 如果仍偏小，可以在保存时应用修正系数")
    print("3. 重新训练时考虑使用幅值加权的损失函数")
    print("4. 检查模型架构，确保输出层能够学习到正确的幅值范围")
    
    print("\n分析完成！")
