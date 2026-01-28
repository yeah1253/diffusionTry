"""
使用 Variance Frequency (VF) 指标评价生成信号质量
根据论文公式: VF(p, n) = Σ_p (Σ_n |f_p(n) - f̂_p|^2) / (Σ_n f_p(n))
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from scipy.fft import fft
from scipy.signal import find_peaks

# ===== 配置参数 =====
REAL_DATA_PATH = r'D:\NC 2000 40.mat'
REAL_SIGNAL_NAME = 'Bearing_Acc_X'
GENERATED_DIR = './generated_samples_infer'
FS = 25600  # Hz
START_IDX = 1000 # 确保从稳态区域开始取样 (5000之后)
END_IDX = 5000
NUM_PEAKS = 100  # 提取的主要频率峰值数量
FFT_SIZE_MULTIPLIER = 4  # FFT零填充倍数，增加频率分辨率（推荐4-16）

# 辅助函数：从结构体中提取最大的向量
def get_largest_vector(data_dict):
    max_len = 0
    vec = None
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.size > max_len:
            max_len = value.size
            vec = value.flatten()
    if vec is None:
        raise ValueError('无法在真实数据中找到有效数值向量')
    return vec

def extract_dominant_frequencies(signal, fs, num_peaks=10, min_peak_height=None, fft_size_multiplier=8):
    """
    从信号中提取主要频率分量
    
    Parameters:
    -----------
    signal : array
        输入信号
    fs : float
        采样频率
    num_peaks : int
        提取的峰值数量
    min_peak_height : float, optional
        峰值最小高度（相对最大值）
    fft_size_multiplier : int
        FFT零填充倍数，增加频率分辨率
    
    Returns:
    --------
    frequencies : array
        主要频率 (Hz)
    magnitudes : array
        对应的幅值
    """
    # FFT - 使用零填充增加频率分辨率
    signal_len = len(signal)
    # 基础FFT点数（2的幂次）
    base_fft_size = 2 ** int(np.ceil(np.log2(signal_len)))
    # 零填充后的FFT点数
    N = base_fft_size * fft_size_multiplier
    
    Y = fft(signal, N)
    f_vec = fs * np.arange(N // 2 + 1) / N
    P = np.abs(Y[:N // 2 + 1]) / signal_len  # 归一化使用原始信号长度
    
    # 去除直流分量
    P[0] = 0
    
    # 设置峰值检测的最小高度（如果未指定，使用最大值的5%）
    if min_peak_height is None:
        min_peak_height = np.max(P) * 0.05
    
    # 查找峰值
    peaks, properties = find_peaks(P, height=min_peak_height, distance=int(len(P) * 0.01))
    
    # 按幅值排序，选择前num_peaks个
    if len(peaks) > 0:
        peak_magnitudes = P[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[::-1]  # 降序
        top_indices = sorted_indices[:num_peaks]
        
        frequencies = f_vec[peaks[top_indices]]
        magnitudes = P[peaks[top_indices]]
        
        # 按频率排序
        sort_idx = np.argsort(frequencies)
        frequencies = frequencies[sort_idx]
        magnitudes = magnitudes[sort_idx]
    else:
        frequencies = np.array([])
        magnitudes = np.array([])
    
    return frequencies, magnitudes

def calculate_vf(real_signal, generated_signals, fs, num_peaks=10, fft_size_multiplier=8):
    """
    计算 Variance Frequency (VF) 指标
    
    Parameters:
    -----------
    real_signal : array
        真实信号
    generated_signals : list of arrays
        生成的信号列表
    fs : float
        采样频率
    num_peaks : int
        提取的主要频率数量
    
    Returns:
    --------
    vf_value : float
        VF值
    target_freqs : array
        目标频率
    gen_freqs_list : list
        每个生成信号的实际频率列表
    """
    # 从真实信号中提取目标频率 f̂_p
    target_freqs, target_magnitudes = extract_dominant_frequencies(
        real_signal, fs, num_peaks=num_peaks, fft_size_multiplier=fft_size_multiplier
    )
    
    if len(target_freqs) == 0:
        raise ValueError('无法从真实信号中提取频率峰值')
    
    print(f'\n提取到 {len(target_freqs)} 个目标频率:')
    for i, (freq, mag) in enumerate(zip(target_freqs, target_magnitudes)):
        print(f'  频率 {i+1}: {freq:.2f} Hz, 幅值: {mag:.4f}')
    
    # 从每个生成信号中提取实际频率 f_p(n)
    gen_freqs_list = []
    gen_magnitudes_list = []
    
    print(f'\n从 {len(generated_signals)} 个生成信号中提取频率:')
    for i, gen_signal in enumerate(generated_signals):
        gen_freqs, gen_magnitudes = extract_dominant_frequencies(
            gen_signal, fs, num_peaks=num_peaks, fft_size_multiplier=fft_size_multiplier
        )
        gen_freqs_list.append(gen_freqs)
        gen_magnitudes_list.append(gen_magnitudes)
        print(f'  Signal {i}: 提取到 {len(gen_freqs)} 个频率')
    
    # 计算VF值
    # VF(p, n) = Σ_p (Σ_n |f_p(n) - f̂_p|^2) / (Σ_n f_p(n))
    # 根据论文，对于每个频率分量p，计算所有生成信号n的误差方差
    
    vf_components = []
    
    # 对于每个目标频率 p
    for p, target_freq in enumerate(target_freqs):
        numerator_sum = 0.0  # Σ_n |f_p(n) - f̂_p|^2
        denominator_sum = 0.0  # Σ_n f_p(n)
        
        # 对于每个生成信号 n
        for n, gen_freqs in enumerate(gen_freqs_list):
            if len(gen_freqs) > 0:
                # 找到最接近目标频率的实际频率
                freq_diff = np.abs(gen_freqs - target_freq)
                closest_idx = np.argmin(freq_diff)
                actual_freq = gen_freqs[closest_idx]
                
                # 计算 |f_p(n) - f̂_p|^2
                freq_error_sq = (actual_freq - target_freq) ** 2
                
                # 累加分子：频率误差的平方
                numerator_sum += freq_error_sq
                
                # 累加分母：实际频率值
                denominator_sum += actual_freq
            else:
                # 如果没有找到频率，使用目标频率作为惩罚
                freq_error_sq = target_freq ** 2  # 最大误差
                numerator_sum += freq_error_sq
                denominator_sum += target_freq
        
        # 计算当前频率分量p的VF分量
        if denominator_sum > 0:
            vf_component = numerator_sum / denominator_sum
        else:
            vf_component = numerator_sum  # 如果分母为0，只使用分子
        
        vf_components.append(vf_component)
    
    # VF值是所有频率分量p的平均值
    vf_value = np.mean(vf_components) if len(vf_components) > 0 else np.inf
    
    return vf_value, target_freqs, gen_freqs_list, gen_magnitudes_list

# ===== 主程序 =====
if __name__ == '__main__':
    print('=' * 70)
    print('Variance Frequency (VF) 指标计算')
    print('=' * 70)
    
    # 1. 加载真实数据
    print('\n[1] 加载真实数据...')
    try:
        real_struct = loadmat(REAL_DATA_PATH)
        
        # 优先尝试从 Signal.y_values 中提取第一列数据（与训练数据格式一致）
        if 'Signal' in real_struct:
            try:
                signal_data = real_struct['Signal']
                if hasattr(signal_data, 'dtype') and signal_data.dtype.names and 'y_values' in signal_data.dtype.names:
                    # 提取 y_values 的第一列
                    y_values = signal_data['y_values'][0, 0]
                    if 'values' in y_values.dtype.names:
                        real_signal = y_values['values'].item()[:, 0]  # 第一列
                        print('成功从 Signal.y_values 中提取第一列数据')
                    else:
                        raise ValueError('Signal.y_values 中未找到 values 字段')
                else:
                    raise ValueError('Signal 结构不符合预期格式')
            except Exception as e:
                print(f'从 Signal.y_values 提取失败: {e}')
                print('尝试备用方法...')
                # 备用方法：尝试直接使用 REAL_SIGNAL_NAME
                if REAL_SIGNAL_NAME in real_struct:
                    real_signal = real_struct[REAL_SIGNAL_NAME]
                    print(f'成功从真实数据中提取 {REAL_SIGNAL_NAME}')
                else:
                    print('提示：真实数据中未找到同名变量，尝试自动提取主信号...')
                    real_signal = get_largest_vector(real_struct)
        elif REAL_SIGNAL_NAME in real_struct:
            real_signal = real_struct[REAL_SIGNAL_NAME]
            print(f'成功从真实数据中提取 {REAL_SIGNAL_NAME}')
        else:
            print('提示：真实数据中未找到 Signal 或指定变量，尝试自动提取主信号...')
            real_signal = get_largest_vector(real_struct)
        
        real_signal = real_signal.flatten().astype(np.float64)
        print(f'真实信号长度: {len(real_signal)}')
        print(f'真实信号范围: [{real_signal.min():.4f}, {real_signal.max():.4f}]')
    except Exception as e:
        print(f'加载真实数据失败: {e}')
        import traceback
        traceback.print_exc()
        raise
    
    # 2. 加载生成的信号
    print('\n[2] 加载生成的信号...')
    generated_signals = []
    generated_indices = []
    
    for i in range(9):
        file_path = os.path.join(GENERATED_DIR, f'infer_signal_{i}.npy')
        if os.path.exists(file_path):
            signal = np.load(file_path).astype(np.float64)
            generated_signals.append(signal)
            generated_indices.append(i)
            print(f'成功加载 Signal {i}: shape={signal.shape}')
    
    if len(generated_signals) == 0:
        raise ValueError('未找到生成的信号文件！')
    
    # 3. 数据截取
    print(f'\n[3] 截取数据：第 {START_IDX} 点 到 第 {END_IDX} 点...')
    
    if len(real_signal) < END_IDX:
        print(f'警告：真实数据长度 ({len(real_signal)}) 小于 END_IDX ({END_IDX})')
        real_signal_sliced = real_signal[START_IDX-1:] if START_IDX <= len(real_signal) else real_signal
    else:
        real_signal_sliced = real_signal[START_IDX-1:END_IDX]
    
    comparison_length = min(len(real_signal_sliced), min(len(s) for s in generated_signals))
    print(f'比较长度: {comparison_length} 点')
    
    real_signal_sliced = real_signal_sliced[:comparison_length]
    generated_signals_sliced = [sig[:comparison_length] for sig in generated_signals]
    
    # 4. 计算VF指标
    signal_len = len(real_signal_sliced)
    base_fft_size = 2 ** int(np.ceil(np.log2(signal_len)))
    actual_fft_size = base_fft_size * FFT_SIZE_MULTIPLIER
    freq_resolution = FS / actual_fft_size
    
    print(f'\n[4] 计算 VF 指标 (提取 {NUM_PEAKS} 个主要频率)...')
    print(f'  FFT配置:')
    print(f'    - 信号长度: {signal_len} 点')
    print(f'    - 基础FFT点数: {base_fft_size}')
    print(f'    - 零填充后FFT点数: {actual_fft_size} (倍数: {FFT_SIZE_MULTIPLIER}x)')
    print(f'    - 频率分辨率: {freq_resolution:.4f} Hz')
    print(f'    - 最大观测频率: {FS/2:.1f} Hz')
    
    try:
        vf_value, target_freqs, gen_freqs_list, gen_magnitudes_list = calculate_vf(
            real_signal_sliced, generated_signals_sliced, FS, 
            num_peaks=NUM_PEAKS, fft_size_multiplier=FFT_SIZE_MULTIPLIER
        )

        print('\n' + '=' * 70)
        print('VF 指标结果')
        print('=' * 70)
        print(f'VF值: {vf_value:.6f}')

        
    except Exception as e:
        print(f'计算VF指标失败: {e}')
        import traceback
        traceback.print_exc()
        raise
    
    # 5. 可视化
    print('\n[5] 生成可视化图表...')
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Variance Frequency (VF) Analysis (VF = {vf_value:.6f})', 
                 fontsize=16, fontweight='bold')
    
    # 子图1: 真实信号的频谱
    ax1 = plt.subplot(3, 2, 1)
    # 使用相同的FFT配置
    signal_len = len(real_signal_sliced)
    base_fft_size = 2 ** int(np.ceil(np.log2(signal_len)))
    N = base_fft_size * FFT_SIZE_MULTIPLIER
    Y_real = fft(real_signal_sliced, N)
    f_vec = FS * np.arange(N // 2 + 1) / N
    P_real = np.abs(Y_real[:N // 2 + 1]) / signal_len
    P_real[0] = 0
    
    ax1.plot(f_vec, P_real, 'b-', linewidth=1, label='Real Signal')
    ax1.scatter(target_freqs, 
                [P_real[np.argmin(np.abs(f_vec - f))] for f in target_freqs],
                color='red', s=100, marker='*', zorder=5, label='Target Frequencies')
    ax1.set_title('Real Signal Spectrum (Target Frequencies)', fontweight='bold')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, min(2000, FS/2)])
    
    # 子图2-5: 每个生成信号的频谱对比
    for idx, (gen_signal, gen_freqs, gen_magnitudes, sig_idx) in enumerate(
        zip(generated_signals_sliced, gen_freqs_list, gen_magnitudes_list, generated_indices)
    ):
        if idx >= 4:  # 只显示前4个
            break
        
        ax = plt.subplot(3, 2, idx + 2)
        # 使用相同的FFT配置
        gen_signal_len = len(gen_signal)
        gen_base_fft_size = 2 ** int(np.ceil(np.log2(gen_signal_len)))
        gen_N = gen_base_fft_size * FFT_SIZE_MULTIPLIER
        Y_gen = fft(gen_signal, gen_N)
        gen_f_vec = FS * np.arange(gen_N // 2 + 1) / gen_N
        P_gen = np.abs(Y_gen[:gen_N // 2 + 1]) / gen_signal_len
        P_gen[0] = 0
        # 使用相同的频率向量
        gen_f_vec = f_vec  # 使用相同的f_vec确保一致性
        
        ax.plot(gen_f_vec, P_gen, 'r--', linewidth=1, alpha=0.7, label='Generated')
        ax.plot(f_vec, P_real, 'b-', linewidth=1, alpha=0.5, label='Real')
        
        if len(gen_freqs) > 0:
            ax.scatter(gen_freqs, gen_magnitudes,
                      color='orange', s=80, marker='o', zorder=5, 
                      label='Extracted Frequencies', alpha=0.7)
        
        ax.scatter(target_freqs, 
                  [P_real[np.argmin(np.abs(f_vec - f))] for f in target_freqs],
                  color='red', s=80, marker='*', zorder=5, 
                  label='Target Frequencies', alpha=0.5)
        
        ax.set_title(f'Signal {sig_idx} Spectrum', fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(2000, FS/2)])
    
    # 子图6: 频率误差对比
    ax6 = plt.subplot(3, 2, 6)
    freq_errors = []
    signal_labels = []
    
    for idx, (gen_freqs, sig_idx) in enumerate(zip(gen_freqs_list, generated_indices)):
        if len(gen_freqs) > 0 and len(target_freqs) > 0:
            # 计算每个目标频率的平均误差
            errors = []
            for target_freq in target_freqs:
                freq_diff = np.abs(gen_freqs - target_freq)
                min_diff = np.min(freq_diff)
                errors.append(min_diff)
            avg_error = np.mean(errors)
            freq_errors.append(avg_error)
            signal_labels.append(f'Sig {sig_idx}')
    
    if len(freq_errors) > 0:
        ax6.bar(range(len(freq_errors)), freq_errors, alpha=0.7)
        ax6.set_xticks(range(len(signal_labels)))
        ax6.set_xticklabels(signal_labels)
        ax6.set_title('Average Frequency Error by Signal', fontweight='bold')
        ax6.set_ylabel('Frequency Error (Hz)')
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = './vf_analysis10000.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'可视化图表已保存到: {output_path}')
    plt.show()
    
    print('\n' + '=' * 70)
    print('分析完成！')
    print('=' * 70)
