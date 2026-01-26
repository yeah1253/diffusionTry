"""
修正生成信号的幅值问题
基于统计特性（标准差或RMS）来放大生成信号，使其与真实信号匹配
"""
import numpy as np
import os
from scipy.io import loadmat

# ===== 配置参数 =====
REAL_DATA_PATH = r'D:\山东科技大学轴承齿轮数据集\轴承数据集\NC\NC 2000 0.mat'
GENERATED_DIR = './generated_samples'
OUTPUT_DIR = './generated_samples_fixed'

def load_real_signal():
    """加载真实信号用于计算修正系数"""
    print("加载真实信号...")
    data = loadmat(REAL_DATA_PATH)
    
    if 'Signal' in data:
        signal_data = data['Signal']
        if hasattr(signal_data, 'dtype') and signal_data.dtype.names and 'y_values' in signal_data.dtype.names:
            y_values = signal_data['y_values'][0, 0]
            if 'values' in y_values.dtype.names:
                signal = y_values['values'].item()[:, 0]
                print(f"  成功提取，长度: {len(signal)}")
                return signal.flatten().astype(np.float64)
    
    raise ValueError("无法从真实数据中提取信号")

def calculate_correction_factor(real_signal, gen_signals, method='std'):
    """
    计算修正系数
    
    Parameters:
    -----------
    real_signal : array
        真实信号
    gen_signals : list of arrays
        生成信号列表
    method : str
        修正方法: 'std' (标准差), 'rms' (均方根), 'peak_to_peak' (峰峰值)
    
    Returns:
    --------
    correction_factor : float
        修正系数
    """
    if method == 'std':
        real_stat = np.std(real_signal)
        gen_stat = np.mean([np.std(s) for s in gen_signals])
    elif method == 'rms':
        real_stat = np.sqrt(np.mean(real_signal**2))
        gen_stat = np.mean([np.sqrt(np.mean(s**2)) for s in gen_signals])
    elif method == 'peak_to_peak':
        real_stat = np.max(real_signal) - np.min(real_signal)
        gen_stat = np.mean([np.max(s) - np.min(s) for s in gen_signals])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if gen_stat == 0:
        return 1.0
    
    correction_factor = real_stat / gen_stat
    print(f"\n修正方法: {method}")
    print(f"  真实信号 {method}: {real_stat:.4f}")
    print(f"  生成信号 {method} (平均): {gen_stat:.4f}")
    print(f"  修正系数: {correction_factor:.4f}")
    
    return correction_factor

def apply_correction(gen_signal, correction_factor, method='scale'):
    """
    应用修正
    
    Parameters:
    -----------
    gen_signal : array
        生成信号
    correction_factor : float
        修正系数
    method : str
        修正方式: 'scale' (直接缩放), 'scale_centered' (保持均值不变缩放)
    
    Returns:
    --------
    corrected_signal : array
        修正后的信号
    """
    if method == 'scale':
        # 直接缩放
        return gen_signal * correction_factor
    elif method == 'scale_centered':
        # 保持均值不变，只缩放波动部分
        mean = np.mean(gen_signal)
        centered = gen_signal - mean
        return mean + centered * correction_factor
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    print('=' * 70)
    print('生成信号幅值修正')
    print('=' * 70)
    
    # 1. 加载真实信号
    real_signal = load_real_signal()
    
    # 2. 加载生成的信号
    print("\n加载生成的信号...")
    gen_signals = []
    gen_indices = []
    
    for i in range(100):  # 检查更多信号
        file_path = os.path.join(GENERATED_DIR, f'generated_signal_{i}.npy')
        if os.path.exists(file_path):
            signal = np.load(file_path).astype(np.float64)
            gen_signals.append(signal)
            gen_indices.append(i)
    
    if len(gen_signals) == 0:
        raise ValueError("未找到生成的信号文件！")
    
    print(f"  加载了 {len(gen_signals)} 个生成信号")
    
    # 3. 截取相同长度
    min_len = min(len(real_signal), min(len(s) for s in gen_signals))
    real_signal = real_signal[:min_len]
    gen_signals = [s[:min_len] for s in gen_signals]
    
    # 4. 计算修正系数（使用标准差方法，通常最稳定）
    correction_factor = calculate_correction_factor(real_signal, gen_signals, method='std')
    
    # 也可以尝试RMS方法
    correction_factor_rms = calculate_correction_factor(real_signal, gen_signals, method='rms')
    
    # 使用标准差方法（更稳定）
    print(f"\n使用修正系数: {correction_factor:.4f} (基于标准差)")
    
    # 5. 应用修正并保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n应用修正并保存到 {OUTPUT_DIR}...")
    for i, (gen_sig, idx) in enumerate(zip(gen_signals, gen_indices)):
        # 使用 centered 方法保持均值不变
        corrected_sig = apply_correction(gen_sig, correction_factor, method='scale_centered')
        
        output_path = os.path.join(OUTPUT_DIR, f'generated_signal_{i}.npy')
        np.save(output_path, corrected_sig)
        
        if i < 5:  # 只打印前5个
            print(f"  Signal {i} (orig idx={idx}): "
                  f"原始范围=[{gen_sig.min():.4f}, {gen_sig.max():.4f}], "
                  f"修正后范围=[{corrected_sig.min():.4f}, {corrected_sig.max():.4f}]")
    
    # 6. 保存修正参数
    correction_params = {
        'correction_factor': correction_factor,
        'correction_factor_rms': correction_factor_rms,
        'method': 'scale_centered',
        'based_on': 'std'
    }
    np.save(os.path.join(OUTPUT_DIR, 'correction_params.npy'), correction_params)
    
    print(f"\n修正完成！")
    print(f"  修正系数: {correction_factor:.4f}")
    print(f"  修正后的信号已保存到: {OUTPUT_DIR}")
    print(f"  修正参数已保存到: {os.path.join(OUTPUT_DIR, 'correction_params.npy')}")
    
    # 7. 验证修正效果
    print("\n验证修正效果...")
    corrected_stats = {
        'std': np.mean([np.std(apply_correction(s, correction_factor, method='scale_centered')) 
                       for s in gen_signals]),
        'rms': np.mean([np.sqrt(np.mean(apply_correction(s, correction_factor, method='scale_centered')**2)) 
                       for s in gen_signals])
    }
    real_stats = {
        'std': np.std(real_signal),
        'rms': np.sqrt(np.mean(real_signal**2))
    }
    
    print(f"  真实信号 std: {real_stats['std']:.4f}, RMS: {real_stats['rms']:.4f}")
    print(f"  修正后 std: {corrected_stats['std']:.4f} ({corrected_stats['std']/real_stats['std']*100:.2f}%)")
    print(f"  修正后 RMS: {corrected_stats['rms']:.4f} ({corrected_stats['rms']/real_stats['rms']*100:.2f}%)")

if __name__ == '__main__':
    main()
