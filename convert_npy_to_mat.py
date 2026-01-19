"""
将生成的 .npy 信号文件转换为 .mat 文件
"""
import numpy as np
from scipy.io import savemat
import os
import glob

# ===== 配置参数 =====
INPUT_DIR = './generated_samples'  # 输入目录
OUTPUT_DIR = './generated_samples_mat'  # 输出目录
INPUT_FILE = None  # 如果指定，只转换单个文件；如果为None，转换目录下所有.npy文件

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_npy_to_mat(npy_path, output_path=None):
    """
    将单个 .npy 文件转换为 .mat 文件
    
    Parameters:
    -----------
    npy_path : str
        输入的 .npy 文件路径
    output_path : str, optional
        输出的 .mat 文件路径，如果为None则自动生成
    """
    # 加载 .npy 文件
    print(f"正在加载: {npy_path}")
    signal = np.load(npy_path)
    
    print(f"  信号形状: {signal.shape}")
    print(f"  信号范围: [{signal.min():.4f}, {signal.max():.4f}]")
    
    # 确定输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.mat")
    
    # 准备MATLAB变量（使用与原始数据相同的变量名）
    mat_data = {
        'Bearing_Acc_X': signal,  # 使用与原始数据相同的变量名
        'signal_length': len(signal),
        'signal_shape': signal.shape
    }
    
    # 保存为 .mat 文件
    savemat(output_path, mat_data)
    print(f"  已保存到: {output_path}")
    print()

if __name__ == '__main__':
    print("=" * 60)
    print("NPY 转 MAT 文件转换工具")
    print("=" * 60)
    print()
    
    if INPUT_FILE:
        # 转换单个文件
        if os.path.exists(INPUT_FILE):
            output_path = os.path.join(
                OUTPUT_DIR, 
                os.path.splitext(os.path.basename(INPUT_FILE))[0] + '.mat'
            )
            convert_npy_to_mat(INPUT_FILE, output_path)
        else:
            print(f"错误: 文件不存在 - {INPUT_FILE}")
    else:
        # 转换目录下所有 .npy 文件
        npy_files = glob.glob(os.path.join(INPUT_DIR, '*.npy'))
        
        if len(npy_files) == 0:
            print(f"警告: 在 {INPUT_DIR} 中未找到 .npy 文件")
        else:
            print(f"找到 {len(npy_files)} 个 .npy 文件")
            print()
            
            for npy_file in sorted(npy_files):
                convert_npy_to_mat(npy_file)
            
            print("=" * 60)
            print(f"转换完成！所有文件已保存到: {OUTPUT_DIR}")
            print("=" * 60)
