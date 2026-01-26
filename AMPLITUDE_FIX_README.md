# 生成信号幅值修正方案

## 问题诊断

根据分析结果，生成信号存在以下问题：

1. **幅值偏小**：生成信号的幅值只有真实信号的 10-15%
   - 标准差比例：11.94%
   - RMS 比例：12.27%
   - 频域幅值比例：3-6%

2. **原因分析**：
   - 扩散模型在采样过程中可能没有充分利用 [-1, 1] 的完整范围
   - 模型输出的动态范围被压缩
   - 虽然反归一化公式正确，但模型输出本身幅值就偏小

## 解决方案

### 方案1：在训练脚本中自动修正（推荐）

已修改 `train_1d_vibration.py`，在生成信号时自动应用幅值修正：

1. **修正方法**：基于训练数据的标准差计算修正系数
2. **修正公式**：
   ```python
   correction_factor = train_std / gen_std
   corrected_signal = mean + (signal - mean) * correction_factor
   ```
3. **优点**：
   - 保持信号的均值不变
   - 只缩放波动部分（标准差）
   - 自动应用，无需额外步骤

### 方案2：使用独立修正脚本

如果已经生成了信号，可以使用 `fix_amplitude.py` 进行后处理：

```bash
python fix_amplitude.py
```

该脚本会：
1. 加载真实信号和生成信号
2. 计算修正系数（基于标准差或RMS）
3. 应用修正并保存到 `./generated_samples_fixed/`

## 使用方法

### 方法1：重新生成信号（推荐）

直接运行训练脚本，修正会自动应用：

```bash
python train_1d_vibration.py
```

生成的信號會自動應用幅值修正。

### 方法2：修正已生成的信号

如果已有生成信号，运行修正脚本：

```bash
python fix_amplitude.py
```

修正后的信号会保存到 `./generated_samples_fixed/` 目录。

## 验证修正效果

运行分析脚本查看修正效果：

```bash
python analyze_amplitude_issue.py
```

检查：
- 标准差比例是否接近 100%
- RMS 比例是否接近 100%
- 频域幅值是否提升

## 技术细节

### 修正系数计算

```python
# 基于标准差（推荐，更稳定）
correction_factor = real_std / gen_std

# 或基于RMS
correction_factor = real_rms / gen_rms
```

### 修正应用方式

1. **直接缩放**（可能改变均值）：
   ```python
   corrected = signal * correction_factor
   ```

2. **保持均值缩放**（推荐）：
   ```python
   mean = np.mean(signal)
   centered = signal - mean
   corrected = mean + centered * correction_factor
   ```

## 注意事项

1. **修正系数**：通常为 7-10 倍（根据实际数据而定）
2. **保持均值**：使用 `scale_centered` 方法保持信号的直流分量
3. **验证**：修正后应检查信号是否在合理范围内

## 未来改进

1. **训练时改进**：
   - 使用幅值加权的损失函数
   - 在损失函数中强调高频分量的幅值
   - 调整模型架构以更好地学习幅值范围

2. **采样时改进**：
   - 调整扩散过程的噪声调度
   - 使用更激进的采样策略

3. **后处理改进**：
   - 基于频域的幅值修正
   - 自适应修正系数（不同频率使用不同系数）
