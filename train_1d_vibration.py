import torch
import numpy as np
import os
# from scipy.io import loadmat  # no longer needed for synthetic dataset

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

# Synthetic SDUST-like dataset implementation
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

    # ---------- dataset configuration (synthetic SDUST) ----------
    SEQ_LENGTH = 1024
    FS = 25600.0
    N_SAMPLES = 2048

    print("Creating synthetic SDUST-like dataset...")
    dataset = SyntheticSDUSTDataset(n_samples=N_SAMPLES, seq_length=SEQ_LENGTH, fs=FS, seed=42)
    print(f"Dataset size: {len(dataset)}, seq_length={SEQ_LENGTH}, fs={FS}")

    # ---------- model and diffusion (enable cond_dim=2) ----------
    CHANNELS = 1
    COND_DIM = 2

    print("Building PhysiNet (cond_dim=2)...")
    model = PhysiNet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = CHANNELS,
        cond_dim = COND_DIM,
        dropout = 0.0,
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = SEQ_LENGTH,
        timesteps = 1000,
        objective = 'pred_v',
        auto_normalize = False,
    )

    # ---------- trainer (small default steps for quick verification) ----------
    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 10000,        # demo default; increase for full training
        gradient_accumulate_every = 1,
        ema_decay = 0.995,
        amp = False,                  # disable AMP for small demo stability
        save_and_sample_every = 1000,
        num_samples = 16,
        results_folder = './results_vibration',
    )

    print("Starting training on synthetic dataset...")
    trainer.train()
    print("Training finished.")

    # ---------- physical-consistency check: sample conditioned on two RPMs ----------
    def norm_cond(rpm, load):
        return np.array([(rpm - 1000.0) / 2000.0, load / 60.0], dtype=np.float32)

    condA = norm_cond(1000.0, 30.0)  # low rpm
    condB = norm_cond(3000.0, 30.0)  # high rpm

    batch_size = 4
    condA_batch = torch.tensor([condA] * batch_size, dtype=torch.float32)
    condB_batch = torch.tensor([condB] * batch_size, dtype=torch.float32)

    print("Sampling conditioned signals for physical-consistency check...")
    try:
        samplesA = trainer.sample_with_condition(batch_size=batch_size, cond=condA_batch)
        samplesB = trainer.sample_with_condition(batch_size=batch_size, cond=condB_batch)
    except Exception as e:
        print("trainer.sample_with_condition failed, falling back to diffusion.sample:", e)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        samplesA = diffusion.sample(batch_size=batch_size, model_forward_kwargs={'cond': condA_batch.to(device)})
        samplesB = diffusion.sample(batch_size=batch_size, model_forward_kwargs={'cond': condB_batch.to(device)})

    samplesA = samplesA.detach().cpu().numpy()
    samplesB = samplesB.detach().cpu().numpy()

    sigA = samplesA[0, 0, :]
    sigB = samplesB[0, 0, :]


    def compute_fft(sig, fs, n_fft=8192):  # 补0到 8192 点
        L = sig.shape[-1]
        # n=n_fft 表示进行 FFT 时使用的点数，不足的部分 numpy 会自动补 0
        fft_vals = np.fft.rfft(sig, n=n_fft)

        # 计算对应的频率轴，注意这里要用 n_fft
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

        mag = np.abs(fft_vals)
        return freqs, mag

    freqsA, magA = compute_fft(sigA, FS)
    freqsB, magB = compute_fft(sigB, FS)

    def dominant_peak(freqs, mag, fmin=1.0):
        mask = freqs > fmin
        idx = np.argmax(mag[mask])
        masked_idxs = np.nonzero(mask)[0]
        return freqs[masked_idxs[idx]], mag[masked_idxs[idx]]

    peakA_freq, peakA_mag = dominant_peak(freqsA, magA)
    peakB_freq, peakB_mag = dominant_peak(freqsB, magB)

    print(f"Dominant peak A (1000 RPM): {peakA_freq:.2f} Hz, mag {peakA_mag:.3f}")
    print(f"Dominant peak B (3000 RPM): {peakB_freq:.2f} Hz, mag {peakB_mag:.3f}")
    if peakA_freq > 0:
        print(f"Frequency ratio B/A: {peakB_freq/peakA_freq:.3f} (expect ~3.0)")

    # plot FFT comparison
    plt.figure(figsize=(10, 6))
    plt.plot(freqsA, magA / (magA.max() + 1e-12), label='Cond A (1000 RPM)', alpha=0.8)
    plt.plot(freqsB, magB / (magB.max() + 1e-12), label='Cond B (3000 RPM)', alpha=0.8)
    plt.xlim(0, 500)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized magnitude')
    plt.title('FFT: Cond A (1000 RPM) vs Cond B (3000 RPM)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('./results_vibration', exist_ok=True)
    plt.savefig('./results_vibration/fft_condition_comparison.png', dpi=200)
    plt.close()

    print("Saved FFT comparison to './results_vibration/fft_condition_comparison.png'")

    # ---------- generate larger batch and run UCFilter as before ----------
    print("Generating bulk samples and running UCFilter...")
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

    filtered_folder = './generated_samples'
    os.makedirs(filtered_folder, exist_ok=True)
    np.save(os.path.join(filtered_folder, 'selected_idx.npy'), selected_idx_np)
    try:
        kl_np = kl_scores.numpy()
    except Exception:
        kl_np = np.array(kl_scores)
    np.save(os.path.join(filtered_folder, 'kl_scores.npy'), kl_np)

    for i, (idx, gen_signal) in enumerate(zip(selected_idx_np, sampled_filtered)):
        np.save(os.path.join(filtered_folder, f'generated_signal_{i}.npy'), gen_signal)
        print(f"Saved UCFilter-selected generated signal {i} (orig idx={idx}) to {os.path.join(filtered_folder, f'generated_signal_{i}.npy')}")

    print("\nScript completed.")
