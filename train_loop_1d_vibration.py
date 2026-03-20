"""
train_loop_1d_vibration.py — 循环训练十种故障类型的扩散模型

对每种故障类型单独训练，训练完成后将最后一个 checkpoint 复制到
checkpoints_by_fault/{故障类别}.pt，便于后续按故障类型加载推理。

真实数据目录结构（如图所示）：
  REAL_DATA_ROOT/
    IF0.2/
    IF0.4/
    IF0.6/
    NC/
    OF0.2/
    OF0.4/
    OF0.6/
    RF0.2/
    RF0.4/
    RF0.6/

每个子目录内需包含 .mat 文件（与 train_1d_vibration.py 一致）。
"""

import os
import glob
import shutil

from train_1d_vibration import (
    RealSDUSTDataset,
    FAULT_TYPE_MAP,
)
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import (
    PhysiNet,
    GaussianDiffusion1D,
    Trainer1D,
)


def find_latest_checkpoint(results_folder: str) -> str:
    """在 results_folder 中查找最新的 model-*.pt 文件。"""
    pattern = os.path.join(results_folder, "model-*.pt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No checkpoint found in {results_folder}")

    def get_milestone(p: str) -> int:
        base = os.path.basename(p)
        name, _ = os.path.splitext(base)
        parts = name.split("-")
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            return -1

    paths = sorted(paths, key=get_milestone)
    return paths[-1]


def train_one_fault(
    fault_key: str,
    data_path: str,
    results_folder: str,
    seq_length: int = 1024,
    overlap: float = 0.5,
    use_condition: bool = True,
    use_phys_signal: bool = True,
    train_num_steps: int = 3000,
    save_and_sample_every: int = 150,
    train_batch_size: int = 48,
    train_lr: float = 4e-5,
    gradient_accumulate_every: int = 2,
) -> str:
    """
    对单个故障类型进行训练，返回最后一个 checkpoint 路径。
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  训练故障类型: {fault_key}")
    print(f"  数据路径: {data_path}")
    print(f"  结果目录: {results_folder}")
    print(f"{'='*60}\n")

    dataset = RealSDUSTDataset(
        data_path=data_path,
        seq_length=seq_length,
        overlap=overlap,
        use_condition=use_condition,
        use_phys_signal=use_phys_signal,
    )
    print(f"Dataset size: {len(dataset)}")

    CHANNELS = 1
    COND_DIM = dataset.cond_dim

    model = PhysiNet(
        dim=128,
        dim_mults=(1, 2, 4, 8),
        channels=CHANNELS,
        cond_dim=COND_DIM,
        dropout=0.1,
        attn_dim_head=64,
        attn_heads=8,
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=seq_length,
        timesteps=1000,
        objective="pred_v",
        auto_normalize=False,
    )

    trainer = Trainer1D(
        diffusion,
        dataset=dataset,
        train_batch_size=train_batch_size,
        train_lr=train_lr,
        train_num_steps=train_num_steps,
        gradient_accumulate_every=gradient_accumulate_every,
        ema_decay=0.9995,
        amp=True,
        save_and_sample_every=save_and_sample_every,
        num_samples=16,
        results_folder=results_folder,
        denorm_min=dataset.signal_min,
        denorm_max=dataset.signal_max,
    )

    trainer.train()
    latest_ckpt = find_latest_checkpoint(results_folder)
    print(f"  训练完成，最后 checkpoint: {latest_ckpt}")
    return latest_ckpt


def main():
    # 真实数据根目录（含 10 个故障子文件夹）
    REAL_DATA_ROOT = r"D:\data\轴承数据集"
    # 每个故障类型的训练结果临时目录
    RESULTS_BASE = "./results_vibration"
    # 最终保存目录：每个故障的最后一个 .pt 按故障名保存
    CHECKPOINTS_BY_FAULT_DIR = "./checkpoints_by_fault"

    # 10 种故障类型（与目录名一致，无空格）
    FAULT_KEYS = [
        "IF0.2", "IF0.4", "IF0.6",
        "NC",
        "OF0.2", "OF0.4", "OF0.6",
        "RF0.2", "RF0.4", "RF0.6",
    ]

    os.makedirs(CHECKPOINTS_BY_FAULT_DIR, exist_ok=True)

    for i, fault_key in enumerate(FAULT_KEYS):
        data_path = os.path.join(REAL_DATA_ROOT, fault_key)
        if not os.path.isdir(data_path):
            print(f"[{i+1}/10] 跳过 {fault_key}：目录不存在 {data_path}")
            continue

        results_folder = os.path.join(RESULTS_BASE, fault_key)
        os.makedirs(results_folder, exist_ok=True)

        try:
            latest_ckpt = train_one_fault(
                fault_key=fault_key,
                data_path=data_path,
                results_folder=results_folder,
            )
            dest_path = os.path.join(CHECKPOINTS_BY_FAULT_DIR, f"{fault_key}.pt")
            shutil.copy2(latest_ckpt, dest_path)
            print(f"  已复制到 {dest_path}\n")
        except Exception as e:
            print(f"  训练 {fault_key} 失败: {e}\n")
            raise

    print(f"\n{'='*60}")
    print(f"  全部完成！各故障模型已保存至: {CHECKPOINTS_BY_FAULT_DIR}")
    print(f"{'='*60}")
    for f in sorted(os.listdir(CHECKPOINTS_BY_FAULT_DIR)):
        if f.endswith(".pt"):
            p = os.path.join(CHECKPOINTS_BY_FAULT_DIR, f)
            size_mb = os.path.getsize(p) / (1024 * 1024)
            print(f"    {f}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
