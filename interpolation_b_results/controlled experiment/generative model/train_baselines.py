"""
第一阶段：微观生成工具对比（工况条件版）

重构目标：
1) 条件由“类别”改为“工况 (rpm, load)”，对齐扩散模型 cond=[norm_rpm, norm_load] 思想。
2) 每个故障类别单独训练一套 CVAE 与 WGAN-GP。
3) 生成结果按 class_i/<load> <rpm>/sample_xxxxx.npy 保存，便于评估侧按工况解析。

运行:

    cd ".../diffusionTry_cond/controlled experiment/generative model"
    python train_baselines.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _THIS_DIR.parent.parent
_DIFFTRY_ROOT = _WORKSPACE_ROOT / "diffusionTry"

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_DIFFTRY_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIFFTRY_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset

from dataset import BearingSignalDataset

# ---------------------------------------------------------------------------
# 全局超参数
# ---------------------------------------------------------------------------
SEQ_LENGTH = 1024
LATENT_DIM = 128
BATCH_SIZE = 64
EPOCHS = 120

# 若 True，则每个类别在每个工况上生成数量=该类别真实训练中该工况样本数
GENERATE_MATCH_REAL_COUNTS = True
# 若 False，使用每工况固定生成条数
GEN_SAMPLES_PER_GROUP = 40

REAL_DATA_PATH = os.environ.get("REAL_DATA_PATH", r"D:\data\轴承数据集")
PROJECT_ROOT = _THIS_DIR

# 工况归一化范围（与扩散脚本一致）
RPM_MIN, RPM_MAX = 1000.0, 3000.0
LOAD_MIN, LOAD_MAX = 0.0, 60.0


# =============================================================================
# 数据准备：每类单独数据集，条件为 [norm_rpm, norm_load]
# =============================================================================


def normalize_condition(load: np.ndarray, rpm: np.ndarray) -> np.ndarray:
    """输入 load/rpm，输出 [norm_rpm, norm_load]。"""
    norm_rpm = (rpm - RPM_MIN) / max(RPM_MAX - RPM_MIN, 1e-8)
    norm_load = (load - LOAD_MIN) / max(LOAD_MAX - LOAD_MIN, 1e-8)
    cond = np.stack([norm_rpm, norm_load], axis=1).astype(np.float32)
    return np.clip(cond, 0.0, 1.0)


class SignalConditionDataset(Dataset):
    """单类别数据集：返回 (signal, cond)；signal 采用实例级 z-score。"""

    def __init__(self, signals: np.ndarray, conds: np.ndarray) -> None:
        assert signals.ndim == 3 and signals.shape[1] == 1, f"signals shape 应为 (N,1,L)，实际 {signals.shape}"
        assert conds.ndim == 2 and conds.shape[1] == 2, f"conds shape 应为 (N,2)，实际 {conds.shape}"
        assert len(signals) == len(conds), "signals 与 conds 数量不一致"
        self.signals = signals.astype(np.float32)
        self.conds = conds.astype(np.float32)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int):
        sig = torch.from_numpy(self.signals[idx]).clone()
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        cond = torch.from_numpy(self.conds[idx])
        return sig, cond


# =============================================================================
# 1D-CVAE（工况条件）
# =============================================================================


class CVAEEncoder(nn.Module):
    """输入 (B,1,L) 与 cond (B,2)，将 cond 投影到长度 L 后与信号拼接。"""

    def __init__(self, latent_dim: int, seq_len: int = SEQ_LENGTH) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.cond_proj = nn.Linear(2, seq_len)
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        flat_dim = 256 * 64
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cond_proj(cond).unsqueeze(1)
        h = torch.cat([x, c], dim=1)
        h = self.conv(h).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class CVAEDecoder(nn.Module):
    """z 与 cond 向量拼接后反卷积生成 (B,1,1024)。"""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        cond_dim = 64
        self.cond_mlp = nn.Sequential(
            nn.Linear(2, cond_dim),
            nn.ReLU(True),
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(latent_dim + cond_dim, 256 * 64)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        ec = self.cond_mlp(cond)
        h = torch.cat([z, ec], dim=1)
        h = self.fc(h).view(z.size(0), 256, 64)
        return self.deconv(h)


class CVAE(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = CVAEEncoder(latent_dim)
        self.decoder = CVAEDecoder(latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, cond)
        return recon, mu, logvar


# =============================================================================
# 1D-WGAN-GP（工况条件）
# =============================================================================


class WGANGenerator(nn.Module):
    """z 与 cond 向量拼接后生成 (B,1,1024)。"""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        cond_dim = 64
        self.cond_mlp = nn.Sequential(
            nn.Linear(2, cond_dim),
            nn.ReLU(True),
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(latent_dim + cond_dim, 256 * 64)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        ec = self.cond_mlp(cond)
        h = torch.cat([z, ec], dim=1)
        h = self.fc(h).view(z.size(0), 256, 64)
        return self.deconv(h)


class WGANCritic(nn.Module):
    """输入信号 + cond 映射，不使用 BN。"""

    def __init__(self, seq_len: int = SEQ_LENGTH) -> None:
        super().__init__()
        self.cond_proj = nn.Linear(2, seq_len)
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = nn.Linear(256 * 64, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = self.cond_proj(cond).unsqueeze(1)
        h = torch.cat([x, c], dim=1)
        h = self.conv(h).flatten(1)
        return self.out(h).squeeze(-1)


def calc_gradient_penalty(
    netD: WGANCritic,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    cond: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    b = real_data.size(0)
    device = real_data.device
    alpha = torch.rand(b, 1, 1, device=device, dtype=real_data.dtype)
    interpolates = (alpha * real_data + (1.0 - alpha) * fake_data).detach()
    interpolates.requires_grad_(True)
    d_interp = netD(interpolates, cond)
    grad_out = torch.ones_like(d_interp, device=device, dtype=d_interp.dtype)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolates,
        grad_outputs=grad_out,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = grads.flatten(1).norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1.0) ** 2).mean()


# =============================================================================
# 工具函数
# =============================================================================


def cvae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


def _collect_class_names(root: str) -> List[str]:
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def _load_full_dataset(class_names: List[str]) -> BearingSignalDataset:
    return BearingSignalDataset.from_class_folders_with_subsample(
        REAL_DATA_PATH,
        class_names=class_names,
        max_per_class=1000,
        max_per_group=100,
        balance_groups=True,
        groups_per_class=None,
        overlap=0.5,
    )


def build_class_dataset(
    full_ds: BearingSignalDataset,
    class_idx: int,
) -> tuple[SignalConditionDataset, Dict[Tuple[int, int], int]]:
    """抽取单类别数据，并返回该类每个工况的真实样本计数。"""
    labels = full_ds.labels.numpy()
    groups = full_ds.group_labels
    if groups is None:
        raise ValueError("group_labels 为空，无法按工况建模")

    idx = np.where(labels == class_idx)[0]
    if len(idx) == 0:
        raise ValueError(f"类别 {class_idx} 无样本")

    sig = full_ds.signals.numpy()[idx]
    grp = groups[idx]  # [load, rpm]

    # 丢弃工况未知样本，确保条件语义纯净
    known_mask = (grp[:, 0] >= 0) & (grp[:, 1] >= 0)
    sig = sig[known_mask]
    grp = grp[known_mask]
    if len(sig) == 0:
        raise ValueError(f"类别 {class_idx} 无可解析工况样本")

    load = grp[:, 0].astype(np.float32)
    rpm = grp[:, 1].astype(np.float32)
    conds = normalize_condition(load, rpm)

    group_counts: Dict[Tuple[int, int], int] = {}
    for g in grp:
        key = (int(g[0]), int(g[1]))
        group_counts[key] = group_counts.get(key, 0) + 1

    return SignalConditionDataset(sig, conds), group_counts


def save_generated_by_group(
    samples: torch.Tensor,
    group_keys: List[Tuple[int, int]],
    out_class_root: Path,
) -> None:
    """按 class_i/load rpm 子目录保存。"""
    arr = samples.detach().cpu().numpy().astype(np.float32)
    for i, (load, rpm) in enumerate(group_keys):
        subdir = out_class_root / f"{int(load)} {int(rpm)}"
        subdir.mkdir(parents=True, exist_ok=True)
        np.save(subdir / f"sample_{i:05d}.npy", arr[i])


# =============================================================================
# 按类别训练并生成
# =============================================================================


def train_and_generate_cvae_per_class(
    class_idx: int,
    ds_cls: SignalConditionDataset,
    group_counts: Dict[Tuple[int, int], int],
    device: torch.device,
) -> None:
    loader = DataLoader(ds_cls, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = CVAE(LATENT_DIM).to(device)
    opt = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        ep_loss = 0.0
        ep_recon = 0.0
        ep_kl = 0.0
        n_batch = 0

        # KL 退火：前 40 轮线性升到 0.05
        beta = min(0.05, 0.05 * epoch / 40.0)

        for x, cond in loader:
            x = x.to(device)
            cond = cond.to(device)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x, cond)
            loss, r, k = cvae_loss(recon, x, mu, logvar, beta=beta)
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            ep_recon += r.item()
            ep_kl += k.item()
            n_batch += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  [CVAE][class_{class_idx}] Epoch {epoch}/{EPOCHS} "
                f"loss={ep_loss/max(n_batch,1):.4f} mse={ep_recon/max(n_batch,1):.4f} "
                f"kl={ep_kl/max(n_batch,1):.4f} beta={beta:.4f}"
            )

    model.eval()
    out_class_root = PROJECT_ROOT / "generated_samples_cvae" / f"class_{class_idx}"

    all_fake: List[torch.Tensor] = []
    all_keys: List[Tuple[int, int]] = []
    rng = torch.Generator(device=device)
    rng.manual_seed(1000 + class_idx)

    with torch.no_grad():
        for (load, rpm), n_real in sorted(group_counts.items()):
            n_gen = n_real if GENERATE_MATCH_REAL_COUNTS else GEN_SAMPLES_PER_GROUP
            cond_np = normalize_condition(
                np.full((n_gen,), float(load), dtype=np.float32),
                np.full((n_gen,), float(rpm), dtype=np.float32),
            )
            cond = torch.from_numpy(cond_np).to(device)
            z = torch.randn(n_gen, LATENT_DIM, device=device, generator=rng)
            fake = model.decoder(z, cond)
            all_fake.append(fake)
            all_keys.extend([(load, rpm)] * n_gen)

    fake_cat = torch.cat(all_fake, dim=0)
    save_generated_by_group(fake_cat, all_keys, out_class_root)
    print(f"[CVAE] class_{class_idx} 生成完成: {len(all_keys)} 条 -> {out_class_root}")


def train_and_generate_wgan_per_class(
    class_idx: int,
    ds_cls: SignalConditionDataset,
    group_counts: Dict[Tuple[int, int], int],
    device: torch.device,
) -> None:
    loader = DataLoader(ds_cls, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    G = WGANGenerator(LATENT_DIM).to(device)
    D = WGANCritic().to(device)
    opt_g = Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_d = Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    n_critic = 3
    G.train()
    D.train()

    for epoch in range(1, EPOCHS + 1):
        ep_d = 0.0
        ep_g = 0.0
        ep_gp = 0.0
        n_g = 0
        n_d = 0

        for x_real, cond in loader:
            x_real = x_real.to(device)
            cond = cond.to(device)
            b = x_real.size(0)

            for _ in range(n_critic):
                z = torch.randn(b, LATENT_DIM, device=device)
                with torch.no_grad():
                    x_fake = G(z, cond)
                opt_d.zero_grad(set_to_none=True)
                d_real = D(x_real, cond)
                d_fake = D(x_fake.detach(), cond)
                gp = calc_gradient_penalty(D, x_real, x_fake.detach(), cond, lambda_gp=10.0)
                loss_d = d_fake.mean() - d_real.mean() + gp
                loss_d.backward()
                opt_d.step()

                ep_d += loss_d.item()
                ep_gp += gp.item()
                n_d += 1

            z = torch.randn(b, LATENT_DIM, device=device)
            opt_g.zero_grad(set_to_none=True)
            x_gen = G(z, cond)
            loss_g = -D(x_gen, cond).mean()
            loss_g.backward()
            opt_g.step()

            ep_g += loss_g.item()
            n_g += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  [WGAN][class_{class_idx}] Epoch {epoch}/{EPOCHS} "
                f"d_loss={ep_d/max(n_d,1):.4f} g_loss={ep_g/max(n_g,1):.4f} gp={ep_gp/max(n_d,1):.4f}"
            )

    G.eval()
    out_class_root = PROJECT_ROOT / "generated_samples_wgan" / f"class_{class_idx}"

    all_fake: List[torch.Tensor] = []
    all_keys: List[Tuple[int, int]] = []
    rng = torch.Generator(device=device)
    rng.manual_seed(2000 + class_idx)

    with torch.no_grad():
        for (load, rpm), n_real in sorted(group_counts.items()):
            n_gen = n_real if GENERATE_MATCH_REAL_COUNTS else GEN_SAMPLES_PER_GROUP
            cond_np = normalize_condition(
                np.full((n_gen,), float(load), dtype=np.float32),
                np.full((n_gen,), float(rpm), dtype=np.float32),
            )
            cond = torch.from_numpy(cond_np).to(device)
            z = torch.randn(n_gen, LATENT_DIM, device=device, generator=rng)
            fake = G(z, cond)
            all_fake.append(fake)
            all_keys.extend([(load, rpm)] * n_gen)

    fake_cat = torch.cat(all_fake, dim=0)
    save_generated_by_group(fake_cat, all_keys, out_class_root)
    print(f"[WGAN] class_{class_idx} 生成完成: {len(all_keys)} 条 -> {out_class_root}")


# =============================================================================
# 主入口
# =============================================================================


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"REAL_DATA_PATH={REAL_DATA_PATH}")
    if not os.path.isdir(REAL_DATA_PATH):
        raise FileNotFoundError(f"真实数据目录不存在: {REAL_DATA_PATH}")

    class_names = _collect_class_names(REAL_DATA_PATH)
    if not class_names:
        raise FileNotFoundError(f"未在 {REAL_DATA_PATH} 下找到故障类别子目录")

    print(f"检测到类别数: {len(class_names)}")
    for i, c in enumerate(class_names):
        print(f"  class_{i} <- {c}")

    full_ds = _load_full_dataset(class_names)

    # 每次重跑前清理旧结果，避免新旧样本混杂
    for base in [PROJECT_ROOT / "generated_samples_cvae", PROJECT_ROOT / "generated_samples_wgan"]:
        base.mkdir(parents=True, exist_ok=True)

    for class_idx in range(len(class_names)):
        print("=" * 72)
        print(f"处理 class_{class_idx} ({class_names[class_idx]})")
        ds_cls, group_counts = build_class_dataset(full_ds, class_idx)
        print(f"  样本数: {len(ds_cls)}  工况数: {len(group_counts)}")

        train_and_generate_cvae_per_class(class_idx, ds_cls, group_counts, device)
        train_and_generate_wgan_per_class(class_idx, ds_cls, group_counts, device)

    print("全部完成。")


if __name__ == "__main__":
    main()
