import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import (
    Dataset1D,
    PhysiNet,
    GaussianDiffusion1D,
    Trainer1D,
)


def run_minimal_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 4
    seq_length = 16
    channels = 1
    cond_dim = 2

    # create tiny model (PhysiNet) with cond_dim
    model = PhysiNet(
        dim=8,
        init_dim=8,
        channels=channels,
        cond_dim=cond_dim,
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=seq_length,
        timesteps=10,  # small for test
        sampling_timesteps=10,
    )

    # construct fake dataset: 2 * batch
    N = batch_size * 2
    signals = torch.randn(N, channels, seq_length)
    # conditions shape (N, cond_dim)
    conditions = torch.rand(N, cond_dim)

    dataset = Dataset1D(signals, conditions)

    trainer = Trainer1D(
        diffusion_model=diffusion,
        dataset=dataset,
        train_batch_size=batch_size,
        train_num_steps=1,  # run only 1 step
        save_and_sample_every=100000,  # don't save during test
        num_samples=4,
    )

    # run one training step (should not raise)
    try:
        trainer.train()
        print('Train step completed')
    except Exception as e:
        print('Train failed:', e)
        raise

    # run conditional sampling
    cond = torch.tensor([[0.5, 0.5]] * batch_size)
    try:
        samples = trainer.sample_with_condition(batch_size=batch_size, cond=cond)
        print('Samples shape:', samples.shape if samples is not None else None)
    except Exception as e:
        print('Sampling failed:', e)
        raise

    # run sampling with None cond
    try:
        samples2 = trainer.sample_with_condition(batch_size=batch_size, cond=None)
        print('Samples2 shape:', samples2.shape if samples2 is not None else None)
    except Exception as e:
        print('Sampling (None cond) failed:', e)
        raise

    print('MINIMAL COND TEST PASSED')


if __name__ == '__main__':
    run_minimal_test()
