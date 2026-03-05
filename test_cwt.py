"""Test script for the modified UFourierLayer with CWT"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import UFourierLayer, PhysiNet

def test_ufourier_layer():
    print("Testing UFourierLayer with CWT and Cumulative Energy Ratio...")

    # Test parameters
    batch_size = 2
    channels = 64
    seq_length = 128
    time_dim = 256

    # Create layer with default energy_threshold=0.9
    layer = UFourierLayer(
        channels=channels,
        time_dim=time_dim,
        energy_threshold=0.9,
        num_scales=32
    )

    # Create test input
    x = torch.randn(batch_size, channels, seq_length)
    time_emb = torch.randn(batch_size, time_dim)

    print(f"Input shape: {x.shape}")
    print(f"Time embedding shape: {time_emb.shape}")

    # Forward pass
    out = layer(x, time_emb)

    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")

    # Verify shapes match
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert out.dtype == x.dtype, f"Dtype mismatch: {out.dtype} vs {x.dtype}"

    print("✓ UFourierLayer test passed!")
    return True

def test_physinet():
    print("\nTesting PhysiNet with modified UFourierLayer...")

    # Test parameters
    batch_size = 2
    channels = 1
    seq_length = 256
    dim = 64
    cond_dim = 4

    # Create model
    model = PhysiNet(
        dim=dim,
        channels=channels,
        cond_dim=cond_dim
    )

    # Create test input
    x = torch.randn(batch_size, channels, seq_length)
    time = torch.randint(0, 1000, (batch_size,))
    cond = torch.randn(batch_size, cond_dim)

    print(f"Input shape: {x.shape}")
    print(f"Time shape: {time.shape}")
    print(f"Condition shape: {cond.shape}")

    # Forward pass
    out = model(x, time, cond)

    print(f"Output shape: {out.shape}")

    # Verify output shape
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    print("✓ PhysiNet test passed!")
    return True

def test_gradient_flow():
    print("\nTesting gradient flow through CWT...")

    layer = UFourierLayer(channels=32, time_dim=128, energy_threshold=0.9)

    x = torch.randn(2, 32, 64, requires_grad=True)
    time_emb = torch.randn(2, 128)

    out = layer(x, time_emb)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient computed for input"
    print(f"Gradient shape: {x.grad.shape}")
    print("✓ Gradient flow test passed!")
    return True

def test_amp_compatibility():
    print("\nTesting AMP (Automatic Mixed Precision) compatibility...")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping AMP test")
        return True

    device = torch.device('cuda')
    layer = UFourierLayer(channels=32, time_dim=128, energy_threshold=0.9).to(device)

    x = torch.randn(2, 32, 64, device=device)
    time_emb = torch.randn(2, 128, device=device)

    # Test with autocast
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = layer(x.half(), time_emb)

    print(f"Input dtype: {x.half().dtype}, Output dtype: {out.dtype}")
    print("✓ AMP compatibility test passed!")
    return True

if __name__ == "__main__":
    all_passed = True

    all_passed &= test_ufourier_layer()
    all_passed &= test_physinet()
    all_passed &= test_gradient_flow()
    all_passed &= test_amp_compatibility()

    if all_passed:
        print("\n" + "="*50)
        print("All tests passed successfully!")
        print("="*50)
    else:
        print("\nSome tests failed!")

