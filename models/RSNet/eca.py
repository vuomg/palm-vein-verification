"""
ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks

Implementation of Efficient Channel Attention (ECA) module.

Reference: Wang et al., "ECA-Net: Efficient Channel Attention for Deep 
Convolutional Neural Networks," CVPR 2020

Key advantages over SE-Net and CBAM:
- No dimensionality reduction → preserves channel information
- 1D convolution on channel dimension → captures local channel interactions  
- Zero additional parameters (just one 1D conv layer)
- More efficient than SE-Net's FC layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    """
    Efficient Channel Attention Module
    
    Uses 1D convolution to capture local cross-channel interactions
    without dimensionality reduction, making it more efficient than SE-Net.
    
    For palm vein recognition:
    - Adaptively recalibrates channel-wise feature responses
    - Emphasizes vein pattern channels, suppresses noise channels
    - Zero parameters overhead (just convolution kernel)
    
    Args:
        channels: Number of input channels
        k_size: Adaptive kernel size for 1D convolution
                Larger k_size = capture longer-range channel interactions
                Paper formula: k_size = |log2(C)/γ + b/γ|_odd
                For simplicity, we use fixed k_size (3, 5, or 7)
    
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W) - same shape, channel-wise recalibrated
    
    Example:
        >>> eca = ECA(channels=128, k_size=3)
        >>> x = torch.randn(2, 128, 7, 7)
        >>> out = eca(x)
        >>> assert out.shape == x.shape
    """
    
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        
        # Ensure kernel size is odd for symmetric padding
        assert k_size % 2 == 1, f"kernel size must be odd, got {k_size}"
        
        self.channels = channels
        self.k_size = k_size
        
        # Global average pooling (no parameters)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 1D convolution on channel dimension
        # This is the ONLY learnable component, but it's just a conv kernel
        # For k_size=3: only 3 parameters per output channel
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass with efficient channel attention.
        
        Steps:
            1. Global average pooling: (B, C, H, W) → (B, C, 1, 1)
            2. Squeeze spatial dims: (B, C, 1, 1) → (B, C)
            3. 1D conv on channels: (B, 1, C) → (B, 1, C)
            4. Sigmoid activation: (B, C) → (B, C) (weights in [0,1])
            5. Unsqueeze and apply: (B, C, 1, 1) * (B, C, H, W)
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Channel-attention weighted features (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Step 1: Global average pooling
        y = self.avg_pool(x)  # (B, C, 1, 1)
        
        # Step 2: Squeeze spatial dimensions
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        
        # Step 3: 1D convolution on channel dimension
        y = self.conv(y)  # (B, 1, C)
        
        # Step 4: Sigmoid activation
        y = self.sigmoid(y)  # (B, 1, C)
        
        # Step 5: Reshape and apply to input
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        
        # Element-wise multiplication (broadcasting)
        return x * y.expand_as(x)


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ECA Module - Comprehensive Test")
    print("=" * 70)
    
    # Test 1: Basic functionality
    print("\n[1] Testing ECA with different kernel sizes...")
    test_configs = [
        (128, 3, 7, 7),   # Typical for upper regions
        (128, 5, 7, 7),   # Typical for lower region
        (256, 3, 14, 14), # Higher resolution
    ]
    
    for channels, k_size, h, w in test_configs:
        eca = ECA(channels=channels, k_size=k_size)
        x = torch.randn(2, channels, h, w)
        out = eca(x)
        
        print(f"  Input: (2, {channels}, {h}, {w}), k_size={k_size}")
        print(f"    Output shape: {out.shape}")
        assert out.shape == x.shape, f"Shape mismatch!"
        print(f"    ✓ Shape correct")
    
    # Test 2: Parameter count
    print("\n[2] Verifying zero parameter overhead...")
    eca = ECA(channels=128, k_size=3)
    total_params = sum(p.numel() for p in eca.parameters())
    print(f"  Total parameters: {total_params}")
    print(f"  Expected: k_size = {eca.k_size} (just conv kernel)")
    assert total_params == eca.k_size, f"Parameter count mismatch!"
    print(f"  ✓ Minimal parameters confirmed")
    
    # Test 3: Attention weights range
    print("\n[3] Verifying attention weights in [0, 1]...")
    eca = ECA(channels=128, k_size=3)
    x = torch.randn(2, 128, 7, 7)
    
    with torch.no_grad():
        # Get attention weights (before multiplication)
        y = eca.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = eca.conv(y)
        weights = eca.sigmoid(y)
        
        print(f"  Attention weights - min: {weights.min():.4f}, max: {weights.max():.4f}")
        assert 0 <= weights.min() and weights.max() <= 1, "Weights out of range!"
        print(f"  ✓ All weights in [0, 1]")
    
    # Test 4: Gradient flow
    print("\n[4] Testing gradient flow...")
    eca = ECA(channels=128, k_size=3)
    x = torch.randn(2, 128, 7, 7, requires_grad=True)
    out = eca(x)
    loss = out.sum()
    loss.backward()
    
    print(f"  Input gradient: {x.grad is not None}")
    print(f"  Conv weight gradient: {eca.conv.weight.grad is not None}")
    assert x.grad is not None, "Gradient not flowing to input!"
    assert eca.conv.weight.grad is not None, "Gradient not flowing to conv!"
    print(f"  ✓ Gradients flow correctly")
    
    # Test 5: Different kernel sizes
    print("\n[5] Testing different kernel sizes...")
    for k in [3, 5, 7]:
        eca = ECA(channels=128, k_size=k)
        x = torch.randn(4, 128, 7, 7)
        out = eca(x)
        params = sum(p.numel() for p in eca.parameters())
        
        print(f"  k_size={k}: params={params}, output_shape={out.shape}")
        assert params == k, f"Expected {k} params, got {params}"
    print(f"  ✓ All kernel sizes work")
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ All ECA tests passed!")
    print("=" * 70)
    print("\nECA Module Summary:")
    print("  • Zero dimensionality reduction (unlike SE-Net)")
    print("  • Minimal parameters (k_size only)")
    print("  • 1D conv captures local channel interactions")
    print("  • Efficient for palm vein region-specific attention")
    print("\nReady for integration into Enhanced RLEB.")
