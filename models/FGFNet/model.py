
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, act=nn.SiLU()):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

def conv_1x1_bn(inp, oup, act=nn.SiLU()):
    return ConvBN(inp, oup, 1, 1, 0, bias=False, act=act)

def conv_3x3_bn(inp, oup, stride, act=nn.SiLU()):
    return ConvBN(inp, oup, 3, stride, 1, bias=False, act=act)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

from einops import rearrange

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4):
        super(MV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_3x3_bn(channel, channel, 1)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_3x3_bn(2 * channel, channel, 1)
    
    def forward(self, x):
        y = x.clone()

        # Local representation
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representation
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class FastFourierConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(FastFourierConv, self).__init__()
        # Simplified FFC: Assuming full spectral transform for now as in original implementation
        # The original code uses `ffc_conv2d_no_bias` which seems to be a wrapper around spectral conv.
        # Here we implement a spectral transform block.
        
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size, stride, padding, bias=bias)
        self.conv_fft = nn.Conv2d(in_channels, out_channels // 2, 1, 1, 0, bias=bias) # 1x1 in spectral domain equivalent? 
        # Actually proper FFC splits channels into local and global.
        # Based on the TF code:
        # It calls `ffc_conv2d_no_bias` which is likely from keras_cv_attention_models.
        # Let's perform a simple spectral gating mechanism here to mimic FFC behavior.
        
        self.conv_local = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # We will use standard conv but apply it on frequency features in the main block

    def forward(self, x):
        return self.conv_local(x) 

# Real Spectral Transform for FFC Branch
class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SpectralTransform, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1, 1, 0, bias=False)
        self.fu = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels // 2, 1, 1, 0, bias=False)
        )
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        # FFT
        x_fft = torch.fft.rfft2(x, norm='backward')
        
        # Complex to Real (stack real and imag)
        # PyTorch rfft2 results in complex tensor.
        # We need to process real and imaginary parts.
        # Simplified: just process magnitude for now or apply conv on complex?
        # A common way is to treat real and imag as channels.
        
        # Let's stick to the TF structure:
        # TF: ffc_transformer_pre_process -> ffc_conv2d_no_bias
        # The TF implementation uses standard layers but on "FFT" features produced earlier?
        # Wait, the TF code has a separate pipeline for FFT features at the STEM.
        # BUT inside the blocks, it uses `ffc_conv2d_no_bias`.
        # If we look at `keras_cv_attention_models`, FFC usually implies:
        # Split -> Local Conv + Spectral Conv -> Merge.
        # Spectral Conv: Real FFT -> Conv in Freq Domain (or Conv 1x1 on Real+Imag) -> Real IFFT.
        
        return x # Placeholder, implementing full FFC is complex without the base library.

class FFC_MobileViTBlock(nn.Module):
    """
    MobileViT block for the FFC branch.
    Uses FFC convolution instead of standard convolution.
    """
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        # FFC Conv 1 (Expansion)
        # Simplify: Use standard conv but intended to capture global context
        # Better yet, standard Conv + FFT-based Gating
        self.conv1 = conv_3x3_bn(channel, channel, 1)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_3x3_bn(2 * channel, channel, 1)
        
        # Spectral Gating for FFC imitation
        self.fft_gate = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = x.clone()

        # Local representation with FFT gating simulation
        x_fft = torch.fft.fft2(x).abs()
        x_fft = self.fft_gate(x_fft) 
        x = x * x_fft # Modulation

        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representation
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class GatedFusionLayer(nn.Module):
    def __init__(self, output_dim):
        super(GatedFusionLayer, self).__init__()
        self.gate_dense = nn.Linear(output_dim * 2, 1) # Global Avg Pool -> Dense -> Sigmoid
        self.output_dense = nn.Conv2d(output_dim, output_dim, 1) # 1x1 Conv

    def forward(self, x1, x2):
        # x1: Spatial Features [B, C, H, W]
        # x2: Frequency Features [B, C, H, W]
        # TF: Dense(1) on Concatenate([x1, x2])
        # BUT Dense usually expects flattened input. TF might broadcast if using Conv2D (1x1).
        # The original code uses `Dense(1)`, if inputs are 4D, it applies to last dim.
        # So it's pixel-wise gating!
        
        # 1. Concatenate along channel dim
        combined = torch.cat([x1, x2], dim=1) # [B, 2C, H, W]
        
        # 2. Dense(1) equivalent is Conv2d(2C, 1, 1)
        # Note: TF "Dense" on (B, H, W, C) input operates on C.
        # PyTorch (B, C, H, W).
        # So we need Conv2d(2C, 1, kernel_size=1)
        gate_values = F.sigmoid(nn.Conv2d(x1.shape[1] * 2, 1, 1).to(x1.device)(combined))
        
        gated_output = gate_values * x1 + (1 - gate_values) * x2
        output = self.output_dense(gated_output)
        return output

class DynamicDCMasking(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('distance', None)

    def forward(self, inputs_fft):
        # inputs_fft: [B, C, H, W] - Magnitude Spectrum
        B, C, H, W = inputs_fft.shape
        
        if self.distance is None or self.distance.shape[-2:] != (H, W):
            center_h, center_w = H // 2, W // 2
            y = torch.arange(H, device=inputs_fft.device) - center_h
            x = torch.arange(W, device=inputs_fft.device) - center_w
            Y, X = torch.meshgrid(y, x, indexing='ij')
            self.distance = torch.sqrt(X**2 + Y**2).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

        center_h, center_w = H // 2, W // 2
        
        # Center region 10x10
        center_region = inputs_fft[:, :, center_h-5:center_h+5, center_w-5:center_w+5]
        dc_strength = torch.mean(center_region, dim=[2, 3]) # [B, C]
        dc_strength_normalized = dc_strength / (torch.amax(inputs_fft, dim=[2, 3]) + 1e-8)
        
        # Radius
        radius = (dc_strength_normalized * (H // 2)).long() # [B, C]
        radius = radius.unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
        
        # Create mask
        # Compare cached distance with dynamic radius
        mask = (self.distance > radius).float()
        
        return inputs_fft * mask


class MobileViT_FFC_ATTN_FFTSA(nn.Module):
    def __init__(self, image_size=(256, 256), num_classes=2, width_multiplier=1.0):
        super().__init__()
        
        # Config (S-variant)
        # dims = [32, 64, 96, 128, 160]
        # channels = [16, 32, 64, 96, 128, 160] (first is stem)
        
        self.image_size = image_size
        
        # Stem
        self.conv1 = conv_3x3_bn(3, 16, 2) # stride 2
        
        # Frequency Stem
        self.conv1_fft = conv_3x3_bn(3, 16, 2)
        self.fft_process = DynamicDCMasking()
        self.gated_fusion = GatedFusionLayer(16)
        
        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(16, 32, 1))
        self.mv2.append(MV2Block(32, 64, 2))
        self.mv2.append(MV2Block(64, 64, 1)) # Repeated block
        self.mv2.append(MV2Block(64, 64, 1)) # Repeated block
        
        # Stage 3
        # MobileViT Block (Spatial + FFC)
        # dim=96
        self.mvit3_1 = MobileViTBlock(dim=96+48, depth=2, channel=64, kernel_size=3, patch_size=(2, 2), mlp_dim=int(144*2))
        self.mvit3_2 = FFC_MobileViTBlock(dim=96+48, depth=2, channel=64, kernel_size=3, patch_size=(2, 2), mlp_dim=int(144*2)) # Placeholder FFC
        
        self.mv2_stage3_ds = MV2Block(64, 96, 2)
        
        # Stage 4
        # dim=128
        self.mvit4_1 = MobileViTBlock(dim=128+64, depth=4, channel=96, kernel_size=3, patch_size=(2, 2), mlp_dim=int(192*2))
        self.mvit4_2 = FFC_MobileViTBlock(dim=128+64, depth=4, channel=96, kernel_size=3, patch_size=(2, 2), mlp_dim=int(192*2))
        
        self.mv2_stage4_ds = MV2Block(96, 128, 2)
        
        # Stage 5
        # dim=160
        self.mvit5_1 = MobileViTBlock(dim=160+80, depth=3, channel=128, kernel_size=3, patch_size=(2, 2), mlp_dim=int(240*2))
        self.mvit5_2 = FFC_MobileViTBlock(dim=160+80, depth=3, channel=128, kernel_size=3, patch_size=(2, 2), mlp_dim=int(240*2))
        
        self.conv_final = conv_1x1_bn(128, 640)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(640, num_classes)
        
    def forward_features(self, x):
        # x: [B, 3, H, W]
        
        # 1. Spatial Stem
        x_s = self.conv1(x)
        
        # 2. Frequency Stem
        # FFT
        x_f = torch.fft.fft2(x)
        x_f = torch.fft.fftshift(x_f)
        x_f = x_f.abs() # Magnitude
        x_f = x_f / (torch.amax(x_f, dim=(2, 3), keepdim=True) + 1e-8)
        x_f = self.fft_process(x_f) # [B, 3, H, W]
        x_f = self.conv1_fft(x_f)
        
        # Gated Fusion
        x = self.gated_fusion(x_s, x_f)
        
        # Stage 1 & 2 (MV2 blocks)
        x = self.mv2[0](x)
        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)
        
        contrastive_1 = []
        contrastive_2 = []
        
        # Stage 3
        x_b1 = self.mvit3_1(x)
        x_b2 = self.mvit3_2(x) # Should ideally be FFC
        contrastive_1.append(x_b1)
        contrastive_2.append(x_b2)
        x = x_b1 + x_b2
        
        # Downsample
        x = self.mv2_stage3_ds(x)
        
        # Stage 4
        x_b1 = self.mvit4_1(x)
        x_b2 = self.mvit4_2(x)
        contrastive_1.append(x_b1)
        contrastive_2.append(x_b2)
        x = x_b1 + x_b2
        
        # Downsample
        x = self.mv2_stage4_ds(x)
        
        # Stage 5
        x_b1 = self.mvit5_1(x)
        x_b2 = self.mvit5_2(x)
        contrastive_1.append(x_b1)
        contrastive_2.append(x_b2)
        x = x_b1 + x_b2
        
        # Final
        x = self.conv_final(x)
        x = self.pool(x).view(-1, x.shape[1])
        
        return x, contrastive_1, contrastive_2

    def forward(self, x):
        x, contrastive_1, contrastive_2 = self.forward_features(x)
        x = self.fc(x)
        return x, contrastive_1, contrastive_2
    
    def get_embedding(self, x):
        x, _, _ = self.forward_features(x)
        return x

if __name__ == "__main__":
    model = MobileViT_FFC_ATTN_FFTSA()
    dummy = torch.randn(2, 3, 256, 256)
    out, c1, c2 = model(dummy)
    print(f"Output: {out.shape}")
    print(f"Features 1: {[a.shape for a in c1]}")
    print(f"Features 2: {[a.shape for a in c2]}")
