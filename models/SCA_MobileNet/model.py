"""
SCA-MobileNet: Spatial-Coordinate Attention MobileNet
=====================================================
A lightweight palm vein authentication network with self-contained backbone.

Architecture:
    Input → [STN] → Backbone (Modified MobileNetV3-Small, Variant H) → [SCIB] → Embedding

Key Components:
    - STN: Spatial Transformer Network with Geometric Regularization
    - Backbone: 9 InvertedResidual blocks (Variant H: layers 10-11 removed)
    - SCIB: Scale-Coordinate Invariant Block (unified CA + SPP)

Reference:
    Howard et al., "Searching for MobileNetV3", ICCV 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from typing import List, Optional, Callable

# =============================================================================
# Utility
# =============================================================================
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Ensure that all layers have a channel number divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# =============================================================================
# 1. Basic Building Blocks
# =============================================================================
class Conv2dBnAct(nn.Sequential):
    """
    Conv2d + BatchNorm2d + Activation.
    
    Standard convolution block used throughout the backbone. Supports
    regular and depthwise separable convolutions via the groups parameter.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        bias: bool = False,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation, groups=groups, bias=bias)
        ]
        layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer())
        
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block.
    
    Adaptively recalibrates channel-wise feature responses by modelling
    inter-channel dependencies. Uses ReLU for squeeze and HardSigmoid
    for excitation (following MobileNetV3 convention).
    """
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        scale_activation: Callable[..., nn.Module] = nn.Hardsigmoid,
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.ReLU()
        self.scale_activation = scale_activation()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale


class InvertedResidual(nn.Module):
    """
    MobileNetV3 Inverted Residual Block.
    
    Structure: Expand (1×1) → Depthwise (k×k) → SE → Project (1×1)
    With residual connection when stride=1 and in_channels==out_channels.
    
    Args:
        in_channels: Input channels
        expanded_channels: Channels after expansion
        out_channels: Output channels
        kernel_size: Depthwise conv kernel size (3 or 5)
        stride: Stride for depthwise conv
        use_se: Whether to use Squeeze-and-Excitation
        use_hs: If True use HardSwish, else ReLU
        dilation: Dilation for depthwise conv
    """
    def __init__(
        self,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = False,
        use_hs: bool = False,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        activation_layer = nn.Hardswish if use_hs else nn.ReLU
        
        layers: List[nn.Module] = []
        
        # 1. Expand (pointwise 1×1) — Skip if no expansion needed
        if expanded_channels != in_channels:
            layers.append(Conv2dBnAct(
                in_channels, expanded_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=activation_layer
            ))
        
        # 2. Depthwise convolution
        dw_stride = 1 if dilation > 1 else stride
        layers.append(Conv2dBnAct(
            expanded_channels, expanded_channels, kernel_size=kernel_size,
            stride=dw_stride, dilation=dilation, groups=expanded_channels,
            norm_layer=norm_layer, activation_layer=activation_layer
        ))
        
        # 3. Squeeze-and-Excitation
        if use_se:
            squeeze_channels = _make_divisible(expanded_channels // 4, 8)
            layers.append(SqueezeExcitation(expanded_channels, squeeze_channels))
        
        # 4. Project (pointwise 1×1, no activation)
        layers.append(Conv2dBnAct(
            expanded_channels, out_channels, kernel_size=1,
            norm_layer=norm_layer, activation_layer=None
        ))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = result + x
        return result


# =============================================================================
# 2. Spatial Transformer Network (STN) — Geometric Alignment Module
# =============================================================================
class STN_Block(nn.Module):
    """
    Spatial Transformer Network with geometric parameter output.
    
    Learns an affine transformation to spatially align input images,
    compensating for pose and scale variations in contactless palm vein
    capture. Returns both the aligned image and the affine parameters
    (theta) for geometric regularization.
    
    Identity initialization ensures stable training start — the STN
    begins as a no-op and gradually learns necessary corrections.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize to identity transform
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = self.adaptive_pool(xs)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_aligned = F.grid_sample(x, grid, align_corners=False)
        return x_aligned, theta


# =============================================================================
# 3. Coordinate Attention (CA)
# =============================================================================
class CoordAtt(nn.Module):
    """
    Coordinate Attention module.
    
    Encodes long-range spatial dependencies with precise positional
    information along horizontal and vertical directions independently.
    Unlike SE which only models channel-wise attention, CA preserves
    coordinate information essential for vein minutiae localization.
    
    Formulation:
        a_h = σ(f_h(δ(BN(f_1([pool_h(F); pool_w(F)])))))  ∈ R^{C×H×1}
        a_w = σ(f_w(δ(BN(f_1([pool_h(F); pool_w(F)])))))  ∈ R^{C×1×W}
        out = F ⊙ a_h ⊙ a_w
    """
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        mip = max(8, inp // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w


# =============================================================================
# 4. Spatial Pyramid Pooling (SPP)
# =============================================================================
class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling module.
    
    Aggregates features at multiple spatial resolutions [1×1, 2×2]
    to capture both fine-grained and coarse-grained spatial patterns
    in vein structures, enabling scale-invariant representation.
    """
    def __init__(self, pool_list=[1, 2]):
        super().__init__()
        self.pool_list = pool_list

    def forward(self, x):
        B = x.size(0)
        outputs = []
        for size in self.pool_list:
            pooled = F.adaptive_max_pool2d(x, output_size=(size, size))
            outputs.append(pooled.view(B, -1))
        return torch.cat(outputs, dim=1)

    def get_output_dim(self, in_channels):
        return in_channels * sum(s * s for s in self.pool_list)


# =============================================================================
# 5. SCIB: Scale-Coordinate Invariant Block
# =============================================================================
class SCIB(nn.Module):
    """
    Scale-Coordinate Invariant Block (SCIB).
    
    Unified architectural block combining Coordinate Attention with
    Spatial Pyramid Pooling. Applying CA BEFORE SPP ensures that
    positional information of vein minutiae (bifurcation/ending points)
    is encoded into the feature map BEFORE spatial downsampling at
    different pyramid levels.
    
    Formulation:
        F' = CA(F)                    — coordinate-weighted features
        SCIB(F) = SPP(F')             — multi-scale pooling
        Output ∈ R^{C × Σ(k²)}       — e.g., 576 × 5 = 2,880 for [1,2]
    
    Args:
        in_channels: Number of input feature channels
        reduction: Channel reduction ratio for CA (default: 32)
        pool_list: SPP pyramid sizes (default: [1, 2])
    """
    def __init__(self, in_channels, reduction=32, pool_list=[1, 2]):
        super().__init__()
        self.coord_att = CoordAtt(in_channels, in_channels, reduction=reduction)
        self.spp = SpatialPyramidPooling(pool_list=pool_list)
        self.pool_list = pool_list
        self.in_channels = in_channels

    def forward(self, x):
        x = self.coord_att(x)   # Step 1: Coordinate Attention
        x = self.spp(x)         # Step 2: Multi-scale Pyramid Pooling
        return x

    def get_output_dim(self):
        return self.in_channels * sum(s * s for s in self.pool_list)


# =============================================================================
# 6. Geometric Regularization Loss
# =============================================================================
class GeometricRegularizationLoss(nn.Module):
    """
    Geometric Regularization Loss for STN.
    
    Penalizes affine transformations deviating too far from identity:
        L_geo = λ_r·max(0, |θ| - θ_max)² + λ_s·||s-1||² + λ_t·||t||²
    
    where θ = rotation angle, s = scale factors, t = translation vector.
    
    Args:
        lambda_rot: Rotation penalty weight
        lambda_scale: Scale penalty weight  
        lambda_trans: Translation penalty weight
        max_rotation_deg: Maximum allowed rotation (degrees)
        max_scale_deviation: Maximum allowed |scale - 1.0|
    """
    def __init__(self, lambda_rot=1.0, lambda_scale=1.0, lambda_trans=0.5,
                 max_rotation_deg=30.0, max_scale_deviation=0.3):
        super().__init__()
        self.lambda_rot = lambda_rot
        self.lambda_scale = lambda_scale
        self.lambda_trans = lambda_trans
        self.max_rotation_rad = max_rotation_deg * math.pi / 180.0
        self.max_scale_deviation = max_scale_deviation

    def forward(self, theta):
        """
        Args:
            theta: [B, 2, 3] affine matrix [[a, b, tx], [c, d, ty]]
        """
        a, b, tx = theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2]
        c, d, ty = theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2]

        # Rotation: penalize beyond max_rotation_rad
        angle = torch.atan2(c, a)
        rot_excess = F.relu(torch.abs(angle) - self.max_rotation_rad)
        loss_rot = (rot_excess ** 2).mean()

        # Scale: penalize deviation from 1.0
        sx = torch.sqrt(a**2 + c**2 + 1e-8)
        sy = torch.sqrt(b**2 + d**2 + 1e-8)
        loss_scale = (F.relu(torch.abs(sx - 1.0) - self.max_scale_deviation) ** 2 +
                      F.relu(torch.abs(sy - 1.0) - self.max_scale_deviation) ** 2).mean()

        # Translation: penalize large shifts
        loss_trans = (tx**2 + ty**2).mean()

        return (self.lambda_rot * loss_rot +
                self.lambda_scale * loss_scale +
                self.lambda_trans * loss_trans)


# =============================================================================
# 7. Backbone Configuration — MobileNetV3-Small Variant H
# =============================================================================
# MobileNetV3-Small architecture (Table 2 from paper)
# Format: (in_ch, kernel, exp_ch, out_ch, SE, HardSwish, stride)
# Variant H: Block 9-10 removed (last two 96ch→96ch blocks)
BACKBONE_CONFIG = [
    # Block 0: in=16, k=3, exp=16,  out=16, SE=T, act=RE, s=2
    (16, 3, 16,  16, True,  False, 2),
    # Block 1: in=16, k=3, exp=72,  out=24, SE=F, act=RE, s=2
    (16, 3, 72,  24, False, False, 2),
    # Block 2: in=24, k=3, exp=88,  out=24, SE=F, act=RE, s=1
    (24, 3, 88,  24, False, False, 1),
    # Block 3: in=24, k=5, exp=96,  out=40, SE=T, act=HS, s=2
    (24, 5, 96,  40, True,  True,  2),
    # Block 4: in=40, k=5, exp=240, out=40, SE=T, act=HS, s=1
    (40, 5, 240, 40, True,  True,  1),
    # Block 5: in=40, k=5, exp=240, out=40, SE=T, act=HS, s=1
    (40, 5, 240, 40, True,  True,  1),
    # Block 6: in=40, k=5, exp=120, out=48, SE=T, act=HS, s=1
    (40, 5, 120, 48, True,  True,  1),
    # Block 7: in=48, k=5, exp=144, out=48, SE=T, act=HS, s=1
    (48, 5, 144, 48, True,  True,  1),
    # Block 8: in=48, k=5, exp=288, out=96, SE=T, act=HS, s=2
    (48, 5, 288, 96, True,  True,  2),
    # Block 9-10: REMOVED (Variant H — reduced depth)
]

# Last conv: 96 -> 576 (6x expansion, 1x1 conv + BN + HardSwish)
LAST_CONV_CHANNELS = 576

# Channel bottleneck before SPP: 576 -> 128 (matches MPSNet style)
# Keeps SPP[1,2,4] output at 128*21 = 2,688-d instead of 576*21 = 12,096-d
SPP_BOTTLENECK_CHANNELS = 128


# =============================================================================
# 8. SCAMobileNet — Main Model
# =============================================================================
class SCAMobileNet(nn.Module):
    """
    SCA-MobileNet: Spatial-Coordinate Attention MobileNet.
    
    A self-contained lightweight palm vein authentication network.
    
    Architecture:
        Input(3×224×224)
          → STN (spatial alignment, returns theta for geo-reg)
          → Stem Conv (3→16, stride=2)
          → 9× InvertedResidual blocks (Variant H)
          -> LastConv (96->576, 1x1)
          -> ChannelBottleneck (576->128, 1x1)  [MPSNet-style]
          -> SCIB (CA -> SPP[1,2,4]) -> 2,688-d
          -> Dropout -> FC -> BN -> 1024-d embedding
    
    Args:
        embedding_size: Output embedding dimension (default: 1024)
        class_size: Number of classes (for optional classifier head)
        pretrained: Load ImageNet pretrained weights (default: True)
        only_embeddings: Return embeddings only (default: True)  
        l2_normed: Apply L2 norm at inference (default: True)
        use_stn: Enable STN (default: True)
        use_ca: Enable CA in SCIB (default: True)
        use_spp: Enable SPP in SCIB (default: True)
        dropout: Dropout rate (default: 0.5)
    """
    def __init__(self, embedding_size=1024, class_size=None, pretrained=True,
                 only_embeddings=True, l2_normed=True, use_stn=True,
                 use_ca=True, use_spp=True, dropout=0.5, use_bottleneck=True,
                 ca_reduction=8):
        super().__init__()
        self.only_embeddings = only_embeddings
        self.l2_normed = l2_normed
        self.use_stn = use_stn
        self.use_ca = use_ca
        self.use_spp = use_spp
        self.use_bottleneck = use_bottleneck
        self.ca_reduction = ca_reduction

        # --- A. STN ---
        if self.use_stn:
            self.stn = STN_Block(in_channels=3)
            self.geo_reg_loss = GeometricRegularizationLoss()

        # --- B. Backbone ---
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        
        # Stem: 3 → 16, stride=2
        backbone_layers: List[nn.Module] = [
            Conv2dBnAct(3, 16, kernel_size=3, stride=2,
                        norm_layer=norm_layer, activation_layer=nn.Hardswish)
        ]
        
        # InvertedResidual blocks (Variant H config)
        for (in_ch, kernel, exp_ch, out_ch, use_se, use_hs, stride) in BACKBONE_CONFIG:
            backbone_layers.append(InvertedResidual(
                in_channels=in_ch, expanded_channels=exp_ch, out_channels=out_ch,
                kernel_size=kernel, stride=stride, use_se=use_se, use_hs=use_hs,
                norm_layer=norm_layer
            ))
        
        # CA inside backbone (fallback when SPP disabled)  
        if self.use_ca and not self.use_spp:
            backbone_layers.append(CoordAtt(inp=48, oup=48))
        
        # Last Conv: out_ch(96) → 576
        last_block_out = BACKBONE_CONFIG[-1][3]  # 96
        backbone_layers.append(Conv2dBnAct(
            last_block_out, LAST_CONV_CHANNELS, kernel_size=1,
            norm_layer=norm_layer, activation_layer=nn.Hardswish
        ))
        
        self.features = nn.Sequential(*backbone_layers)
        print(f"  Backbone: SCA-MobileNet (Variant H, {len(BACKBONE_CONFIG)} blocks)")

        # --- C. Channel Bottleneck (MPSNet-style: 576 -> 128) ---
        # Reduces SPP input channels so [1,2,4] stays affordable:
        #   576 * 21 = 12,096-d  →  128 * 21 = 2,688-d  (same as MPSNet)
        if self.use_bottleneck:
            self.channel_bottleneck = nn.Sequential(
                nn.Conv2d(LAST_CONV_CHANNELS, SPP_BOTTLENECK_CHANNELS,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(SPP_BOTTLENECK_CHANNELS),
                nn.Hardswish()
            )
            spp_in_channels = SPP_BOTTLENECK_CHANNELS
            print(f"  Bottleneck: {LAST_CONV_CHANNELS} -> {SPP_BOTTLENECK_CHANNELS} ch (1x1 conv)")
        else:
            self.channel_bottleneck = None
            spp_in_channels = LAST_CONV_CHANNELS
            print(f"  Bottleneck: DISABLED (SPP input = {LAST_CONV_CHANNELS} ch)")

        # --- D. Pooling ---
        if self.use_spp and self.use_ca:
            self.scib = SCIB(in_channels=spp_in_channels,
                             reduction=ca_reduction, pool_list=[1, 2, 4])
            pooling_dim = self.scib.get_output_dim()
            print(f"  Pooling: SCIB (CA->SPP [1,2,4]) -> dim={pooling_dim}")
        elif self.use_spp:
            self.spp_module = SpatialPyramidPooling(pool_list=[1, 2, 4])
            pooling_dim = self.spp_module.get_output_dim(spp_in_channels)
            print(f"  Pooling: SPP [1,2,4] -> dim={pooling_dim}")
        else:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            pooling_dim = spp_in_channels
            print(f"  Pooling: GAP -> dim={pooling_dim}")

        # --- E. Embedding Head ---
        # Note: pooling_bn removed (redundant — embedder starts with BN)
        self.embedder = nn.Sequential(
            nn.BatchNorm1d(pooling_dim),
            nn.Dropout(p=dropout),
            nn.Linear(pooling_dim, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )
        print(f"  Embedder: {pooling_dim} -> {embedding_size}-d")

        self.final = None
        if not self.only_embeddings:
            self.final = nn.Linear(embedding_size, class_size)

        # --- E. Initialize & Load Pretrained ---
        self._init_weights()
        if pretrained:
            self._load_pretrained_weights()

    def _init_weights(self):
        """Initialize only the embedder head weights.
        
        NOTE: We do NOT re-initialize the entire model here because:
        - Backbone: gets pretrained ImageNet weights via _load_pretrained_weights()
        - STN: has critical identity initialization in STN_Block.__init__()
        - SCIB/CA: has its own initialization
        Re-initializing all modules would destroy the STN identity init,
        causing the STN to apply random warping from epoch 1.
        """
        for m in self.embedder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self):
        """Load ImageNet pretrained weights from torchvision MobileNetV3-Small."""
        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            pretrained_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            pretrained_sd = pretrained_model.features.state_dict()
            
            # Build key mapping: pretrained features → our features
            # Pretrained has 13 layers (0=stem, 1-11=blocks, 12=lastconv)
            # Our Variant H has 11 layers (0=stem, 1-9=blocks, 10=lastconv)
            # Pretrained blocks 10,11 (indices 10,11) are removed in Variant H
            key_mapping = {}
            for key, value in pretrained_sd.items():
                parts = key.split('.')
                layer_idx = int(parts[0])
                rest = '.'.join(parts[1:])
                
                if layer_idx <= 9:
                    # Stem (0) + Blocks 0-8 (1-9): direct mapping
                    new_key = f"{layer_idx}.{rest}"
                    key_mapping[new_key] = value
                elif layer_idx == 12:
                    # Last conv: pretrained index 12 → our index 10
                    new_key = f"10.{rest}"
                    key_mapping[new_key] = value
                # Skip layer_idx 10, 11 (removed in Variant H)
            
            # Load mapped weights
            own_sd = self.features.state_dict()
            loaded = 0
            for key, value in key_mapping.items():
                if key in own_sd and own_sd[key].shape == value.shape:
                    own_sd[key] = value
                    loaded += 1
            
            self.features.load_state_dict(own_sd)
            total = len(own_sd)
            print(f"  Pretrained: Loaded {loaded}/{total} backbone parameters from ImageNet")
            
        except Exception as e:
            print(f"  Warning: Could not load pretrained weights: {e}")
            print(f"  Training from scratch.")

    def forward(self, x, train=True):
        """
        Forward pass.
        
        Returns:
            Training with STN: (embeddings, theta) — theta for geo-reg loss
            Otherwise: embeddings (or logits if not only_embeddings)
        """
        theta = None

        # 1. STN alignment
        if self.use_stn:
            x, theta = self.stn(x)

        # 2. Backbone
        x = self.features(x)

        # 3. Channel bottleneck (576 -> 128, MPSNet-style)
        if self.channel_bottleneck is not None:
            x = self.channel_bottleneck(x)

        # 4. Pooling
        if self.use_spp and self.use_ca:
            x = self.scib(x)
        elif self.use_spp:
            x = self.spp_module(x)
        else:
            x = self.gap(x).view(x.size(0), -1)

        # 5. Embedding
        embeddings = self.embedder(x)

        # 5. Output
        if train:
            output = embeddings if self.only_embeddings else self.final(embeddings)
        else:
            output = F.normalize(embeddings, p=2, dim=1) if self.l2_normed else embeddings

        if self.use_stn and train and theta is not None:
            return output, theta
        return output
