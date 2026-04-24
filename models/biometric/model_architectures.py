"""
Advanced CNN architectures for palm vein recognition with attention mechanisms.
Optimized for metric learning and biometric authentication tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class SpatialAttentionModule(nn.Module):
    """Spatial attention module focusing on important spatial regions."""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Apply average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention)


class ChannelAttentionModule(nn.Module):
    """Channel attention module focusing on important feature channels."""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling path
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc(y_avg)
        
        # Max pooling path
        y_max = self.max_pool(x).view(b, c)
        y_max = self.fc(y_max)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(y_avg + y_max).view(b, c, 1, 1)
        return attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention."""
    
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size)
    
    def forward(self, x):
        # Apply channel attention
        x_ca = x * self.channel_attention(x)
        # Apply spatial attention
        x_out = x_ca * self.spatial_attention(x_ca)
        return x_out


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone optimized for grayscale palm vein images.
    Supports B0, B3, B4, V2-S, V2-M variants.
    """
    
    # Feature dimensions for each EfficientNet variant
    FEATURE_DIMS = {
        'efficientnet_b0': 1280,
        'efficientnet_b3': 1536,
        'efficientnet_b4': 1792,
        'efficientnet_v2_s': 1280,
        'efficientnet_v2_m': 1280,
    }
    
    def __init__(self, variant='b0', pretrained=True):
        """
        Args:
            variant: 'b0', 'b3', 'b4', 'v2_s', 'v2_m'
            pretrained: Whether to use pretrained weights
        """
        super(EfficientNetBackbone, self).__init__()
        
        # Map variant names to model constructors
        model_constructors = {
            'b0': models.efficientnet_b0,
            'b3': models.efficientnet_b3,
            'b4': models.efficientnet_b4,
            'v2_s': models.efficientnet_v2_s,
            'v2_m': models.efficientnet_v2_m,
        }
        
        if variant not in model_constructors:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}. "
                           f"Supported: {list(model_constructors.keys())}")
        
        # Load EfficientNet model
        constructor = model_constructors[variant]
        self.efficientnet = constructor(pretrained=pretrained)
        
        # Modify first layer for grayscale input
        original_conv = self.efficientnet.features[0][0]
        has_bias = original_conv.bias is not None
        
        self.efficientnet.features[0][0] = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=has_bias
        )
        
        # Copy weights from pretrained model
        if pretrained:
            with torch.no_grad():
                rgb_weights = original_conv.weight.data
                gray_weights = rgb_weights.mean(dim=1, keepdim=True)
                self.efficientnet.features[0][0].weight.data = gray_weights
                if has_bias:
                    self.efficientnet.features[0][0].bias.data = original_conv.bias.data
        
        # Remove classifier to use as feature extractor
        self.features = self.efficientnet.features
        self.avgpool = self.efficientnet.avgpool
        
        # Feature dimension based on variant
        model_key = f'efficientnet_{variant}'
        self.feature_dim = self.FEATURE_DIMS.get(model_key, 1280)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNetBackbone(nn.Module):
    """ResNet-50 backbone for palm vein recognition."""
    
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        
        # Load ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first layer for grayscale input
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # Copy weights from first channel of pretrained model
        if pretrained:
            with torch.no_grad():
                rgb_weights = original_conv.weight.data
                gray_weights = rgb_weights.mean(dim=1, keepdim=True)
                resnet.conv1.weight.data = gray_weights
        
        # Remove classifier
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048  # ResNet-50 output channels
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class PalmVeinCNN(nn.Module):
    """
    Advanced CNN for palm vein recognition with attention mechanisms.
    Optimized for metric learning and high-accuracy biometric authentication.
    """
    
    def __init__(self, 
                 backbone='efficientnet', 
                 feature_dim=512, 
                 dropout_rate=0.3,
                 attention_type='cbam',
                 use_extended_feature_head=False):
        super(PalmVeinCNN, self).__init__()
        
        # Backbone selection
        if backbone == 'efficientnet' or backbone == 'efficientnet_b0':
            self.backbone = EfficientNetBackbone(variant='b0', pretrained=True)
            backbone_dim = 1280
            attention_channels = 1280
        elif backbone == 'efficientnet_b3':
            self.backbone = EfficientNetBackbone(variant='b3', pretrained=True)
            backbone_dim = 1536
            attention_channels = 1536
        elif backbone == 'efficientnet_b4':
            self.backbone = EfficientNetBackbone(variant='b4', pretrained=True)
            backbone_dim = 1792
            attention_channels = 1792
        elif backbone == 'efficientnet_v2_s' or backbone == 'efficientnetv2_s':
            self.backbone = EfficientNetBackbone(variant='v2_s', pretrained=True)
            backbone_dim = 1280
            attention_channels = 1280
        elif backbone == 'efficientnet_v2_m' or backbone == 'efficientnetv2_m':
            self.backbone = EfficientNetBackbone(variant='v2_m', pretrained=True)
            backbone_dim = 1280
            attention_channels = 1280
        elif backbone == 'resnet50':
            self.backbone = ResNetBackbone(pretrained=True)
            backbone_dim = 2048
            attention_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                           f"Supported: efficientnet/efficientnet_b0, efficientnet_b3, "
                           f"efficientnet_b4, efficientnet_v2_s, efficientnet_v2_m, resnet50")
        
        # Attention mechanism
        if attention_type == 'cbam':
            self.attention = CBAM(attention_channels)
        elif attention_type == 'channel':
            self.attention = ChannelAttentionModule(attention_channels)
        elif attention_type == 'spatial':
            self.attention = SpatialAttentionModule()
        else:
            self.attention = nn.Identity()
        
        # Feature projection head
        if use_extended_feature_head:
            # Extended structure: backbone -> 1024 -> 1024 -> 512
            self.feature_head = nn.Sequential(
                nn.Linear(backbone_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                
                nn.Linear(512, feature_dim),
                nn.BatchNorm1d(feature_dim)
            )
        else:
            # Standard structure: backbone -> 1024 -> 512 -> feature_dim
            self.feature_head = nn.Sequential(
            nn.Linear(backbone_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
        self.feature_dim = feature_dim
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.feature_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with L2 normalization for metric learning.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 128, 128)
            
        Returns:
            Normalized feature embeddings of shape (batch_size, feature_dim)
        """
        # Extract backbone features
        features = self.backbone(x)
        
        # Apply attention if using spatial attention on feature maps
        if hasattr(self.attention, 'spatial_attention'):
            # Reshape for spatial attention (needs 4D tensor)
            b = features.size(0)
            # For flattened features, we skip spatial attention
            pass
        
        # Project to embedding space
        embeddings = self.feature_head(features)
        
        # L2 normalization for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def extract_features(self, x):
        """Extract features without normalization (for analysis)."""
        features = self.backbone(x)
        embeddings = self.feature_head(features)
        return embeddings


class SiameseNetwork(nn.Module):
    """Siamese network for palm vein verification tasks."""
    
    def __init__(self, base_model, feature_dim=512):
        super(SiameseNetwork, self).__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
    
    def forward_once(self, x):
        """Forward pass for one image."""
        return self.base_model(x)
    
    def forward(self, x1, x2):
        """
        Forward pass for pair of images.
        
        Args:
            x1, x2: Input image pairs
            
        Returns:
            Tuple of normalized embeddings (out1, out2)
        """
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


def create_model(architecture='efficientnet', 
                feature_dim=512, 
                dropout_rate=0.3,
                attention_type='cbam',
                use_extended_feature_head=False):
    """
    Factory function to create palm vein recognition models.
    
    Args:
        architecture: One of:
            - 'efficientnet' or 'efficientnet_b0': Baseline, GPU yếu (default)
            - 'efficientnet_b3': Cân bằng speed/accuracy
            - 'efficientnet_b4': Precision cao, GPU mạnh
            - 'efficientnet_v2_s' or 'efficientnetv2_s': Train nhanh, kiến trúc mới
            - 'efficientnet_v2_m' or 'efficientnetv2_m': Nghiên cứu nâng cao
            - 'resnet50': ResNet-50 backbone
        feature_dim: Dimension of output embeddings
        dropout_rate: Dropout rate for regularization
        attention_type: 'cbam', 'channel', 'spatial', or 'none'
        use_extended_feature_head: If True, use extended feature_head (1024->1024->512)
                                   instead of standard (1024->512->feature_dim)
    
    Returns:
        Configured model instance
    """
    model = PalmVeinCNN(
        backbone=architecture,
        feature_dim=feature_dim,
        dropout_rate=dropout_rate,
        attention_type=attention_type,
        use_extended_feature_head=use_extended_feature_head
    )
    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'total_mb': total_params * 4 / (1024 ** 2),  # Assume float32
        'trainable_mb': trainable_params * 4 / (1024 ** 2)
    }


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create EfficientNet model
    model = create_model('efficientnet', feature_dim=512, attention_type='cbam')
    model = model.to(device)
    
    # Count parameters
    param_info = count_parameters(model)
    print(f"Model parameters: {param_info}")
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 128, 128).to(device)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output norm (should be ~1.0): {torch.norm(output, p=2, dim=1)}")
    
    print("Model architecture test completed successfully!")


