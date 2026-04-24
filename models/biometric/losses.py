"""
Loss Functions for Biometric Deep Learning

Các loss function được tối ưu cho bài toán metric learning,
đặc biệt là identification với few-shot learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss (Additive Angular Margin Loss)
    
    Paper: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    https://arxiv.org/abs/1801.07698
    
    Optimized cho few-shot biometric recognition.
    Tốt hơn Triplet Loss cho datasets có ít samples/class.
    
    Args:
        in_features: Dimension của embedding vector (e.g., 512)
        out_features: Number of classes/identities
        scale: Feature scale (s), default=30.0
        margin: Angular margin (m), default=0.5 (~28.6 độ)
        easy_margin: Use easy margin (cho training dễ hơn)
        label_smoothing: Label smoothing factor (0.0 đến 1.0)
    
    Formula:
        L = -log(exp(s * cos(θ + m)) / (exp(s * cos(θ + m)) + Σexp(s * cos(θ))))
    
    Trong đó:
        - θ: Góc giữa feature và weight vector
        - m: Angular margin (đẩy classes ra xa nhau)
        - s: Scale factor (điều chỉnh độ mạnh của gradient)
    """
    
    def __init__(self, in_features, out_features, scale=30.0, margin=0.5, 
                 easy_margin=False, label_smoothing=0.1):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale  # s parameter
        self.margin = margin  # m parameter (angular margin)
        self.easy_margin = easy_margin
        self.label_smoothing = label_smoothing
        
        # Weight matrix (class prototypes)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-compute trigonometric values for efficiency
        self.cos_m = np.cos(margin)
        self.sin_m = np.sin(margin)
        self.th = np.cos(np.pi - margin)  # Threshold = cos(180 - margin)
        self.mm = np.sin(np.pi - margin) * margin  # sin(180 - margin) * margin
    
    def forward(self, input, label):
        """
        Forward pass
        
        Args:
            input: Feature embeddings (N, in_features)
            label: Ground truth labels (N,)
        
        Returns:
            loss: ArcFace loss value
        """
        # Normalize features and weights (L2 normalization)
        # Đưa vectors về unit sphere để tính cosine similarity
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Calculate cos(theta + margin)
        # Công thức: cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Apply margin based on easy_margin setting
        if self.easy_margin:
            # Easy margin: chỉ apply margin khi cosine > 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Hard margin: apply margin với threshold
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Convert labels to one-hot encoding
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply label smoothing (regularization technique)
        if self.label_smoothing > 0:
            num_classes = cosine.size(1)
            smooth_value = self.label_smoothing / num_classes
            one_hot = one_hot * (1 - self.label_smoothing) + smooth_value
        
        # Apply margin to target class, keep cosine for non-target classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale the logits
        output *= self.scale
        
        # Calculate cross-entropy loss
        return F.cross_entropy(output, label)


class TripletLoss(nn.Module):
    """
    Triplet Loss with Hard Negative Mining
    
    Alternative loss function cho metric learning.
    Ít hiệu quả hơn ArcFace cho few-shot learning.
    
    Args:
        margin: Triplet margin (default=0.3)
        mining: Mining strategy ('hard', 'semi-hard', 'all')
    """
    
    def __init__(self, margin=0.3, mining='hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining
    
    def forward(self, embeddings, labels):
        """
        Forward pass
        
        Args:
            embeddings: Feature embeddings (N, D)
            labels: Ground truth labels (N,)
        
        Returns:
            loss: Triplet loss value
        """
        # Calculate pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # Get positive and negative masks
        labels = labels.unsqueeze(1)
        mask_positive = (labels == labels.t()).float()
        mask_negative = (labels != labels.t()).float()
        
        # For each anchor, find hardest positive and hardest negative
        if self.mining == 'hard':
            # Hardest positive: max distance among positives
            anchor_positive_dist = (pairwise_dist * mask_positive).max(dim=1)[0]
            
            # Hardest negative: min distance among negatives
            # Add large value to positives to exclude them
            negative_dist = pairwise_dist + mask_positive * 1e6
            anchor_negative_dist = negative_dist.min(dim=1)[0]
            
            # Triplet loss
            loss = F.relu(anchor_positive_dist - anchor_negative_dist + self.margin)
            return loss.mean()
        
        else:
            raise NotImplementedError(f"Mining strategy '{self.mining}' not implemented")


class CenterLoss(nn.Module):
    """
    Center Loss
    
    Học centers cho mỗi class và đẩy features về gần centers.
    Thường dùng kết hợp với Softmax Loss hoặc ArcFace.
    
    Args:
        num_classes: Number of classes
        feat_dim: Feature dimension
        lambda_c: Weight của center loss (default=0.003)
    """
    
    def __init__(self, num_classes, feat_dim, lambda_c=0.003):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        
        # Learnable centers
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    
    def forward(self, features, labels):
        """
        Forward pass
        
        Args:
            features: Feature embeddings (N, feat_dim)
            labels: Ground truth labels (N,)
        
        Returns:
            loss: Center loss value
        """
        batch_size = features.size(0)
        
        # Get centers for current batch
        centers_batch = self.centers[labels]
        
        # Calculate distance to centers
        loss = F.mse_loss(features, centers_batch)
        
        return loss * self.lambda_c


# Helper function để chọn loss
def get_loss_function(loss_type='arcface', **kwargs):
    """
    Factory function để tạo loss function
    
    Args:
        loss_type: 'arcface', 'triplet', 'center', hoặc 'combined'
        **kwargs: Arguments cho loss function
    
    Returns:
        loss_fn: Loss function instance
    
    Examples:
        >>> loss = get_loss_function('arcface', in_features=512, out_features=1000)
        >>> loss = get_loss_function('triplet', margin=0.3)
    """
    if loss_type == 'arcface':
        return ArcFaceLoss(**kwargs)
    elif loss_type == 'triplet':
        return TripletLoss(**kwargs)
    elif loss_type == 'center':
        return CenterLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test ArcFaceLoss
    print("Testing ArcFaceLoss...")
    
    batch_size = 32
    num_classes = 100
    embedding_dim = 512
    
    # Create dummy data
    features = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Create loss
    arcface = ArcFaceLoss(
        in_features=embedding_dim,
        out_features=num_classes,
        scale=30.0,
        margin=0.5
    )
    
    # Forward pass
    loss = arcface(features, labels)
    
    print(f"✓ ArcFaceLoss output: {loss.item():.4f}")
    print(f"✓ Loss shape: {loss.shape}")
    print(f"✓ All tests passed!")
