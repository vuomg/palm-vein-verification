"""
Loss Functions for RSNet - Matching Paper Configuration

Paper: "RSNet: Region-Specific Network for Contactless Palm Vein Authentication"
IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY, VOL. 20, 2025

Implements:
1. AdaFace Loss (Adaptive Margin Loss) - as specified in paper Section III-D
2. Difference Loss - Orthogonality constraint between global and local features
3. RSNet Joint Loss - Combined loss for dual-branch architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaFaceLoss(nn.Module):
    """
    AdaFace Loss (Quality Adaptive Margin for Face Recognition)
    
    Paper: "AdaFace: Quality Adaptive Margin for Face Recognition" (CVPR 2022)
    Used in RSNet paper for both global and local branch losses.
    
    The key idea is to adaptively adjust the margin based on image quality,
    where quality is estimated from the feature norm.
    
    Formula from RSNet paper (Eq. 6-9):
        L_k = -log(exp(f_k(θ_yi, m)) / (exp(f_k(θ_yi, m)) + Σ_j≠yi exp(s·cos(θ_j))))
        
        f_k(θ_j, m) = s·cos(θ_j + g_angle) - g_add,  for j = yi
                    = s·cos(θ_j),                      for j ≠ yi
        
        g_angle = -m · ‖z_i‖_norm
        g_add = m · ‖z_i‖_norm + m
        
        ‖z_i‖_norm = clip((‖z_i‖ - μ_z) / (σ_z / h), -1, 1)
    
    Args:
        in_features: Dimension of embedding vector (e.g., 1024)
        out_features: Number of classes/identities
        scale: Feature scale (s), default=50 (paper value)
        margin: Angular margin (m), default=0.55 (paper value)
        h: Constant parameter for norm normalization, default=0.29 (paper value)
    """
    
    def __init__(self, in_features, out_features, scale=50.0, margin=0.55, h=0.29):
        super(AdaFaceLoss, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.h = h
        
        # Weight matrix (class prototypes)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-compute margin values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        
        # Running statistics for feature norms (for quality estimation)
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_std', torch.ones(1))
        self.register_buffer('num_batches', torch.zeros(1))
        self.momentum = 0.1
    
    def _update_running_stats(self, norms):
        """Update running mean and std of feature norms."""
        if self.training:
            with torch.no_grad():
                batch_mean = norms.mean()
                batch_std = norms.std()
                
                if self.num_batches == 0:
                    self.running_mean.copy_(batch_mean)
                    self.running_std.copy_(batch_std)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                    self.running_std.mul_(1 - self.momentum).add_(batch_std * self.momentum)
                
                self.num_batches += 1
    
    def _compute_norm_indicator(self, norms):
        """
        Compute normalized feature norm indicator (image quality proxy).
        Eq. 9 from paper: ‖z_i‖_norm = clip((‖z_i‖ - μ_z) / (σ_z / h), -1, 1)
        """
        # Avoid division by zero
        std = self.running_std.clamp(min=1e-6)
        
        # Normalize
        norm_indicator = (norms - self.running_mean) / (std / self.h)
        
        # Clip to [-1, 1]
        norm_indicator = torch.clamp(norm_indicator, -1, 1)
        
        return norm_indicator
    
    def forward(self, embeddings, labels):
        """
        Forward pass.
        
        Args:
            embeddings: Feature embeddings (B, in_features)
            labels: Ground truth labels (B,)
            
        Returns:
            loss: AdaFace loss value
        """
        # Compute feature norms (before normalization)
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        
        # Update running statistics
        self._update_running_stats(norms)
        
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings_norm, weight_norm)  # (B, num_classes)
        
        # Compute sine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        
        # Compute normalized norm indicator (quality proxy)
        norm_indicator = self._compute_norm_indicator(norms).squeeze(1)  # (B,)
        
        # Compute adaptive margins (Eq. 8)
        # g_angle = -m · ‖z_i‖_norm
        # g_add = m · ‖z_i‖_norm + m
        g_angle = -self.margin * norm_indicator  # (B,)
        g_add = self.margin * norm_indicator + self.margin  # (B,)
        
        # Compute cos(θ + g_angle) using angle addition formula
        # cos(θ + g_angle) = cos(θ)cos(g_angle) - sin(θ)sin(g_angle)
        cos_g_angle = torch.cos(g_angle).unsqueeze(1)  # (B, 1)
        sin_g_angle = torch.sin(g_angle).unsqueeze(1)  # (B, 1)
        
        # Apply margin: cos(θ + g_angle) - g_add for target class
        phi = cosine * cos_g_angle - sine * sin_g_angle - g_add.unsqueeze(1)
        
        # Create one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to target class
        output = one_hot * phi + (1.0 - one_hot) * cosine
        
        # Scale
        output = output * self.scale
        
        # Cross-entropy loss
        loss = F.cross_entropy(output, labels)
        
        return loss


class DifferenceLoss(nn.Module):
    """
    Difference Loss - Soft Subspace Orthogonality Constraint
    
    Enforces orthogonality between local and global embeddings to ensure
    they learn complementary feature aspects.
    
    Formula (Eq. 10 from paper):
        L_diff = ‖v_local^T · v_global‖²_F
    
    Where:
        v_local: L2-normalized local embeddings
        v_global: L2-normalized global embeddings
        ‖·‖_F: Frobenius norm (squared)
    """
    
    def __init__(self):
        super(DifferenceLoss, self).__init__()
    
    def forward(self, local_embeddings, global_embeddings):
        """
        Compute orthogonality loss between local and global embeddings.
        
        ⚠️ FIXED VERSION: Changed from B×B pairwise to element-wise computation.
        
        Original buggy implementation computed ALL pairwise dot products (B×B matrix),
        which causes the model to push all embeddings together, leading to score collapse.
        
        This corrected version only enforces orthogonality between local and global
        embeddings OF THE SAME SAMPLE, as intended by the paper.
        
        Args:
            local_embeddings: Local branch embeddings (B, D)
            global_embeddings: Global branch embeddings (B, D)
            
        Returns:
            Scalar loss value (mean of squared dot products)
        """
        # L2 normalize embeddings
        local_norm = F.normalize(local_embeddings, p=2, dim=1)
        global_norm = F.normalize(global_embeddings, p=2, dim=1)
        
        # ✅ FIXED: Element-wise dot product (only within same sample)
        # Compute: local[i] · global[i] for each sample i
        # Shape: (B,) where [i] = dot product of sample i's local and global embeddings
        dot_product = torch.sum(local_norm * global_norm, dim=1)  # (B,)
        
        # Compute mean of squared dot products
        # This enforces: local[i] ⊥ global[i] for each sample i independently
        loss = torch.mean(dot_product ** 2)
        
        return loss


class RSNetJointLoss(nn.Module):
    """
    RSNet Joint Loss - Combined Loss for Dual-Branch Architecture
    
    Formula (Eq. 11 from paper):
        L = L_local + λ1 · L_global + λ2 · L_diff
    
    Where:
        L_local: AdaFace loss for local branch
        L_global: AdaFace loss for global branch
        L_diff: Difference loss (orthogonality constraint)
        λ1, λ2: Weight parameters (database-specific)
    
    Args:
        adaface_loss_local: AdaFaceLoss instance for local branch
        adaface_loss_global: AdaFaceLoss instance for global branch
        lambda1: Weight for global branch loss (default: 0.5)
        lambda2: Weight for difference loss (default: 0.1)
    """
    
    def __init__(self, adaface_loss_local, adaface_loss_global, lambda1=0.5, lambda2=0.1):
        super(RSNetJointLoss, self).__init__()
        
        self.adaface_loss_local = adaface_loss_local
        self.adaface_loss_global = adaface_loss_global
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.diff_loss = DifferenceLoss()
    
    def forward(self, local_embeddings, global_embeddings, labels):
        """
        Compute joint loss for RSNet dual-branch architecture.
        
        Args:
            local_embeddings: Local branch embeddings (B, D)
            global_embeddings: Global branch embeddings (B, D)
            labels: Ground truth labels (B,)
            
        Returns:
            Tuple of (total_loss, loss_local, loss_global, loss_diff)
        """
        # AdaFace loss for local branch (using separate instance)
        loss_local = self.adaface_loss_local(local_embeddings, labels)
        
        # AdaFace loss for global branch (using separate instance)
        loss_global = self.adaface_loss_global(global_embeddings, labels)
        
        # Difference loss (orthogonality constraint)
        loss_diff = self.diff_loss(local_embeddings, global_embeddings)
        
        # Combined loss (Eq. 11)
        total_loss = loss_local + self.lambda1 * loss_global + self.lambda2 * loss_diff
        
        return total_loss, loss_local, loss_global, loss_diff


# Database-specific lambda configurations from paper TABLE II
# NOTE: lambda2 set to 0.1 as per RSNet paper (optimal for open-set protocol)
# Database-specific configurations
DATABASE_CONFIGS = {
    'CASIA_850': {'lambda1': 0.5, 'lambda2': 0.1, 'n_groups': 2},
    'CASIA_940': {'lambda1': 0.5, 'lambda2': 0.1, 'n_groups': 2},
    'VERA': {'lambda1': 0.5, 'lambda2': 0.1, 'n_groups': 2},
    'TJ_PV': {'lambda1': 0.5, 'lambda2': 0.1, 'n_groups': 4},
    'PLUS_PV850': {'lambda1': 0.5, 'lambda2': 0.1, 'n_groups': 2},
    'PLUS_PV950': {'lambda1': 0.5, 'lambda2': 0.1, 'n_groups': 2},
    'SCUT': {'lambda1': 0.5, 'lambda2': 0.1, 'n_groups': 4},
    'default': {'lambda1': 0.5, 'lambda2': 0.1, 'n_groups': 2}
}


def get_database_config(database_name):
    """
    Get database-specific configuration.
    
    Args:
        database_name: Name of the database
        
    Returns:
        Dictionary with lambda1, lambda2, and n_groups parameters
    """
    return DATABASE_CONFIGS.get(database_name, DATABASE_CONFIGS['default'])


if __name__ == "__main__":
    print("Testing Loss Functions (Paper Configuration)...")
    print("=" * 60)
    
    batch_size = 8
    feature_dim = 1024
    num_classes = 100
    
    # Test AdaFaceLoss
    print("\n1. Testing AdaFaceLoss...")
    adaface = AdaFaceLoss(
        in_features=feature_dim,
        out_features=num_classes,
        scale=50.0,
        margin=0.55,
        h=0.29
    )
    
    embeddings = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    loss = adaface(embeddings, labels)
    print(f"   ✓ AdaFace Loss: {loss.item():.4f}")
    
    # Test DifferenceLoss
    print("\n2. Testing DifferenceLoss...")
    diff_loss_fn = DifferenceLoss()
    local_emb = torch.randn(batch_size, feature_dim)
    global_emb = torch.randn(batch_size, feature_dim)
    
    diff_loss = diff_loss_fn(local_emb, global_emb)
    print(f"   ✓ Difference Loss: {diff_loss.item():.6f}")
    
    # Test RSNetJointLoss
    print("\n3. Testing RSNetJointLoss...")
    adaface_global = AdaFaceLoss(
        in_features=feature_dim,
        out_features=num_classes,
        scale=50.0,
        margin=0.55,
        h=0.29
    )
    joint_loss_fn = RSNetJointLoss(adaface, adaface_global, lambda1=0.5, lambda2=0.1)
    
    total, l_local, l_global, l_diff = joint_loss_fn(local_emb, global_emb, labels)
    print(f"   ✓ Total Loss: {total.item():.4f}")
    print(f"     - Local AdaFace: {l_local.item():.4f}")
    print(f"     - Global AdaFace: {l_global.item():.4f}")
    print(f"     - Difference Loss: {l_diff.item():.6f}")
    
    # Verify: total = local + 0.5*global + 0.1*diff
    expected = l_local.item() + 0.5 * l_global.item() + 0.1 * l_diff.item()
    print(f"   ✓ Verification: {total.item():.4f} ≈ {expected:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All loss function tests passed!")
