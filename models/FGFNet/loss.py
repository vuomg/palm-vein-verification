
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features1, features2):
        """
        Compute contrastive loss between two lists of features (from spatial and frequency branches).
        features1: List[Tensor]
        features2: List[Tensor]
        """
        total_loss = 0.0
        n_layers = len(features1)
        
        for f1, f2 in zip(features1, features2):
            # Features are [B, C, H, W]
            # Flatten to [B, D] where D = C*H*W 
            # OR Global Average Pool first?
            # The original TF code:
            # normalized_features1 = tf.math.l2_normalize(features1[i], axis=-1)
            # It seems to operate on the flattened or specific axis.
            # In TF MobileViT, the output of Transformer block is [B, H, W, C] (channels last)
            # So l2_normalize(axis=-1) normalizes the channel vector at each pixel.
            # Then matmul is performed.
            
            # Let's inspect TF code again:
            # similarity_matrix = tf.matmul(normalized_features1, normalized_features2, transpose_b=True)
            # If shape is [B, H, W, C], matmul behavior depends on rank.
            # If rank > 2, matmul is batch-wise.
            # So it computes similarity pixel-wise? Or patch-wise?
            # Actually, `features1[i]` in the loop refers to the output of a stage.
            # If it's 4D, `l2_normalize` works on last axis.
            # `matmul` on 4D tensors `[..., r, c] x [..., c, m] -> [..., r, m]`
            # This implies it's computing similarity map?
            
            # BUT:
            # `exp_similarity_matrix = tf.exp(similarity_matrix / temperature)`
            # `sum_exp_similarity_matrix = tf.reduce_sum(...)`
            # `log_probabilities = ...`
            # `loss = -tf.reduce_mean(tf.linalg.diag_part(log_probabilities))`
            
            # `diag_part` suggests we are looking for diagonal elements matching.
            # This usually implies [Batch, Feature] vs [Batch, Feature] -> [Batch, Batch] similarity.
            # If so, features must be flattened to 2D [B, D] first.
            
            # Let's assume we flatten spatial dims and normalize.
            b = f1.shape[0]
            f1_flat = f1.view(b, -1)
            f2_flat = f2.view(b, -1)
            
            f1_norm = F.normalize(f1_flat, p=2, dim=1)
            f2_norm = F.normalize(f2_flat, p=2, dim=1)
            
            # Similarity matrix [B, B]
            sim_matrix = torch.matmul(f1_norm, f2_norm.T) # [B, B]
            
            # Contrastive loss:
            # We want diagonal elements to be 1 (loss 0), others to be small.
            # This looks like NT-Xent or similar, but between two views of SAME image.
            # So f1[i] and f2[i] should match.
            # The TF code implementation seems to be:
            # Probability of f1[i] matching f2[i] given f1[i] matching all f2[j].
            
            logits = sim_matrix / self.temperature
            
            # Cross Entropy with labels 0, 1, 2... B-1
            # i.e. f1[i] should match f2[i]
            labels = torch.arange(b).to(f1.device)
            loss = F.cross_entropy(logits, labels)
            
            total_loss += loss
            
        return total_loss / n_layers

class FGFNetLoss(nn.Module):
    def __init__(self, contrastive_weight=0.1, temperature=0.1):
        super(FGFNetLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.contrastive_weight = contrastive_weight
        
    def forward(self, outputs, targets):
        """
        outputs: (logits, features1, features2)
        targets: labels
        """
        logits, feats1, feats2 = outputs
        
        # Cross Entropy Loss
        loss_ce = self.ce_loss(logits, targets)
        
        # Contrastive Loss
        loss_cont = self.contrastive_loss(feats1, feats2)
        
        total_loss = loss_ce + self.contrastive_weight * loss_cont
        
        return total_loss, loss_ce, loss_cont
