# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np
from itertools import combinations


class ArcMarginProduct(nn.Module):
    r"""
    Implement of Additive Angular Margin Loss (ArcFace)
    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    Args:
        in_features: size of each input sample (embedding size, e.g., 512)
        out_features: size of each output sample (number of classes / IDs)
        s: norm of input feature (scale factor, default: 30.0)
        m: margin (default: 0.50)
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # Trọng số W của lớp Loss (Centers), kích thước [Class_Size, Embedding_Size]
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # 1. Cosine similarity: cos(theta) = (Input_norm * Weight_norm)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 2. Tính sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # 3. Tính cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # 4. Chuyển label thành one-hot encoding
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # 5. Output
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


class AdaCos(nn.Module):
    """
    AdaCos - Adaptive Cosine Loss
    """
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        
        if label is None:
            return logits * self.s
        
        return logits * self.s


# ============================================================================
# GSCL FusionLoss Components (from GSCL paper)
# ============================================================================

class CosFace(nn.Module):
    """
    CosFace Loss (from GSCL)
    Paper: CosFace: Large Margin Cosine Loss for Deep Face Recognition
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.2):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        # Weight matrix for classification
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, labels):
        # Normalize input and weight
        input_normalized = F.normalize(input, p=2, dim=1)
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cos = F.linear(input_normalized, weight_normalized)
        
        # Add margin to target class
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = self.s * (cos - one_hot * self.m)

        # Compute cross entropy loss
        softmax_output = F.log_softmax(output, dim=1)
        loss = -1 * softmax_output.gather(1, labels.view(-1, 1).long())
        loss = loss.mean()

        return loss


class NormLinear(nn.Module):
    """
    Normalized Linear layer for classification head
    """
    def __init__(self, in_features, out_features):
        super(NormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        input_normalized = F.normalize(input, p=2, dim=1)
        output = F.linear(input_normalized, weight_normalized)
        return output


def pdist(vectors, is_distance=True):
    """Compute pairwise distance/similarity matrix"""
    if is_distance:
        # L2 distance for normalized vectors
        distance_matrix = - 2 * np.matmul(vectors, vectors.T) + 2.0
    else:
        # Cosine similarity
        distance_matrix = np.matmul(vectors, vectors.T)
    return distance_matrix


class TripletSelector:
    """Base class for triplet selection"""
    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, selects negative samples based on the provided function
    """
    def __init__(self, margin, negative_selection_fn, is_distance=True, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.is_distance = is_distance

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
        distance_matrix = pdist(embeddings, self.is_distance)

        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            negative_indices = np.where(np.logical_not(label_mask))[0]
            if len(label_indices) < 2:
                continue
            
            anchor_positives = np.array(list(combinations(label_indices, 2)) + 
                                        list(combinations(label_indices[::-1], 2)))
            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                if self.is_distance:
                    loss_values = ap_distance - distance_matrix[anchor_positive[0], negative_indices] + self.margin
                else:
                    loss_values = distance_matrix[anchor_positive[0], negative_indices] - ap_distance + self.margin
                
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
        
        if len(triplets) == 0:
            # Fallback: create at least one triplet
            triplets.append([anchor_positives[0][0], anchor_positives[0][1], negative_indices[0]])
        
        triplets = torch.LongTensor(triplets)
        return triplets


def random_hard_negative(loss_values):
    """Select random hard negative with positive loss"""
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def RandomNegativeTripletSelector(margin, is_distance=True, cpu=True):
    """Factory function for RandomNegativeTripletSelector"""
    return FunctionNegativeTripletSelector(
        margin=margin,
        negative_selection_fn=random_hard_negative,
        is_distance=is_distance,
        cpu=cpu
    )


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Loss (from GSCL)
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector
    """
    def __init__(self, margin=0.2, is_distance=True):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = RandomNegativeTripletSelector(margin=margin, is_distance=is_distance)
        self.is_distance = is_distance

    def forward(self, embeddings, target):
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        triplets = self.triplet_selector.get_triplets(embeddings_normalized, target).to(embeddings.device)

        if self.is_distance:
            ap_distances = (embeddings_normalized[triplets[:, 0]] - embeddings_normalized[triplets[:, 1]]).pow(2).sum(1)
            an_distances = (embeddings_normalized[triplets[:, 0]] - embeddings_normalized[triplets[:, 2]]).pow(2).sum(1)
            losses = F.relu(ap_distances - an_distances + self.margin)
        else:
            ap_distances = (embeddings_normalized[triplets[:, 0]] * embeddings_normalized[triplets[:, 1]]).sum(1)
            an_distances = (embeddings_normalized[triplets[:, 0]] * embeddings_normalized[triplets[:, 2]]).sum(1)
            losses = 2 * F.relu(an_distances - ap_distances + self.margin)
        
        return losses.mean()


class FusionLoss(nn.Module):
    """
    FusionLoss from GSCL Paper
    Combines Classification Loss + OnlineTripletLoss (metric learning)
    
    Args:
        in_features: embedding size
        num_classes: number of classes
        cls_type: classification loss type, 'cosface' or 'adacos' (default: 'cosface')
        s: scale factor for CosFace (default: 30.0, ignored if adacos)
        m: margin for CosFace/AdaCos (default: 0.2)
        triplet_margin: margin for triplet loss (default: 0.2)
        w_cls: weight for classification loss (default: 1.0)
        w_metric: weight for metric learning loss (default: 4.0)
    """
    def __init__(self, in_features, num_classes, cls_type='cosface', s=30.0, m=0.2, 
                 triplet_margin=0.2, w_cls=1.0, w_metric=4.0):
        super(FusionLoss, self).__init__()
        self.w_cls = w_cls
        self.w_metric = w_metric
        self.cls_type = cls_type.lower()
        
        # Classification loss: CosFace or AdaCos
        if self.cls_type == 'cosface':
            self.cls_loss = CosFace(in_features, num_classes, s=s, m=m)
            self.ce_loss = None  # CosFace already includes CE
        elif self.cls_type == 'adacos':
            self.cls_loss = AdaCos(num_features=in_features, num_classes=num_classes, m=m)
            self.ce_loss = nn.CrossEntropyLoss()  # AdaCos returns logits
        else:
            raise ValueError(f"cls_type must be 'cosface' or 'adacos', got '{cls_type}'")
        
        # Metric learning loss
        self.metric_loss = OnlineTripletLoss(margin=triplet_margin, is_distance=True)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: feature embeddings [B, embedding_dim]
            labels: class labels [B]
        Returns:
            loss: combined loss value
        """
        # Classification loss
        if self.cls_type == 'cosface':
            cls_loss = self.cls_loss(embeddings, labels)
        else:  # adacos
            logits = self.cls_loss(embeddings, labels)
            cls_loss = self.ce_loss(logits, labels)
        
        # Metric learning loss
        metric_loss = self.metric_loss(embeddings, labels)
        
        # Combined loss
        loss = self.w_cls * cls_loss + self.w_metric * metric_loss
        return loss


# ============================================================================
# Utility: Balanced Batch Sampler for FusionLoss training
# ============================================================================

class BalancedBatchSampler(torch.utils.data.BatchSampler):
    """
    BatchSampler for FusionLoss training.
    Samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples.
    
    Required for OnlineTripletLoss to work properly.
    """
    def __init__(self, labels, n_classes, n_samples):
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_samples * self.n_classes
        self.n_batches = len(self.labels) // self.batch_size

    def __iter__(self):
        count = 0
        while count < self.n_batches:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(np.random.choice(
                    self.label_to_indices[class_], 
                    self.n_samples, 
                    replace=len(self.label_to_indices[class_]) < self.n_samples
                ))
            yield indices
            count += 1

    def __len__(self):
        return self.n_batches