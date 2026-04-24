"""
Comprehensive evaluation metrics for biometric identification systems.
Includes EER, AUC, FNIR, FPIR, and other standard biometric metrics.
Uses ISO/IEC 19795 terminology for identification (1:N) systems.

Terminology:
- FNIR (False Non-Match Identification Rate): Tỷ lệ genuine bị từ chối
- FPIR (False Positive Identification Rate): Tỷ lệ impostor được chấp nhận
- EER (Equal Error Rate): Điểm FNIR = FPIR
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Union
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def calculate_similarity_scores(embeddings1: np.ndarray, 
                              embeddings2: np.ndarray, 
                              metric: str = 'cosine') -> np.ndarray:
    """
    Calculate similarity scores between two sets of embeddings.
    
    Args:
        embeddings1, embeddings2: Feature embeddings arrays
        metric: 'cosine', 'euclidean', or 'manhattan'
    
    Returns:
        Array of similarity scores
    """
    if metric == 'cosine':
        # Cosine similarity
        similarities = cosine_similarity(embeddings1, embeddings2)
        return np.diag(similarities)
    
    elif metric == 'euclidean':
        # Negative euclidean distance (higher = more similar)
        distances = euclidean_distances(embeddings1, embeddings2)
        return -np.diag(distances)
    
    elif metric == 'manhattan':
        # Negative Manhattan distance
        distances = np.sum(np.abs(embeddings1 - embeddings2), axis=1)
        return -distances
    
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def generate_pairs_and_labels(embeddings: np.ndarray, 
                            labels: np.ndarray,
                            max_pairs: Optional[int] = None,
                            final_evaluation: bool = False,
                            balanced_sampling: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate pairs of embeddings with corresponding labels.
    
    Strategy:
    1. GENUINE: Lấy HẾT tất cả cặp genuine (không giới hạn)
    2. IMPOSTOR:
       - Nếu balanced_sampling=True: Sample số impostor = số genuine (1:1 ratio)
       - Nếu final_evaluation=True: Lấy HẾT (cho báo cáo/thesis)
       - Nếu tổng số impostor < max_pairs: Lấy HẾT
       - Ngược lại: Random sampling đúng max_pairs cặp
    
    Args:
        embeddings: Feature embeddings (N, D)
        labels: Identity labels (N,)
        max_pairs: Maximum impostor pairs limit (default: 20000, ignored if balanced_sampling=True)
        final_evaluation: If True, take ALL impostor pairs (for final evaluation/thesis)
                         Recommended only for final run (slow but accurate)
        balanced_sampling: If True, use 1:1 genuine-impostor ratio (default: True)
    
    Returns:
        Tuple of (genuine_scores, impostor_scores)
    """
    n_samples = len(embeddings)
    
    if max_pairs is None:
        max_pairs = 20000  # Default impostor limit
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # ==================================================================
    # 1. GENUINE PAIRS - LẤY HẾT (không giới hạn)
    # ==================================================================
    print("  [1/2] Generating GENUINE pairs (ALL)...")
    genuine_scores = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        
        if len(indices) >= 2:
            # Lấy TẤT CẢ các cặp combinations trong class này
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    score = cosine_similarity([embeddings[indices[i]]], 
                                            [embeddings[indices[j]]])[0, 0]
                    genuine_scores.append(score)
    
    print(f"    → Generated {len(genuine_scores):,} genuine pairs (ALL)")
    
    # ==================================================================
    # 2. IMPOSTOR PAIRS - Logic thông minh
    # ==================================================================
    print("  [2/2] Generating IMPOSTOR pairs...")
    
    # Tính tổng số impostor pairs có thể sinh ra
    n_impostor_possible = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if labels[i] != labels[j]:
                n_impostor_possible += 1
    
    print(f"    → Total possible impostor pairs: {n_impostor_possible:,}")
    
    impostor_scores = []
    n_genuine = len(genuine_scores)
    
    # BALANCED SAMPLING MODE: 1:1 genuine-impostor ratio
    if balanced_sampling:
        target_impostor = n_genuine  # Match number of genuine pairs
        print(f"    → BALANCED SAMPLING: Target {target_impostor:,} impostor pairs (1:1 ratio)")
        
        sampled = 0
        max_attempts = target_impostor * 5  # Tránh vòng lặp vô hạn
        attempts = 0
        
        while sampled < target_impostor and attempts < max_attempts:
            i, j = np.random.choice(n_samples, 2, replace=False)
            if labels[i] != labels[j]:
                score = cosine_similarity([embeddings[i]], 
                                        [embeddings[j]])[0, 0]
                impostor_scores.append(score)
                sampled += 1
            attempts += 1
        
        print(f"    ✓ Generated {len(impostor_scores):,} impostor pairs (1:1 ratio with genuine)")
    
    # FINAL EVALUATION MODE: Lấy HẾT cho báo cáo/thesis
    elif final_evaluation:
        print(f"    ⚠️  FINAL EVALUATION MODE: Taking ALL {n_impostor_possible:,} impostor pairs")
        print(f"    ⚠️  This may take 1-2 minutes for accurate FPIR@0.001% calculation...")
        
        from tqdm import tqdm
        for i in tqdm(range(n_samples), desc="    Processing"):
            for j in range(i + 1, n_samples):
                if labels[i] != labels[j]:
                    score = cosine_similarity([embeddings[i]], 
                                            [embeddings[j]])[0, 0]
                    impostor_scores.append(score)
        
        print(f"    ✓ Generated ALL {len(impostor_scores):,} impostor pairs")
    
    elif n_impostor_possible <= max_pairs:
        # TH1: Số impostor có thể sinh < limit → LẤY HẾT
        print(f"    → Taking ALL {n_impostor_possible:,} impostor pairs")
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if labels[i] != labels[j]:
                    score = cosine_similarity([embeddings[i]], 
                                            [embeddings[j]])[0, 0]
                    impostor_scores.append(score)
    else:
        # TH2: Số impostor có thể sinh > limit → RANDOM SAMPLING
        print(f"    → Random sampling {max_pairs:,} impostor pairs (from {n_impostor_possible:,})")
        
        sampled = 0
        max_attempts = max_pairs * 5  # Tránh vòng lặp vô hạn
        attempts = 0
        
        while sampled < max_pairs and attempts < max_attempts:
            i, j = np.random.choice(n_samples, 2, replace=False)
            if labels[i] != labels[j]:
                score = cosine_similarity([embeddings[i]], 
                                        [embeddings[j]])[0, 0]
                impostor_scores.append(score)
                sampled += 1
            attempts += 1
    
    print(f"    → Generated {len(impostor_scores):,} impostor pairs")
    
    return np.array(genuine_scores), np.array(impostor_scores)


def calculate_eer(genuine_scores: np.ndarray, 
                 imposter_scores: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER) and corresponding threshold.
    
    Args:
        genuine_scores: Similarity scores for genuine pairs
        imposter_scores: Similarity scores for imposter pairs
    
    Returns:
        Tuple of (EER, threshold)
    """
    # Create labels (1 for genuine, 0 for imposter)
    y_true = np.concatenate([np.ones(len(genuine_scores)), 
                           np.zeros(len(imposter_scores))])
    y_scores = np.concatenate([genuine_scores, imposter_scores])
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calculate FNIR (False Non-Match Identification Rate) = 1 - TPR
    fnir = 1 - tpr
    
    # Find EER point where FNIR = FPIR (FPR)
    eer_idx = np.nanargmin(np.absolute(fnir - fpr))
    eer = (fnir[eer_idx] + fpr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold


def calculate_fnir_fpir_at_threshold(genuine_scores: np.ndarray,
                                     imposter_scores: np.ndarray,
                                     threshold: float) -> Tuple[float, float]:
    """
    Calculate FNIR and FPIR at a specific threshold.
    ISO/IEC 19795 terminology for identification systems.
    
    Args:
        genuine_scores: Similarity scores for genuine pairs
        imposter_scores: Similarity scores for imposter pairs  
        threshold: Decision threshold
    
    Returns:
        Tuple of (FNIR, FPIR)
    """
    # FPIR: False Positive Identification Rate (imposters accepted)
    fpir = np.mean(imposter_scores >= threshold)
    
    # FNIR: False Non-Match Identification Rate (genuines rejected)  
    fnir = np.mean(genuine_scores < threshold)
    
    return fnir, fpir


# Backward compatibility alias
calculate_fnir_fpir_at_threshold = calculate_fnir_fpir_at_threshold


def calculate_auc(genuine_scores: np.ndarray, 
                 imposter_scores: np.ndarray) -> float:
    """
    Calculate Area Under the ROC Curve.
    
    Args:
        genuine_scores: Similarity scores for genuine pairs
        imposter_scores: Similarity scores for imposter pairs
    
    Returns:
        AUC score
    """
    y_true = np.concatenate([np.ones(len(genuine_scores)), 
                           np.zeros(len(imposter_scores))])
    y_scores = np.concatenate([genuine_scores, imposter_scores])
    
    return roc_auc_score(y_true, y_scores)


def calculate_rank_n_accuracy(query_embeddings: np.ndarray,
                            gallery_embeddings: np.ndarray,
                            query_labels: np.ndarray,
                            gallery_labels: np.ndarray,
                            n: int = 1) -> float:
    """
    Calculate Rank-N identification accuracy.
    
    Args:
        query_embeddings: Query feature embeddings
        gallery_embeddings: Gallery feature embeddings
        query_labels: Query identity labels
        gallery_labels: Gallery identity labels
        n: Rank to calculate (default: Rank-1)
    
    Returns:
        Rank-N accuracy
    """
    correct = 0
    total = len(query_embeddings)
    
    for i, query_emb in enumerate(query_embeddings):
        # Calculate similarities to all gallery samples
        similarities = cosine_similarity([query_emb], gallery_embeddings)[0]
        
        # Get top-N matches
        top_n_indices = np.argsort(similarities)[-n:][::-1]
        top_n_labels = gallery_labels[top_n_indices]
        
        # Check if query label is in top-N
        if query_labels[i] in top_n_labels:
            correct += 1
    
    return correct / total


class BiometricEvaluator:
    """
    Comprehensive biometric evaluation class.
    """
    
    def __init__(self, similarity_metric: str = 'cosine'):
        self.similarity_metric = similarity_metric
        self.results = {}
    
    def evaluate_verification(self, 
                            embeddings: np.ndarray,
                            labels: np.ndarray,
                            max_pairs: Optional[int] = 100000,
                            final_evaluation: bool = False,
                            balanced_sampling: bool = True) -> Dict[str, float]:
        """
        Comprehensive verification evaluation.
        
        Args:
            embeddings: Feature embeddings (N, D)
            labels: Identity labels (N,)
            max_pairs: Maximum pairs to evaluate (for efficiency, ignored if balanced_sampling=True)
            final_evaluation: If True, take ALL impostor pairs for final thesis evaluation
            balanced_sampling: If True, use 1:1 genuine-impostor ratio (default: True)
        
        Returns:
            Dictionary with evaluation results
        """
        print("Generating pairs and calculating scores...")
        genuine_scores, imposter_scores = generate_pairs_and_labels(
            embeddings, labels, max_pairs, final_evaluation=final_evaluation,
            balanced_sampling=balanced_sampling
        )
        
        print(f"Generated {len(genuine_scores)} genuine pairs and {len(imposter_scores)} imposter pairs")
        
        # Calculate main metrics
        eer, eer_threshold = calculate_eer(genuine_scores, imposter_scores)
        auc_score = calculate_auc(genuine_scores, imposter_scores)
        
        # Calculate FNIR and FPIR at EER threshold (most important metric)
        fpir_at_eer, fnir_at_eer = calculate_fnir_fpir_at_threshold(genuine_scores, imposter_scores, eer_threshold)
        
        # Calculate FNIR/FPIR at different operating points
        fpir_001, fnir_001 = calculate_fnir_fpir_at_threshold(genuine_scores, imposter_scores, 
                                                        np.percentile(genuine_scores, 99.9))
        fpir_01, fnir_01 = calculate_fnir_fpir_at_threshold(genuine_scores, imposter_scores,
                                                      np.percentile(genuine_scores, 99))
        fpir_1, fnir_1 = calculate_fnir_fpir_at_threshold(genuine_scores, imposter_scores,
                                                    np.percentile(genuine_scores, 90))
        
        # Calculate FNIR at specific FPIR thresholds (reverse direction)
        # Find thresholds that give specific FPIR values using percentiles
        # For FPIR = 0.01%, we want threshold where 99.99% of imposters are below (rejected)
        # So threshold = 99.99th percentile of imposter scores
        if len(imposter_scores) > 0:
            fpir_001_threshold = np.percentile(imposter_scores, 99.99)
            fpir_01_threshold = np.percentile(imposter_scores, 99.9)
            fpir_1_threshold = np.percentile(imposter_scores, 99)
        else:
            fpir_001_threshold = fpir_01_threshold = fpir_1_threshold = 0.0
        
        frr_at_fpir_001, _ = calculate_fnir_fpir_at_threshold(genuine_scores, imposter_scores, fpir_001_threshold)
        frr_at_fpir_01, _ = calculate_fnir_fpir_at_threshold(genuine_scores, imposter_scores, fpir_01_threshold)
        frr_at_fpir_1, _ = calculate_fnir_fpir_at_threshold(genuine_scores, imposter_scores, fpir_1_threshold)
        
        # Calculate score statistics
        genuine_mean = np.mean(genuine_scores)
        genuine_std = np.std(genuine_scores)
        imposter_mean = np.mean(imposter_scores)
        imposter_std = np.std(imposter_scores)
        
        # D-prime (separability measure)
        d_prime = abs(genuine_mean - imposter_mean) / np.sqrt(0.5 * (genuine_std**2 + imposter_std**2))
        
        results = {
            'EER': eer,
            'EER_threshold': eer_threshold,
            'FPIR_at_EER': fpir_at_eer,
            'FNIR_at_EER': fnir_at_eer,
            'AUC': auc_score,
            'FPIR@0.01%FNIR': fpir_001,
            'FPIR@0.1%FNIR': fpir_01, 
            'FPIR@1%FNIR': fpir_1,
            'FNIR@0.01%FPIR': frr_at_fpir_001,
            'FNIR@0.1%FPIR': frr_at_fpir_01,
            'FNIR@1%FPIR': frr_at_fpir_1,
            # Add TAR (True Acceptance Rate) = 1 - FNIR (False Rejection Rate)
            'TAR@0.01%FAR': 1.0 - frr_at_fpir_001,
            'TAR@0.1%FAR': 1.0 - frr_at_fpir_01,
            'TAR@1%FAR': 1.0 - frr_at_fpir_1,
            # Add Thresholds
            'threshold_001_far': fpir_001_threshold,
            'threshold_01_far': fpir_01_threshold,
            'threshold_1_far': fpir_1_threshold,
            'genuine_mean': genuine_mean,
            'genuine_std': genuine_std,
            'imposter_mean': imposter_mean,
            'imposter_std': imposter_std,
            'd_prime': d_prime,
            'n_genuine_pairs': len(genuine_scores),
            'n_imposter_pairs': len(imposter_scores)
        }
        
        # Store scores for plotting
        self.genuine_scores = genuine_scores
        self.imposter_scores = imposter_scores
        self.results = results
        
        return results
    
    def evaluate_identification(self,
                              query_embeddings: np.ndarray,
                              gallery_embeddings: np.ndarray, 
                              query_labels: np.ndarray,
                              gallery_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate identification performance.
        
        Returns:
            Dictionary with identification results
        """
        rank_1 = calculate_rank_n_accuracy(query_embeddings, gallery_embeddings,
                                         query_labels, gallery_labels, n=1)
        rank_5 = calculate_rank_n_accuracy(query_embeddings, gallery_embeddings,
                                         query_labels, gallery_labels, n=5) 
        rank_10 = calculate_rank_n_accuracy(query_embeddings, gallery_embeddings,
                                          query_labels, gallery_labels, n=10)
        
        return {
            'Rank-1_Accuracy': rank_1,
            'Rank-5_Accuracy': rank_5,
            'Rank-10_Accuracy': rank_10
        }
    
    def analyze_user_rejections(self, 
                               embeddings: np.ndarray,
                               labels: np.ndarray,
                               thresholds: Dict[str, float],
                               image_paths: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Analyze which users/samples would be rejected at various thresholds.
        
        Args:
            embeddings: Feature embeddings (N, D)
            labels: Identity labels (N,)
            thresholds: Dict of threshold names to threshold values
            image_paths: Optional array of image paths for detailed reporting
            
        Returns:
            Dictionary with rejection analysis per threshold
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        unique_labels = np.unique(labels)
        results = {}
        
        for threshold_name, threshold_value in thresholds.items():
            rejected_users = []
            rejected_samples = []
            
            for label in unique_labels:
                # Get all samples for this user
                user_indices = np.where(labels == label)[0]
                
                if len(user_indices) < 2:
                    continue
                
                # Calculate intra-class (genuine) similarities
                user_embeddings = embeddings[user_indices]
                sim_matrix = cosine_similarity(user_embeddings)
                
                # Get upper triangle (excluding diagonal)
                n = len(user_indices)
                genuine_scores = []
                for i in range(n):
                    for j in range(i + 1, n):
                        genuine_scores.append(sim_matrix[i, j])
                
                if len(genuine_scores) == 0:
                    continue
                
                # Check if any genuine pair is below threshold (would be rejected)
                min_genuine = min(genuine_scores)
                avg_genuine = np.mean(genuine_scores)
                
                if min_genuine < threshold_value:
                    rejected_users.append({
                        'label': int(label),
                        'n_samples': len(user_indices),
                        'min_genuine_score': float(min_genuine),
                        'avg_genuine_score': float(avg_genuine),
                        'rejection_rate': float(np.mean(np.array(genuine_scores) < threshold_value))
                    })
            
            results[threshold_name] = {
                'threshold': float(threshold_value),
                'n_rejected_users': len(rejected_users),
                'n_total_users': len(unique_labels),
                'rejection_rate': len(rejected_users) / len(unique_labels) if len(unique_labels) > 0 else 0,
                'rejected_users': rejected_users[:10]  # Only keep top 10 for brevity
            }
        
        return results
    
    def plot_score_distributions(self, save_path: Optional[str] = None):
        """Plot genuine and imposter score distributions."""
        if not hasattr(self, 'genuine_scores') or not hasattr(self, 'imposter_scores'):
            raise ValueError("Need to run evaluate_verification first")
        
        plt.figure(figsize=(12, 5))
        
        # Score distributions
        plt.subplot(1, 2, 1)
        plt.hist(self.imposter_scores, bins=50, alpha=0.7, label='Imposter', color='red', density=True)
        plt.hist(self.genuine_scores, bins=50, alpha=0.7, label='Genuine', color='blue', density=True)
        plt.axvline(self.results['EER_threshold'], color='green', linestyle='--', 
                   label=f'EER Threshold: {self.results["EER_threshold"]:.3f}')
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.title('Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # ROC curve
        plt.subplot(1, 2, 2)
        y_true = np.concatenate([np.ones(len(self.genuine_scores)), 
                               np.zeros(len(self.imposter_scores))])
        y_scores = np.concatenate([self.genuine_scores, self.imposter_scores])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {self.results["AUC"]:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Random')
        plt.xlabel('False Positive Rate (FPIR)')
        plt.ylabel('True Positive Rate (1-FRR)')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_results(self):
        """Print formatted evaluation results."""
        if not self.results:
            raise ValueError("Need to run evaluation first")
        
        print("\n" + "="*60)
        print("BIOMETRIC EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n🎯 PRIMARY METRICS:")
        print(f"   EER (Equal Error Rate):     {self.results['EER']:.4f} ({self.results['EER']*100:.2f}%)")
        print(f"   AUC (Area Under Curve):     {self.results['AUC']:.4f}")
        print(f"   EER Threshold:              {self.results['EER_threshold']:.4f}")
        
        print(f"\n📊 FNIR & FPIR AT EER THRESHOLD:")
        print(f"   FPIR (False Positive Identification Rate):    {self.results['FPIR_at_EER']:.6f} ({self.results['FPIR_at_EER']*100:.4f}%)")
        print(f"   FNIR (False Non-Match Identification Rate):    {self.results['FNIR_at_EER']:.6f} ({self.results['FNIR_at_EER']*100:.4f}%)")
        
        print(f"\n📊 OPERATING POINTS (FAR at fixed FRR):")
        print(f"   FPIR @ 0.01% FNIR:           {self.results['FPIR@0.01%FNIR']:.6f} ({self.results['FPIR@0.01%FNIR']*100:.4f}%)")
        print(f"   FPIR @ 0.1% FNIR:            {self.results['FPIR@0.1%FNIR']:.6f} ({self.results['FPIR@0.1%FNIR']*100:.4f}%)")
        print(f"   FPIR @ 1% FNIR:              {self.results['FPIR@1%FNIR']:.6f} ({self.results['FPIR@1%FNIR']*100:.4f}%)")
        
        print(f"\n📊 OPERATING POINTS (FRR at fixed FAR):")
        print(f"   FNIR @ 0.01% FPIR:           {self.results['FNIR@0.01%FPIR']:.6f} ({self.results['FNIR@0.01%FPIR']*100:.4f}%)")
        print(f"   FNIR @ 0.1% FPIR:            {self.results['FNIR@0.1%FPIR']:.6f} ({self.results['FNIR@0.1%FPIR']*100:.4f}%)")
        print(f"   FNIR @ 1% FPIR:              {self.results['FNIR@1%FPIR']:.6f} ({self.results['FNIR@1%FPIR']*100:.4f}%)")
        
        print(f"\n📈 SCORE STATISTICS:")
        print(f"   Genuine Mean ± Std:        {self.results['genuine_mean']:.4f} ± {self.results['genuine_std']:.4f}")
        print(f"   Imposter Mean ± Std:       {self.results['imposter_mean']:.4f} ± {self.results['imposter_std']:.4f}")
        print(f"   D-prime (Separability):    {self.results['d_prime']:.4f}")
        
        print(f"\n📋 DATA STATISTICS:")
        print(f"   Genuine Pairs:             {self.results['n_genuine_pairs']:,}")
        print(f"   Imposter Pairs:            {self.results['n_imposter_pairs']:,}")
        
        print("="*60 + "\n")


class StatisticalEvaluator:
    """
    Statistical evaluation using Genuine/Imposter score distributions.
    Finds optimal threshold based on mean and standard deviation overlap.
    """
    
    def __init__(self, embeddings, labels):
        """
        Args:
            embeddings: Feature embeddings (N, D)
            labels: Identity labels (N,)
        """
        self.embeddings = embeddings
        self.labels = labels
        self.genuine_scores = []
        self.imposter_scores = []
        self.distribution_stats = {}
        self.optimal_threshold = None
        self.optimal_method = None
        
    def compute_all_pair_scores(self):
        """
        Compute ALL pairwise similarity scores (not sampled).
        - Genuine: same identity (label_i == label_j, i != j)
        - Imposter: different identity (label_i != label_j)
        """
        print("Computing exhaustive pairwise similarity scores...")
        n_samples = len(self.embeddings)
        genuine_scores = []
        imposter_scores = []
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute all pairwise similarities efficiently
        # Use matrix multiplication for cosine similarity: X @ X.T
        print("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings_norm)
        
        # Extract genuine and imposter pairs
        print("Extracting genuine and imposter pairs...")
        for i in tqdm(range(n_samples), desc="Processing pairs"):
            for j in range(i + 1, n_samples):  # Only upper triangle to avoid duplicates
                score = similarity_matrix[i, j]
                
                if self.labels[i] == self.labels[j]:
                    # Genuine pair
                    genuine_scores.append(score)
                else:
                    # Imposter pair
                    imposter_scores.append(score)
        
        self.genuine_scores = np.array(genuine_scores)
        self.imposter_scores = np.array(imposter_scores)
        
        print(f"Computed {len(self.genuine_scores)} genuine pairs and {len(self.imposter_scores)} imposter pairs")
        
        return self.genuine_scores, self.imposter_scores
    
    def compute_distribution_statistics(self):
        """
        Calculate mean, std, min, max for both distributions.
        
        Returns:
            dict with keys:
            - genuine_mean, genuine_std, genuine_min, genuine_max
            - imposter_mean, imposter_std, imposter_min, imposter_max
            - d_prime: (genuine_mean - imposter_mean) / sqrt(0.5*(g_std^2 + i_std^2))
        """
        if len(self.genuine_scores) == 0 or len(self.imposter_scores) == 0:
            raise ValueError("Must compute pair scores first using compute_all_pair_scores()")
        
        genuine_mean = np.mean(self.genuine_scores)
        genuine_std = np.std(self.genuine_scores)
        genuine_min = np.min(self.genuine_scores)
        genuine_max = np.max(self.genuine_scores)
        
        imposter_mean = np.mean(self.imposter_scores)
        imposter_std = np.std(self.imposter_scores)
        imposter_min = np.min(self.imposter_scores)
        imposter_max = np.max(self.imposter_scores)
        
        # D-prime (separability measure)
        d_prime = abs(genuine_mean - imposter_mean) / np.sqrt(0.5 * (genuine_std**2 + imposter_std**2))
        
        self.distribution_stats = {
            'genuine_mean': genuine_mean,
            'genuine_std': genuine_std,
            'genuine_min': genuine_min,
            'genuine_max': genuine_max,
            'imposter_mean': imposter_mean,
            'imposter_std': imposter_std,
            'imposter_min': imposter_min,
            'imposter_max': imposter_max,
            'd_prime': d_prime
        }
        
        return self.distribution_stats
    
    def find_optimal_threshold_statistical(self, method='equal_error'):
        """
        Find optimal threshold using statistical methods.
        
        Methods:
        1. 'equal_error': Where FAR(t) = FRR(t) (EER point)
        2. 'mean_midpoint': (genuine_mean + imposter_mean) / 2
        3. 'three_sigma': genuine_mean - 3*genuine_std
        4. 'max_separation': Maximize (genuine_mean - imposter_mean) / (genuine_std + imposter_std)
        
        Returns:
            optimal_threshold, method_name, metrics_at_threshold
        """
        if not self.distribution_stats:
            self.compute_distribution_statistics()
        
        stats = self.distribution_stats
        
        if method == 'equal_error':
            # Find EER threshold
            eer, eer_threshold = calculate_eer(self.genuine_scores, self.imposter_scores)
            far, frr = calculate_fnir_fpir_at_threshold(self.genuine_scores, self.imposter_scores, eer_threshold)
            self.optimal_threshold = eer_threshold
            self.optimal_method = 'equal_error'
            return eer_threshold, 'equal_error', {'FPIR': far, 'FNIR': frr, 'EER': eer}
        
        elif method == 'mean_midpoint':
            threshold = (stats['genuine_mean'] + stats['imposter_mean']) / 2
            far, frr = calculate_fnir_fpir_at_threshold(self.genuine_scores, self.imposter_scores, threshold)
            self.optimal_threshold = threshold
            self.optimal_method = 'mean_midpoint'
            return threshold, 'mean_midpoint', {'FPIR': far, 'FNIR': frr}
        
        elif method == 'three_sigma':
            threshold = stats['genuine_mean'] - 3 * stats['genuine_std']
            far, frr = calculate_fnir_fpir_at_threshold(self.genuine_scores, self.imposter_scores, threshold)
            self.optimal_threshold = threshold
            self.optimal_method = 'three_sigma'
            return threshold, 'three_sigma', {'FPIR': far, 'FNIR': frr}
        
        elif method == 'max_separation':
            # Find threshold that maximizes separation
            # We'll search through a range of thresholds
            threshold_range = np.linspace(stats['imposter_mean'], stats['genuine_mean'], 1000)
            best_separation = -np.inf
            best_threshold = threshold_range[0]
            
            for t in threshold_range:
                separation = (stats['genuine_mean'] - stats['imposter_mean']) / (stats['genuine_std'] + stats['imposter_std'])
                if separation > best_separation:
                    best_separation = separation
                    best_threshold = t
            
            # Actually, max_separation is independent of threshold, so use mean_midpoint
            threshold = (stats['genuine_mean'] + stats['imposter_mean']) / 2
            far, frr = calculate_fnir_fpir_at_threshold(self.genuine_scores, self.imposter_scores, threshold)
            self.optimal_threshold = threshold
            self.optimal_method = 'max_separation'
            return threshold, 'max_separation', {'FPIR': far, 'FNIR': frr}
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def plot_distributions(self, save_path=None):
        """
        Plot overlapping distributions with:
        - Histogram for genuine scores (blue)
        - Histogram for imposter scores (red)
        - Gaussian fit curves
        - Mean ± 1σ, 2σ, 3σ lines
        - Optimal threshold line
        - Overlap region shaded
        """
        if len(self.genuine_scores) == 0 or len(self.imposter_scores) == 0:
            raise ValueError("Must compute pair scores first")
        
        if not self.distribution_stats:
            self.compute_distribution_statistics()
        
        stats = self.distribution_stats
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histograms
        n_bins = 100
        ax.hist(self.imposter_scores, bins=n_bins, alpha=0.6, label='Imposter', 
                color='red', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(self.genuine_scores, bins=n_bins, alpha=0.6, label='Genuine', 
                color='blue', density=True, edgecolor='black', linewidth=0.5)
        
        # Fit Gaussian distributions
        try:
            from scipy import stats as scipy_stats
            use_scipy = True
        except ImportError:
            use_scipy = False
        
        # Generate x values for smooth curves
        x_min = min(stats['imposter_min'], stats['genuine_min'])
        x_max = max(stats['imposter_max'], stats['genuine_max'])
        x = np.linspace(x_min, x_max, 1000)
        
        # Gaussian PDFs
        if use_scipy:
            genuine_pdf = scipy_stats.norm.pdf(x, stats['genuine_mean'], stats['genuine_std'])
            imposter_pdf = scipy_stats.norm.pdf(x, stats['imposter_mean'], stats['imposter_std'])
        else:
            # Manual Gaussian PDF calculation
            genuine_pdf = (1 / (stats['genuine_std'] * np.sqrt(2 * np.pi))) * \
                         np.exp(-0.5 * ((x - stats['genuine_mean']) / stats['genuine_std'])**2)
            imposter_pdf = (1 / (stats['imposter_std'] * np.sqrt(2 * np.pi))) * \
                          np.exp(-0.5 * ((x - stats['imposter_mean']) / stats['imposter_std'])**2)
        
        ax.plot(x, genuine_pdf, 'b-', linewidth=2, label='Genuine Gaussian Fit', alpha=0.8)
        ax.plot(x, imposter_pdf, 'r-', linewidth=2, label='Imposter Gaussian Fit', alpha=0.8)
        
        # Mean lines
        ax.axvline(stats['genuine_mean'], color='blue', linestyle='--', linewidth=1.5, 
                  label=f"Genuine Mean: {stats['genuine_mean']:.4f}")
        ax.axvline(stats['imposter_mean'], color='red', linestyle='--', linewidth=1.5,
                  label=f"Imposter Mean: {stats['imposter_mean']:.4f}")
        
        # Standard deviation bands
        for sigma, alpha in [(1, 0.3), (2, 0.2), (3, 0.1)]:
            ax.axvspan(stats['genuine_mean'] - sigma * stats['genuine_std'],
                      stats['genuine_mean'] + sigma * stats['genuine_std'],
                      alpha=alpha, color='blue')
            ax.axvspan(stats['imposter_mean'] - sigma * stats['imposter_std'],
                      stats['imposter_mean'] + sigma * stats['imposter_std'],
                      alpha=alpha, color='red')
        
        # Optimal threshold line
        if self.optimal_threshold is not None:
            ax.axvline(self.optimal_threshold, color='green', linestyle='-', 
                      linewidth=2, label=f'Optimal Threshold ({self.optimal_method}): {self.optimal_threshold:.4f}')
        
        # Shade overlap region
        overlap_start = max(stats['imposter_mean'] - 3*stats['imposter_std'], 
                           stats['genuine_mean'] - 3*stats['genuine_std'])
        overlap_end = min(stats['imposter_mean'] + 3*stats['imposter_std'],
                         stats['genuine_mean'] + 3*stats['genuine_std'])
        if overlap_start < overlap_end:
            x_overlap = np.linspace(overlap_start, overlap_end, 100)
            if use_scipy:
                genuine_overlap = scipy_stats.norm.pdf(x_overlap, stats['genuine_mean'], stats['genuine_std'])
                imposter_overlap = scipy_stats.norm.pdf(x_overlap, stats['imposter_mean'], stats['imposter_std'])
            else:
                genuine_overlap = (1 / (stats['genuine_std'] * np.sqrt(2 * np.pi))) * \
                                 np.exp(-0.5 * ((x_overlap - stats['genuine_mean']) / stats['genuine_std'])**2)
                imposter_overlap = (1 / (stats['imposter_std'] * np.sqrt(2 * np.pi))) * \
                                  np.exp(-0.5 * ((x_overlap - stats['imposter_mean']) / stats['imposter_std'])**2)
            overlap_area = np.minimum(genuine_overlap, imposter_overlap)
            ax.fill_between(x_overlap, overlap_area, alpha=0.3, color='purple', 
                           label='Overlap Region')
        
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Genuine vs Imposter Score Distributions', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def evaluate_at_threshold(self, threshold):
        """
        Calculate FAR, FRR, accuracy at given threshold.
        """
        far = np.mean(self.imposter_scores >= threshold)
        frr = np.mean(self.genuine_scores < threshold)
        accuracy = 1 - (far + frr) / 2
        return {'FPIR': far, 'FNIR': frr, 'accuracy': accuracy}
    
    def print_statistical_report(self):
        """Print comprehensive statistical evaluation report."""
        if not self.distribution_stats:
            self.compute_distribution_statistics()
        
        stats = self.distribution_stats
        
        # Find optimal threshold if not already found
        if self.optimal_threshold is None:
            self.find_optimal_threshold_statistical('equal_error')
        
        metrics = self.evaluate_at_threshold(self.optimal_threshold)
        
        # Calculate overlap percentage
        overlap_count = np.sum((self.imposter_scores >= self.optimal_threshold) | 
                              (self.genuine_scores < self.optimal_threshold))
        total_count = len(self.genuine_scores) + len(self.imposter_scores)
        overlap_percentage = (overlap_count / total_count) * 100
        
        print("\n" + "="*70)
        print("STATISTICAL EVALUATION RESULTS")
        print("="*70)
        print(f"\nGenuine Distribution:")
        print(f"  Mean: {stats['genuine_mean']:.4f} ± {stats['genuine_std']:.4f}")
        print(f"  Range: [{stats['genuine_min']:.4f}, {stats['genuine_max']:.4f}]")
        
        print(f"\nImposter Distribution:")
        print(f"  Mean: {stats['imposter_mean']:.4f} ± {stats['imposter_std']:.4f}")
        print(f"  Range: [{stats['imposter_min']:.4f}, {stats['imposter_max']:.4f}]")
        
        print(f"\nD-prime (Separability): {stats['d_prime']:.4f}")
        
        print(f"\nOptimal Threshold ({self.optimal_method}):")
        print(f"  Threshold: {self.optimal_threshold:.4f}")
        print(f"  FPIR: {metrics['FPIR']:.6f} ({metrics['FPIR']*100:.4f}%)")
        print(f"  FNIR: {metrics['FNIR']:.6f} ({metrics['FNIR']*100:.4f}%)")
        if 'EER' in metrics:
            print(f"  EER: {metrics['EER']:.6f} ({metrics['EER']*100:.4f}%)")
        
        print(f"\nDistribution Overlap: {overlap_percentage:.2f}% of scores")
        print("="*70 + "\n")


class EnrollmentTestEvaluator:
    """
    Enrollment-Test evaluation protocol (7-enroll, 3-test).
    Simulates real-world biometric authentication deployment.
    """
    
    def __init__(self, embeddings, labels, n_enroll=7, n_test=3, fusion_method='average'):
        """
        Args:
            embeddings: Feature embeddings (N, D)
            labels: Identity labels (N,)
            n_enroll: Number of samples per user for enrollment (default: 7)
            n_test: Number of samples per user for testing (default: 3)
            fusion_method: Template fusion method ('average', 'median', 'best_quality')
        """
        self.embeddings = embeddings
        self.labels = labels
        self.n_enroll = n_enroll
        self.n_test = n_test
        self.fusion_method = fusion_method
        
        self.enroll_embeddings = None
        self.enroll_labels = None
        self.test_embeddings = None
        self.test_labels = None
        self.templates = {}
        self.genuine_scores = []
        self.imposter_scores = []
        self.eer_threshold = None
        self.fpir = None
        self.fnir = None
        self.auc = None
        self.accuracy = None
        
    def split_enroll_test(self):
        """
        Split dataset into enrollment and test sets.
        For each user:
        - First n_enroll samples → enrollment set
        - Remaining n_test samples → test set
        
        Returns:
            enroll_embeddings, enroll_labels, test_embeddings, test_labels
        """
        
        unique_labels = np.unique(self.labels)
        enroll_embeddings_list = []
        enroll_labels_list = []
        test_embeddings_list = []
        test_labels_list = []
        
        skipped_users = 0
        
        for label in unique_labels:
            # Get all samples for this user
            user_indices = np.where(self.labels == label)[0]
            n_samples = len(user_indices)
            
            # Check if user has enough samples
            if n_samples < (self.n_enroll + self.n_test):
                skipped_users += 1
                continue
            
            # Shuffle indices for this user to randomize selection
            np.random.seed(42)  # For reproducibility
            shuffled_indices = np.random.permutation(user_indices)
            
            # Split: first n_enroll for enrollment, next n_test for testing
            enroll_indices = shuffled_indices[:self.n_enroll]
            test_indices = shuffled_indices[self.n_enroll:self.n_enroll + self.n_test]
            
            enroll_embeddings_list.append(self.embeddings[enroll_indices])
            enroll_labels_list.extend([label] * len(enroll_indices))
            
            test_embeddings_list.append(self.embeddings[test_indices])
            test_labels_list.extend([label] * len(test_indices))
        
        if skipped_users > 0:
            pass  # Suppress warning message
        
        self.enroll_embeddings = np.concatenate(enroll_embeddings_list, axis=0) if enroll_embeddings_list else np.array([])
        self.enroll_labels = np.array(enroll_labels_list)
        self.test_embeddings = np.concatenate(test_embeddings_list, axis=0) if test_embeddings_list else np.array([])
        self.test_labels = np.array(test_labels_list)
        
        
        return self.enroll_embeddings, self.enroll_labels, self.test_embeddings, self.test_labels
    
    def create_enrollment_templates(self, enroll_embeddings=None, enroll_labels=None):
        """
        Create enrollment template for each user.
        
        Template fusion methods:
        1. 'average': Mean of n_enroll embeddings (default)
        2. 'median': Median of n_enroll embeddings
        3. 'best_quality': Select embedding closest to mean
        4. 'weighted_average': Weighted average based on quality (distance to mean)
        
        Returns:
            dict: {user_id: template_embedding}
        """
        if enroll_embeddings is None:
            enroll_embeddings = self.enroll_embeddings
        if enroll_labels is None:
            enroll_labels = self.enroll_labels
        
        if len(enroll_embeddings) == 0:
            raise ValueError("Must split enroll/test sets first using split_enroll_test()")
        
        
        unique_labels = np.unique(enroll_labels)
        templates = {}
        
        for label in unique_labels:
            # Get all enrollment embeddings for this user
            user_mask = (enroll_labels == label)
            user_embeddings = enroll_embeddings[user_mask]
            
            # Normalize embeddings
            user_embeddings_norm = user_embeddings / (np.linalg.norm(user_embeddings, axis=1, keepdims=True) + 1e-8)
            
            if self.fusion_method == 'average':
                template = np.mean(user_embeddings_norm, axis=0)
            elif self.fusion_method == 'median':
                template = np.median(user_embeddings_norm, axis=0)
            elif self.fusion_method == 'best_quality':
                # Find embedding closest to mean
                mean_emb = np.mean(user_embeddings_norm, axis=0)
                distances = np.linalg.norm(user_embeddings_norm - mean_emb, axis=1)
                best_idx = np.argmin(distances)
                template = user_embeddings_norm[best_idx]
            elif self.fusion_method == 'weighted_average':
                # Weighted average based on quality (inverse distance to mean)
                mean_emb = np.mean(user_embeddings_norm, axis=0)
                distances = np.linalg.norm(user_embeddings_norm - mean_emb, axis=1)
                # Convert distances to weights (closer = higher weight)
                # Add small epsilon to avoid division by zero
                epsilon = 1e-8
                weights = 1.0 / (distances + epsilon)
                weights = weights / np.sum(weights)  # Normalize
                template = np.average(user_embeddings_norm, axis=0, weights=weights)
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")
            
            # Normalize template
            template = template / (np.linalg.norm(template) + 1e-8)
            templates[label] = template
        
        self.templates = templates
        
        return templates
    
    def evaluate_authentication(self, test_embeddings=None, test_labels=None, 
                                templates=None, threshold=None):
        """
        Test authentication against enrolled templates.
        
        For each test sample:
        - Genuine attempt: Compare with own template
        - Imposter attempt: Compare with all other templates
        
        Returns:
            genuine_scores, imposter_scores, FAR, FRR
        """
        if test_embeddings is None:
            test_embeddings = self.test_embeddings
        if test_labels is None:
            test_labels = self.test_labels
        if templates is None:
            templates = self.templates
        
        if len(test_embeddings) == 0 or len(templates) == 0:
            raise ValueError("Must create templates first using create_enrollment_templates()")
        
        
        genuine_scores = []
        imposter_scores = []
        
        # Normalize test embeddings
        test_embeddings_norm = test_embeddings / (np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-8)
        
        for test_emb, test_label in tqdm(zip(test_embeddings_norm, test_labels), 
                                         desc="Authentication", total=len(test_embeddings)):
            # Genuine: compare with own template
            if test_label in templates:
                own_template = templates[test_label]
                genuine_score = np.dot(test_emb, own_template)
                genuine_scores.append(genuine_score)
            
            # Imposter: compare with all other templates
            for user_id, template in templates.items():
                if user_id != test_label:
                    imposter_score = np.dot(test_emb, template)
                    imposter_scores.append(imposter_score)
        
        self.genuine_scores = np.array(genuine_scores)
        self.imposter_scores = np.array(imposter_scores)
        
        # Find EER threshold if not provided
        if threshold is None:
            self.eer_threshold = self.find_eer_threshold(self.genuine_scores, self.imposter_scores)
            threshold = self.eer_threshold
        else:
            self.eer_threshold = threshold
        
        # Calculate FNIR and FPIR
        self.fpir = np.mean(self.imposter_scores >= threshold)
        self.fnir = np.mean(self.genuine_scores < threshold)
        
        # Calculate AUC
        self.auc = calculate_auc(self.genuine_scores, self.imposter_scores)
        
        # Calculate Accuracy at threshold
        # Accuracy = (True Positives + True Negatives) / Total
        # TP = genuine accepted (genuine_scores >= threshold)
        # TN = imposter rejected (imposter_scores < threshold)
        tp = np.sum(self.genuine_scores >= threshold)
        tn = np.sum(self.imposter_scores < threshold)
        total = len(self.genuine_scores) + len(self.imposter_scores)
        self.accuracy = (tp + tn) / total
        
        
        return self.genuine_scores, self.imposter_scores, self.far, self.frr
    
    def find_eer_threshold(self, genuine_scores, imposter_scores):
        """Find threshold where FPIR = FRR."""
        eer, eer_threshold = calculate_eer(genuine_scores, imposter_scores)
        return eer_threshold
    
    def generate_det_curve(self, genuine_scores=None, imposter_scores=None, save_path=None):
        """
        Generate DET (Detection Error Tradeoff) curve.
        Plot FPIR vs FNIR at different thresholds.
        """
        if genuine_scores is None:
            genuine_scores = self.genuine_scores
        if imposter_scores is None:
            imposter_scores = self.imposter_scores
        
        if len(genuine_scores) == 0 or len(imposter_scores) == 0:
            raise ValueError("Must run authentication evaluation first")
        
        # Calculate FNIR and FPIR at different thresholds
        all_scores = np.concatenate([genuine_scores, imposter_scores])
        thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 1000)
        
        far_values = []
        frr_values = []
        
        for threshold in thresholds:
            far = np.mean(imposter_scores >= threshold)
            frr = np.mean(genuine_scores < threshold)
            far_values.append(far)
            frr_values.append(frr)
        
        # Plot DET curve
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(far_values, frr_values, 'b-', linewidth=2, label='DET Curve')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equal Error Line')
        
        # Mark EER point
        if self.eer_threshold is not None:
            eer_far = np.mean(imposter_scores >= self.eer_threshold)
            eer_frr = np.mean(genuine_scores < self.eer_threshold)
            ax.plot(eer_far, eer_frr, 'go', markersize=10, 
                   label=f'EER Point (FAR={eer_far:.4f}, FRR={eer_frr:.4f})')
        
        ax.set_xlabel('False Positive Identification Rate (FPIR)', fontsize=12)
        ax.set_ylabel('False Non-Match Identification Rate (FNIR)', fontsize=12)
        ax.set_title('Detection Error Tradeoff (DET) Curve', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"DET curve saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_evaluation_report(self):
        """Print comprehensive evaluation report."""
        if self.far is None or self.frr is None:
            raise ValueError("Must run authentication evaluation first")
        
        print("\n" + "="*70)
        print("ENROLLMENT-TEST EVALUATION REPORT (7-Enroll, 3-Test)")
        print("="*70)
        print(f"\nDataset Configuration:")
        print(f"  Total Users: {len(np.unique(self.labels))}")
        print(f"  Enrollment Samples/User: {self.n_enroll}")
        print(f"  Test Samples/User: {self.n_test}")
        print(f"  Total Enrollment Samples: {len(self.enroll_embeddings)}")
        print(f"  Total Test Samples: {len(self.test_embeddings)}")
        
        print(f"\nTemplate Statistics:")
        print(f"  Template Fusion Method: {self.fusion_method.capitalize()}")
        print(f"  Number of Templates: {len(self.templates)}")
        
        print(f"\nAuthentication Results:")
        print(f"  Genuine Attempts: {len(self.genuine_scores):,}")
        print(f"  Imposter Attempts: {len(self.imposter_scores):,}")
        print(f"  Optimal Threshold (EER): {self.eer_threshold:.4f}")
        print(f"  FPIR at EER: {self.far:.6f} ({self.far*100:.4f}%)")
        print(f"  FNIR at EER: {self.frr:.6f} ({self.frr*100:.4f}%)")
        print(f"  EER: {(self.far + self.frr)/2:.6f} ({(self.far + self.frr)/2*100:.4f}%)")
        print(f"  AUC (Area Under ROC Curve): {self.auc:.6f}")
        print(f"  Accuracy at EER Threshold: {self.accuracy:.6f} ({self.accuracy*100:.4f}%)")
        
        print("="*70 + "\n")


def evaluate_model(model, dataloader, device='cuda', max_samples=None):
    """
    Extract embeddings and evaluate a trained model.
    
    Args:
        model: Trained palm vein recognition model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        max_samples: Maximum samples to evaluate (for efficiency)
    
    Returns:
        Evaluation results dictionary
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for i, (images, batch_labels) in enumerate(dataloader):
            if max_samples and len(embeddings) >= max_samples:
                break
                
            images = images.to(device)
            batch_embeddings = model(images)
            
            embeddings.append(batch_embeddings.cpu().numpy())
            labels.append(batch_labels.numpy())
    
    # Concatenate all embeddings and labels
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"Extracted {len(embeddings)} embeddings from {len(np.unique(labels))} unique identities")
    
    # Evaluate
    evaluator = BiometricEvaluator()
    results = evaluator.evaluate_verification(embeddings, labels)
    evaluator.print_results()
    
    return results, evaluator


if __name__ == "__main__":
    # Test evaluation with synthetic data
    print("Testing biometric evaluation metrics...")
    
    # Generate synthetic embeddings (simulate good vs poor models)
    np.random.seed(42)
    
    n_identities = 100
    n_samples_per_identity = 5
    embedding_dim = 512
    
    embeddings = []
    labels = []
    
    for identity_id in range(n_identities):
        # Generate identity center
        identity_center = np.random.randn(embedding_dim)
        identity_center = identity_center / np.linalg.norm(identity_center)
        
        for _ in range(n_samples_per_identity):
            # Add some noise around center
            sample = identity_center + 0.1 * np.random.randn(embedding_dim)
            sample = sample / np.linalg.norm(sample)  # L2 normalize
            
            embeddings.append(sample)
            labels.append(identity_id)
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"Generated {len(embeddings)} synthetic embeddings for {n_identities} identities")
    
    # Evaluate
    evaluator = BiometricEvaluator()
    results = evaluator.evaluate_verification(embeddings, labels, max_pairs=50000)
    evaluator.print_results()
    
    print("Metrics evaluation test completed successfully!")
