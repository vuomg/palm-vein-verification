"""
Comprehensive Training Visualization Module for Biometric Systems.

Generates the following charts:
    fig1_loss_convergence.png      - Training and validation loss over epochs
    fig2_eer_evolution.png         - EER evolution during training  
    fig3_tar_far_evolution.png     - TAR@FAR metrics evolution
    fig4_dprime_separability.png   - D-prime separability metric
    fig5_score_distribution.png    - Score distribution (genuine vs imposter)
    fig6_security_benchmarks.png   - Security benchmarks comparison
    fig7_roc_curve.png             - Receiver Operating Characteristic curve
    fig8_score_histogram.png       - Score distribution histogram
    fig9_det_curve.png             - Detection Error Tradeoff curve
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, auc

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


class TrainingVisualizer:
    """Generate comprehensive training visualization charts."""
    
    def __init__(self, output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / 'charts'
        self.charts_dir.mkdir(exist_ok=True, parents=True)
        
        # Color palette
        self.colors = {
            'primary': '#2563eb',       # Blue
            'secondary': '#dc2626',     # Red
            'success': '#16a34a',       # Green
            'warning': '#d97706',       # Orange
            'purple': '#9333ea',        # Purple
            'cyan': '#0891b2',          # Cyan
            'genuine': '#2563eb',       # Blue for genuine
            'imposter': '#dc2626',      # Red for imposter
        }
        
    def load_metrics(self) -> Dict:
        """Load training metrics from JSON file."""
        metrics_file = self.output_dir / 'training_metrics.json'
        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    def generate_all_charts(self, 
                           genuine_scores: Optional[np.ndarray] = None,
                           imposter_scores: Optional[np.ndarray] = None,
                           eer_threshold: Optional[float] = None):
        """
        Generate all visualization charts.
        
        Args:
            genuine_scores: Optional genuine similarity scores for distribution plots
            imposter_scores: Optional imposter similarity scores for distribution plots
            eer_threshold: Optional EER threshold for visualization
        """
        print("\n" + "="*60)
        print("GENERATING TRAINING VISUALIZATION CHARTS")
        print("="*60)
        
        try:
            metrics = self.load_metrics()
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            return
        
        epochs_data = metrics.get('epochs', [])
        if not epochs_data:
            print("No epoch data found in metrics file.")
            return
        
        # Extract data arrays
        epochs = [e['epoch'] for e in epochs_data]
        
        # Generate each chart
        self._plot_loss_convergence(epochs_data, epochs)
        self._plot_eer_evolution(epochs_data, epochs)
        self._plot_tar_far_evolution(epochs_data, epochs)
        self._plot_dprime_separability(epochs_data, epochs)
        self._plot_security_benchmarks(epochs_data)
        
        # Score-based plots (require genuine/imposter scores)
        if genuine_scores is not None and imposter_scores is not None:
            self._plot_score_distribution(genuine_scores, imposter_scores, eer_threshold)
            self._plot_roc_curve(genuine_scores, imposter_scores)
            self._plot_score_histogram(genuine_scores, imposter_scores, eer_threshold)
            self._plot_det_curve(genuine_scores, imposter_scores)
        else:
            print("  >> Skipping score-based plots (no score data provided)")
        
        print(f"\n✓ All charts saved to: {self.charts_dir}")
        print("="*60 + "\n")
    
    def _plot_loss_convergence(self, epochs_data: List[Dict], epochs: List[int]):
        """Fig 1: Training and validation loss convergence."""
        print("  >> Generating fig1_loss_convergence.png...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_loss = [e.get('train_loss', 0) for e in epochs_data]
        test_loss = [e.get('test_loss', 0) for e in epochs_data]
        
        ax.plot(epochs, train_loss, '-', color=self.colors['primary'], 
                linewidth=2, label='Training Loss', marker='o', markersize=4)
        ax.plot(epochs, test_loss, '--', color=self.colors['secondary'], 
                linewidth=2, label='Validation Loss', marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Convergence During Training', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Annotate best epoch
        best_idx = np.argmin(test_loss)
        ax.annotate(f'Best: {test_loss[best_idx]:.4f}', 
                   xy=(epochs[best_idx], test_loss[best_idx]),
                   xytext=(10, 20), textcoords='offset points',
                   fontsize=9, color=self.colors['success'],
                   arrowprops=dict(arrowstyle='->', color=self.colors['success']))
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'fig1_loss_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_eer_evolution(self, epochs_data: List[Dict], epochs: List[int]):
        """Fig 2: EER evolution during training."""
        print("  >> Generating fig2_eer_evolution.png...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        eer_values = [e.get('eer', 0) * 100 for e in epochs_data]  # Convert to percentage
        
        ax.plot(epochs, eer_values, '-', color=self.colors['primary'], 
                linewidth=2.5, marker='o', markersize=6)
        ax.fill_between(epochs, eer_values, alpha=0.2, color=self.colors['primary'])
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('EER (%)', fontsize=12)
        ax.set_title('Equal Error Rate (EER) Evolution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Highlight best EER
        best_idx = np.argmin(eer_values)
        ax.scatter([epochs[best_idx]], [eer_values[best_idx]], 
                  color=self.colors['success'], s=150, zorder=5, marker='*')
        ax.annotate(f'Best EER: {eer_values[best_idx]:.4f}%', 
                   xy=(epochs[best_idx], eer_values[best_idx]),
                   xytext=(15, 15), textcoords='offset points',
                   fontsize=10, fontweight='bold', color=self.colors['success'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=self.colors['success']))
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'fig2_eer_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tar_far_evolution(self, epochs_data: List[Dict], epochs: List[int]):
        """Fig 3: TAR@FAR metrics evolution."""
        print("  >> Generating fig3_tar_far_evolution.png...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tar_001 = [e.get('tar_at_001_far', 0) * 100 for e in epochs_data]
        tar_01 = [e.get('tar_at_01_far', 0) * 100 for e in epochs_data]
        tar_1 = [e.get('tar_at_1_far', 0) * 100 for e in epochs_data]
        
        ax.plot(epochs, tar_001, '-', color=self.colors['primary'], linewidth=2, 
                label='TAR@FAR=0.01%', marker='o', markersize=4)
        ax.plot(epochs, tar_01, '-', color=self.colors['success'], linewidth=2, 
                label='TAR@FAR=0.1%', marker='s', markersize=4)
        ax.plot(epochs, tar_1, '-', color=self.colors['warning'], linewidth=2, 
                label='TAR@FAR=1%', marker='^', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('TAR (%)', fontsize=12)
        ax.set_title('True Acceptance Rate at Various FAR Thresholds', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'fig3_tar_far_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dprime_separability(self, epochs_data: List[Dict], epochs: List[int]):
        """Fig 4: D-prime separability metric evolution."""
        print("  >> Generating fig4_dprime_separability.png...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        d_prime = [e.get('d_prime', 0) for e in epochs_data]
        
        ax.plot(epochs, d_prime, '-', color=self.colors['purple'], 
                linewidth=2.5, marker='D', markersize=5)
        ax.fill_between(epochs, d_prime, alpha=0.2, color=self.colors['purple'])
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel("D-prime (d')", fontsize=12)
        ax.set_title("D-prime Separability Evolution", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add reference lines
        ax.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5, label="d'=3.0 (Good)")
        ax.axhline(y=4.0, color=self.colors['success'], linestyle='--', alpha=0.5, label="d'=4.0 (Excellent)")
        ax.legend(loc='lower right', fontsize=9)
        
        # Highlight best d-prime
        best_idx = np.argmax(d_prime)
        ax.scatter([epochs[best_idx]], [d_prime[best_idx]], 
                  color=self.colors['success'], s=150, zorder=5, marker='*')
        ax.annotate(f"Best d': {d_prime[best_idx]:.2f}", 
                   xy=(epochs[best_idx], d_prime[best_idx]),
                   xytext=(10, -20), textcoords='offset points',
                   fontsize=10, fontweight='bold', color=self.colors['success'])
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'fig4_dprime_separability.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_score_distribution(self, genuine_scores: np.ndarray, 
                                 imposter_scores: np.ndarray,
                                 eer_threshold: Optional[float] = None):
        """Fig 5: Score distribution (genuine vs imposter)."""
        print("  >> Generating fig5_score_distribution.png...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        ax.hist(imposter_scores, bins=100, alpha=0.7, label='Imposter', 
                color=self.colors['imposter'], density=True, edgecolor='white', linewidth=0.5)
        ax.hist(genuine_scores, bins=100, alpha=0.7, label='Genuine', 
                color=self.colors['genuine'], density=True, edgecolor='white', linewidth=0.5)
        
        # Add EER threshold line
        if eer_threshold is not None:
            ax.axvline(eer_threshold, color=self.colors['success'], linestyle='--', 
                      linewidth=2, label=f'EER Threshold: {eer_threshold:.4f}')
        
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Score Distribution: Genuine vs Imposter', fontsize=14, fontweight='bold')
        ax.legend(loc='upper center', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics annotation
        stats_text = (f"Genuine: μ={np.mean(genuine_scores):.3f}, σ={np.std(genuine_scores):.3f}\n"
                     f"Imposter: μ={np.mean(imposter_scores):.3f}, σ={np.std(imposter_scores):.3f}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'fig5_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_security_benchmarks(self, epochs_data: List[Dict]):
        """Fig 6: Security benchmarks comparison (bar chart of final metrics)."""
        print("  >> Generating fig6_security_benchmarks.png...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get final epoch data
        final = epochs_data[-1]
        
        metrics = {
            'EER': final.get('eer', 0) * 100,
            'FNIR@0.01%': final.get('fnir_at_001_fpir', 0) * 100,
            'FNIR@0.1%': final.get('fnir_at_01_fpir', 0) * 100,
            'FNIR@1%': final.get('fnir_at_1_fpir', 0) * 100,
        }
        
        x = np.arange(len(metrics))
        bars = ax.bar(x, list(metrics.values()), color=[
            self.colors['primary'],
            self.colors['secondary'],
            self.colors['warning'],
            self.colors['success']
        ], alpha=0.8, edgecolor='white', linewidth=1.5)
        
        ax.set_ylabel('Error Rate (%)', fontsize=12)
        ax.set_title('Security Benchmarks (Final Epoch)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics.keys(), fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'fig6_security_benchmarks.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, genuine_scores: np.ndarray, imposter_scores: np.ndarray):
        """Fig 7: ROC Curve."""
        print("  >> Generating fig7_roc_curve.png...")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create labels
        y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))])
        y_scores = np.concatenate([genuine_scores, imposter_scores])
        
        # Compute ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color=self.colors['primary'], linewidth=2.5, 
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        # Fill area under curve
        ax.fill_between(fpr, tpr, alpha=0.2, color=self.colors['primary'])
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'fig7_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_score_histogram(self, genuine_scores: np.ndarray, 
                              imposter_scores: np.ndarray,
                              eer_threshold: Optional[float] = None):
        """Fig 8: Score Distribution Histogram (alternative view)."""
        print("  >> Generating fig8_score_histogram.png...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Genuine histogram
        ax1.hist(genuine_scores, bins=80, color=self.colors['genuine'], 
                alpha=0.8, edgecolor='white', linewidth=0.5)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Genuine Score Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        if eer_threshold is not None:
            ax1.axvline(eer_threshold, color=self.colors['warning'], linestyle='--', linewidth=2)
        
        # Imposter histogram  
        ax2.hist(imposter_scores, bins=80, color=self.colors['imposter'], 
                alpha=0.8, edgecolor='white', linewidth=0.5)
        ax2.set_xlabel('Similarity Score', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Imposter Score Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        if eer_threshold is not None:
            ax2.axvline(eer_threshold, color=self.colors['warning'], linestyle='--', 
                       linewidth=2, label=f'EER Threshold: {eer_threshold:.4f}')
            ax2.legend(loc='upper right', fontsize=10)
        
        fig.suptitle('Score Distribution Histogram', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'fig8_score_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_det_curve(self, genuine_scores: np.ndarray, imposter_scores: np.ndarray):
        """Fig 9: Detection Error Tradeoff (DET) Curve."""
        print("  >> Generating fig9_det_curve.png...")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create labels
        y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))])
        y_scores = np.concatenate([genuine_scores, imposter_scores])
        
        # Compute ROC to get FPR and FNR
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        
        # DET curve uses normal deviate scale but we'll use log scale for simplicity
        # Filter out zeros for log scale
        mask = (fpr > 0) & (fnr > 0)
        fpr_plot = fpr[mask]
        fnr_plot = fnr[mask]
        
        ax.loglog(fpr_plot * 100, fnr_plot * 100, color=self.colors['primary'], 
                 linewidth=2.5, label='DET Curve')
        
        # Add diagonal reference
        ax.loglog([0.1, 50], [0.1, 50], 'k--', linewidth=1, label='EER Line')
        
        ax.set_xlim([0.1, 50])
        ax.set_ylim([0.1, 50])
        ax.set_xlabel('False Positive Rate (%)', fontsize=12)
        ax.set_ylabel('False Negative Rate (%)', fontsize=12)
        ax.set_title('Detection Error Tradeoff (DET) Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'fig9_det_curve.png', dpi=300, bbox_inches='tight')
        plt.close()


def generate_training_charts(output_dir: str,
                            genuine_scores: Optional[np.ndarray] = None,
                            imposter_scores: Optional[np.ndarray] = None,
                            eer_threshold: Optional[float] = None):
    """
    Convenience function to generate all training charts.
    
    Args:
        output_dir: Directory containing training_metrics.json
        genuine_scores: Optional genuine similarity scores
        imposter_scores: Optional imposter similarity scores
        eer_threshold: Optional EER threshold value
    """
    visualizer = TrainingVisualizer(output_dir)
    visualizer.generate_all_charts(genuine_scores, imposter_scores, eer_threshold)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate training visualization charts')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory with training_metrics.json')
    args = parser.parse_args()
    
    generate_training_charts(args.output_dir)
