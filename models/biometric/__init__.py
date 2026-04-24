"""
Biometric - Biometric Recognition Package

Core utilities for biometric recognition systems including palm vein, 
fingerprint, iris, and face recognition.

Includes data augmentation, loss functions, metrics, and training utilities
optimized for few-shot biometric learning.
"""

__version__ = '1.0.0'
__author__ = 'Biometric Research Team'

# Import core modules for convenience
from .losses import ArcFaceLoss, TripletLoss, CenterLoss, get_loss_function
from .metrics import BiometricEvaluator, calculate_eer, calculate_fnir_fpir_at_threshold
from .early_stopping import EarlyStopping, LearningRateScheduler, TrainingMonitor

__all__ = [
    # Loss functions
    'ArcFaceLoss',
    'TripletLoss',
    'CenterLoss',
    'get_loss_function',
    
    # Metrics
    'BiometricEvaluator',
    'calculate_eer',
    'calculate_fnir_fpir_at_threshold',
    
    # Training utilities
    'EarlyStopping',
    'LearningRateScheduler',
    'TrainingMonitor',
]
