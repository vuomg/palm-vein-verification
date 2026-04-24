"""
RSNet Core Module

Paper: "RSNet: Region-Specific Network for Contactless Palm Vein Authentication"
IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY, VOL. 20, 2025

Components:
- RSNet: Main model with lightweight CNN backbone (IR blocks + MAB blocks)
- MAB: Multi-scale Aggregation Block (Res2Net-inspired)
- RLEB: Region-based Local Enhancement Block
- AdaFaceLoss: Quality-adaptive margin loss
- DifferenceLoss: Orthogonality constraint loss
- RSNetJointLoss: Combined training loss

Backbone Architecture (Paper Section III-C):
- Stage 1: Inverted Residual (IR) blocks from MobileNetV2
- Stage 2-3: Multi-scale Aggregation Blocks (MAB) replacing IR blocks
- Stage 4: Dual-branch (Local: RLEB | Global: 1×1 conv)
"""

from .model import RSNet, MAB, RLEB, ChannelShuffle, InvertedResidual
from .losses import AdaFaceLoss, DifferenceLoss, RSNetJointLoss, get_database_config, DATABASE_CONFIGS

__all__ = [
    # Model components
    'RSNet',
    'MAB',
    'RLEB',
    'ChannelShuffle',
    'InvertedResidual',
    
    # Loss functions
    'AdaFaceLoss',
    'DifferenceLoss',
    'RSNetJointLoss',
    
    # Configuration
    'get_database_config',
    'DATABASE_CONFIGS'
]
