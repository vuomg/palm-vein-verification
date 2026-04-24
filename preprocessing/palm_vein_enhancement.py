"""
palm_vein_enhancement.py

Pipeline xử lý ảnh tĩnh mạch lòng bàn tay (Palm Vein Enhancement):
PHẦN 2: CÁC BƯỚC 8-9 (Illumination Enhancement và Output)

PIPELINE (BƯỚC 8-9):
8. Illumination enhancement (CLAHE)
9. Output cho model: 128×128 scale-normalized ROI

Input: 128×128 ROI từ preprocessing (bước 1-7)
Output: 128×128 enhanced ROI ready for CNN model
"""

import numpy as np
import cv2
from typing import Tuple


# ===================================================================
# BƯỚC 8: ILLUMINATION ENHANCEMENT
# ===================================================================

def enhance_palm_veins(roi: np.ndarray, 
                       use_denoise: bool = True, 
                       clahe_clip: float = 8.0, 
                       clahe_grid: Tuple[int, int] = (8, 8), 
                       unsharp: bool = True) -> np.ndarray:
    """
    BƯỚC 8: Tăng cường độ rõ nét của tĩnh mạch lòng bàn tay
    
    Quy trình:
    1. Denoising nhẹ (giữ lại cạnh tĩnh mạch)
    2. Background subtraction (loại bỏ ánh sáng không đều)
    3. CLAHE (tăng tương phản cục bộ) - BƯỚC CHÍNH
    4. Unsharp mask (làm sắc nét)
    
    Args:
        roi: ROI cần tăng cường (128×128 từ preprocessing)
        use_denoise: Có denoise không
        clahe_clip: Clip limit cho CLAHE (mặc định: 8.0)
        clahe_grid: Grid size cho CLAHE (mặc định: 8×8)
        unsharp: Có dùng unsharp mask không
    
    Returns:
        enhanced: ROI đã tăng cường (128×128)
    """
    if roi is None or roi.size == 0:
        raise ValueError("Invalid ROI")
    
    img = roi.astype(np.float32)
    
    # Bước 1: Denoise nhẹ
    if use_denoise:
        den = cv2.fastNlMeansDenoising(np.uint8(img), None, h=0.5,
                                       templateWindowSize=7, searchWindowSize=21)
        img = cv2.bilateralFilter(den, d=3, sigmaColor=10, sigmaSpace=10)
        img = img.astype(np.float32)
    
    # Bước 2: Background subtraction
    h, w = img.shape
    block_size = 32
    step = block_size - 3
    hr = max(1, (h + step - 1) // step)
    wr = max(1, (w + step - 1) // step)
    small = cv2.resize(img, (wr, hr), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    
    sub = img - bg
    sub = cv2.normalize(sub, None, 0, 255, cv2.NORM_MINMAX)
    sub = np.clip(sub, 0, 255).astype(np.uint8)
    
    # Bước 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # ĐÂY LÀ BƯỚC CHÍNH CỦA ENHANCEMENT
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(sub)
    
    # Bước 4: Unsharp mask (làm sắc nét)
    if unsharp:
        blur = cv2.GaussianBlur(enhanced, (3, 3), 0.8)
        enhanced = cv2.addWeighted(enhanced, 1.25, blur, -0.25, 3)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced


def enhance_palm_veins_simple(roi: np.ndarray, clahe_clip: float = 8.0) -> np.ndarray:
    """
    BƯỚC 8 (Phiên bản đơn giản): Chỉ áp dụng CLAHE
    
    Sử dụng khi muốn enhancement nhanh và đơn giản hơn
    
    Args:
        roi: ROI cần tăng cường (128×128)
        clahe_clip: Clip limit cho CLAHE
    
    Returns:
        enhanced: ROI đã tăng cường bằng CLAHE
    """
    if roi is None or roi.size == 0:
        raise ValueError("Invalid ROI")
    
    # Normalize về [0, 255]
    normalized = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Áp dụng CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    
    return enhanced


# ===================================================================
# BƯỚC 9: OUTPUT CHO MODEL
# ===================================================================

def prepare_for_model(enhanced_roi: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    BƯỚC 9: Chuẩn bị ROI cho CNN model
    
    Args:
        enhanced_roi: ROI đã enhance (128×128)
        normalize: Có normalize về [0, 1] không
    
    Returns:
        model_input: Ảnh ready cho CNN (128×128, optionally normalized)
    """
    if enhanced_roi is None or enhanced_roi.size == 0:
        raise ValueError("Invalid enhanced ROI")
    
    # Đảm bảo là uint8
    if enhanced_roi.dtype != np.uint8:
        model_input = np.clip(enhanced_roi, 0, 255).astype(np.uint8)
    else:
        model_input = enhanced_roi.copy()
    
    # Normalize về [0, 1] nếu cần (cho PyTorch/TensorFlow)
    if normalize:
        model_input = model_input.astype(np.float32) / 255.0
    
    return model_input


# ===================================================================
# PIPELINE ĐẦY ĐỦ (BƯỚC 8-9)
# ===================================================================

def enhance_and_prepare(roi: np.ndarray, 
                       use_denoise: bool = True,
                       clahe_clip: float = 8.0,
                       normalize_output: bool = False) -> np.ndarray:
    """
    Pipeline đầy đủ cho bước 8-9: Enhancement + Preparation
    
    Args:
        roi: ROI 128×128 từ preprocessing
        use_denoise: Có denoise trước CLAHE không
        clahe_clip: Clip limit cho CLAHE
        normalize_output: Có normalize output về [0,1] không
    
    Returns:
        output: Enhanced ROI ready for model (128×128)
    """
    # BƯỚC 8: Enhancement
    enhanced = enhance_palm_veins(roi, use_denoise=use_denoise, clahe_clip=clahe_clip)
    
    # BƯỚC 9: Prepare for model
    output = prepare_for_model(enhanced, normalize=normalize_output)
    
    return output
