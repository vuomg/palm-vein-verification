"""
palm_vein_processing.py

Pipeline xử lý ảnh tĩnh mạch lòng bàn tay (Palm Vein Processing):
THEO CHUẨN SCALE NORMALIZATION VỚI REFERENCE LENGTH

KIẾN TRÚC MỚI (MODULAR):
- palm_vein_preprocessing.py: Bước 1-7 (Segmentation → Resize)
- palm_vein_enhancement.py: Bước 8-9 (CLAHE → Output)
- palm_vein_processing.py: Main script + Dataset processing

PIPELINE (9 BƯỚC):
1. Segmentation (GrabCut algorithm) → Binary mask
2. Orientation normalization (Optional) → Xoay về hướng chuẩn
3. Xác định Palm Center (Distance Transform)
4. Tính Reference Length L = max(distance_transform)
5. Scale normalization: ROI size = k × L
6. Padding & boundary handling
7. Resize về 128×128 (duy nhất 1 lần)
8. Illumination enhancement (CLAHE)
9. Output cho model: 128×128 scale-normalized ROI

Chạy mặc định: python -u palm_vein_processing.py
  → Input: dataset_palm_vein/
  → Output: dataset_palm_roi/ (128×128 PNG files)
"""

import numpy as np
import cv2
import sys
import glob
import os
import time
from datetime import datetime
from typing import Tuple, List

# Import modules mới (Bước 1-7 và Bước 8-9)
from palm_vein_preprocessing import (
    preprocess_image, segment_hand, normalize_hand_orientation,
    compute_palm_center_and_reference_length, extract_roi_scale_normalized,
    preprocess_palm_vein
)
from palm_vein_enhancement import (
    enhance_palm_veins, enhance_and_prepare
)


# ===================================================================
# LEGACY FUNCTIONS (Kept for backward compatibility)
# ===================================================================
# Các hàm cũ được giữ lại để tương thích với code cũ.
# Pipeline HIỆN TẠI sử dụng các hàm từ:
# - palm_vein_preprocessing.py (Bước 1-7)
# - palm_vein_enhancement.py (Bước 8-9)
# ===================================================================

def detect_reference_points(mask: np.ndarray, contour: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    [LEGACY] Phát hiện các điểm tham chiếu trên bàn tay
    
    Điểm tham chiếu bao gồm:
    - Fingertip points (đầu ngón tay)
    - Valley points (điểm trũng giữa các ngón)
    - Palm center (tâm lòng bàn tay)
    
    Args:
        mask: Binary mask của bàn tay
        contour: Contour của bàn tay
    
    Returns:
        ref_points: Danh sách các điểm tham chiếu (x, y)
        vis: Ảnh visualization
    """
    if mask is None or contour is None:
        raise ValueError("Mask or contour is None")
    
    # Tìm convex hull và defects
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) == 0:
        raise ValueError("Failed to compute convex hull")
    
    defects = cv2.convexityDefects(contour, hull)
    
    # Phát hiện fingertip points từ convexity defects
    fingertips = []
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Lọc dựa trên góc
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = np.arccos((b**2 + c**2 - a**2)/(2*b*c)) * 180/np.pi
            
            if angle < 90:  # Valley point (giữa các ngón)
                fingertips.extend([start, end])
    
    fingertips = list(set(fingertips))  # Loại bỏ trùng lặp
    
    # Tìm palm center bằng distance transform
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
    palm_center = max_loc
    
    # Gộp tất cả reference points
    ref_points = fingertips + [palm_center]
    
    # Tạo visualization
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for pt in fingertips:
        cv2.circle(vis, pt, 5, (0,255,0), -1)
    cv2.circle(vis, palm_center, 5, (0,0,255), -1)
    
    return ref_points, vis


def extract_roi(img: np.ndarray, mask: np.ndarray, ref_points: List[Tuple[int, int]], 
                roi_size: Tuple[int, int] = (180, 180)) -> Tuple[np.ndarray, np.ndarray]:
    """
    [LEGACY - KHÔNG DÙNG] Trích xuất ROI với kích thước CỐ ĐỊNH (phương pháp cũ)
    
    ⚠️ Hàm này chỉ được giữ lại để tương thích code cũ.
    ⚠️ Pipeline HIỆN TẠI sử dụng extract_roi_scale_normalized() với:
        - Kích thước ĐỘNG: ROI_size = k × L (scale normalization)
        - L tự động adapt theo khoảng cách camera
        - Target size: 128×128 (sau khi resize)
    
    Phương pháp cũ (hàm này):
        - Kích thước CỐ ĐỊNH: 180×180 pixels (không thay đổi)
        - Không adapt theo khoảng cách camera
        - ROI có thể quá lớn/nhỏ tùy khoảng cách
    
    Args:
        img: Ảnh grayscale
        mask: Binary mask của bàn tay
        ref_points: Danh sách điểm tham chiếu
        roi_size: Kích thước CỐ ĐỊNH (mặc định 180×180) - DEPRECATED
    
    Returns:
        roi: ROI đã trích xuất với kích thước cố định
        vis: Ảnh visualization
    """
    if img is None or mask is None or ref_points is None:
        raise ValueError("Invalid input")
    
    # Lấy palm center (điểm cuối cùng trong ref_points)
    palm_center = ref_points[-1]
    
    # Tính bounding box cho ROI
    x, y = palm_center
    w, h = roi_size
    half_w, half_h = w//2, h//2
    
    # Đảm bảo ROI nằm trong ảnh
    x1, y1 = max(0, x-half_w), max(0, y-half_h)
    x2, y2 = min(img.shape[1], x+half_w), min(img.shape[0], y+half_h)
    
    # Trích xuất ROI
    roi = img[y1:y2, x1:x2]
    
    # Resize về kích thước chuẩn nếu cần
    if roi.shape[:2] != roi_size:
        roi = cv2.resize(roi, roi_size)
    
    # Tạo visualization
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,0), 2)
    
    return roi, vis


# ===================================================================
# HÀM XỬ LÝ ẢNH ĐƠN (Single Image Processing)
# ===================================================================

def process_palm_vein_image(image_path: str, output_dir: str = None, 
                            normalize_angle: bool = True, debug: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Xử lý một ảnh đơn lẻ qua pipeline chuẩn
    
    Args:
        image_path: Đường dẫn ảnh input
        output_dir: Thư mục lưu ảnh debug (nếu debug=True)
        normalize_angle: Có chuẩn hóa góc xoay không
        debug: Có lưu ảnh visualization không
        
    Returns:
        enhanced_roi: Ảnh kết quả 128x128
        results: Dictionary chứa các bước trung gian
    """
    try:
        # 1. Read Image
        if image_path.lower().endswith('.raw'):
            with open(image_path, 'rb') as f:
                raw_data = f.read()
            pixels = np.frombuffer(raw_data, dtype=np.uint8)
            expected_size = 320 * 960
            if pixels.size < expected_size:
                raise ValueError(f"Raw image size {pixels.size} is smaller than expected {expected_size}")
            
            # Update to match C# logic (480x640 by default for 307200 bytes)
            # Old logic (320x960) seems to be for a specific split-sensor mode not currently used by C#
            try:
                # Try reshaping as 640x480 (Height x Width) - Portrait mode matching C# Bitmap(480, 640)
                # Note: C# Bitmap(width, height) vs numpy reshape(rows, cols) -> reshape(640, 480)
                img = pixels[:expected_size].reshape(640, 480)
            except:
                # Fallback to old logic if reshape fails (unlikely given size check)
                image = pixels[:expected_size].reshape(320, 960)
                img = cv2.resize(image[:, 0:480], (480, 640), interpolation=cv2.INTER_CUBIC)
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
        # 2. Preprocess
        preprocessed = preprocess_image(img)
        
        # 3. Segment
        mask, contour = segment_hand(preprocessed, debug=debug)
        
        results = {
            'original': img,
            'preprocessed': preprocessed,
            'mask': mask,
            'contour': contour
        }
        
        # 4. Normalize Orientation (Optional)
        if normalize_angle:
            preprocessed, mask, angle = normalize_hand_orientation(
                preprocessed, mask, contour, debug=debug
            )
            # Update results
            results['orientation_angle'] = angle
            if debug:
                # Save orientation visualization if needed
                pass
            
            # Re-compute contour after rotation
            contours_new, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_new:
                contour = max(contours_new, key=cv2.contourArea)
            results['rotated_mask'] = mask
        
        # 5. Palm Center & Ref Length
        palm_center, L = compute_palm_center_and_reference_length(mask, method='percentile')
        results['palm_center'] = palm_center
        results['reference_length'] = L
        
        # 6. Extract ROI
        roi, roi_vis = extract_roi_scale_normalized(
            preprocessed, mask, palm_center, L,
            k=2.4, target_size=(128, 128), debug=debug
        )
        results['roi'] = roi
        results['roi_vis'] = roi_vis
        
        # 7. Enhance
        enhanced_roi = enhance_palm_veins(roi)
        results['enhanced_roi'] = enhanced_roi
        
        # Save debug images if requested
        if debug and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            try:
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_1_preprocess.png"), preprocessed)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_2_mask.png"), mask)
                if roi_vis is not None:
                    cv2.imwrite(os.path.join(output_dir, f"{base_name}_3_roi_vis.png"), roi_vis)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_4_enhanced.png"), enhanced_roi)
            except Exception as e:
                print(f"Warning: Failed to save debug images: {e}")
            
        return enhanced_roi, results
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ===================================================================
# 5. HÀM XỬ LÝ DATASET (Main Processing Function)
# ===================================================================

def get_timestamp() -> str:
    """Lấy timestamp hiện tại dạng [HH:MM:SS]"""
    return datetime.now().strftime("[%H:%M:%S]")


def process_dataset(input_dir: str, output_dir: str, normalize_orientation: bool = False):
    """
    Xử lý toàn bộ dataset palm vein theo CHUẨN SCALE NORMALIZATION
    
    PIPELINE (9 BƯỚC KHOA HỌC):
    1. Segmentation (GrabCut) → Binary mask
    2. [OPTIONAL] Orientation normalization → Xoay về hướng chuẩn
    3-4. Tính Palm Center & Reference Length L (Distance Transform)
    5. Scale normalization: ROI size = k × L
    6. Padding & boundary handling
    7. Resize về 128×128 (duy nhất 1 lần)
    8. Illumination enhancement (CLAHE)
    9. Output cho model (128×128 scale-normalized ROI)
    
    Args:
        input_dir: Thư mục chứa dataset raw
        output_dir: Thư mục lưu kết quả
        normalize_orientation: Có chuẩn hóa hướng bàn tay không (mặc định: False)
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")
    
    # Tạo output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm tất cả file .raw
    image_files = []
    for ext in ['.raw']:
        image_files.extend(glob.glob(os.path.join(input_dir, '**', f"*{ext}"), recursive=True))
    
    # Fallback sang PNG/JPG nếu không có .raw
    if not image_files:
        for ext in ['.png', '.jpg', '.jpeg']:
            image_files.extend(glob.glob(os.path.join(input_dir, '**', f"*{ext}"), recursive=True))
    
    if not image_files:
        raise ValueError(f"No images found in directory: {input_dir}")
    
    print(f"{get_timestamp()} Processing {len(image_files)} images from dataset...", flush=True)
    sys.stdout.flush()
    
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    for idx, img_path in enumerate(image_files, 1):
        # Lấy relative path để giữ cấu trúc thư mục
        rel_path = os.path.relpath(img_path, input_dir)
        rel_dir = os.path.dirname(rel_path)
        
        # Tạo output directory tương ứng
        if rel_dir:
            output_subdir = os.path.join(output_dir, rel_dir)
            os.makedirs(output_subdir, exist_ok=True)
        else:
            output_subdir = output_dir
        
        # Tạo tên file output (chuyển sang .png)
        base_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        base_name = base_name_no_ext + '.png'
        output_path = os.path.join(output_subdir, base_name)
        
        print(f"{get_timestamp()} [{idx}/{len(image_files)}] Processing: {rel_path}...", end=' ', flush=True)
        sys.stdout.flush()
        
        try:
            # Đọc ảnh (raw hoặc regular image)
            if img_path.lower().endswith('.raw'):
                # Đọc file .raw (320x960 binary)
                with open(img_path, 'rb') as f:
                    raw_data = f.read()
                
                pixels = np.frombuffer(raw_data, dtype=np.uint8)
                image = pixels[:320 * 960].reshape(320, 960)
                
                # Lấy left frame và resize
                left = image[:, 0:480]
                img = cv2.resize(left, (480, 640), interpolation=cv2.INTER_CUBIC)
            else:
                # Đọc ảnh thường
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Could not read image: {img_path}")
                
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Pipeline xử lý
            preprocessed = preprocess_image(img)
            mask, contour = segment_hand(preprocessed)
            
            # Chuẩn hóa hướng bàn tay (chỉ khi normalize_orientation=True)
            if normalize_orientation:
                preprocessed, mask, _ = normalize_hand_orientation(
                    preprocessed, mask, contour, debug=False
                )
                
                # Cập nhật contour sau khi xoay
                contours_new, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours_new:
                    contour = max(contours_new, key=cv2.contourArea)
            
            # BƯỚC 3 & 4: Tính Palm Center & Reference Length L
            # Dùng method='percentile' để tránh L quá lớn do ngón tay
            palm_center, L = compute_palm_center_and_reference_length(mask, method='percentile')
            
            # BƯỚC 5-7: Scale Normalization → Padding → Resize về 128×128
            roi, _ = extract_roi_scale_normalized(
                preprocessed, mask, palm_center, L,
                k=2.6,  # Hệ số scale:
                target_size=(128, 128),  # Kích thước chuẩn cho CNN
                debug=False
            )
            
            # BƯỚC 8: Illumination enhancement (CLAHE)
            enhanced_roi = enhance_palm_veins(roi)
            
            # Lưu kết quả
            cv2.imwrite(output_path, enhanced_roi)
            
            print("✓", flush=True)
            sys.stdout.flush()
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error: {str(e)}", flush=True)
            sys.stdout.flush()
            error_count += 1
        
        # Hiển thị progress summary mỗi 10 ảnh
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta_seconds = (len(image_files) - idx) / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            print(f"\n{get_timestamp()} --- Progress: {idx}/{len(image_files)} ({idx*100/len(image_files):.1f}%) | "
                  f"Speed: {rate:.1f} img/s | ETA: {eta_minutes:.1f} min | "
                  f"Success: {success_count} | Errors: {error_count} ---", flush=True)
            sys.stdout.flush()
    
    # In kết quả cuối cùng
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"{get_timestamp()} Processing complete!")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Enhanced ROI images saved in: {output_dir}")
    print(f"{'='*60}")


# ===================================================================
# 6. COMMAND-LINE INTERFACE
# ===================================================================

def _cli_main():
    """
    Command-line interface - chạy mặc định với dataset_palm_vein
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Palm vein image preprocessing pipeline",
        epilog="Default: python -u palm_vein_processing.py → processes dataset_palm_vein → dataset_palm_roi (NO rotation)"
    )
    parser.add_argument("--input", default="dataset_palm_vein",
                       help="Input directory (default: dataset_palm_vein)")
    parser.add_argument("--output", default="dataset_palm_roi",
                       help="Output directory (default: dataset_palm_roi)")
    parser.add_argument("--normalize-orientation", action="store_true",
                       help="Chuẩn hóa hướng bàn tay (xoay về hướng chuẩn). Mặc định: KHÔNG xoay")
    
    args = parser.parse_args()
    
    print("="*60)
    print("PALM VEIN PROCESSING PIPELINE")
    print("="*60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Normalize Orientation: {'YES (xoay bàn tay)' if args.normalize_orientation else 'NO (không xoay)'}")
    print("="*60)
    
    try:
        process_dataset(args.input, args.output, normalize_orientation=args.normalize_orientation)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())