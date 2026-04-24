"""
palm_vein_preprocessing.py

Pipeline xử lý ảnh tĩnh mạch lòng bàn tay (Palm Vein Preprocessing):
PHẦN 1: CÁC BƯỚC 1-7 (Preprocessing đến Resize)

PIPELINE (BƯỚC 1-7):
1. Segmentation (GrabCut algorithm) → Binary mask
2. Orientation normalization (Optional) → Xoay về hướng chuẩn
3. Xác định Palm Center (Distance Transform)
4. Tính Reference Length L = max(distance_transform)
5. Scale normalization: ROI size = k × L
6. Padding & boundary handling
7. Resize về 128×128 (duy nhất 1 lần)

Output: 128×128 ROI ready for enhancement (bước 8-9)
"""

import numpy as np
import cv2
from typing import Tuple, List


# ===================================================================
# 1. CÁC HÀM XỬ LÝ HƯỚNG BÀN TAY (Hand Orientation)
# ===================================================================

def detect_hand_orientation(mask: np.ndarray, contour: np.ndarray, debug: bool = False) -> Tuple[float, np.ndarray]:
    """
    Phát hiện góc nghiêng của bàn tay sử dụng Weight Vector Method
    
    Nguyên lý: Vector từ tâm lòng bàn tay (Palm Center) đến trọng tâm (Centroid) 
    chỉ hướng của ngón tay
    
    Args:
        mask: Binary mask của bàn tay
        contour: Contour của bàn tay
        debug: Có tạo ảnh visualization không
    
    Returns:
        angle: Góc cần xoay (độ)
        vis: Ảnh visualization (nếu debug=True)
    """
    # Bước 1: Tìm tâm lòng bàn tay bằng Distance Transform
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, palm_center_tuple = cv2.minMaxLoc(dist_transform)
    palm_center = np.array(palm_center_tuple, dtype=np.float32)
    
    # Bước 2: Tính trọng tâm (centroid) của tất cả pixel trắng
    M = cv2.moments(mask)
    if M["m00"] == 0:
        centroid = palm_center.copy()
    else:
        centroid = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float32)
    
    # Bước 3: Tính Weight Vector (Palm Center → Centroid)
    weight_vector = centroid - palm_center
    vector_angle = np.arctan2(weight_vector[1], weight_vector[0]) * 180 / np.pi
    
    # Bước 4: Tính góc cần xoay để ngón tay hướng lên (-90°)
    angle_to_rotate = vector_angle - (-90)
    angle_to_rotate = ((angle_to_rotate + 180) % 360) - 180  # Normalize về [-180, 180]
    
    # Bước 5: Kiểm tra nếu vector quá nhỏ (bàn tay nắm/đóng)
    vector_magnitude = np.linalg.norm(weight_vector)
    if vector_magnitude < 10.0:
        angle_to_rotate = 0.0
    
    # Tạo visualization (optional)
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    palm_center_int = tuple(palm_center.astype(int))
    centroid_int = tuple(centroid.astype(int))
    cv2.arrowedLine(vis, palm_center_int, centroid_int, (0, 0, 255), 3, tipLength=0.3)
    cv2.circle(vis, palm_center_int, 10, (255, 0, 0), -1)
    cv2.circle(vis, centroid_int, 8, (0, 255, 0), -1)
    
    return angle_to_rotate, vis


def rotate_image_and_mask(img: np.ndarray, mask: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Xoay ảnh và mask cùng một góc
    
    Args:
        img: Ảnh grayscale cần xoay
        mask: Binary mask cần xoay
        angle: Góc xoay (độ, dương = ngược chiều kim đồng hồ)
    
    Returns:
        rotated_img: Ảnh đã xoay
        rotated_mask: Mask đã xoay
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Tạo ma trận xoay
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Xoay ảnh (dùng bilinear interpolation)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    
    # Xoay mask (dùng nearest neighbor để giữ giá trị binary)
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
    
    return rotated_img, rotated_mask


def normalize_hand_orientation(img: np.ndarray, mask: np.ndarray, contour: np.ndarray, 
                               angle_threshold: float = 5.0, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Chuẩn hóa hướng bàn tay (xoay để ngón tay hướng lên)
    
    Args:
        img: Ảnh grayscale
        mask: Binary mask của bàn tay
        contour: Contour của bàn tay
        angle_threshold: Bỏ qua xoay nếu góc < threshold (độ)
        debug: Có lưu ảnh debug không
    
    Returns:
        normalized_img: Ảnh đã chuẩn hóa
        normalized_mask: Mask đã chuẩn hóa  
        rotation_angle: Góc đã xoay
    """
    # Phát hiện góc nghiêng
    angle, orientation_vis = detect_hand_orientation(mask, contour, debug)
    
    # Nếu góc nhỏ, không cần xoay
    if abs(angle) < angle_threshold:
        return img, mask, 0.0
    
    # Xoay ảnh và mask
    rotated_img, rotated_mask = rotate_image_and_mask(img, mask, angle)
    
    return rotated_img, rotated_mask, angle


# ===================================================================
# 2. CÁC HÀM TIỀN XỬ LÝ VÀ PHÂN ĐOẠN (Preprocessing & Segmentation)
# ===================================================================

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý ảnh cơ bản
    
    Args:
        img: Ảnh đầu vào
    
    Returns:
        gray: Ảnh grayscale đã normalize
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Chuyển sang grayscale nếu cần
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Normalize contrast
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    return gray


def segment_hand(img: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phân đoạn bàn tay khỏi background sử dụng GrabCut algorithm
    
    Quy trình:
    1. Modified Otsu thresholding (giảm ngưỡng 15% để giữ đầy đủ bàn tay)
    2. GrabCut refinement
    3. Morphological operations để làm sạch
    
    Args:
        img: Ảnh grayscale
        debug: Có lưu ảnh debug không
    
    Returns:
        mask: Binary mask của bàn tay
        contour: Contour của bàn tay
    """
    # Bước 1: Modified Otsu - giảm threshold 15% để giữ đầy đủ bàn tay
    otsu_thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower_thresh = otsu_thresh * 0.85
    _, binary_init = cv2.threshold(img, lower_thresh, 255, cv2.THRESH_BINARY)
    
    # Làm sạch binary mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    binary_init = cv2.morphologyEx(binary_init, cv2.MORPH_CLOSE, kernel)
    
    # Tìm contour để lấy bounding box
    contours_init, _ = cv2.findContours(binary_init, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_init:
        raise ValueError("No hand contour found")
    
    # Lấy bounding box với margin
    largest_contour = max(contours_init, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2*margin)
    h = min(img.shape[0] - y, h + 2*margin)
    rect = (x, y, w, h)
    
    # Bước 2: GrabCut refinement
    img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    mask_grabcut = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img_3ch, mask_grabcut, rect, bgd_model, fgd_model, 
                iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
    
    # Chuyển GrabCut mask thành binary (lấy cả FG và Probably FG)
    binary = np.where((mask_grabcut == 2) | (mask_grabcut == 0), 0, 1).astype('uint8') * 255
    
    # Bước 3: Làm sạch mask cuối cùng
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Tìm contour cuối cùng
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No hand contour found after GrabCut")
    
    hand_contour = max(contours, key=cv2.contourArea)
    
    # Tạo mask sạch từ contour
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [hand_contour], -1, 255, -1)
    
    return mask, hand_contour


# ===================================================================
# 3. CÁC HÀM TRÍCH XUẤT ĐẶC TRƯNG - SCALE NORMALIZATION (BƯỚC 3-7)
# ===================================================================

def compute_palm_center_and_reference_length(mask: np.ndarray, 
                                              method: str = 'percentile') -> Tuple[Tuple[int, int], float]:
    """
    BƯỚC 3 & 4: Xác định Palm Center và tính Reference Length L
    
    Sử dụng Distance Transform để:
    - Tìm palm center (điểm có khoảng cách lớn nhất đến biên)
    - Tính reference length L (QUAN TRỌNG: phải phản ánh kích thước LÒNG BÀN TAY, không phải ngón tay)
    
    L phản ánh kích thước tay trong ảnh và tự động thay đổi theo khoảng cách chụp.
    
    Args:
        mask: Binary mask của bàn tay (sau orientation normalization)
        method: Phương pháp tính L
                - 'percentile' (mặc định, khuyến nghị): Dùng 80th percentile của distance transform
                - 'max': Dùng max distance (có thể bị ảnh hưởng bởi ngón tay)
    
    Returns:
        palm_center: (cx, cy) - Tọa độ tâm lòng bàn tay
        L: Reference length (pixels) - Đại diện cho bán kính lòng bàn tay
    """
    if mask is None or mask.sum() == 0:
        raise ValueError("Invalid or empty mask")
    
    # Distance Transform
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Tìm điểm có distance lớn nhất → Palm Center
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
    palm_center = max_loc  # (x, y)
    
    # Tính Reference Length L
    if method == 'max':
        # Phương pháp cũ: Dùng max distance
        # ⚠️ Có thể cho L quá lớn nếu mask bao gồm ngón tay dài
        L = max_val
    
    elif method == 'percentile':
        # Phương pháp khuyến nghị: Dùng percentile
        # Lọc chỉ lấy vùng có distance > 0 (inside mask)
        distances_inside = dist_transform[dist_transform > 0]
        
        if len(distances_inside) == 0:
            raise ValueError("No valid distances in mask")
        
        # Dùng 80th percentile thay vì max
        # Lý do: Loại bỏ outliers (vùng ngón tay có distance rất lớn)
        # 80% thay vì 85% để tập trung hơn vào vùng lòng bàn tay
        L = np.percentile(distances_inside, 80)
        
        # Đảm bảo L không nhỏ hơn 70% max_val (tránh quá nhỏ)
        L = max(L, 0.7 * max_val)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'percentile' or 'max'")
    
    return palm_center, L


def extract_roi_scale_normalized(img: np.ndarray, 
                                  mask: np.ndarray,
                                  palm_center: Tuple[int, int],
                                  L: float,
                                  k: float = 2.5,
                                  target_size: Tuple[int, int] = (128, 128),
                                  debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    BƯỚC 5-7: Scale Normalization → Padding → Resize
    
    Quy trình:
    1. Tính side = k × L (scale-normalized ROI size)
    2. Cắt ROI vuông tâm palm_center, cạnh = side
    3. Padding nếu vượt biên
    4. Resize về target_size (128×128)
    
    Args:
        img: Ảnh grayscale (sau orientation normalization)
        mask: Binary mask
        palm_center: (cx, cy) từ compute_palm_center_and_reference_length
        L: Reference length
        k: Hệ số scale (mặc định 2.5, range khuyến nghị: 2.3-2.8)
        target_size: Kích thước cuối cùng cho CNN (128×128)
        debug: Có tạo visualization không
    
    Returns:
        roi: ROI scale-normalized, resized về target_size
        vis: Ảnh visualization (nếu debug=True)
    """
    if img is None or palm_center is None or L <= 0:
        raise ValueError("Invalid input for ROI extraction")
    
    h, w = img.shape[:2]
    cx, cy = palm_center
    
    # BƯỚC 5: Scale Normalization
    # Tính cạnh ROI dựa trên reference length
    side = int(k * L)
    half_side = side // 2
    
    # Tính tọa độ ROI
    x1 = cx - half_side
    y1 = cy - half_side
    x2 = cx + half_side
    y2 = cy + half_side
    
    # BƯỚC 6: Padding & Boundary Handling
    # Xử lý trường hợp ROI vượt biên
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    
    # Điều chỉnh tọa độ cắt
    x1_crop = max(0, x1)
    y1_crop = max(0, y1)
    x2_crop = min(w, x2)
    y2_crop = min(h, y2)
    
    # Cắt ROI
    roi = img[y1_crop:y2_crop, x1_crop:x2_crop]
    
    # Padding nếu cần (đảm bảo ROI là hình vuông side×side)
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        roi = cv2.copyMakeBorder(
            roi,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REPLICATE  # Replicate edge pixels
        )
    
    # Đảm bảo ROI đúng kích thước side×side
    if roi.shape[0] != side or roi.shape[1] != side:
        roi = cv2.resize(roi, (side, side), interpolation=cv2.INTER_AREA)
    
    # BƯỚC 7: Resize về kích thước chuẩn (128×128)
    # Resize duy nhất 1 lần để CNN thấy scale nhất quán
    if (side, side) != target_size:
        # Dùng INTER_AREA cho downscale (tốt hơn INTER_LINEAR)
        interp = cv2.INTER_AREA if side > target_size[0] else cv2.INTER_LINEAR
        roi = cv2.resize(roi, target_size, interpolation=interp)
    
    # Visualization (nếu debug)
    vis = None
    if debug:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Vẽ ROI bounding box
        cv2.rectangle(vis, (x1_crop, y1_crop), (x2_crop, y2_crop), (0, 255, 0), 2)
        # Vẽ palm center
        cv2.circle(vis, palm_center, 5, (0, 0, 255), -1)
        # Vẽ reference length circle
        cv2.circle(vis, palm_center, int(L), (255, 0, 0), 2)
    
    return roi, vis


# ===================================================================
# 4. HÀM PIPELINE ĐẦY ĐỦ (Bước 1-7)
# ===================================================================

def preprocess_palm_vein(img: np.ndarray, 
                         normalize_orientation: bool = False,
                         k: float = 2.5,
                         target_size: Tuple[int, int] = (128, 128),
                         debug: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Pipeline đầy đủ cho bước 1-7: Preprocessing đến Resize
    
    Args:
        img: Ảnh đầu vào (grayscale hoặc color)
        normalize_orientation: Có chuẩn hóa hướng bàn tay không
        k: Hệ số scale cho ROI extraction
        target_size: Kích thước target (128x128)
        debug: Có return thông tin debug không
    
    Returns:
        roi: ROI 128×128 đã được scale-normalized (chưa enhance)
        info: Dictionary chứa thông tin debug nếu debug=True
    """
    # BƯỚC 1: Preprocess
    preprocessed = preprocess_image(img)
    
    # BƯỚC 2: Segment
    mask, contour = segment_hand(preprocessed, debug=debug)
    
    # BƯỚC 3: Normalize orientation (optional)
    rotation_angle = 0.0
    if normalize_orientation:
        preprocessed, mask, rotation_angle = normalize_hand_orientation(
            preprocessed, mask, contour, debug=debug
        )
        # Re-compute contour after rotation
        contours_new, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_new:
            contour = max(contours_new, key=cv2.contourArea)
    
    # BƯỚC 4 & 5: Palm Center và Reference Length
    palm_center, L = compute_palm_center_and_reference_length(mask, method='percentile')
    
    # BƯỚC 6 & 7: Extract ROI với scale normalization + Resize
    roi, roi_vis = extract_roi_scale_normalized(
        preprocessed, mask, palm_center, L,
        k=k, target_size=target_size, debug=debug
    )
    
    # Prepare debug info
    info = {}
    if debug:
        info = {
            'preprocessed': preprocessed,
            'mask': mask,
            'contour': contour,
            'rotation_angle': rotation_angle,
            'palm_center': palm_center,
            'reference_length': L,
            'roi_vis': roi_vis
        }
    
    return roi, info
