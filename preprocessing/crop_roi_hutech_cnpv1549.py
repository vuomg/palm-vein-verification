"""
Cắt ROI cho vài user trong C:\\tmp\\project\\auto (dữ liệu auto-capture, đã có sẵn
tọa độ ROI trong roi_N.txt).

CHỈ CẮT ROI theo tọa độ có sẵn (perspective warp) — KHÔNG áp dụng CLAHE hay xử lý
gì khác — dùng để test nhanh model trên vài user thật, không phải build dataset
chính thức (dùng setup_internal_dataset.py cho việc đó).

Tái sử dụng đúng hàm đọc raw (_read_raw_image) và cắt ROI (_crop_roi_from_coords)
từ setup_internal_dataset.py — đã verify khớp định dạng thật của thiết bị
(raw layout 320x960, chỉ dùng nửa trái, resize lên 640x480).

Chạy:  python preprocessing/crop_roi_auto.py --n-users 3
"""
import os
import sys
import argparse
import glob as glob_module

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup_internal_dataset import _read_raw_image, _crop_roi_from_coords
import cv2


def main(src, out, n_users, target_size):
    users = sorted(
        [d for d in os.listdir(src) if d.startswith("autoUser")
         and os.path.isdir(os.path.join(src, d))],
        key=lambda s: int(s.replace("autoUser", ""))
    )[:n_users]

    total = 0
    for u in users:
        udir = os.path.join(src, u)
        outdir = os.path.join(out, u)
        os.makedirs(outdir, exist_ok=True)

        raw_files = sorted(
            glob_module.glob(os.path.join(udir, "img_*.raw")),
            key=lambda p: int(os.path.basename(p).replace("img_", "").replace(".raw", ""))
        )

        cropped = 0
        for raw_path in raw_files:
            num = os.path.basename(raw_path).replace("img_", "").replace(".raw", "")
            roi_txt = os.path.join(udir, f"roi_{num}.txt")
            if not os.path.exists(roi_txt):
                print(f"  [bỏ qua] {raw_path}: không có roi_{num}.txt")
                continue

            img = _read_raw_image(raw_path)          # raw -> ảnh 640x480 đúng chuẩn
            roi = _crop_roi_from_coords(img, roi_txt, target_size=target_size)  # chỉ cắt, không CLAHE

            out_path = os.path.join(outdir, f"roi_{num}.bmp")
            cv2.imwrite(out_path, roi)
            cropped += 1

        print(f"{u}: cắt {cropped} ảnh ROI -> {outdir}")
        total += cropped

    print(f"\nTổng: {len(users)} user, {total} ảnh ROI đã cắt -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"C:\tmp\project\auto",
                    help="thư mục dữ liệu auto-capture gốc")
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "datasets", "AUTO_ROI_test"),
                    help="thư mục xuất ROI đã cắt")
    ap.add_argument("--n-users", type=int, default=20,
                    help="số user cần cắt (mặc định 3, để test nhanh)")
    ap.add_argument("--size", type=int, default=128,
                    help="kích thước ROI vuông đầu ra (mặc định 128, khớp convention project)")
    args = ap.parse_args()
    main(args.src, args.out, args.n_users, (args.size, args.size))
