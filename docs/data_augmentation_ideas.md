# Các Kỹ Thuật Tăng Cường Dữ Liệu (Data Augmentation) Cho Palm Vein

Dưới đây là danh sách các kỹ thuật tăng cường dữ liệu được phân loại từ cơ bản đến nâng cao, đặc biệt phù hợp cho bài toán nhận diện tĩnh mạch lòng bàn tay (Palm Vein Recognition).

## 1. Biến Đổi Hình Học (Geometric Transformations)
Các kỹ thuật này mô phỏng sự thay đổi về vị trí và góc độ của bàn tay khi đặt lên thiết bị thu nhận.

- **Random Rotation**: Xoay ảnh ngẫu nhiên trong một khoảng nhỏ (ví dụ: ±10° đến ±15°). Mô phỏng việc người dùng đặt tay hơi nghiêng.
- **Random Translation (Shift)**: Dịch chuyển ảnh theo chiều ngang/dọc. Mô phỏng việc đặt tay không chính giữa.
- **Random Scale (Zoom)**: Phóng to/thu nhỏ nhẹ (ví dụ: 0.9x - 1.1x). Mô phỏng khoảng cách thay đổi giữa tay và camera.
- **Random Perspective / Shearing**: Biến đổi phối cảnh hoặc làm méo ảnh. Mô phỏng khi bàn tay không phẳng hoàn toàn hoặc camera chụp ở góc nghiêng.
- **Elastic Transform**: Biến dạng đàn hồi cục bộ. Giả lập sự co giãn của da tay hoặc tĩnh mạch (rất hiệu quả cho dữ liệu sinh trắc học).

## 2. Biến Đổi Quang Học & Nhiễu (Photometric & Noise)
Mô phỏng các điều kiện ánh sáng và chất lượng sensor khác nhau.

- **Random Brightness / Contrast**: Thay đổi độ sáng và độ tương phản. Rất quan trọng vì ảnh hồng ngoại (NIR) nhạy cảm với cường độ sáng.
- **Random Gamma**: Điều chỉnh gamma correction. Giúp mô hình bền vững với các điều kiện phơi sáng phi tuyến tính.
- **Gaussian Noise / Salt-and-Pepper Noise**: Thêm nhiễu hạt. Giả lập nhiễu từ cảm biến camera (sensor noise).
- **Gaussian Blur / Motion Blur**: Làm mờ ảnh. Mô phỏng việc lấy nét sai (out-of-focus) hoặc tay di chuyển khi chụp (motion blur).
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: *Khuyên dùng*. Tăng cường độ tương phản cục bộ, giúp làm nổi bật đường vân tĩnh mạch trước khi đưa vào mô hình.

## 3. Các Kỹ Thuật Che Chắn (Occlusion & Erasure)
Giúp mô hình học các đặc trưng cục bộ thay vì chỉ nhớ toàn bộ ảnh, tăng khả năng chống chịu khi một phần lòng bàn tay bị che khuất hoặc lỗi.

- **Random Erasing**: Chọn ngẫu nhiên một hình chữ nhật trong ảnh và xóa giá trị pixel (gán bằng 0 hoặc noise).
- **Cutout**: Tương tự Random Erasing nhưng thường là hình vuông cố định.
- **GridMask**: Che ảnh bằng một lưới các ô vuông. Giữ lại thông tin không gian tốt hơn Cutout thông thường.
- **Hide-and-Seek**: Chia ảnh thành lưới và ngẫu nhiên ẩn một số ô.

## 4. Kỹ Thuật Trộn Ảnh (Mixing Techniques)
Các kỹ thuật hiện đại giúp làm mịn biên quyết định (decision boundary) và chống over-fitting cực tốt.

- **MixUp**: Trộn 2 ảnh và 2 nhãn theo tỷ lệ alpha.
  `Image = λ * Img1 + (1-λ) * Img2`
- **CutMix**: Cắt một phần của ảnh A dán đè lên ảnh B, nhãn cũng được trộn theo tỷ lệ diện tích.
  *Lưu ý*: Với bài toán nhận diện (Verification/Identification), cần cẩn thận khi dùng MixUp/CutMix với Loss function dạng Margin (ArcFace, CosFace) vì nhãn không còn là one-hot vector thuần túy. Tuy nhiên, có thể dùng để train backbone cho khỏe hơn.

## 5. Chiến Lược Tự Động (AutoML Styles)
Nếu không chắc chắn tham số nào tốt nhất, có thể dùng các chiến lược đã được tối ưu hóa.

- **RandAugment**: Chọn ngẫu nhiên N phép biến đổi từ một tập hợp K phép biến đổi có sẵn với cường độ M. Dễ cài đặt và hiệu quả cao.
- **AutoAugment**: Một policy phức tạp hơn được học từ dữ liệu (thường pretrained trên ImageNet/CIFAR).
- **TrivialAugment**: Phiên bản đơn giản hóa của RandAugment, không cần hyperparameter tuning nhiều.

## 6. Gợi Ý Cụ Thể Cho Dự Án Hiện Tại (MobileNetV3 + SPP)

Dựa trên code hiện tại của bạn (`train.py`), bạn đang dùng:
`Rotation, Translation, GaussianNoise, Contrast, Brightness, Scale`.

**Nên thử thêm theo thứ tự ưu tiên:**
1.  **GridMask hoặc Random Erasing**: Rất đơn giản để thêm vào và thường tăng độ chính xác cho các mạng CNN (MobileNet).
2.  **Elastic Transform**: Nếu thư viện `albumentations` có sẵn, cái này rất tốt cho vân tay/tĩnh mạch.
3.  **RandAugment**: Thay thế chuỗi `BiometricCompose` thủ công hiện tại bằng RandAugment tiêu chuẩn để xem hiệu quả có tốt hơn không.
4.  **Motion Blur**: Thực tế khi quét tay người dùng thường rung nhẹ.

---
### Ví dụ Code (PyTorch Transforms)

```python
import torchvision.transforms as T

# Ví dụ thêm Random Erasing
transform_train = T.Compose([
    T.Resize((224, 224)),
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)) # Thêm dòng này
])
```
