import cv2
import numpy as np
from core.interfaces import SegmentationModel

class ColorBasedSegmentation(SegmentationModel):
    """
    [MOCK CLASS] - ĐÂY LÀ PHẦN GIẢ LẬP MODEL CỦA TEAMATE.
    Mục đích: Tự động tạo mask dựa trên màu sắc để bạn test inpainting.
    Ví dụ: Tự động tìm vật thể màu Đỏ để xóa.
    """
    def __init__(self, color_range='red'):
        self.color_range = color_range

    def get_mask(self, image: np.ndarray) -> np.ndarray:
        # Chuyển sang không gian màu HSV để lọc màu tốt hơn
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if self.color_range == 'red':
            # Định nghĩa khoảng màu đỏ (Red)
            lower1 = np.array([0, 70, 50])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 70, 50])
            upper2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = mask1 + mask2
        elif self.color_range == 'blue':
            lower = np.array([100, 150, 0])
            upper = np.array([140, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
        else:
            # Mặc định tạo mask rỗng
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Xử lý nhiễu một chút cho giống model thật (Morphology)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        print(f"[Segmentation Mock] Đã tạo mask dựa trên màu: {self.color_range}")
        return mask