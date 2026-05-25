import cv2
import numpy as np

img = cv2.imread("test_bedroom.png", cv2.IMREAD_GRAYSCALE)

# Bản vẽ kiến trúc: nền trắng = trong phòng, tường xám/đen = tường
# Cần đảo lại: trong phòng = 85 (floor), ngoài = 0 (background)

out = np.zeros_like(img)

# Vùng sáng (trong phòng) → floor = 85
out[img > 200] = 85

# Vùng rất tối (tường dày, ngoài phòng) → background = 0
out[img < 50] = 0

# Vùng trung gian (tường mỏng) → giữ là floor
out[(img >= 50) & (img <= 200)] = 85

cv2.imwrite("test_bedroom_fixed.png", out)
