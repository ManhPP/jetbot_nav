# color_tuner.py
import cv2
import numpy as np
from jetbot import Camera

# Hàm rỗng, cần cho trackbar
def nothing(x):
    pass

# Khởi tạo camera
camera = Camera.instance(width=300, height=300)

# Tạo một cửa sổ để chứa các thanh trượt
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 180, 255, nothing) # Bắt đầu V ở mức cao
cv2.createTrackbar("U - H", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing) # Bắt đầu S ở mức thấp
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

print("Di chuyển các thanh trượt cho đến khi vạch trắng hiện rõ trên cửa sổ 'Mask'.")
print("Bấm 'q' để thoát.")

while True:
    # Đọc ảnh
    img = camera.value
    
    # Chuyển sang HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Lấy giá trị từ thanh trượt
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    
    # Tạo mask và hiển thị
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    
    # Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()
print(f"Giá trị cuối cùng:\nLower: {lower_range}\nUpper: {upper_range}")