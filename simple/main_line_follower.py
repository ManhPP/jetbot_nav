# main_line_follower.py

import cv2
import numpy as np
import traitlets
from jetbot import Robot, Camera
import time

# --- CẤU HÌNH ---

# 1. Kích thước ảnh từ camera
WIDTH = 300
HEIGHT = 300

# 2. Vùng quan tâm (Region of Interest - ROI)
# Robot chỉ cần nhìn vào một phần ảnh ngay phía trước nó.
# ROI_Y: Vị trí bắt đầu của ROI theo chiều dọc (tính từ trên xuống)
# ROI_H: Chiều cao của ROI
ROI_Y = int(HEIGHT * 0.6) # Bắt đầu từ 60% chiều cao của ảnh
ROI_H = int(HEIGHT * 0.4) # Lấy 40% chiều cao còn lại

# 3. Màu của vạch kẻ (dạng HSV)
# Đây là giá trị cho vạch MÀU ĐEN. Bạn cần thay đổi nếu vạch có màu khác.
LINE_COLOR_LOWER = np.array([95, 80, 50])
LINE_COLOR_UPPER = np.array([125, 255, 255])

# 4. Tốc độ của Robot
BASE_SPEED = 0.1 # Tốc độ cơ bản khi đi thẳng
KP = 0.7          # Hệ số bẻ lái (Proportional Gain). Tăng để bẻ lái "gắt" hơn.

# --- KHỞI TẠO ROBOT VÀ CAMERA ---
robot = Robot()
camera = Camera.instance(width=WIDTH, height=HEIGHT)

print("Khởi tạo hoàn tất. Đặt robot lên vạch kẻ và đợi 3 giây...")
time.sleep(3)
print("Bắt đầu dò line!")

# --- HÀM CHÍNH ---
try:
    while True:
        # 1. Đọc ảnh từ camera
        image = camera.value
        
        # 2. Xử lý ảnh
        # Cắt lấy vùng quan tâm (ROI)
        roi = image[ROI_Y : ROI_Y + ROI_H, :]
        
        # Chuyển từ hệ màu BGR sang HSV (dễ dàng cho việc lọc màu)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Lọc ảnh để chỉ giữ lại màu của vạch kẻ
        mask = cv2.inRange(hsv, LINE_COLOR_LOWER, LINE_COLOR_UPPER)
        
        # Tìm các đường viền trong ảnh đã lọc
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Tìm vị trí vạch kẻ và ra quyết định
        if len(contours) > 0:
            # Tìm đường viền lớn nhất (chính là vạch kẻ)
            c = max(contours, key=cv2.contourArea)
            
            # Tính toán tâm của vạch kẻ
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # --- THUẬT TOÁN ĐIỀU KHIỂN ---
                # Tính toán sai số (error): khoảng cách từ tâm vạch kẻ đến tâm màn hình
                error = cx - (WIDTH / 2)
                
                # Tính toán tốc độ bẻ lái, tỉ lệ thuận với sai số
                # Đây là bộ điều khiển tỉ lệ (P-Controller) đơn giản
                turn_speed = error / (WIDTH / 2) * KP
                
                # Đặt tốc độ cho động cơ
                # Nếu vạch kẻ bên phải (error > 0), robot cần rẽ phải (giảm tốc độ bánh phải, tăng bánh trái)
                # Nếu vạch kẻ bên trái (error < 0), robot cần rẽ trái (giảm tốc độ bánh trái, tăng bánh phải)
                left_speed = BASE_SPEED + turn_speed
                right_speed = BASE_SPEED - turn_speed
                
                robot.left_motor.value = max(0, min(1, left_speed))
                robot.right_motor.value = max(0, min(1, right_speed))
                
            else:
                # Không tìm thấy tâm, dừng lại
                robot.stop()
        else:
            # Mất vạch kẻ, dừng lại
            robot.stop()

except KeyboardInterrupt:
    # Bấm Ctrl+C để dừng chương trình
    print("Đã dừng chương trình.")
finally:
    # Đảm bảo robot dừng lại và giải phóng camera
    robot.stop()
    camera.stop()