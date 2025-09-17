#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import os
import json
import math
from enum import Enum

from jetbot import Robot
import onnxruntime as ort
from pyzbar.pyzbar import decode
import paho.mqtt.client as mqtt

# Import các message type của ROS
from sensor_msgs.msg import LaserScan

# =============================================================================
# --- LỚP ĐIỀU KHIỂN CHÍNH ---
# =============================================================================

class JetBotLidarController:
    def __init__(self):
        """
        Hàm khởi tạo, thiết lập tất cả các thành phần của robot.
        """
        rospy.loginfo("Đang khởi tạo JetBot Lidar Controller...")

        # --- CẤU HÌNH ---
        self.setup_parameters()

        # --- KHỞI TẠO PHẦN CỨNG VÀ AI ---
        self.initialize_hardware()
        self.initialize_yolo()
        self.initialize_mqtt()

        # --- KHỞI TẠO ROS ---
        self.latest_scan = None
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.loginfo("Đã đăng ký vào topic /scan của LiDAR.")
        
        rospy.loginfo("Khởi tạo hoàn tất. Sẵn sàng hoạt động.")

    def setup_parameters(self):
        """Tập trung tất cả các tham số cấu hình vào một nơi."""
        # Thông số Robot và Dò line
        self.WIDTH, self.HEIGHT = 300, 300
        self.ROI_Y, self.ROI_H = int(self.HEIGHT * 0.6), int(self.HEIGHT * 0.4)
        self.LINE_COLOR_LOWER = np.array([95, 80, 50])
        self.LINE_COLOR_UPPER = np.array([125, 255, 255])
        self.BASE_SPEED, self.STEERING_GAIN, self.TURN_SPEED = 0.15, 0.5, 0.2
        self.TURN_DURATION_90_DEG = 1.5

        # Thông số LiDAR để phát hiện giao lộ
        self.LIDAR_MIN_DISTANCE = 0.15
        self.LIDAR_MAX_DISTANCE = 1.5
        self.LIDAR_OBJECT_MIN_POINTS = 5
        self.LIDAR_OPPOSITE_TOLERANCE = 15.0
        self.LIDAR_MIN_OPPOSITE_DISTANCE = 150.0

        # Thông số quét đường bằng camera
        self.SCAN_REGION_WIDTH_PERCENT = 0.3
        self.SCAN_PIXEL_THRESHOLD = 100

        # Thông số YOLO
        self.YOLO_MODEL_PATH = "yolo_model.onnx"
        self.YOLO_CONF_THRESHOLD = 0.6
        self.YOLO_INPUT_SIZE = (640, 640)
        self.YOLO_CLASS_NAMES = ['turn_left', 'turn_right', 'go_straight', 'qr_code', 'math_problem']

        # Thông số MQTT
        self.MQTT_BROKER = "localhost"
        self.MQTT_PORT = 1883
        self.MQTT_DATA_TOPIC = "jetbot/lidar_yolo_data"

        # Trạng thái và Hướng
        self.current_state = RobotState.FOLLOWING_LINE
        self.DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 0
        self.ANGLE_TO_FACE_SIGN_MAP = {d: a for d, a in zip(self.DIRECTIONS, [45, -45, -135, 135])}

    def initialize_hardware(self):
        """Khởi tạo camera và động cơ của JetBot."""
        try:
            self.robot = Robot()
            self.camera = Camera.instance(width=self.WIDTH, height=self.HEIGHT)
            rospy.loginfo("Phần cứng JetBot đã được khởi tạo.")
        except Exception as e:
            rospy.logwarn(f"Không tìm thấy phần cứng JetBot, sử dụng Mock objects. Lỗi: {e}")
            class MockRobot:
                def stop(self): pass
                def set_motors(self, l, r): pass
            class MockCamera:
                def __init__(self, w, h): self.value = np.zeros((h, w, 3), dtype=np.uint8)
                def stop(self): pass
            self.robot, self.camera = MockRobot(), MockCamera(self.WIDTH, self.HEIGHT)

    def initialize_yolo(self):
        """Tải mô hình YOLO vào ONNX Runtime."""
        try:
            self.yolo_session = ort.InferenceSession(self.YOLO_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            rospy.loginfo("Tải mô hình YOLO thành công.")
        except Exception as e:
            rospy.logerr(f"Không thể tải mô hình YOLO từ '{self.YOLO_MODEL_PATH}'. Lỗi: {e}")
            self.yolo_session = None

    def initialize_mqtt(self):
        """Khởi tạo và kết nối tới MQTT Broker."""
        self.mqtt_client = mqtt.Client()
        def on_connect(client, userdata, flags, rc):
            rospy.loginfo(f"Kết nối MQTT: {'Thành công' if rc == 0 else f'Thất bại, mã lỗi: {rc}'}")
        self.mqtt_client.on_connect = on_connect
        try:
            self.mqtt_client.connect(self.MQTT_BROKER, self.MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            rospy.logerr(f"Không thể kết nối tới MQTT Broker: {e}")
            
    # --- CÁC HÀM CALLBACK VÀ VÒNG LẶP CHÍNH ---

    def scan_callback(self, scan_msg):
        """Lưu lại dữ liệu scan mới nhất từ LiDAR."""
        self.latest_scan = scan_msg

    def run(self):
        """Vòng lặp điều khiển chính của robot."""
        rospy.loginfo("Bắt đầu vòng lặp điều khiển chính. Đợi 3 giây...")
        time.sleep(3)
        rospy.loginfo("Hành trình bắt đầu!")
        
        rate = rospy.Rate(20) # 20Hz

        while not rospy.is_shutdown():
            if self.current_state == RobotState.FOLLOWING_LINE:
                if self.process_lidar_for_intersection():
                    rospy.loginfo("!!! LiDAR ĐÃ PHÁT HIỆN GIAO LỘ !!!")
                    self.current_state = RobotState.AT_INTERSECTION
                    self.handle_intersection()
                    self.latest_scan = None
                else:
                    self.go_straight(self.camera.value)
            
            elif self.current_state == RobotState.DEAD_END:
                rospy.logwarn("Robot ở trạng thái ngõ cụt. Dừng hoạt động.")
                break
            
            rate.sleep()

        self.cleanup()

    def cleanup(self):
        """Hàm dọn dẹp tài nguyên khi node tắt."""
        rospy.loginfo("Dừng robot và giải phóng tài nguyên...")
        self.robot.stop()
        if self.camera: self.camera.stop()
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        rospy.loginfo("Đã giải phóng tài nguyên. Chương trình kết thúc.")

    # --- LOGIC XỬ LÝ LIDAR ---

    def process_lidar_for_intersection(self):
        """
        Phát hiện giao lộ.
        """
        if self.latest_scan is None:
            return False

        scan = self.latest_scan
        ranges = np.array(scan.ranges)

        # 1. Tìm tất cả các vật thể (cụm điểm hợp lệ) trong toàn bộ scan
        all_objects = []
        valid_indices = np.where((ranges > self.LIDAR_MIN_DISTANCE) & 
                                 (ranges < self.LIDAR_MAX_DISTANCE) & 
                                 np.isfinite(ranges))[0]
        
        if len(valid_indices) < self.LIDAR_OBJECT_MIN_POINTS * 2:
            return False

        # Thuật toán tìm cụm (clustering)
        current_cluster_indices = [valid_indices[0]]
        for i in range(1, len(valid_indices)):
            # Kiểm tra xem điểm có gần nhau về góc (chỉ số) và khoảng cách không
            is_consecutive = (valid_indices[i] - valid_indices[i-1] <= 2)
            distance_diff = abs(ranges[valid_indices[i]] - ranges[valid_indices[i-1]])
            
            if is_consecutive and distance_diff < 0.1: # 0.1m distance threshold
                current_cluster_indices.append(valid_indices[i])
            else:
                if len(current_cluster_indices) >= self.LIDAR_OBJECT_MIN_POINTS:
                    all_objects.append(current_cluster_indices)
                current_cluster_indices = [valid_indices[i]]
        
        if len(current_cluster_indices) >= self.LIDAR_OBJECT_MIN_POINTS:
            all_objects.append(current_cluster_indices)

        if len(all_objects) < 2:
            return False

        # 2. Tính toán góc trung tâm cho mỗi vật thể
        object_angles = []
        for obj_indices in all_objects:
            center_index = obj_indices[len(obj_indices) // 2]
            angle_rad = scan.angle_min + center_index * scan.angle_increment
            object_angles.append(math.degrees(angle_rad))
        
        # 3. Tìm các cặp vật thể đối diện nhau
        for i in range(len(object_angles)):
            for j in range(i + 1, len(object_angles)):
                angle1 = object_angles[i]
                angle2 = object_angles[j]
                
                diff = abs(angle1 - angle2)
                angle_diff = min(diff, 360 - diff)

                if angle_diff > self.LIDAR_MIN_OPPOSITE_DISTANCE and \
                   abs(angle_diff - 180.0) < self.LIDAR_OPPOSITE_TOLERANCE:
                    rospy.loginfo(f"[Robust] Phát hiện cặp đối diện: {angle1:.1f}° và {angle2:.1f}°, chênh lệch {angle_diff:.1f}°")
                    return True
        
        return False
        
    # --- CÁC HÀM LOGIC ROBOT ---
    
    def go_straight(self, image):
        """Điều khiển robot đi thẳng theo vạch kẻ."""
        roi = image[self.ROI_Y : self.ROI_Y + self.ROI_H, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                error = cx - (self.WIDTH / 2)
                adj = error / (self.WIDTH / 2) * self.STEERING_GAIN
                self.robot.set_motors(self.BASE_SPEED + adj, self.BASE_SPEED - adj)
            else: self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
        else: self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)

    def turn_robot(self, degrees):
        """Xoay robot tại chỗ một góc xác định."""
        duration = abs(degrees) / 90.0 * self.TURN_DURATION_90_DEG
        if degrees > 0: self.robot.set_motors(self.TURN_SPEED, -self.TURN_SPEED)
        elif degrees < 0: self.robot.set_motors(-self.TURN_SPEED, self.TURN_SPEED)
        if degrees != 0: time.sleep(duration)
        self.robot.stop()
        if degrees % 90 == 0:
            num_turns = round(degrees / 90)
            self.current_direction_index = (self.current_direction_index + num_turns) % 4
            rospy.loginfo(f"Hướng đi mới: {self.DIRECTIONS[self.current_direction_index].name}")
        time.sleep(0.5)

    def scan_for_available_paths(self, image):
        """Quét ảnh để tìm các lối đi có vạch kẻ (logic đầy đủ đã được khôi phục)."""
        roi = image[self.ROI_Y : self.ROI_Y + self.ROI_H, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        w = int(self.WIDTH * self.SCAN_REGION_WIDTH_PERCENT)
        cs, ce = int(self.WIDTH/2 - w/2), int(self.WIDTH/2 + w/2)
        paths = {
            "left": np.sum(mask[:, 0:w] > 0) > self.SCAN_PIXEL_THRESHOLD,
            "straight": np.sum(mask[:, cs:ce] > 0) > self.SCAN_PIXEL_THRESHOLD,
            "right": np.sum(mask[:, self.WIDTH-w:self.WIDTH] > 0) > self.SCAN_PIXEL_THRESHOLD
        }
        rospy.loginfo(f"Quét đường: Trái({paths['left']}), Thẳng({paths['straight']}), Phải({paths['right']})")
        return paths

    def publish_data(self, payload_dict):
        """Chuyển đổi dictionary thành JSON và publish lên MQTT."""
        try:
            json_payload = json.dumps(payload_dict)
            self.mqtt_client.publish(self.MQTT_DATA_TOPIC, json_payload)
            rospy.loginfo(f"Đã publish MQTT: {json_payload}")
        except Exception as e: rospy.logerr(f"Lỗi khi publish MQTT: {e}")
        
    def solve_math_problem(self, problem_string):
        """Giải một phép toán đơn giản từ chuỗi, an toàn hơn `eval`."""
        try:
            safe_string = "".join(c for c in problem_string if c in "0123456789.+-*/ ")
            return eval(safe_string)
        except: return None

    def detect_with_yolo(self, image):
        """Thực hiện nhận diện đối tượng bằng YOLO."""
        if self.yolo_session is None: return []
        original_height, original_width = image.shape[:2]
        img_resized = cv2.resize(image, self.YOLO_INPUT_SIZE)
        img_data = np.array(img_resized, dtype=np.float32) / 255.0
        img_data = np.transpose(img_data, (2, 0, 1))
        input_tensor = np.expand_dims(img_data, axis=0)
        input_name = self.yolo_session.get_inputs()[0].name
        outputs = self.yolo_session.run(None, {input_name: input_tensor})
        detections = []
        output_data = outputs[0][0]
        for box_data in output_data:
            confidence = box_data[4]
            if confidence > self.YOLO_CONF_THRESHOLD:
                class_id = int(box_data[5])
                class_name = self.YOLO_CLASS_NAMES[class_id]
                cx, cy, w, h = box_data[0], box_data[1], box_data[2], box_data[3]
                x_scale = original_width / self.YOLO_INPUT_SIZE[0]
                y_scale = original_height / self.YOLO_INPUT_SIZE[1]
                x1 = int((cx - w / 2) * x_scale)
                y1 = int((cy - h / 2) * y_scale)
                x2 = int((cx + w / 2) * x_scale)
                y2 = int((cy + h / 2) * y_scale)
                detections.append({'class_name': class_name, 'confidence': float(confidence), 'box': [x1, y1, x2, y2]})
        rospy.loginfo(f"YOLO đã phát hiện {len(detections)} đối tượng.")
        return detections

    def handle_intersection(self):
        """Hàm chính xử lý toàn bộ logic khi robot dừng tại một giao lộ."""
        rospy.loginfo("\n[GIAO LỘ] Đã đến! Dừng và xử lý.")
        self.robot.stop(); time.sleep(1)
        current_direction = self.DIRECTIONS[self.current_direction_index]
        angle_to_sign = self.ANGLE_TO_FACE_SIGN_MAP.get(current_direction, 0)
        self.turn_robot(angle_to_sign)
        image_info = self.camera.value
        detections = self.detect_with_yolo(image_info)
        nav_action = None; data_published = False
        for det in detections:
            if det['class_name'] in ['turn_left', 'turn_right', 'go_straight']:
                rospy.loginfo(f"[QUYẾT ĐỊNH] Theo biển báo: {det['class_name'].upper()}")
                if det['class_name'] == 'turn_left': nav_action = -90
                elif det['class_name'] == 'turn_right': nav_action = 90
                else: nav_action = 0
                break
        if nav_action is None:
            for det in detections:
                if det['class_name'] == 'qr_code':
                    box = det['box']
                    qr_image = image_info[box[1]:box[3], box[0]:box[2]]
                    decoded = decode(qr_image)
                    if decoded:
                        qr_data = decoded[0].data.decode('utf-8')
                        self.publish_data({'type': 'QR_CODE', 'value': qr_data})
                        data_published = True
                    break
        if nav_action is None and not data_published:
             for det in detections:
                if det['class_name'] == 'math_problem':
                    # TODO: Cần tích hợp OCR ở đây
                    problem_text = "25 + 17" 
                    solution = self.solve_math_problem(problem_text)
                    if solution is not None:
                        self.publish_data({'type': 'MATH_PROBLEM', 'problem': problem_text, 'solution': solution})
                    break
        self.turn_robot(-angle_to_sign)
        if nav_action is not None:
            self.turn_robot(nav_action)
        else:
            available_paths = self.scan_for_available_paths(self.camera.value)
            if available_paths['straight']: self.turn_robot(0)
            elif available_paths['right']: self.turn_robot(90)
            elif available_paths['left']: self.turn_robot(-90)
            else:
                rospy.logwarn("[!!!] NGÕ CỤT! Dừng hoạt động.")
                self.current_state = RobotState.DEAD_END
                return
        rospy.loginfo("[GIAO LỘ] Hoàn tất xử lý. Tiếp tục dò line.")
        self.current_state = RobotState.FOLLOWING_LINE

class RobotState(Enum):
    FOLLOWING_LINE, AT_INTERSECTION, DEAD_END = 1, 2, 3

class Direction(Enum):
    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

def main():
    """Hàm chính để khởi tạo node và chạy controller."""
    rospy.init_node('jetbot_lidar_follower_node', anonymous=True)
    try:
        controller = JetBotLidarController()
        if controller.yolo_session: # Chỉ chạy nếu model được tải thành công
            controller.run()
        else:
            rospy.logerr("Không thể bắt đầu vì mô hình YOLO chưa được tải.")
    except rospy.ROSInterruptException:
        rospy.loginfo("Node đã bị ngắt.")
    except Exception as e:
        rospy.logerr(f"Lỗi không xác định trong main: {e}", exc_info=True)

if __name__ == '__main__':
    main()