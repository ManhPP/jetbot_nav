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
from sensor_msgs.msg import LaserScan, Image
from opposite_detector import SimpleOppositeDetector

class RobotState(Enum):
    DRIVING_STRAIGHT = 1
    HANDLING_EVENT = 2
    LEAVING_INTERSECTION = 3
    DEAD_END = 4

class Direction(Enum):
    NORTH, EAST, SOUTH, WEST = 1, 2, 3, 4

class JetBotCorrectedController:
    def __init__(self):
        rospy.loginfo("Đang khởi tạo JetBot Corrected Event-Driven Controller...")
        self.setup_parameters()
        self.initialize_hardware()
        self.initialize_yolo()
        self.initialize_mqtt()
        self.latest_scan = None
        self.latest_image = None
        self.detector = SimpleOppositeDetector()
        rospy.Subscriber('/scan', LaserScan, self.detector.callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        rospy.loginfo("Đã đăng ký vào các topic /scan và /csi_cam_0/image_raw.")
        self.state_change_time = rospy.get_time()
        self._set_state(RobotState.DRIVING_STRAIGHT, initial=True)
        rospy.loginfo("Khởi tạo hoàn tất. Sẵn sàng hoạt động.")

    def setup_parameters(self):
        self.WIDTH, self.HEIGHT = 300, 300
        self.BASE_SPEED = 0.16
        self.TURN_SPEED = 0.2
        self.TURN_DURATION_90_DEG = 0.8
        self.ROI_Y = int(self.HEIGHT * 0.85)
        self.ROI_H = int(self.HEIGHT * 0.15)
        self.CORRECTION_GAIN = 0.5
        self.SAFE_ZONE_PERCENT = 0.40
        self.LINE_COLOR_LOWER = np.array([95, 80, 50])
        self.LINE_COLOR_UPPER = np.array([125, 255, 255])
        self.INTERSECTION_CLEARANCE_DURATION = 1.5
        self.SCAN_PIXEL_THRESHOLD = 100
        self.YOLO_MODEL_PATH = "models/best.onnx"
        self.YOLO_CONF_THRESHOLD = 0.6
        self.YOLO_INPUT_SIZE = (640, 640)
        self.YOLO_CLASS_NAMES = ['F', 'L', 'NF', 'NL', 'NR', 'R', 'math']
        self.MQTT_BROKER = "localhost"; self.MQTT_PORT = 1883
        self.MQTT_DATA_TOPIC = "jetbot/corrected_event_data"
        self.current_state = None
        self.DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 1
        self.ANGLE_TO_FACE_SIGN_MAP = {d: a for d, a in zip(self.DIRECTIONS, [45, -45, -135, 135])}

    def initialize_hardware(self):
        try:
            self.robot = Robot()
            rospy.loginfo("Phần cứng JetBot (động cơ) đã được khởi tạo.")
        except Exception as e:
            rospy.logwarn(f"Không tìm thấy phần cứng JetBot, sử dụng Mock object. Lỗi: {e}")
            from unittest.mock import Mock
            self.robot = Mock()

    def initialize_yolo(self):
        """Tải mô hình YOLO vào ONNX Runtime."""
        try:
            self.yolo_session = ort.InferenceSession(self.YOLO_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            rospy.loginfo("Tải mô hình YOLO thành công.")
        except Exception as e:
            rospy.logerr(f"Không thể tải mô hình YOLO từ '{self.YOLO_MODEL_PATH}'. Lỗi: {e}")
            self.yolo_session = None

    def detect_with_yolo(self, image):
        """
        Thực hiện nhận diện đối tượng bằng YOLOv8 và hậu xử lý kết quả đúng cách.
        """
        if self.yolo_session is None: return []

        original_height, original_width = image.shape[:2]

        img_resized = cv2.resize(image, self.YOLO_INPUT_SIZE)
        img_data = np.array(img_resized, dtype=np.float32) / 255.0
        img_data = np.transpose(img_data, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(img_data, axis=0)  # Add batch dimension

        input_name = self.yolo_session.get_inputs()[0].name
        outputs = self.yolo_session.run(None, {input_name: input_tensor})

        # Lấy output thô, output của YOLOv8 thường có shape (1, 84, 8400) hoặc tương tự
        # Chúng ta cần transpose nó thành (1, 8400, 84) để dễ xử lý
        predictions = np.squeeze(outputs[0]).T

        # Lọc các box có điểm tin cậy (objectness score) thấp
        # Cột 4 trong predictions là điểm tin cậy tổng thể của box
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.YOLO_CONF_THRESHOLD, :]
        scores = scores[scores > self.YOLO_CONF_THRESHOLD]

        if predictions.shape[0] == 0:
            rospy.loginfo("YOLO không phát hiện đối tượng nào vượt ngưỡng tin cậy.")
            return []

        # Lấy class_id có điểm cao nhất
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Lấy tọa độ box và chuyển đổi về ảnh gốc
        x, y, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        
        # Tính toán tỷ lệ scale để chuyển đổi tọa độ
        x_scale = original_width / self.YOLO_INPUT_SIZE[0]
        y_scale = original_height / self.YOLO_INPUT_SIZE[1]

        # Chuyển từ [center_x, center_y, width, height] sang [x1, y1, x2, y2]
        x1 = (x - w / 2) * x_scale
        y1 = (y - h / 2) * y_scale
        x2 = (x + w / 2) * x_scale
        y2 = (y + h / 2) * y_scale
        
        # Chuyển thành list các box và scores
        boxes = np.column_stack((x1, y1, x2, y2)).tolist()
        
        # 4. Thực hiện Non-Maximum Suppression (NMS)
        # Đây là một bước cực kỳ quan trọng để loại bỏ các box trùng lặp
        # OpenCV cung cấp một hàm NMS hiệu quả
        nms_threshold = 0.45 # Ngưỡng IOU để loại bỏ box
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.YOLO_CONF_THRESHOLD, nms_threshold)
        
        if len(indices) == 0:
            rospy.loginfo("YOLO: Sau NMS, không còn đối tượng nào.")
            return []

        # 5. Tạo danh sách kết quả cuối cùng
        final_detections = []
        for i in indices.flatten():
            final_detections.append({
                'class_name': self.YOLO_CLASS_NAMES[class_ids[i]],
                'confidence': float(scores[i]),
                'box': [int(coord) for coord in boxes[i]] # Chuyển tọa độ sang int
            })

        rospy.loginfo(f"YOLO đã phát hiện {len(final_detections)} đối tượng cuối cùng.")
        return final_detections

    def initialize_mqtt(self):
        self.mqtt_client = mqtt.Client()
        def on_connect(client, userdata, flags, rc): rospy.loginfo(f"Kết nối MQTT: {'Thành công' if rc == 0 else 'Thất bại'}")
        self.mqtt_client.on_connect = on_connect
        try:
            self.mqtt_client.connect(self.MQTT_BROKER, self.MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e: rospy.logerr(f"Không thể kết nối MQTT: {e}")
    
    def _set_state(self, new_state, initial=False):
        if self.current_state != new_state:
            if not initial: rospy.loginfo(f"Chuyển trạng thái: {self.current_state.name} -> {new_state.name}")
            self.current_state = new_state
            self.state_change_time = rospy.get_time()

    def camera_callback(self, image_msg):
        try:
            if image_msg.encoding.endswith('compressed'):
                np_arr = np.frombuffer(image_msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(image_msg.height, image_msg.width, -1)
            if 'rgb' in image_msg.encoding: cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.latest_image = cv2.resize(cv_image, (self.WIDTH, self.HEIGHT))
        except Exception as e: rospy.logerr(f"Lỗi chuyển đổi ảnh: {e}")

    def run(self):
        rospy.loginfo("Bắt đầu vòng lặp. Đợi 3 giây..."); time.sleep(3); rospy.loginfo("Hành trình bắt đầu!")
        self.detector.start_scanning()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.current_state == RobotState.DRIVING_STRAIGHT:
                if self.latest_image is None:
                    rospy.logwarn_throttle(5, "Đang chờ dữ liệu hình ảnh từ topic camera...")
                    rate.sleep()
                    continue
                if self.detector.process_detection():
                    rospy.loginfo("SỰ KIỆN: LiDAR phát hiện giao lộ.")
                    self._set_state(RobotState.HANDLING_EVENT)
                    self.handle_intersection()
                    continue
                line_center_x = self._get_line_center(self.latest_image)
                if line_center_x is None:
                    rospy.logwarn("SỰ KIỆN: Camera không còn thấy vạch kẻ (cuối đường hoặc góc cua).")
                    self._set_state(RobotState.HANDLING_EVENT)
                    self.handle_end_of_line()
                else:
                    self.correct_course(line_center_x)
            elif self.current_state == RobotState.LEAVING_INTERSECTION:
                self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
                if rospy.get_time() - self.state_change_time > self.INTERSECTION_CLEARANCE_DURATION:
                    rospy.loginfo("Đã thoát khỏi khu vực sự kiện, quay lại đi thẳng.")
                    self._set_state(RobotState.DRIVING_STRAIGHT)
            elif self.current_state == RobotState.DEAD_END:
                rospy.logwarn("Đã đến ngõ cụt. Dừng hoạt động vĩnh viễn."); self.robot.stop(); break
            rate.sleep()
        self.cleanup()

    def cleanup(self):
        rospy.loginfo("Dừng robot và giải phóng tài nguyên..."); self.robot.stop()
        self.detector.stop_scanning(); self.mqtt_client.loop_stop(); self.mqtt_client.disconnect()
        rospy.loginfo("Đã giải phóng tài nguyên. Chương trình kết thúc.")
    
    def _get_line_center(self, image):
        roi = image[self.ROI_Y : self.ROI_Y + self.ROI_H, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        _img, contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                return int(M["m10"] / M["m00"])
        return None

    def correct_course(self, line_center_x):
        error = line_center_x - (self.WIDTH / 2)
        safe_zone_pixels = (self.WIDTH / 2) * self.SAFE_ZONE_PERCENT
        if abs(error) < safe_zone_pixels:
            self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
        else:
            adj = error / (self.WIDTH / 2) * self.CORRECTION_GAIN
            self.robot.set_motors(self.BASE_SPEED + adj, self.BASE_SPEED - adj)

    def handle_end_of_line(self):
        rospy.loginfo("[CUỐI ĐƯỜNG] Dừng lại và quét tìm lối rẽ...")
        self.robot.stop(); time.sleep(0.5)
        available_paths = self.scan_for_available_paths_proactive()
        if available_paths['right']:
            rospy.loginfo("[CUỐI ĐƯỜNG] Quyết định: Rẽ PHẢI.")
            self.turn_robot(90, update_main_direction=True)
        elif available_paths['left']:
            rospy.loginfo("[CUỐI ĐƯỜNG] Quyết định: Rẽ TRÁI.")
            self.turn_robot(-90, update_main_direction=True)
        else:
            self._set_state(RobotState.DEAD_END); return
        self._set_state(RobotState.LEAVING_INTERSECTION)
        
    def handle_intersection(self):
        rospy.loginfo("\n[GIAO LỘ] Dừng lại và xử lý...")
        self.robot.stop(); time.sleep(0.5)
        nav_action = None

        current_direction = self.DIRECTIONS[self.current_direction_index]
        angle_to_sign = self.ANGLE_TO_FACE_SIGN_MAP.get(current_direction, 0)
        self.turn_robot(angle_to_sign, False)

        image_info = self.latest_image
        detections = self.detect_with_yolo(image_info)

        for det in detections:
            if det['class_name'] in ['F', 'L', 'R']:
                rospy.loginfo(f"[QUYẾT ĐỊNH] Theo biển báo: {det['class_name'].upper()}")
                if det['class_name'] == 'L': nav_action = -90
                elif det['class_name'] == 'R': nav_action = 90
                else: nav_action = 0
                break
        if nav_action is None:
            for det in detections:
                if det['class_name'] == 'qr':
                    box = det['box']
                    qr_image = image_info[box[1]:box[3], box[0]:box[2]]
                    decoded = decode(qr_image)
                    if decoded:
                        qr_data = decoded[0].data.decode('utf-8')
                        self.publish_data({'type': 'QR_CODE', 'value': qr_data})
                    break
        if nav_action is None:
             for det in detections:
                if det['class_name'] == 'math':
                    # TODO: Cần tích hợp OCR ở đây
                    problem_text = "25 + 17" 
                    solution = self.solve_math_problem(problem_text)
                    if solution is not None:
                        self.publish_data({'type': 'MATH_PROBLEM', 'problem': problem_text, 'solution': solution})
                    break

        self.turn_robot(-angle_to_sign, False)

        if nav_action is not None:
            self.turn_robot(nav_action, update_main_direction=True)
        else:
            available_paths = self.scan_for_available_paths_proactive()
            if available_paths['straight']:
                rospy.loginfo("[GIAO LỘ] Quyết định: Đi THẲNG.")
            elif available_paths['right']:
                rospy.loginfo("[GIAO LỘ] Quyết định: Rẽ PHẢI."); self.turn_robot(90, update_main_direction=True)
            elif available_paths['left']:
                rospy.loginfo("[GIAO LỘ] Quyết định: Rẽ TRÁI."); self.turn_robot(-90, update_main_direction=True)
            else:
                self._set_state(RobotState.DEAD_END); return
        self._set_state(RobotState.LEAVING_INTERSECTION)
    
    def turn_robot(self, degrees, update_main_direction=True):
        duration = abs(degrees) / 90.0 * self.TURN_DURATION_90_DEG
        if degrees > 0: self.robot.set_motors(self.TURN_SPEED, -self.TURN_SPEED)
        elif degrees < 0: self.robot.set_motors(-self.TURN_SPEED, self.TURN_SPEED)
        if degrees != 0: time.sleep(duration)
        self.robot.stop()
        if update_main_direction and degrees % 90 == 0 and degrees != 0:
            num_turns = round(degrees / 90)
            self.current_direction_index = (self.current_direction_index + num_turns) % 4
            rospy.loginfo(f"==> Hướng đi MỚI: {self.DIRECTIONS[self.current_direction_index].name}")
        time.sleep(0.5)
    
    def _does_path_exist_in_frame(self, image):
        if image is None: return False
        roi = image[self.ROI_Y : self.ROI_Y + self.ROI_H, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        _img, contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return bool(contours) and cv2.contourArea(max(contours, key=cv2.contourArea)) > self.SCAN_PIXEL_THRESHOLD
    
    def scan_for_available_paths_proactive(self):
        rospy.loginfo("[SCAN] Bắt đầu quét chủ động...")
        paths = {"straight": False, "right": False, "left": False}
        if self.latest_image is not None:
            paths["straight"] = self._does_path_exist_in_frame(self.latest_image)
        self.turn_robot(90, update_main_direction=False); time.sleep(0.5)
        if self.latest_image is not None:
            paths["right"] = self._does_path_exist_in_frame(self.latest_image)
        self.turn_robot(-180, update_main_direction=False); time.sleep(0.5)
        if self.latest_image is not None:
            paths["left"] = self._does_path_exist_in_frame(self.latest_image)
        self.turn_robot(90, update_main_direction=False)
        rospy.loginfo(f"[SCAN] Kết quả: {paths}")
        return paths

def main():
    rospy.init_node('jetbot_corrected_follower_node', anonymous=True)
    try:
        controller = JetBotCorrectedController()
        controller.run()
    except rospy.ROSInterruptException: rospy.loginfo("Node đã bị ngắt.")
    except Exception as e: rospy.logerr(f"Lỗi không xác định: {e}", exc_info=True)

if __name__ == '__main__':
    main()