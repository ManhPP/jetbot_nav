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
# import onnxruntime as ort
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
        # self.initialize_yolo()
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
        self.YOLO_MODEL_PATH = "yolo_model.onnx"
        self.YOLO_CONF_THRESHOLD = 0.6
        self.YOLO_INPUT_SIZE = (640, 640)
        self.YOLO_CLASS_NAMES = ['turn_left', 'turn_right', 'go_straight', 'qr_code', 'math_problem']
        self.MQTT_BROKER = "localhost"; self.MQTT_PORT = 1883
        self.MQTT_DATA_TOPIC = "jetbot/corrected_event_data"
        self.current_state = None
        self.DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 0
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
        self.yolo_session = None

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