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

from map_navigator import MapNavigator

class RobotState(Enum):
    DRIVING_STRAIGHT = 1
    HANDLING_EVENT = 2
    LEAVING_INTERSECTION = 3
    DEAD_END = 4
    GOAL_REACHED = 5

class Direction(Enum):
    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

class JetBotController:
    def __init__(self):
        rospy.loginfo("Đang khởi tạo JetBot Event-Driven Controller...")
        self.setup_parameters()
        self.initialize_hardware()
        self.initialize_yolo()
        self.initialize_mqtt()

        self.navigator = MapNavigator(self.MAP_FILE_PATH)
        self.current_node_id = self.navigator.start_node
        self.planned_path = None
        self.banned_edges = []
        self.plan_initial_route()

        self.latest_scan = None
        self.latest_image = None
        self.detector = SimpleOppositeDetector()
        rospy.Subscriber('/scan', LaserScan, self.detector.callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        rospy.loginfo("Đã đăng ký vào các topic /scan và /csi_cam_0/image_raw.")
        self.state_change_time = rospy.get_time()
        self._set_state(RobotState.DRIVING_STRAIGHT, initial=True)
        rospy.loginfo("Khởi tạo hoàn tất. Sẵn sàng hoạt động.")

    def plan_initial_route(self): 
        """Lập kế hoạch đường đi ban đầu từ điểm xuất phát đến đích."""
        rospy.loginfo(f"Đang lập kế hoạch từ node {self.navigator.start_node} đến {self.navigator.end_node}...")
        self.planned_path = self.navigator.find_path(
            self.navigator.start_node, 
            self.navigator.end_node,
            self.banned_edges
        )
        if self.planned_path:
            rospy.loginfo(f"Đã tìm thấy đường đi: {self.planned_path}")
        else:
            rospy.logerr("Không tìm thấy đường đi đến đích!")
            self._set_state(RobotState.DEAD_END)

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
        self.PRESCRIPTIVE_SIGNS = {'L', 'R', 'F'}
        self.PROHIBITIVE_SIGNS = {'NL', 'NR', 'NF'}
        self.DATA_ITEMS = {'qr_code', 'math_problem'}
        self.MQTT_BROKER = "localhost"; self.MQTT_PORT = 1883
        self.MQTT_DATA_TOPIC = "jetbot/corrected_event_data"
        self.current_state = None
        self.DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 1
        self.ANGLE_TO_FACE_SIGN_MAP = {d: a for d, a in zip(self.DIRECTIONS, [45, -45, -135, 135])}

        self.MAX_ACCEPTABLE_ERROR_PERCENT = 0.85 
        self.MAP_FILE_PATH = "map.json"
        self.LABEL_TO_DIRECTION_ENUM = {'N': Direction.NORTH, 'E': Direction.EAST, 'S': Direction.SOUTH, 'W': Direction.WEST}

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

    def numpy_nms(self, boxes, scores, iou_threshold):
        """
        Thực hiện Non-Maximum Suppression (NMS) bằng NumPy.
        :param boxes: list các bounding box, mỗi box là [x1, y1, x2, y2]
        :param scores: list các điểm tin cậy tương ứng
        :param iou_threshold: ngưỡng IoU để loại bỏ các box trùng lặp
        :return: list các chỉ số (indices) của các box được giữ lại
        """
        # Chuyển đổi sang NumPy array để tính toán vector hóa
        x1 = np.array([b[0] for b in boxes])
        y1 = np.array([b[1] for b in boxes])
        x2 = np.array([b[2] for b in boxes])
        y2 = np.array([b[3] for b in boxes])

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # Sắp xếp các box theo điểm tin cậy giảm dần
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Tính toán IoU (Intersection over Union)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Giữ lại các box có IoU nhỏ hơn ngưỡng
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

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
        indices = self.numpy_nms(np.array(boxes), scores, nms_threshold)
        
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

                lidar_detected = self.detector.process_detection()
                line_lost = False
                line_center_x = self._get_line_center(self.latest_image)
                if line_center_x is None:
                    line_lost = True
                else:
                    error = line_center_x - (self.WIDTH / 2)
                    max_error_pixels = (self.WIDTH / 2) * self.MAX_ACCEPTABLE_ERROR_PERCENT
                    if abs(error) > max_error_pixels:
                        line_lost = True
                
                if lidar_detected or line_lost:
                    if lidar_detected:
                        rospy.loginfo("SỰ KIỆN: LiDAR phát hiện giao lộ.")
                    if line_lost:
                        rospy.logwarn("SỰ KIỆN: Vạch kẻ đường biến mất (ngã ba/ngã tư).")

                    # Kiểm tra xem đã đến đích chưa
                    if self.current_node_id == self.navigator.end_node:
                        rospy.loginfo("ĐÃ ĐẾN ĐÍCH!")
                        self._set_state(RobotState.GOAL_REACHED)
                        continue
                    
                    self._set_state(RobotState.HANDLING_EVENT)
                    self.handle_intersection() 
                    continue

                if line_center_x is not None:
                    self.correct_course(line_center_x)

            elif self.current_state == RobotState.LEAVING_INTERSECTION:
                self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
                if rospy.get_time() - self.state_change_time > self.INTERSECTION_CLEARANCE_DURATION:
                    rospy.loginfo("Đã thoát khỏi khu vực sự kiện, quay lại đi thẳng.")
                    self._set_state(RobotState.DRIVING_STRAIGHT)
            elif self.current_state == RobotState.DEAD_END:
                rospy.logwarn("Đã đến ngõ cụt. Dừng hoạt động vĩnh viễn."); self.robot.stop(); break
            elif self.current_state == RobotState.GOAL_REACHED: 
                rospy.loginfo("ĐÃ ĐẾN ĐÍCH. Dừng hoạt động."); self.robot.stop(); break
            rate.sleep()
        self.cleanup()

    def cleanup(self):
        rospy.loginfo("Dừng robot và giải phóng tài nguyên..."); self.robot.stop()
        self.detector.stop_scanning(); self.mqtt_client.loop_stop(); self.mqtt_client.disconnect()
        rospy.loginfo("Đã giải phóng tài nguyên. Chương trình kết thúc.")

    def map_absolute_to_relative(self, target_direction_label, current_robot_direction):
        """
        Chuyển đổi hướng tuyệt đối ('N', 'E', 'S', 'W') thành hành động tương đối ('straight', 'left', 'right').
        Ví dụ: robot đang hướng BẮC (NORTH), mục tiêu là đi hướng ĐÔNG (EAST) -> hành động là 'right'.
        """
        target_dir = self.LABEL_TO_DIRECTION_ENUM.get(target_direction_label)
        if target_dir is None: return None

        current_idx = current_robot_direction.value
        target_idx = target_dir.value
        
        diff = (target_idx - current_idx + 4) % 4 
        
        if diff == 0:
            return 'straight'
        elif diff == 1:
            return 'right'
        elif diff == 3: 
            return 'left'
        else: 
            return 'turn_around'
        
    def map_relative_to_absolute(self, relative_action, current_robot_direction):
        """
        Chuyển đổi hành động tương đối ('straight', 'left', 'right') thành hướng tuyệt đối ('N', 'E', 'S', 'W').
        """
        current_idx = current_robot_direction.value
        if relative_action == 'straight':
            target_idx = current_idx
        elif relative_action == 'right':
            target_idx = (current_idx + 1) % 4
        elif relative_action == 'left':
            target_idx = (current_idx - 1 + 4) % 4
        else:
            return None
        
        for label, direction in self.LABEL_TO_DIRECTION_ENUM.items():
            if direction.value == target_idx:
                return label
        return None
    
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
        
    def handle_intersection(self):
        rospy.loginfo("\n[GIAO LỘ] Dừng lại và xử lý...")
        self.robot.stop(); time.sleep(0.5)

        current_direction = self.DIRECTIONS[self.current_direction_index]
        angle_to_sign = self.ANGLE_TO_FACE_SIGN_MAP.get(current_direction, 0)
        self.turn_robot(angle_to_sign, False)
        image_info = self.latest_image
        detections = self.detect_with_yolo(image_info)
        self.turn_robot(-angle_to_sign, False)
        
        prescriptive_cmds = {det['class_name'] for det in detections if det['class_name'] in self.PRESCRIPTIVE_SIGNS}
        prohibitive_cmds = {det['class_name'] for det in detections if det['class_name'] in self.PROHIBITIVE_SIGNS}
        data_items = [det for det in detections if det['class_name'] in self.DATA_ITEMS]

        # 2. Xử lý các mục dữ liệu (QR, Toán) và Publish
        rospy.loginfo("[STEP 2] Processing data items...")
        for item in data_items:
            if item['class_name'] == 'qr_code':
                # Code đọc QR thật
                # box = item['box']; qr_image = self.latest_image[box[1]:box[3], box[0]:box[2]]
                # decoded = decode(qr_image)
                # if decoded: qr_data = decoded[0].data.decode('utf-8'); self.publish_data(...)
                rospy.loginfo("Found QR Code. Publishing data...")
                self.publish_data({'type': 'QR_CODE', 'value': 'simulated_data_123'})
            elif item['class_name'] == 'math_problem':
                rospy.loginfo("Found Math Problem. Solving and publishing...")
                self.publish_data({'type': 'MATH_PROBLEM', 'value': '2+2=4'})
        
        
        rospy.loginfo("[STEP 3] Lập kế hoạch điều hướng theo bản đồ...")
        # 3. Lập kế hoạch Điều hướng
        final_decision = None
        is_deviation = False 

        while True:
            planned_direction_label = self.navigator.get_next_direction_label(self.current_node_id, self.planned_path)
            if not planned_direction_label:
                rospy.logerr("Lỗi kế hoạch: Không tìm thấy bước tiếp theo."); self._set_state(RobotState.DEAD_END); return
            
            planned_action = self.map_absolute_to_relative(planned_direction_label, current_direction)
            rospy.loginfo(f"Kế hoạch A* đề xuất: Đi {planned_action} (hướng {planned_direction_label})")

            # Ưu tiên 1: Biển báo bắt buộc
            intended_action = None
            if 'L' in prescriptive_cmds: intended_action = 'left'
            elif 'R' in prescriptive_cmds: intended_action = 'right'
            elif 'F' in prescriptive_cmds: intended_action = 'straight'
            
            # Ưu tiên 2: Plan
            if intended_action is None:
                intended_action = planned_action
            else:
                # Nếu hành động bắt buộc khác với kế hoạch, đánh dấu là đi chệch hướng
                if intended_action != planned_action:
                    is_deviation = True
                    rospy.logwarn(f"CHỆCH HƯỚNG! Biển báo bắt buộc ({intended_action}) khác với kế hoạch ({planned_action}).")

            # 3.3. Veto bởi biển báo cấm
            is_prohibited = (intended_action == 'straight' and 'NF' in prohibitive_cmds) or \
                            (intended_action == 'right' and 'NR' in prohibitive_cmds) or \
                            (intended_action == 'left' and 'NL' in prohibitive_cmds)

            if is_prohibited:
                rospy.logwarn(f"Hành động dự định '{intended_action}' bị CẤM!")
                
                # Nếu hành động bị cấm đến từ biển báo bắt buộc -> Lỗi bản đồ
                if is_deviation:
                    rospy.logerr("LỖI BẢN ĐỒ! Biển báo bắt buộc mâu thuẫn với biển báo cấm. Không thể đi tiếp.")
                    self._set_state(RobotState.DEAD_END); return
                
                # Nếu hành động bị cấm đến từ kế hoạch A* -> Tìm đường lại
                banned_edge = (self.current_node_id, self.planned_path[self.planned_path.index(self.current_node_id) + 1])
                if banned_edge not in self.banned_edges:
                    self.banned_edges.append(banned_edge)
                
                rospy.loginfo(f"Thêm cạnh cấm {banned_edge} và tìm đường lại...")
                new_path = self.navigator.find_path(self.current_node_id, self.navigator.end_node, self.banned_edges)
                
                if new_path:
                    self.planned_path = new_path
                    rospy.loginfo(f"Đã tìm thấy đường đi mới: {self.planned_path}")
                    continue # Quay lại đầu vòng lặp để kiểm tra với kế hoạch mới
                else:
                    rospy.logerr("Không thể tìm đường đi mới sau khi gặp biển cấm.")
                    self._set_state(RobotState.DEAD_END); return
            
            final_decision = intended_action
            break 

        # 4. Thực thi quyết định
        if final_decision == 'straight': rospy.loginfo("[FINAL] Decision: Go STRAIGHT.")
        elif final_decision == 'right': rospy.loginfo("[FINAL] Decision: Turn RIGHT."); self.turn_robot(90, True)
        elif final_decision == 'left': rospy.loginfo("[FINAL] Decision: Turn LEFT."); self.turn_robot(-90, True)
        else:
            rospy.logwarn("[!!!] DEAD END! No valid paths found."); self._set_state(RobotState.DEAD_END); return
        
        # 5. Cập nhật trạng thái robot sau khi thực hiện
        # 5.1. Xác định node tiếp theo
        next_node_id = None
        if not is_deviation:
            # Nếu đi theo kế hoạch, chỉ cần lấy node tiếp theo từ path
            path_index = self.planned_path.index(self.current_node_id)
            next_node_id = self.planned_path[path_index + 1]
        else:
            # Nếu chệch hướng, phải tìm node tiếp theo dựa trên hành động đã thực hiện
            
            new_robot_direction = self.DIRECTIONS[self.current_direction_index] 
            executed_direction_label = self.map_relative_to_absolute(final_decision, new_robot_direction)
            next_node_id = self.navigator.get_neighbor_by_direction(self.current_node_id, executed_direction_label)
            if next_node_id is None:
                 rospy.logerr("LỖI BẢN ĐỒ! Đã thực hiện rẽ nhưng không có node tương ứng."); self._set_state(RobotState.DEAD_END); return
            
            # Quan trọng: Lập kế hoạch lại từ vị trí mới
            rospy.loginfo(f"Đã đi chệch kế hoạch. Lập lại đường đi từ node mới {next_node_id}...")
            new_path = self.navigator.find_path(next_node_id, self.navigator.end_node, self.banned_edges)
            if new_path:
                self.planned_path = new_path
                rospy.loginfo(f"Đường đi mới sau khi chệch hướng: {self.planned_path}")
            else:
                rospy.logerr("Không thể tìm đường về đích từ vị trí mới."); self._set_state(RobotState.DEAD_END); return

        self.current_node_id = next_node_id
        rospy.loginfo(f"==> Đang di chuyển đến node tiếp theo: {self.current_node_id}")
        self._set_state(RobotState.LEAVING_INTERSECTION)
    
    def turn_robot(self, degrees, update_main_direction=True):
        duration = abs(degrees) / 90.0 * self.TURN_DURATION_90_DEG
        if degrees > 0: self.robot.set_motors(self.TURN_SPEED, -self.TURN_SPEED)
        elif degrees < 0: self.robot.set_motors(-self.TURN_SPEED, self.TURN_SPEED)
        if degrees != 0: time.sleep(duration)
        self.robot.stop()
        if update_main_direction and degrees % 90 == 0 and degrees != 0:
            num_turns = round(degrees / 90)
            self.current_direction_index = (self.current_direction_index + num_turns + 4) % 4
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
    rospy.init_node('jetbot_controller_node', anonymous=True)
    try:
        controller = JetBotController()
        controller.run()
    except rospy.ROSInterruptException: rospy.loginfo("Node đã bị ngắt.")
    except Exception as e: rospy.logerr(f"Lỗi không xác định: {e}", exc_info=True)

if __name__ == '__main__':
    main()