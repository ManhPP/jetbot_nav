#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String # Thêm String để có thể publish trạng thái chi tiết
import math

# =================================================================
# CÁC HẰNG SỐ CÓ THỂ TINH CHỈNH
# =================================================================

# 1. Khoảng cách quét
SCAN_RANGE_MIN = 0.15
SCAN_RANGE_MAX = 2.5

# 2. Phân cụm
DISTANCE_THRESHOLD_CLUSTER = 0.2
MIN_POINTS_PER_CLUSTER = 5

# 3. Đặc điểm tấm bảng
SIGN_WIDTH_MIN = 0.15
SIGN_WIDTH_MAX = 0.40

# 4. Vùng góc để tìm kiếm (độ)
LEFT_ANGLE_MIN = 15.0
LEFT_ANGLE_MAX = 165.0
RIGHT_ANGLE_MIN = -165.0
RIGHT_ANGLE_MAX = -15.0

# 5. Tiêu chí trung tâm
DISTANCE_SYMMETRY_TOLERANCE = 0.20 # Sai số cho phép về khoảng cách (20cm)
ANGLE_IDEAL = 90.0
ANGLE_SYMMETRY_TOLERANCE = 20.0 # Sai số cho phép về góc (20 độ)


class IntersectionCenterDetector:
    def __init__(self):
        rospy.init_node('intersection_center_detector_node', anonymous=True)
        rospy.loginfo("Node phát hiện trung tâm giao lộ đã khởi động.")

        self.status_pub = rospy.Publisher('/intersection_status', String, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.spin()

    def get_point_cartesian(self, r, theta):
        return r * math.cos(theta), r * math.sin(theta)

    def get_distance_between_points(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def scan_callback(self, scan_msg):
        clusters = []
        current_cluster = []
        
        # --- BƯỚC 1: LỌC VÀ PHÂN CỤM ---
        for i in range(len(scan_msg.ranges)):
            distance = scan_msg.ranges[i]
            if not (SCAN_RANGE_MIN < distance < SCAN_RANGE_MAX):
                if len(current_cluster) >= MIN_POINTS_PER_CLUSTER:
                    clusters.append(list(current_cluster))
                current_cluster = []
                continue

            if not current_cluster:
                current_cluster.append(i)
                continue
            
            p_curr = self.get_point_cartesian(distance, scan_msg.angle_min + i * scan_msg.angle_increment)
            p_prev = self.get_point_cartesian(scan_msg.ranges[current_cluster[-1]], scan_msg.angle_min + current_cluster[-1] * scan_msg.angle_increment)
            
            if self.get_distance_between_points(p_curr, p_prev) < DISTANCE_THRESHOLD_CLUSTER:
                current_cluster.append(i)
            else:
                if len(current_cluster) >= MIN_POINTS_PER_CLUSTER:
                    clusters.append(list(current_cluster))
                current_cluster = [i]
        
        if len(current_cluster) >= MIN_POINTS_PER_CLUSTER:
            clusters.append(list(current_cluster))

        # --- BƯỚC 2: XỬ LÝ WRAP-AROUND ---
        if len(clusters) > 1:
            p_last = self.get_point_cartesian(scan_msg.ranges[clusters[-1][-1]], scan_msg.angle_min + clusters[-1][-1] * scan_msg.angle_increment)
            p_first = self.get_point_cartesian(scan_msg.ranges[clusters[0][0]], scan_msg.angle_min + clusters[0][0] * scan_msg.angle_increment)
            if self.get_distance_between_points(p_last, p_first) < DISTANCE_THRESHOLD_CLUSTER:
                clusters[-1].extend(clusters[0])
                clusters.pop(0)

        # --- BƯỚC 3: TÌM BẢNG VÀ LƯU THÔNG TIN ---
        left_sign_info = None
        right_sign_info = None

        for cluster in clusters:
            p_first_cluster = self.get_point_cartesian(scan_msg.ranges[cluster[0]], scan_msg.angle_min + cluster[0] * scan_msg.angle_increment)
            p_last_cluster = self.get_point_cartesian(scan_msg.ranges[cluster[-1]], scan_msg.angle_min + cluster[-1] * scan_msg.angle_increment)
            cluster_width = self.get_distance_between_points(p_first_cluster, p_last_cluster)

            if SIGN_WIDTH_MIN < cluster_width < SIGN_WIDTH_MAX:
                center_idx = cluster[len(cluster)//2]
                center_dist = scan_msg.ranges[center_idx]
                center_angle = math.degrees(scan_msg.angle_min + center_idx * scan_msg.angle_increment)
                
                if LEFT_ANGLE_MIN < center_angle < LEFT_ANGLE_MAX:
                    if left_sign_info is None or center_dist < left_sign_info[0]:
                        left_sign_info = (center_dist, center_angle)
                elif RIGHT_ANGLE_MIN < center_angle < RIGHT_ANGLE_MAX:
                    if right_sign_info is None or center_dist < right_sign_info[0]:
                        right_sign_info = (center_dist, center_angle)
        
        # --- BƯỚC 4: RA QUYẾT ĐỊNH ---
        current_status = "DRIVING"
        
        if left_sign_info and right_sign_info:
            current_status = "IN_ZONE" # Đã vào vùng giao lộ
            
            left_dist, left_angle = left_sign_info
            right_dist, right_angle = right_sign_info
            
            is_dist_symmetric = abs(left_dist - right_dist) < DISTANCE_SYMMETRY_TOLERANCE
            is_left_angle_ideal = abs(left_angle - ANGLE_IDEAL) < ANGLE_SYMMETRY_TOLERANCE
            is_right_angle_ideal = abs(right_angle - (-ANGLE_IDEAL)) < ANGLE_SYMMETRY_TOLERANCE
            
            if is_dist_symmetric and is_left_angle_ideal and is_right_angle_ideal:
                current_status = "AT_CENTER" # Đang ở chính giữa

        rospy.loginfo(f"Status: {current_status}")
        self.status_pub.publish(current_status)

if __name__ == '__main__':
    try:
        IntersectionCenterDetector()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node phát hiện trung tâm giao lộ đã tắt.")