1.  **Sơ đồ Kiến trúc Tổng quan:** Mô tả các thành phần chính của hệ thống và mối quan hệ giữa chúng.
2.  **Sơ đồ Luồng Hoạt động (State Machine):** Mô tả chi tiết các trạng thái và các sự kiện/hành động gây ra sự chuyển đổi giữa chúng.

---

### 1. Sơ đồ Kiến trúc Tổng quan Hệ thống

Sơ đồ này cho thấy các "khối xây dựng" chính và cách chúng tương tác với nhau.

```plantuml
@startuml
!theme vibrant

title Sơ đồ Kiến trúc Tổng quan Hệ thống Điều khiển JetBot

package "ROS Melodic" {
    [ROS Topics] as Topics
}

package "Phần cứng Robot" {
    [Camera CSI]
    [Cảm biến LiDAR]
    [Động cơ JetBot] as Motors
}

package "Hệ thống Phần mềm Điều khiển" {
    
    node "JetBotController (main)" as Controller {
        
        component "State Machine (FSM)" as FSM
        component "Logic Điều khiển\n(Bám line, Rẽ,...)" as ControlLogic
        
        ControlLogic -> FSM
    }
    
    node "Thành phần Cảm biến" {
        component "Xử lý Ảnh Camera\n(OpenCV)" as Vision
        component "Xử lý Dữ liệu LiDAR\n(SimpleOppositeDetector)" as LidarProc
        component "Nhận diện Đối tượng\n(YOLOv8-ONNX)" as YOLO
    }
    
    node "Thành phần Điều hướng" {
        component "Bản đồ & Tìm đường\n(MapNavigator - A*)" as Navigator
    }
    
    node "Thành phần Giao tiếp" {
        component "MQTT Client" as MQTT
    }
    
    ' -- Mối quan hệ trong hệ thống --
    Controller --> ControlLogic
    Controller --> Navigator
    Controller --> MQTT
    
    ControlLogic --> Vision
    ControlLogic --> LidarProc
    
    ' -- Mối quan hệ với các thành phần ngoài --
    Vision -> YOLO
}


' -- Mối quan hệ giữa các package --
Camera_CSI --> Topics : "/csi_cam_0/image_raw"
LiDAR --> Topics : "/scan"

Topics --> Vision : Đọc ảnh
Topics --> LidarProc: Đọc scan

Vision --> Controller : Cung cấp line_center
LidarProc --> Controller: Cung cấp tín hiệu giao lộ
YOLO --> Controller : Cung cấp kết quả nhận diện

Controller --> Motors : Gửi lệnh tốc độ

@enduml
```

**Giải thích Sơ đồ Kiến trúc:**

*   **ROS Melodic** là nền tảng giao tiếp, truyền dữ liệu từ **Phần cứng Robot** (Camera, LiDAR) đến **Hệ thống Phần mềm Điều khiển** thông qua các `Topics`.
*   **Hệ thống Phần mềm Điều khiển** là phần code chính của bạn, được chia thành các khối logic:
    *   **JetBotController (main):** Là "nhạc trưởng", chứa `State Machine` để quyết định hành vi chung và `Logic Điều khiển` để thực thi các hành động cụ thể.
    *   **Thành phần Cảm biến:** Các khối chuyên xử lý dữ liệu thô từ cảm biến (OpenCV xử lý ảnh, `SimpleOppositeDetector` xử lý LiDAR) và AI (`YOLO`).
    *   **Thành phần Điều hướng:** `MapNavigator` độc lập, chịu trách nhiệm về bản đồ và tìm đường.
    *   **Thành phần Giao tiếp:** `MQTT Client` để gửi dữ liệu ra ngoài.
*   Luồng dữ liệu rất rõ ràng: Cảm biến -> Topics -> Bộ xử lý cảm biến -> Controller -> Động cơ.

---

### 2. Sơ đồ Luồng Hoạt động (State Machine Diagram)

Sơ đồ này mô tả chi tiết các trạng thái và các sự kiện gây ra sự chuyển đổi trạng thái, đây chính là "linh hồn" của hệ thống.

```plantuml
@startuml
!theme vibrant
title Sơ đồ Luồng Hoạt động (State Machine) của JetBot

state DRIVING_STRAIGHT : "Bám line & Dò sự kiện\n- Dùng ROI Dự báo & LiDAR\n- Dùng ROI Chính để bám line"
state APPROACHING_INTERSECTION: "Tiến vào Giao lộ\n- Đi thẳng (thời gian ngắn)\n- Bỏ qua cảm biến line"
state HANDLING_EVENT : "Dừng & Xử lý\n- Quét biển báo (YOLO)\n- Ra quyết định\n- Lập lại kế hoạch (nếu cần)"
state LEAVING_INTERSECTION : "Thoát khỏi Giao lộ\n- Đi thẳng (thời gian dài)"
state REACQUIRING_LINE : "Tìm lại Line\n- Đi thẳng\n- Tìm kiếm vạch kẻ mới"
state GOAL_REACHED <<end>> : "Đã đến Đích"
state DEAD_END <<end>> : "Ngõ cụt / Lỗi"

[*] --> DRIVING_STRAIGHT : Khởi động & Lập kế hoạch xong

' -- Luồng hoạt động chính --

DRIVING_STRAIGHT --> APPROACHING_INTERSECTION : [Chỉ Camera dự báo sự kiện]\n(ROI Dự báo mất line)
DRIVING_STRAIGHT --> HANDLING_EVENT : [LiDAR phát hiện sự kiện]\n(Dừng lại ngay, cập nhật vị trí)

APPROACHING_INTERSECTION --> HANDLING_EVENT : [Hết thời gian đi thẳng]\n(Dừng lại, cập nhật vị trí)

HANDLING_EVENT --> LEAVING_INTERSECTION : [Ra quyết định & hành động xong]\n(Đặt target_node_id mới)

LEAVING_INTERSECTION --> REACQUIRING_LINE : [Hết thời gian thoát ly]

REACQUIRING_LINE --> DRIVING_STRAIGHT : [Đã tìm thấy vạch kẻ mới]

' -- Các luồng kết thúc --

DRIVING_STRAIGHT --> GOAL_REACHED : [Phát hiện sự kiện]\n & [current_node == end_node]
APPROACHING_INTERSECTION --> GOAL_REACHED : [Hết thời gian đi thẳng]\n & [current_node == end_node]

REACQUIRING_LINE --> DEAD_END : [Timeout, không tìm thấy line]
HANDLING_EVENT --> DEAD_END : [Không thể tìm đường đi mới]\nHoặc [Lỗi bản đồ]

@enduml
```

**Giải thích Sơ đồ Luồng Hoạt động:**

*   **Các `state`** đại diện cho các trạng thái trong `RobotState` Enum của bạn.
*   **Các mũi tên** biểu thị sự chuyển đổi trạng thái.
*   **Văn bản trong `[...]`** là các **sự kiện (events)** hoặc **điều kiện (conditions)** kích hoạt sự chuyển đổi đó.
*   **Văn bản trong `(...)`** là các **hành động (actions)** được thực hiện ngay sau khi vào trạng thái mới hoặc ngay trước khi rời đi.

Sơ đồ này thể hiện rất rõ ràng chu trình hoạt động của robot:
1.  Bắt đầu ở `DRIVING_STRAIGHT`.
2.  Khi gặp giao lộ, nó có thể đi qua `APPROACHING_INTERSECTION` (nếu do camera) hoặc nhảy thẳng đến `HANDLING_EVENT` (nếu do LiDAR).
3.  Sau khi xử lý xong, nó đi qua một chuỗi ổn định `LEAVING_INTERSECTION` -> `REACQUIRING_LINE` để đảm bảo nó không bị lỗi.
4.  Cuối cùng, nó quay trở lại `DRIVING_STRAIGHT` để bắt đầu một chu kỳ mới.
5.  Các trạng thái kết thúc `GOAL_REACHED` và `DEAD_END` có thể được kích hoạt từ nhiều điểm khác nhau trong luồng, thể hiện các điểm thoát của hệ thống.