Hãy coi quy trình này như một **nhà máy lọc thông tin**. Đầu vào là một bức ảnh thô, nhiễu loạn và đầy chi tiết. Đầu ra là một con số duy nhất, chính xác: tọa độ X của tâm vạch kẻ. Mỗi bước trong nhà máy là một bộ lọc, loại bỏ dần những thông tin không cần thiết.

---

### **Quy trình Xử lý Ảnh để Dò Line: Chi tiết từng bước**

**Đầu vào:** Một khung hình (`image`) từ camera và tọa độ của một Vùng Quan Tâm (`roi_y`, `roi_h`).

---

#### **Bước 1: Lọc Không gian - Cắt Vùng Quan Tâm (ROI)**

*   **Mục tiêu:** Loại bỏ 80-90% các thông tin nhiễu không liên quan trong ảnh (bầu trời, người, tường, v.v.). Chỉ tập trung vào khu vực ngay trước mặt robot nơi vạch kẻ đường chắc chắn sẽ xuất hiện.
*   **Code:**
    ```python
    roi = image[roi_y : roi_y + roi_h, :]
    ```
*   **Xử lý bên trong:** Đây là một thao tác "cắt lát" (slicing) mảng NumPy. Nó tạo ra một bức ảnh con mới (`roi`) chỉ chứa các hàng pixel từ `roi_y` đến `roi_y + roi_h`.
*   **Kết quả:** Một bức ảnh hình chữ nhật thấp và rộng, chỉ chứa phần sàn nhà và vạch kẻ ngay trước robot. Khối lượng dữ liệu cần xử lý đã giảm đi đáng kể.

---

#### **Bước 2: Lọc Màu sắc - Chuyển đổi sang Không gian màu HSV**

*   **Mục tiêu:** Chuyển đổi dữ liệu màu sắc sang một định dạng dễ dàng lọc hơn, đặc biệt là để đối phó với sự thay đổi về ánh sáng (bóng đổ, блики).
*   **Code:**
    ```python
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ```
*   **Xử lý bên trong:** Thuật toán của OpenCV sẽ chuyển đổi từng pixel từ bộ ba giá trị (Blue, Green, Red) sang bộ ba giá trị (Hue, Saturation, Value).
    *   **Value (Độ sáng)** là thành phần quan trọng nhất đối với chúng ta. Một vạch kẻ đen, dù nằm trong bóng tối hay dưới ánh sáng mạnh, sẽ luôn có giá trị `Value` rất thấp.
*   **Kết quả:** Một bức ảnh `hsv` có cùng kích thước với `roi`, nhưng mỗi pixel giờ đây chứa thông tin về Tông màu, Độ bão hòa và Độ sáng.

---

#### **Bước 3: Tạo Mặt nạ Màu sắc (Color Mask)**

*   **Mục tiêu:** Tạo ra một ảnh nhị phân (chỉ có hai màu đen và trắng), trong đó chỉ những pixel thuộc vạch kẻ mục tiêu (màu đen) được đánh dấu là màu trắng.
*   **Code:**
    ```python
    color_mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
    ```
*   **Xử lý bên trong:** Hàm này quét qua từng pixel của ảnh `hsv`. Đối với mỗi pixel, nó kiểm tra xem cả ba giá trị H, S, V có nằm trong khoảng giữa `LINE_COLOR_LOWER` và `LINE_COLOR_UPPER` hay không.
    *   Nếu pixel thỏa mãn điều kiện (có `Value` thấp), pixel tương ứng trong `color_mask` sẽ được gán giá trị 255 (màu trắng).
    *   Nếu không, nó sẽ được gán giá trị 0 (màu đen).
*   **Kết quả:** Một ảnh đen trắng `color_mask`. Lý tưởng nhất, nó sẽ là một nền đen với một hoặc nhiều hình dạng màu trắng, đại diện cho vạch kẻ và bất kỳ vật thể màu đen nào khác trong ROI.

---

#### **Bước 4: Tạo Mặt nạ Tập trung (Focus Mask)**

*   **Mục tiêu:** Xây dựng một "cặp kính che" ảo để buộc robot chỉ nhìn vào khu vực trung tâm của tầm nhìn, loại bỏ sự phân tâm từ các vạch kẻ ngang ở hai bên tại ngã tư.
*   **Code:**
    ```python
    focus_mask = np.zeros_like(color_mask)
    # ... tính toán start_x, end_x ...
    cv2.rectangle(focus_mask, (start_x, 0), (end_x, roi_height), 255, -1)
    ```
*   **Xử lý bên trong:**
    1.  Tạo một ảnh `focus_mask` hoàn toàn màu đen, có cùng kích thước với `color_mask`.
    2.  Tính toán tọa độ X bắt đầu và kết thúc của một dải dọc hẹp ở giữa (dựa trên `ROI_CENTER_WIDTH_PERCENT`).
    3.  Vẽ một hình chữ nhật màu trắng tô đầy lên `focus_mask` tại vị trí đã tính.
*   **Kết quả:** Một ảnh đen trắng `focus_mask`, trông giống như một cột màu trắng trên nền đen.

---

#### **Bước 5: Lọc Kết hợp - Áp dụng Cả Hai Mặt nạ**

*   **Mục tiêu:** Chỉ giữ lại những thông tin nào thỏa mãn *cả hai* điều kiện: (1) là màu đen của vạch kẻ VÀ (2) nằm trong khu vực tập trung.
*   **Code:**
    ```python
    final_mask = cv2.bitwise_and(color_mask, focus_mask)
    ```
*   **Xử lý bên trong:** Phép toán `bitwise_and` thực hiện một phép logic AND trên từng cặp pixel tương ứng của `color_mask` và `focus_mask`. Một pixel trong `final_mask` chỉ có thể là màu trắng (255) nếu pixel ở cùng vị trí trong *cả hai* ảnh đầu vào đều là màu trắng.
*   **Kết quả:** Một ảnh đen trắng `final_mask` cực kỳ "sạch sẽ". Nó chỉ chứa một hình dạng trắng duy nhất (nếu có), đó chính là đoạn vạch kẻ thẳng mà robot cần bám theo.

---

#### **Bước 6: Nhận dạng Đối tượng - Tìm Contours**

*   **Mục tiêu:** Chuyển đổi từ một "đám mây" các pixel trắng rời rạc thành một đối tượng hình học có thể phân tích được.
*   **Code:**
    ```python
    _, contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ```
*   **Xử lý bên trong:** Thuật toán của OpenCV sẽ quét qua `final_mask` và tìm ra đường viền của tất cả các vùng trắng liền mạch.
*   **Kết quả:** Một danh sách `contours`, mỗi phần tử trong danh sách là một mảng các điểm `(x, y)` định nghĩa đường viền của một hình dạng.

---

#### **Bước 7: Lọc Đối tượng và Tính toán Trọng tâm**

*   **Mục tiêu:** Từ các hình dạng đã tìm được, chọn ra hình dạng chính xác là vạch kẻ và tính toán tọa độ trung tâm của nó.
*   **Code:**
    ```python
    c = max(contours, key=cv2.contourArea)
    # ... kiểm tra diện tích tối thiểu ...
    M = cv2.moments(c)
    line_center_x = int(M["m10"] / M["m00"])
    ```
*   **Xử lý bên trong:**
    1.  `max(contours, ...)`: Tìm ra contour có diện tích lớn nhất. Chúng ta giả định vạch kẻ đường luôn là đối tượng lớn nhất trong `final_mask`.
    2.  `cv2.contourArea(c) < SCAN_PIXEL_THRESHOLD`: Một bước lọc an toàn cuối cùng để loại bỏ các đốm nhiễu nhỏ có thể vô tình lọt qua.
    3.  `cv2.moments(c)`: Tính toán các "mô-men hình học" của contour. Đây là một khái niệm toán học cho phép suy ra các thuộc tính như diện tích, trọng tâm, hướng, v.v.
    4.  `int(M["m10"] / M["m00"])`: Đây là công thức toán học chuẩn để tính tọa độ X của trọng tâm (center of mass/centroid) từ các giá trị mô-men. `m00` là tổng diện tích, và `m10` là tổng của tất cả các tọa độ x của các pixel trong hình dạng.
*   **Kết quả:** **Một con số nguyên duy nhất, `line_center_x`**, đại diện cho vị trí ngang của vạch kẻ trong ROI. Đây chính là đầu ra cuối cùng của toàn bộ nhà máy lọc thông tin này, sẵn sàng để được đưa vào hàm `correct_course`.