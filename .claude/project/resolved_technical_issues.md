# Lịch sử Giải quyết Vấn đề Kỹ thuật chuyên sâu (Deep Technical Resolutions)

Tài liệu này đóng vai trò là phần mở rộng chi tiết của `non_solve_problem.md`. Nó tập trung bóc tách các "tử huyệt" kỹ thuật, lỗi logic, và nghịch lý toán học **đã được phát hiện và giải quyết trực tiếp trong quá trình code hệ thống**. 

Mục tiêu của tài liệu là giải thích cặn kẽ 3 yếu tố cho mỗi vấn đề:
1. **WHAT:** Vấn đề là gì? Triệu chứng khi hệ thống sụp đổ?
2. **WHY:** Nguyên nhân sâu xa về mặt toán học hoặc cấu trúc bộ nhớ?
3. **HOW:** Giải pháp tối ưu đã được thực hiện và lý do nó hoạt động hoàn hảo.

---

## 1. Cú va chạm cấu trúc dữ liệu: Pandas Index Alignment vs Numpy Arrays

### WHAT (Vấn đề)
Trong giai đoạn thiết kế `CustomTargetEncoder` và `DecisionTreeFeatureSelector`, hệ thống liên tục văng lỗi `NaN injection` (tiêm giá trị rỗng một cách vô lý) hoặc lỗi `TypeError: unhashable type: 'slice'` khi lấy dữ liệu.
Cụ thể, dòng code `df['target'] = y` sinh ra toàn bộ cột `NaN`, và dòng `col = X[:, j]` đánh sập chương trình.

### WHY (Nguyên nhân)
*   **Bẫy Index của Pandas:** Để chia K-Fold chống Leakage, ta dùng `np.random.shuffle(indices)`. Tuy nhiên, mảng `y` là Numpy (chỉ hiểu vị trí), còn `df` là Pandas (hiểu theo Index). Khi gán `y` ngược lại vào `df`, Pandas cố gắng khớp Index gốc (0, 1, 2...) với dữ liệu đã bị xáo trộn. Không khớp -> Nó điền `NaN`.
*   **Bẫy Slicing:** Biến `X` được truyền vào dưới dạng `pd.DataFrame`, nhưng thuật toán CART lại dùng cú pháp slicing của mảng 2 chiều Numpy `X[:, j]`. Pandas không hiểu cú pháp này (nó cần `X.iloc[:, j]`).

### HOW (Cách Fix Tối ưu)
Đồng nhất kiểu dữ liệu ngay tại cổng nạp (Entry point) của mọi Class:
```python
X = np.asarray(X, dtype=np.float64)
y = np.asarray(y).flatten()
```
Bằng cách lột bỏ lớp vỏ Pandas ngay từ đầu, ta đưa mọi thứ về ma trận Numpy thuần túy. Mọi thao tác cắt lát (slicing) `X[:, j]` hoạt động hoàn hảo với tốc độ $O(1)$, và triệt tiêu vĩnh viễn rủi ro "Silent Bug" từ Index Alignment.

---

## 2. Nghịch lý Variance của Missing Indicators (Pipeline B)

### WHAT (Vấn đề)
Trong Pipeline B (chuẩn bị dữ liệu cho MLP/GNN), biến liên tục (Continuous) được chuẩn hóa Z-score về khoảng $[-3, 3]$. Tuy nhiên, để đánh dấu dữ liệu bị khuyết, ta tạo ra các Missing Indicators với giá trị nhị phân $\{0, 1\}$. Nếu nạp trực tiếp ma trận này vào `MLPTrainer`, quá trình huấn luyện diễn ra cực kỳ chậm chạp và Gradient bị méo mó.

### WHY (Nguyên nhân)
Mạng Nơ-ron (cụ thể là `He Initialization` và thuật toán `SGD`) dựa trên giả định toán học cốt lõi: **Input features nên có trung bình bằng 0 và phương sai (Variance) bằng 1**.
Cột Missing Indicator $\{0, 1\}$ (ví dụ 95% là 0, 5% là 1) có phương sai rất nhỏ và không có tâm ở 0. Việc trộn lẫn hai không gian scale `[-3, 3]` (phương sai 1.0) và `{0, 1}` (phương sai rất nhỏ) tạo ra một Loss Landscape (bề mặt hàm Loss) hình phễu cực kỳ hẹp. Trọng số $W$ gắn với các cột Indicator không thể bắt kịp tốc độ cập nhật của các cột Continuous.

### HOW (Cách Fix Tối ưu)
**Phá bỏ định dạng ngữ nghĩa của $\{0, 1\}$.**
Trong `MLPPipelinePreprocessor`, ma trận nháp được tạo ra bằng cách ghép Continuous và Indicators. Sau đó, **áp dụng Z-score Normalization lên TOÀN BỘ ma trận**. 
Giá trị `0` có thể biến thành `-0.22`, và giá trị `1` biến thành `+4.5`. Ta hy sinh khả năng đọc hiểu của con người để ép phương sai của cột Indicator về chuẩn $1.0$. Gradient Landscape trở nên đối xứng, mô hình hội tụ nhanh chóng và He Init phát huy 100% công lực.

---

## 3. Lỗ hổng Cụt Đường (Dead-End) và Nút thắt O(N log N) trong Thuật toán CART

### WHAT (Vấn đề)
Trong hàm đệ quy `_build_tree` của Decision Tree, thuật toán dùng vòng lặp `for split_val in range(max_bin):` để tìm điểm cắt (threshold) tốt nhất, dựa trên việc gọi `np.unique()`. Kết quả: Bỏ sót các quy luật quan trọng, và thời gian chạy lâu đến mức treo máy.

### WHY (Nguyên nhân)
*   **Bẫy Off-by-one:** Lệnh `range(max_bin)` trong Python chạy từ $0$ đến $max\_bin - 1$. Điều này khiến cây không bao giờ thử nghiệm điểm phân tách ở ngay sát giá trị lớn nhất, làm mù lòa mô hình trước các giao dịch siêu dị biệt (Super Outliers).
*   **Chi phí O(N log N):** Việc gọi `np.unique()` hoặc `.sort()` tại mỗi node cho 500,000 dòng dữ liệu sẽ nhân chi phí tính toán lên theo cấp số nhân trong hàm đệ quy.

### HOW (Cách Fix Tối ưu)
Thiết kế lại hệ cơ sở dữ liệu nội bộ của Node:
1. **Dùng Quantile Binning:** Chuyển dữ liệu liên tục thành các "rổ" (bins) số nguyên từ sớm (chỉ tốn $O(N \log N)$ **một lần duy nhất**).
2. **Thuật toán O(N):** Sử dụng `np.bincount` thay cho `np.unique`. `bincount` đếm tần suất các số nguyên bằng cách truy xuất index mảng bộ nhớ (Hashing-like), tốc độ $O(N)$.
3. **Sửa biên lặp:** Đổi vòng lặp thành `range(1, num_bins)` — duyệt các vách ngăn *ở giữa* các rổ thay vì duyệt chính cái rổ, đảm bảo chia nhánh vét cạn không trượt phát nào.

---

## 4. Tử huyệt Cào bằng của BCEWithLogitsLoss trên Dữ liệu Imbalanced

### WHAT (Vấn đề)
Huấn luyện mạng MLP tự code (Numpy) trên bộ dữ liệu IEEE-CIS (chỉ có $3.5\%$ Fraud). Mô hình lập tức đạt độ chính xác (Accuracy) $>96\%$ ngay ở epoch 2, nhưng Validation AUC-PR = 0. Mô hình bị tê liệt, chỉ đoán toàn số 0.

### WHY (Nguyên nhân)
Phương trình đạo hàm lùi (Backward Pass) gốc của Binary Cross Entropy là: 
$$dx = \frac{p - y}{N}$$
Gradient này đối xử với lỗi False Negative (đoán 0 khi thực tế là 1) hoàn toàn ngang bằng với lỗi False Positive (đoán 1 khi thực tế là 0). Để tối thiểu hóa Loss nhanh nhất trên tập $96.5\%$ là 0, mạng nơ-ron chọn cách an toàn: dự đoán $p = 0$ cho tất cả.

### HOW (Cách Fix Tối ưu)
Đưa trọng số phạt (Class Weight / Pos_Weight = 27.6) trực tiếp vào nền tảng toán học của Loss:
$$Loss = - \left[ w \cdot y \log(p) + (1-y) \log(1-p) \right]$$
Giải tích đạo hàm lùi bằng tay (Exact Math) để tìm ra một gradient vừa vặn hoàn hảo mà không bị tràn số:
$$dx = \frac{-w \cdot y \cdot (1-p) + (1-y) \cdot p}{N}$$
Lúc này, nếu mô hình bỏ lọt một giao dịch Fraud ($y=1, p=0$), gradient $dx$ sẽ bung lên tới $-27.6 / N$, một cú giật điện cục độ ép các trọng số $W$ phải điều chỉnh để học cách bắt Fraud. Mô hình đã được cứu sống.

---

## 5. Bẫy Kiến trúc "Double Sigmoid"

### WHAT (Vấn đề)
Sự cám dỗ của việc tạo ra lớp `Sigmoid()` và nhét nó vào phía sau lớp `Linear` cuối cùng trước khi gọi hàm Loss. Kết quả: Mạng MLP không thể hội tụ, Loss đứng im.

### WHY (Nguyên nhân)
Để đạt độ ổn định toán học (tránh lỗi `log(0) = -inf`), hàm `BCEWithLogitsLoss` được thiết kế để nhận **Logits** (giá trị thô chưa qua kích hoạt, nằm trong khoảng $[-\infty, \infty]$). Bản thân hàm Loss này đã tích hợp sẵn công thức Log-Sum-Exp (chứa Sigmoid ngầm bên trong).
Nếu ta gọi lớp `Sigmoid` trước khi đưa vào Loss, các logit bị bóp méo thành xác suất $(0, 1)$. Khi đưa $(0, 1)$ vào `BCEWithLogitsLoss`, nó lại áp dụng Sigmoid một lần nữa lên khoảng này (tạo ra kết quả từ $0.5$ đến $0.73$). Vùng giá trị này làm đạo hàm gần như phẳng lì (Vanishing Gradient).

### HOW (Cách Fix Tối ưu)
Phân tách rạch ròi quy trình:
1. **Training (Huấn luyện):** Xóa bỏ hoàn toàn lớp `Sigmoid` khỏi kiến trúc `forward`. Mạng nhả logit thô thẳng vào `BCEWithLogitsLoss`.
2. **Inference (Suy luận):** Tạo một hàm riêng biệt `predict_proba()`. Lúc này, hàm mới gọi `Sigmoid.forward(logits)` để ép logit thành xác suất an toàn $\in [0, 1]$ phục vụ cho tính toán AUC-PR và Threshold. Đặt ghi chú cảnh báo chữ in hoa trong source code của `layers.py`.

---

*(Tài liệu này sẽ liên tục được cập nhật khi kỹ sư phát hiện và bóc tách các vấn đề kiến trúc và toán học phức tạp khác trong quá trình triển khai PyTorch Geometric GNN ở Tuần 6 & 7).*
