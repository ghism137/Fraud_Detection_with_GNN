# 04. Mạng Nơ-ron Đa Tầng (Custom MLP) bằng Numpy Thuần

> **[THÔNG BÁO TIẾN ĐỘ THỰC THI]**
> Tiến trình kiến trúc và code cho Module Custom MLP đã hoàn thiện 100% về mặt toán học và tích hợp thành công với Pipeline B.
> Tuy nhiên, **phần Kết quả & Đánh giá (Mục 3) hiện chưa có dữ liệu thực tế**.
> **Việc cần làm tiếp theo để hoàn thành báo cáo này:** Cần chạy file script (hoặc Notebook `04_mlp_custom.ipynb`) để nạp tập dữ liệu IEEE-CIS, cho phép quá trình Mini-Batch SGD thực thi. Sau đó, lưu lại biểu đồ Loss hội tụ, trích xuất điểm Validation AUC-PR và so sánh tốc độ với LightGBM Baseline để bổ sung vào báo cáo.

---

## 1. Mục tiêu & Cơ sở lý thuyết

Theo yêu cầu cốt lõi của Đồ án (yếu tố "Deep Learning viết tay"), phân hệ này chịu trách nhiệm xây dựng một mạng Nơ-ron truyền thẳng (Feedforward Neural Network) hoàn toàn từ con số 0 bằng thư viện đại số tuyến tính `Numpy`, không sử dụng bất kỳ framework tự động vi phân (autograd) nào như PyTorch hay TensorFlow. 

Việc triển khai phải giải quyết được 3 bài toán lớn về mặt cơ sở lý thuyết toán học:
1. **Thiết kế Lan truyền ngược (Backpropagation) giải tích:** Mọi đạo hàm cấp 1 của các ma trận trọng số $\frac{\partial L}{\partial W}$ và bias $\frac{\partial L}{\partial b}$ phải được tính tay chính xác thông qua quy tắc chuỗi (Chain Rule).
2. **Khắc phục Bất cân bằng Dữ liệu ($3.5\%$ Fraud):** Hàm Loss mặc định của mạng Nơ-ron sẽ có xu hướng "cào bằng", dự đoán $0$ cho toàn bộ giao dịch. Cơ sở lý thuyết yêu cầu phải tích hợp một trọng số phạt (Pos_Weight) trực tiếp vào nền tảng của Binary Cross Entropy.
3. **Chống Triệt tiêu Đạo hàm (Vanishing Gradient):** Lựa chọn hàm kích hoạt (ReLU thay vì Sigmoid cho lớp ẩn) và phương pháp khởi tạo trọng số (He Initialization) để đảm bảo tín hiệu lan truyền được qua nhiều lớp trong mạng.

---

## 2. Chi tiết Triển khai Kiến trúc

### 2.1 Tiền xử lý Dữ liệu đặc thù (Pipeline B)
Mạng Nơ-ron nhạy cảm cực độ với phương sai (Variance) của đầu vào. Chúng ta đã tách biệt hoàn toàn luồng xử lý `Pipeline B` (`MLPPipelinePreprocessor`) với các quyết định kỹ thuật sâu:
*   **Median Imputation & Missing Indicators:** Các giá trị `NaN` được điền bằng số trung vị để duy trì đường cong phân phối, đồng thời lưu lại tín hiệu mất mát thông qua cột nhị phân (Indicator).
*   **Z-score Normalization toàn cục:** Để cứu sống He Initialization, thay vì chỉ chuẩn hóa biến liên tục, chúng ta ép Z-score lên *toàn bộ* ma trận (bao gồm cả Indicator $\{0, 1\}$). Điều này ép phương sai của mọi Feature về đúng mức $1.0$, tạo ra một mặt phẳng Gradient đối xứng hoàn hảo.

### 2.2 Xây dựng Cấu trúc Layer (`layers.py`)
*   **Lớp Linear (Fully Connected):** Khởi tạo trọng số $W$ theo thuật toán **He Initialization** với độ lệch chuẩn $std = \sqrt{2/N_{in}}$. Ở hàm Backward, sử dụng đại số tuyến tính: $dW = X^T \cdot dZ$ và $dx = dZ \cdot W^T$.
*   **Lớp ReLU:** Khóa gradient ở các vùng giá trị âm, giữ gradient là $1$ ở vùng dương. Ngăn chặn hiện tượng bão hòa tín hiệu.
*   **Hàm BCEWithLogitsLoss an toàn:**
    *   **Toán học tích hợp:** Hợp nhất Sigmoid và BCE để sử dụng Log-Sum-Exp, kẹp giá trị Logits bằng hàm `np.clip` (giới hạn ở mức -500 đến 500) giúp ngăn chặn triệt để lỗi tràn bộ nhớ `log(0) = -inf`.
    *   **Tích hợp Pos_Weight:** Phương trình đạo hàm lùi được tính giải tích hoàn toàn: 
    $$dx = \frac{-w_{pos} \cdot y \cdot (1-p) + (1-y) \cdot p}{N}$$
    Điều này tạo ra một lực kéo Gradient (lên đến 27.6 lần) ép mạng phải học cách bắt Fraudster.

### 2.3 Bộ tối ưu hóa & Huấn luyện (`optimizers.py` & `mlp_trainer.py`)
*   **Mini-batch Generator:** Chia nhỏ dữ liệu thành các lô $2048$ dòng để chống OOM (Out of Memory) trên tổng số $590,000$ mẫu.
*   **SGDCosineWarmup:** 
    *   Dùng **Momentum** để giúp trọng số vượt qua các khe hẹp (local minima).
    *   Tích hợp **Weight Decay (L2 Regularization):** Tính năng phạt trọng số $W$ quá lớn, đặc trị Overfitting cho dữ liệu Tabular.
    *   Dùng **Cosine Annealing với Warmup:** Tăng tốc từ từ ở 5 epoch đầu để chống sốc (exploding gradients), sau đó hạ dần độ dài bước nhảy theo đường cong Cosine để hội tụ mịn tại đáy Loss.
*   **Early Stopping:** Báo cáo liên tục AUC-PR trên tập Validation. Tự động ngắt quá trình học nếu mô hình không cải thiện sau 5 epochs (Patience=5) để lưu lại trọng số tốt nhất.

---

## 3. Kết quả & Đánh giá

*(Nội dung này đang chờ dữ liệu chạy thực tế)*

**Dữ liệu mong đợi để hoàn thiện phần này:**
1. **Biểu đồ Hội tụ Loss:** Đồ thị đường cong Training Loss và Validation Loss để chứng minh kiến trúc MLP không bị mắc kẹt tại Local Minima và Weight Decay đã chống Overfitting thành công.
2. **Chỉ số AUC-PR:** Ghi nhận điểm số AUC-PR tốt nhất trên tập Validation trước khi Early Stopping được kích hoạt.
3. **So sánh với LightGBM Baseline:** Phân tích độ chênh lệch hiệu năng và thời gian huấn luyện giữa mô hình toán học ma trận tự viết (MLP) và công cụ tối ưu công nghiệp (LightGBM). Lý giải vì sao cần phải xây thêm Graph Neural Network (GNN) để vượt mặt Baseline.
