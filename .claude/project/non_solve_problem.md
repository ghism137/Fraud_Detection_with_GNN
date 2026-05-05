# Architectural Decision Records (ADR) & Issue Tracker

### [SOLVED] [2026-05-05] 1. Phản biện Kiến trúc Tiền xử lý: Bỏ ngay "Dùng chung" và "MICE"
~~**Vấn đề/Lỗi Logic:** Ở mục 3.1 (Khối Tiền xử lý), bạn ghi rõ là "Dùng chung" và sử dụng thuật toán "MICE (điền khuyết)". Tuy nhiên, ở Tuần 4 (Mục 7) bạn vẫn ghi là làm MICE, trong khi ở Mục 4.1 lại đề xuất "Group theo missing pattern -> median impute per group". Như chúng ta đã phân tích về bản chất NMAR (Missing Not At Random) của dữ liệu Fraud, việc dùng MICE là sai lầm về mặt thống kê và sẽ làm sập hệ thống (OOM) do khối lượng tính toán. Không thể có "Tiền xử lý dùng chung". LightGBM yêu cầu giữ nguyên NaN để tận dụng Native Missing Handling, trong khi MLP/GNN bằng Numpy/PyTorch bắt buộc phải điền khuyết (Impute).~~
~~**Hành động:** Sửa Mục 3.1 thành Kiến trúc Dual-Pipeline: Pipeline A (Tree-based) giữ nguyên NaN và Pipeline B (Vector-based) dùng Điền khuyết (Median Imputation) + Cờ báo khuyết (Missing Indicators). Gạch bỏ hoàn toàn chữ "MICE" khỏi Mục 3.1 và Mục 7.~~
**Resolution [2026-05-05]:** Đã cập nhật toàn bộ `project.md` và `session_state.md` sang mô hình Tiền xử lý Kép (Dual-Pipeline).

### [SOLVED] [2026-05-05] 2. "Quả bom" Bộ nhớ ở Đồ thị Heuristic (Tuần 6)
~~**Vấn đề/Lỗi Logic:** Mục 7 (Tuần 6) ghi rõ: "Heuristic Graph Construction (Nối cạnh qua card1 + card4 + P_emaildomain)". Đây là một cái bẫy chết người. P_emaildomain chứa các domain chung (generic) như 'gmail.com' với hàng trăm nghìn giao dịch. Nếu bạn nối cạnh trực tiếp mọi node có chung 'gmail.com' mà không có giới hạn, bạn sẽ tạo ra một Clique khổng lồ với hàng tỷ cạnh. Đồ thị này sẽ làm tràn RAM trước khi kịp đưa vào PyTorch Geometric.~~
~~**Hành động:** Thay đổi mô tả ở Tuần 6 thành: "Xây dựng Heuristic Graph đa tầng (Multi-tier) tích hợp Cửa sổ thời gian (Temporal Window 30 ngày) để kiểm soát bậc của node (Node Degree)".~~
**Resolution [2026-05-05]:** Đã thay key nối từ `P_emaildomain` sang `addr1` và bổ sung luật giới hạn liên kết theo Cửa sổ thời gian 30 ngày.

### [SOLVED] [2026-05-05] 3. Tàn dư của SMOTE trong Tech Stack
~~**Vấn đề/Lỗi Logic:** Ở Mục 9 (Tech stack), bạn liệt kê thư viện imbalanced-learn (SMOTE). Việc dùng SMOTE (tính toán khoảng cách Euclide) trên bộ dữ liệu chứa hàng tá biến Categorical (đã qua mã hóa) sẽ tạo ra những mẫu gian lận "Frankenstein" (tổ hợp ID thiết bị và mạng không tồn tại trong thực tế), làm nhiễu nghiêm trọng bộ dữ liệu. Bộ công cụ tối ưu cho mất cân bằng lớp ở đây là tham số scale_pos_weight của LightGBM và hàm Focal Loss.~~
~~**Hành động:** Xóa hoàn toàn SMOTE khỏi Mục 9. Thay vào đó, có thể ghi rõ: "Xử lý Imbalanced: scale_pos_weight (LGBM) & Cost-sensitive Learning".~~
**Resolution [2026-05-05]:** Đã xóa SMOTE. Cập nhật chiến lược xử lý Imbalanced mới.

### [SOLVED] [2026-05-05] 4. Bổ sung LayerNorm/GAT vào kiến trúc GNN
~~**Vấn đề/Lỗi Logic:** Mục 3.2 mô tả pipeline của GNN là: GCNConv → ReLU → Dropout → GCNConv → Sigmoid. Với đặc thù của đồ thị giao dịch tài chính (Scale-free network) chứa các Super-nodes, kiến trúc GCNConv thuần túy sẽ gặp hiện tượng Over-smoothing (chỉ sau 2 lớp, thông tin của mọi node bị pha loãng giống hệt nhau).~~
~~**Hành động:** Cập nhật pipeline ở Mục 3.2 thành: GATConv (hoặc GCNConv + LayerNorm) → ReLU → Dropout → GATConv → Sigmoid. Việc bổ sung LayerNorm hoặc nâng cấp lên Graph Attention (GAT) là mấu chốt để hệ thống tự vệ trước sự khuếch đại tín hiệu nhiễu từ các Super-nodes.~~
**Resolution [2026-05-05]:** Cập nhật mô hình từ GCN sang GAT (kèm LayerNorm và Skip connection) trên tất cả các file mô tả.

### [OPEN] [2026-05-05] 5. Câu hỏi bảo vệ: Tính ngưỡng Threshold từ Cost Matrix
**Vấn đề/Lỗi Logic:** Giả sử hội đồng phản biện đặt vấn đề: "Trong trường hợp mô hình Late Fusion Ensemble của em dự đoán một giao dịch có xác suất gian lận là 0.4, dựa vào đâu để hệ thống của em quyết định đóng băng (block) giao dịch này thay vì cho phép nó đi qua?".
**Đề xuất/Hành động:** Cần xây dựng và trình bày công thức nội suy Threshold $\tau^*$ dựa trên Cost Matrix (FN = $120, FP = $5) để thuyết phục hội đồng. Chưa có tài liệu nào diễn giải toán học cho phần này.

### [SOLVED] [2026-05-06] 6. Nghịch lý Variance của Missing Indicators (Pipeline B)
~~**Vấn đề/Lỗi Logic:** Khi gộp các cột Missing Indicator có giá trị $\{0, 1\}$ chung với các cột biến liên tục đã được chuẩn hóa (Z-score nằm trong khoảng $[-3, 3]$), chúng ta tạo ra một sự bất đối xứng khổng lồ về phương sai (Variance Mismatch). Điều này triệt tiêu hoàn toàn tác dụng của He Initialization ở layer đầu tiên, khiến Gradient bị gập ghềnh và tốc độ hội tụ của các feature bị lệch pha.~~
~~**Hành động:** Chấp nhận phá bỏ định dạng ngữ nghĩa $\{0, 1\}$ của cột Indicator. Thiết kế lại `pipeline_b_mlp.py` để quét Z-score lên TOÀN BỘ ma trận kết hợp, ép phương sai của cả biến liên tục lẫn Indicator về đúng $1.0$.~~
**Resolution [2026-05-06]:** Đã implement thành công `MLPPipelinePreprocessor` xử lý đồng nhất variance.

### [SOLVED] [2026-05-06] 7. Tử huyệt của BCEWithLogitsLoss trên Imbalanced Data
~~**Vấn đề/Lỗi Logic:** Tự code `BCEWithLogitsLoss` thuần túy trên bộ dữ liệu chỉ có $3.5\%$ Fraud sẽ khiến MLP học cách "hèn nhát" dự đoán mọi thứ là Legit (0). Hàm Backward `dx = (p - y)/N` đang cào bằng lỗi False Negative và False Positive.~~
~~**Hành động:** Tích hợp trực tiếp trọng số phạt `pos_weight = 27.6` vào phương trình Loss. Giải tích lại đạo hàm Backward để triệt tiêu tràn số: $dx = \frac{-w \cdot y \cdot (1-p) + (1-y) \cdot p}{N}$.~~
**Resolution [2026-05-06]:** Đã cập nhật Toán học an toàn (Stable Math) vào `layers.py`.
