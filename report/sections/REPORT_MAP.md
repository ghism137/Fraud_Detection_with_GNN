# Report Context Mapping

Bảng quy định ngữ cảnh động cho việc viết báo cáo thành phần. Khi AI nhận lệnh viết một phần báo cáo, AI bắt buộc phải đọc file này, tìm đúng `Section`, và dùng tool để đọc CHÍNH XÁC các file được liệt kê trong cột `Context Files` trước khi viết.

| Section Code | Tiêu đề Báo Cáo | Context Files (Nguồn cần đọc) |
|--------------|-----------------|-------------------------------|
| `01_preprocess` | 01. Tiền xử lý & Dual-Pipeline | `src/preprocessing/pipeline_a_lgbm.py`, `src/preprocessing/pipeline_b_mlp.py`, `.claude/project/session_state.md` |
| `02_cart` | 02. Feature Selection (CART) | `src/kdd/cart_tree.py`, `.claude/project/session_state.md` |
| `03_baseline` | 03. LightGBM Baseline | `src/baseline/lgbm_trainer.py`, `.claude/project/session_state.md` |
| `04_mlp` | 04. Custom MLP (Numpy) | `src/mlp_numpy/mlp_trainer.py`, `src/mlp_numpy/layers.py`, `src/mlp_numpy/optimizers.py`, `.claude/project/session_state.md` |
| `05_graph` | 05. Heuristic Graph Construction | `src/graph/builder.py`, `src/graph/dataset.py`, `.claude/project/session_state.md` |
| `06_gnn` | 06. PyG GNN Training | `src/models/gat.py`, `.claude/project/session_state.md` |
| `07_xai` | 07. Explainability & Ensemble | `src/xai/shap_explainer.py`, `src/xai/gnn_explainer.py`, `.claude/project/session_state.md` |
| `08_business` | 08. Cost Analysis & Threshold | `.claude/project/session_state.md`, `report/figures/cost_matrix.png` |

**Lưu ý cho AI:**
1. Chỉ đọc các file được liệt kê ở cột `Context Files`. Không tự ý đọc toàn bộ thư mục.
2. Nếu không tìm thấy file trong `Context Files` (có thể do cấu trúc thư mục thay đổi), hãy báo cáo lại cho user.
3. Luôn tham chiếu `session_state.md` để cập nhật metrics hoặc các quyết định kiến trúc mới nhất.
