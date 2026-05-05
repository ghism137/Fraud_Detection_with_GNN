# Fraud Detection with Graph Neural Network
>
> Đồ án tích hợp 3 môn cuối kỳ: Data Mining · Deep Learning · Business Analysis

---

## 1. Tổng quan

| Mục | Nội dung |
|-----|----------|
| **Chủ đề** | Phát hiện giao dịch gian lận tín dụng bằng Graph Neural Network |
| **Dataset** | [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) |
| **Thời gian** | 10 tuần |
| **Môn tích hợp** | Data Mining / KDD · Deep Learning · Business Analysis |
| **Môi trường** | Kaggle Notebooks (GPU, 30h/tuần) · Colab Free T4 · HuggingFace Spaces (deploy) |

---

## 2. Dataset — IEEE-CIS Fraud Detection

### 2.1 Cấu trúc

Dataset gồm **2 bảng** phải JOIN qua `TransactionID`:

```
train_transaction.csv   (~590,000 rows, ~394 columns)
train_identity.csv      (~144,000 rows, ~41 columns)
```

**Nhóm features trong transaction.csv:**

| Nhóm | Features | Mô tả |
|------|----------|-------|
| Label | `isFraud` | 0 = legit, 1 = fraud (~3.5% fraud) |
| Thời gian | `TransactionDT` | Timedelta từ reference point (không phải timestamp thực) |
| Giá trị | `TransactionAmt` | Số tiền giao dịch (USD) |
| Sản phẩm | `ProductCD` | W / H / C / S / R |
| Thẻ | `card1–card6` | Card number, type, bank, country |
| Địa chỉ | `addr1`, `addr2` | Billing address (mã hóa) |
| Khoảng cách | `dist1`, `dist2` | Khoảng cách liên quan đến địa chỉ |
| Email | `P_emaildomain`, `R_emaildomain` | Email purchaser / recipient |
| Count | `C1–C14` | Count features bí ẩn (Vesta không tiết lộ ý nghĩa) |
| Timedelta | `D1–D15` | Khoảng thời gian giữa các sự kiện |
| Match | `M1–M9` | T / F / NaN — match flags |
| Vesta | `V1–V339` | Vesta-engineered features, ~70–90% missing |

**Nhóm features trong identity.csv:**

| Nhóm | Features | Mô tả |
|------|----------|-------|
| Thiết bị | `DeviceType`, `DeviceInfo` | Mobile / Desktop, tên thiết bị |
| ID | `id_01–id_38` | Network, browser, OS features (mã hóa) |

### 2.2 Đặc điểm quan trọng

- **Imbalanced nặng**: ~3.5% fraud (tốt hơn dataset cũ 0.17%, nhưng vẫn cần xử lý)
- **Missing values thực sự**: V columns ~70–90% missing — đây là dữ liệu thô thực tế
- **Categorical features phong phú**: ProductCD, card4, card6, email domains, M flags → Apriori applicable
- **Graph construction tự nhiên**: card1 + card4 + addr1 → account fingerprint (node ID thực sự)

---

## 3. Kiến trúc kỹ thuật

### 3.1 Luồng tổng thể (Kiến trúc Hệ thống - Bird's Eye View)

Hệ thống rẽ nhánh rõ ràng để phục vụ các mục tiêu học thuật khác nhau, nhưng quy tụ lại ở khâu đánh giá:

- **Khối Tiền xử lý (Kiến trúc Dual-Pipeline)**: 
  - **Pipeline A (Tree-based/LGBM)**: Giữ nguyên `NaN` (tận dụng Native Missing Handling) + Tự code Custom K-Fold Target Encoding bằng Pandas.
  - **Pipeline B (Vector-based/MLP & GNN)**: Điền khuyết (Median Imputation) + Cờ báo khuyết (Missing Indicators) + Z-Score Normalization cho TOÀN BỘ dữ liệu (kể cả Indicators) để đồng nhất Variance.
- **Khối KDD (Data Mining)**: Tự code CART (Numpy) $\rightarrow$ Lọc đặc trưng (Feature Selection) $\rightarrow$ Train LightGBM $\rightarrow$ Giải thích bằng SHAP.
- **Khối DL (Deep Learning)**: Custom MLP (2-3 lớp ẩn, ReLU, SGD + Cosine Warmup) $\rightarrow$ Xây dựng Đồ thị nhân tạo đa tầng $\rightarrow$ Train GNN (GAT + LayerNorm + Residual).
- **Khối Đánh giá (Business Analysis)**: So sánh XAI (Cục bộ vs Cấu trúc) $\rightarrow$ Ensemble (Late Fusion) $\rightarrow$ Cost Matrix.

### 3.2 Deep Learning — 2 tầng

**Tầng 1 — MLP viết tay bằng numpy** *(nộp thầy, thỏa yêu cầu "code tay")*

```
Forward pass (Hidden): Z = W·X + b  →  A = ReLU(Z)
Forward pass (Output): Z_out = W_out·A + b_out
Loss:                  BCE = -[w_p·y·log(σ(Z_out)) + (1-y)·log(1-σ(Z_out))]
Backward pass:         dL/dW, dL/db qua chain rule (tích hợp Weight Decay chống Overfit)
Update:                W -= lr · dW
```

Implement hoàn toàn bằng numpy, không dùng bất kỳ framework nào.

**Tầng 2 — GNN bằng PyTorch + PyTorch Geometric**

Graph construction từ card linkage:

```python
df['uid'] = df['card1'].astype(str) + '_' + \
            df['card4'].astype(str) + '_' + \
            df['addr1'].astype(str)
# uid → node,  TransactionID → edge
```

GCN propagation rule (Kipf & Welling 2017):

$$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-\frac{1}{2}}\,\tilde{A}\,\tilde{D}^{-\frac{1}{2}}\,H^{(l)}\,W^{(l)}\right)$$

Trong đó $\tilde{A} = A + I$ (adjacency với self-loop), $\tilde{D}$ là degree matrix tương ứng.

Model pipeline: `GATConv (hoặc GCNConv + LayerNorm) → ReLU → Dropout → GATConv → Sigmoid`

### 3.3 Ranh giới "code tay"

| Component | Framework | Lý do |
|-----------|-----------|-------|
| CART (Decision Tree) | Numpy thuần | Tự code hàm tính Information Gain để hiểu sâu bản chất Feature Selection. |
| MLP | Numpy thuần | Yêu cầu bắt buộc của thầy (tự code Feedforward & Backprop). |
| GNN | PyTorch + PyG | Sparse matrix GNN bằng numpy vượt scope; giải thích rõ trong báo cáo. |

---

## 4. Data Mining — Chi tiết

### 4.1 Missing value strategy

Không impute toàn bộ một cách đồng nhất — mỗi nhóm có chiến lược riêng:

| Nhóm | Strategy | Lý do |
|------|----------|-------|
| **Pipeline A (LGBM)** | Giữ nguyên `NaN` hoàn toàn | LightGBM xử lý Missing phân nhánh theo Default Direction cực tốt. |
| **Pipeline B (MLP)** | Điền `Median` + Thêm `Missing Indicators` | GNN/MLP không nuốt được `NaN`. Indicator lưu lại tín hiệu (Pattern) bị khuyết của V-columns. |
| Categorical (`M1-M9`, `card`) | Target Encoding (K-Fold + Smoothing) / LOO Encoder | Ánh xạ category thành con số mang thông tin xác suất rủi ro, chống leakage. |

### 4.2 Cây quyết định tự code (CART) & Feature Selection

Thay thế Apriori bằng việc tự code thuật toán Cây quyết định (Decision Tree - CART) bằng Numpy. Tập trung vào hàm tính Information Gain dựa trên Entropy:

$$IG(D, A) = Entropy(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} Entropy(D_v)$$

Đầu ra: Dùng CART tự code để đánh giá Feature Importance, chọn ra Top 50 đặc trưng tốt nhất.

### 4.3 Baseline ML — LightGBM

Đưa 50 features vào huấn luyện mô hình LightGBM làm trục xương sống. Lấy AUC-PR làm mốc Baseline để so sánh với GNN và MLP.

```
LightGBM → Custom MLP (2-3 layers, ReLU) → GCN / GraphSAGE → Ensemble
```

Metric chính: **AUC-PR** (Precision-Recall AUC) — không dùng Accuracy vì imbalanced.

---

## 5. Metrics đánh giá

| Metric | Vai trò | Lý do chọn |
|--------|---------|------------|
| **AUC-PR** | Metric chính | Robust với imbalanced; AUC-ROC bị inflate khi negative >> positive |
| AUC-ROC | Tham khảo | Quen thuộc, dễ so sánh với paper |
| F1 @ threshold | Báo cáo | Cần threshold cụ thể, dùng cho cost analysis |
| Precision / Recall | Threshold analysis | Trade-off rõ ràng: FN cost >> FP cost |

**Cost matrix (Business Analysis):**

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | TN = $0 | FP = $5 (customer friction) |
| **Actual Fraud** | FN = $120 (avg loss) | TP = $0 |

Threshold tối ưu = minimize $5 \cdot FP + 120 \cdot FN$, không phải maximize F1.

---

## 6. Explainability (XAI)

- **GNNExplainer** (có trong PyG): giải thích subgraph nào dẫn đến dự đoán fraud cho từng node
- **SHAP**: feature importance cho node features — V columns nào, C columns nào có weight cao nhất

Case study trong báo cáo: chọn 2–3 giao dịch fraud thực sự, show GNNExplainer output + SHAP waterfall.

---

## 7. Roadmap 10 tuần (Roadmap V2)

| Giai đoạn | Tuần | Nội dung | Deliverable |
|-----------|------|----------|-------------|
| **Toán + NN tay** | 1 | Linear algebra, calculus, chain rule | Numpy matrix ops notebook |
| | 2 | Forward pass từ đầu | MLP 2-layer forward |
| | 3 | Backpropagation từ đầu | MLP numpy hoàn chỉnh — nộp thầy |
| **Khai phá & Làm phẳng** | 4 | KDD & Feature Selection (Code tay CART), Dual-Pipeline (Giữ NaN cho Tree, Impute cho Vector) | Dùng CART lọc ra Top 50 features |
| | 5 | Baseline LightGBM & Tái cấu trúc MLP (2-3 lớp ẩn, ReLU) | Bảng so sánh sớm: LightGBM vs Custom MLP |
| **GNN & Cấu trúc** | 6 | Heuristic Graph Construction (Multi-tier: uid + Temporal Window 30 ngày để chống Clique) | Graph dataset, đo lường sparsity |
| | 7 | PyG Implementation (GAT/LayerNorm) trên Top 50 features | GNN pipeline hoàn chỉnh |
| | 8 | XAI (SHAP, GNNExplainer) & Late Fusion Ensemble ($\alpha \cdot \hat{y}_{LGBM} + (1-\alpha) \cdot \hat{y}_{GNN}$) | Bảng so sánh + Tối ưu hóa hệ số $\alpha$ |
| **BA + Đóng gói** | 9 | Phân tích Chi phí (Cost Matrix) & Streamlit Demo | Demo deploy + Cost report, Threshold curve |
| | 10 | Case Study (LightGBM vs GNN) & Hoàn thiện GitHub | Slide thuyết trình + README chuyên nghiệp |

**Fallback nếu trễ tiến độ**: Bỏ GraphSAGE, chỉ nộp GCN. Không bỏ Business Analysis và XAI.

---

## 8. Tài liệu cốt lõi

### Backpropagation & MLP

- Karpathy — ["The spelled-out intro to neural networks: building micrograd"](https://www.youtube.com/watch?v=VMj-3S1tku0) (video 2h22m)
- Nielsen — [Neural Networks and Deep Learning, ch.1–2](http://neuralnetworksanddeeplearning.com)
- CS231n — [Backpropagation Notes](https://cs231n.github.io/optimization-2/)

### GNN

- distill.pub — ["A Gentle Introduction to Graph Neural Networks"](https://distill.pub/2021/gnn-intro/)
- Kipf & Welling 2017 — [GCN paper gốc](https://arxiv.org/abs/1609.02907)
- Hamilton et al. 2017 — [GraphSAGE](https://arxiv.org/abs/1706.02216)
- PyTorch Geometric — [Introduction by Example](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html)

### GNN for Fraud Detection

- Dou et al. 2020 — ["Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters" (CARE-GNN)](https://arxiv.org/abs/2008.08692)
- Liu et al. 2021 — ["Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection" (PC-GNN)](https://arxiv.org/abs/2108.13798)
- safe-graph — [DGFraud benchmark (GitHub)](https://github.com/safe-graph/DGFraud)

### Explainability

- Ying et al. 2019 — [GNNExplainer](https://arxiv.org/abs/1903.03894)
- PyG — [Explainability module](https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html)

### Data Mining & ML Baseline

- Ke et al. 2017 — [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- Micci-Barreca 2001 — [A Preprocessing Scheme for High-Cardinality Categorical Attributes (Target Encoding)](https://dl.acm.org/doi/10.1145/507533.507538)
- Phân tích Late Fusion Ensemble (Kết hợp nội suy trọng số mô hình).

### Imbalanced & Metrics

- Davis & Goadrich 2006 — ["Precision-Recall vs ROC Curves"](https://dl.acm.org/doi/10.1145/1143844.1143874)
- Lin et al. 2017 — [Focal Loss](https://arxiv.org/abs/1708.02002)

### Toán nền

- 3Blue1Brown — [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- Deisenroth et al. — [Mathematics for Machine Learning (free PDF)](https://mml-book.github.io)

---

## 9. Tech stack

```
Language:     Python 3.10+
Data:         pandas, numpy, scikit-learn
Imbalanced:   scale_pos_weight (LGBM) & Cost-sensitive Learning
Preprocess:   pandas, numpy (Custom K-Fold Target Encoding)
ML Baseline:  LightGBM
ML/DL tay:    numpy thuần (CART, MLP, SGD+Momentum, Cosine Warmup, Weight Decay, Pos_Weight)
DL framework: PyTorch 2.x
GNN:          PyTorch Geometric (PyG - GAT, LayerNorm)
XAI:          SHAP, PyG GNNExplainer
Visualization:matplotlib, seaborn, plotly
Demo:         Streamlit
Deploy:       HuggingFace Spaces
Compute:      Kaggle Notebooks (30h GPU/tuần)
Version ctrl: Git + GitHub
```

---

## 10. Cấu trúc GitHub repo

```
fraud-detection-gnn/
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep              # dataset không commit (quá lớn)
├── notebooks/
│   ├── 01_eda.ipynb           # EDA, MICE, Target Encoding
│   ├── 02_data_mining.ipynb   # Code tay CART + Feature Selection
│   ├── 03_baseline_lgbm.ipynb # LightGBM Baseline
│   ├── 04_mlp_custom.ipynb    # MLP (2-3 lớp, ReLU)
│   ├── 05_graph_construction.ipynb # Heuristic Graph
│   ├── 06_gnn_training.ipynb  # PyG GCN / GraphSAGE
│   └── 07_xai_ensemble.ipynb  # SHAP, GNNExplainer, Late Fusion
├── src/
│   ├── preprocessing/
│   │   ├── pipeline_a_lgbm.py # Custom Target Encoder, Drop columns
│   │   └── pipeline_b_mlp.py  # Median Impute, Indicator, Z-score Normalize
│   ├── kdd/
│   │   └── cart_tree.py       # Tự code CART (Numpy)
│   ├── baseline/
│   │   └── lgbm_trainer.py    # LightGBM Baseline với Threshold tối ưu
│   ├── mlp_numpy/
│   │   ├── layers.py          # Linear, ReLU, Sigmoid, BCEWithLogitsLoss
│   │   ├── optimizers.py      # SGD + Momentum + Cosine Warmup + Weight Decay
│   │   └── mlp_trainer.py     # Training loop với Mini-batch & Early Stopping
│   ├── graph/
│   │   ├── builder.py         # Graph construction từ card linkage
│   │   └── dataset.py         # PyG Dataset wrapper
│   └── models/
│       └── gat.py             # GAT model (Attention + LayerNorm)
├── streamlit_app/
│   └── app.py                 # Demo dashboard
└── report/
    └── figures/               # Charts, XAI visualizations
```

---

## 11. Câu hỏi bảo vệ cần chuẩn bị

**Data Mining:**

- Tại sao dùng AUC-PR thay Accuracy?
- Strategy xử lý missing kép (Dual-Pipeline) chống Leakage thế nào?
- Hàm Information Gain trong CART code tay tính toán ra sao?

**Deep Learning:**

- Tại sao cấu trúc MLP lại sử dụng ReLU thay vì Sigmoid cho bài toán này?
- Tại sao lại cần xây dựng đồ thị Heuristic đa tầng (Multi-tier) và tích hợp cửa sổ thời gian (Temporal Window)?
- Hàm Late Fusion Ensemble nội suy hệ số alpha như thế nào?

**GNN:**

- Tại sao GNN tốt hơn MLP cho fraud detection?
- GCN propagation rule có ý nghĩa gì về mặt toán học?
- Graph được xây từ dữ liệu thế nào?

**Business Analysis:**

- FN cost và FP cost khác nhau thế nào? Ảnh hưởng thế nào đến threshold?
- Trong trường hợp mô hình Late Fusion Ensemble dự đoán một giao dịch có xác suất gian lận là $0.4$, dựa vào đâu để quyết định đóng băng (block) giao dịch này thay vì cho phép nó đi qua? Trình bày công thức nội suy Threshold $\tau^*$ dựa trên Cost Matrix.
- ROI của hệ thống được tính ra sao?
- Model này deploy thực tế có vấn đề gì?
