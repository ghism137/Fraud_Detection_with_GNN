# PIPELINE.md — Bản đồ Luồng Dữ liệu
> Tài liệu này mô tả 3 luồng chạy chính của hệ thống Fraud Detection.
> Đặt cùng thư mục code để dễ tham chiếu khi debug hoặc viết báo cáo.

---

## Sơ đồ tổng quan

```
data/raw/
├── train_transaction.csv  (590k rows, 394 cols)
└── train_identity.csv     (144k rows, 41 cols)
         │
         ▼  LEFT JOIN (TransactionID)
    [Merged DataFrame]
         │
         ├─────────────────────────┐
         ▼                         ▼
   [Pipeline A]               [Pipeline B]
   Tree-based                 Vector-based
   (LGBM)                     (MLP / GNN)
         │                         │
         ▼                         ▼
  Target Encode              Median Impute
  (giữ NaN numeric)         + Missing Indicators
                             + Z-Score (ALL cols)
         │                         │
         └──────────┬──────────────┘
                    ▼
           [CART Feature Selection]
           src/kdd/cart_tree.py
           → Top50 features
           → data/processed/top50_features.json
                    │
         ┌──────────┼──────────────┐
         ▼          ▼              ▼
     [LGBM]      [MLP]          [GAT]
  run_lgbm.py  run_mlp.py    run_gnn.py
         │          │              │
         ▼          ▼              ▼
  lgbm_proba.npy  mlp_proba.npy  gnn_proba.npy
         │          │              │
         └──────────┴──────────────┘
                    ▼
           [Late Fusion Ensemble]
           ŷ = α·ŷ_LGBM + (1-α)·ŷ_GNN
           src/models/ensemble.py
```

---

## Luồng 1: LightGBM Baseline

**File:** `src/pipelines/run_lgbm.py`

```
[1] Load & Merge CSV
      src/kdd/feature_selection.load_and_merge()
      Input:  data/raw/*.csv
      Output: df (DataFrame), y (ndarray)

[2] Chuẩn bị feature matrix
      src/kdd/feature_selection.prepare_for_cart()
      → Target Encode categorical (K-Fold, anti-leakage)
      → Encode M-flags: T=1, F=0, NaN=-1
      → Giữ NaN trong numeric (LGBM native handling)
      Output: X_all (ndarray), feature_names (list)

[3] CART Feature Selection   ← có thể skip với --use_cache
      src/kdd/feature_selection.run_cart_feature_selection()
      src/kdd/cart_tree.DecisionTreeFeatureSelector
      Output: top50_features[_debug].json

[4] Lọc Top50 features
      X_top50 = X_all[:, top50_idx]

[5] Stratified Train/Val split (80/20, seed=42)

[6] Train LightGBM
      src/baseline/lgbm_trainer.LGBMFraudTrainer
      → Early Stopping (50 rounds)
      → scale_pos_weight=27.6

[7] Evaluate: AUC-PR + Cost Matrix threshold τ*
      src/baseline/lgbm_trainer.evaluate()

[8] Lưu artifacts
      data/processed/lgbm_proba[_debug].npy
      data/processed/lgbm_y_val[_debug].npy
```

**Lệnh chạy:**
```bash
# Debug (50k rows, ~2 phút)
python src/pipelines/run_lgbm.py --mode debug

# Production (590k rows)
python src/pipelines/run_lgbm.py --mode full

# Production, bỏ qua CART nếu đã có cache
python src/pipelines/run_lgbm.py --mode full --use_cache
```

---

## Luồng 2: Custom MLP (Numpy)

**File:** `src/pipelines/run_mlp.py`

```
[1] Load & Merge CSV        (giống LGBM)
[2] Chuẩn bị feature matrix (giống LGBM — CART dùng chung encoding)
[3] CART / cache            (giống LGBM — dùng chung JSON)

[4] Lọc Top50 features
      Lưu ý: NaN ĐƯỢC GIỮ NGUYÊN — MLPTrainer tự Impute qua Pipeline B

[5] Stratified Train/Val split (80/20, seed=42)

[6] Train Custom MLP
      src/mlp_numpy/mlp_trainer.MLPTrainer
      → Bên trong fit() tự chạy Pipeline B:
          MLPPipelinePreprocessor.fit_transform(X_train)
          → Median Impute + Missing Indicators + Z-Score (ALL cols)
      → Mini-batch SGD + Cosine Annealing Warmup
      → BCEWithLogitsLoss tích hợp pos_weight=27.6
      → Early Stopping (patience=5)

[7] Evaluate: AUC-PR + Cost Matrix threshold τ*
      (wrapper _evaluate_mlp() trong run_mlp.py)

[8] Lưu artifacts
      data/processed/mlp_proba[_debug].npy
      data/processed/mlp_y_val[_debug].npy
```

**Lệnh chạy:**
```bash
python src/pipelines/run_mlp.py --mode debug
python src/pipelines/run_mlp.py --mode full --use_cache
```

**Điểm khác biệt so với run_lgbm.py:**
- LGBM nhận X có NaN → native handling tự động
- MLP nhận X có NaN → MLPTrainer impute + Z-Score bên trong fit()
- LGBM có `evaluate_full()` tích hợp sẵn trong LGBMFraudTrainer
- MLP dùng wrapper `_evaluate_mlp()` định nghĩa tại chỗ trong run_mlp.py

---

## Luồng 3: GNN (GAT) — *Implement Tuần 6-7*

**File:** `src/pipelines/run_gnn.py`

```
[1] Load & Merge CSV
[2] Load top50 từ cache          (bắt buộc có cache từ LGBM/MLP run trước)
[3] Pipeline B → node features   (src/preprocessing/pipeline_b_mlp.py)
[4] Graph Construction           (src/graph/builder_3tier.py)   ← TODO
      Tier 1: card1 + card4 + addr1 (Hard identity link)
      Tier 2: Temporal window 30 ngày
      Tier 3: Device fingerprint
[5] Graph Validation             (src/graph/validator.py)       ← TODO
      → Kiểm tra max_degree < 500 (chống Super-nodes)
[6] Train GAT                    (src/models/gat_layer.py)      ← TODO
      → GATConv + LayerNorm + Residual + Self-loops
[7] Evaluate: AUC-PR + Cost Matrix
[8] Lưu: data/processed/gnn_proba.npy
```

**Lệnh chạy (sau khi implement):**
```bash
python src/pipelines/run_gnn.py --mode debug --model gat
```

---

## Cấu trúc file liên quan

```
src/
├── config.py                    ← Tất cả hằng số & hyperparameters
├── pipelines/
│   ├── PIPELINE.md              ← File này
│   ├── run_lgbm.py
│   ├── run_mlp.py
│   └── run_gnn.py
├── kdd/
│   ├── cart_tree.py             ← CART tự code (numpy)
│   └── feature_selection.py    ← Pipeline KDD đầy đủ
├── preprocessing/
│   ├── pipeline_a_lgbm.py       ← CustomTargetEncoder (K-Fold)
│   └── pipeline_b_mlp.py        ← MLPPipelinePreprocessor (Z-Score)
├── baseline/
│   └── lgbm_trainer.py          ← LGBMFraudTrainer + Cost Matrix
├── mlp_numpy/
│   ├── layers.py                ← Linear, ReLU, Sigmoid, BCEWithLogitsLoss
│   ├── optimizers.py            ← SGDCosineWarmup
│   └── mlp_trainer.py           ← MLPTrainer (tích hợp Pipeline B)
├── graph/                       ← [TODO Tuần 6]
│   ├── builder_3tier.py
│   ├── validator.py
│   └── dataset.py
└── models/                      ← [TODO Tuần 7]
    ├── gat_layer.py
    └── ensemble.py
```

---

## Output artifacts (data/processed/)

| File | Tạo bởi | Dùng bởi |
|------|---------|---------|
| `top50_features.json` | `run_lgbm.py` hoặc `run_mlp.py` | Tất cả pipelines |
| `lgbm_proba.npy` | `run_lgbm.py` | `ensemble.py` |
| `mlp_proba.npy` | `run_mlp.py` | `ensemble.py` |
| `gnn_proba.npy` | `run_gnn.py` | `ensemble.py` |
| `*_y_val.npy` | Mỗi `run_*.py` | `ensemble.py` |

> **Lưu ý**: Các file `*_debug.npy` và `*_debug.json` được tạo khi chạy `--mode debug`.
> Không commit vào git (đã có trong `.gitignore` qua `data/`).
