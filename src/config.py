"""
config.py — Trung tâm cấu hình duy nhất
========================================
Tất cả hằng số và siêu tham số của project.
Mọi file run_*.py import từ đây.

NGUYÊN TẮC: Các module cũ (lgbm_trainer.py, mlp_trainer.py) KHÔNG bị sửa.
            run_*.py sẽ inject config vào constructor của từng class.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(_SRC_DIR)
DATA_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Tên file cache JSON cho kết quả CART
CART_CACHE_FILE       = "top50_features.json"
CART_CACHE_FILE_DEBUG = "top50_features_debug.json"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── CART Feature Selection ────────────────────────────────────────────────────
CART_TOP_K        = 50
CART_MAX_DEPTH    = 5
CART_MIN_SAMPLES  = 20
CART_MAX_BINS     = 256
CART_CLASS_WEIGHT = 27.6   # N_legit / N_fraud ≈ 569_877 / 20_663

# ── LightGBM ──────────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "average_precision",   # AUC-PR — metric chính
    "scale_pos_weight":  27.6,
    "n_estimators":      3000,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 50,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "reg_lambda":        2.0,
    "reg_alpha":         0.1,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

# ── MLP (Numpy custom) ────────────────────────────────────────────────────────
MLP_HIDDEN_SIZES = [256, 128]
MLP_LR_MAX       = 0.001
MLP_LR_MIN       = 1e-5
MLP_EPOCHS       = 50
MLP_BATCH_SIZE   = 512
MLP_POS_WEIGHT   = 27.6
MLP_WEIGHT_DECAY = 1e-4
MLP_PATIENCE     = 5

# ── GNN (PyTorch Geometric) ───────────────────────────────────────────────────
GNN_HIDDEN_DIM = 64
GNN_HEADS      = 4
GNN_DROPOUT    = 0.3
GNN_EPOCHS     = 100
GNN_LR         = 0.001

# ── Graph Construction ────────────────────────────────────────────────────────
GRAPH_MAX_DEGREE           = 500
GRAPH_TEMPORAL_WINDOW_DAYS = 30

# ── Business: Cost Matrix ─────────────────────────────────────────────────────
COST_FN = 120   # Chi phí bỏ lọt 1 giao dịch Fraud ($)
COST_FP = 5     # Chi phí chặn nhầm 1 giao dịch hợp lệ ($)

# ── Debug mode ────────────────────────────────────────────────────────────────
DEBUG_N_SAMPLES = 50_000   # Số dòng stratified sample khi --mode debug

# ── Train/Val split ───────────────────────────────────────────────────────────
VAL_SIZE = 0.2   # 20% dữ liệu cho Validation
