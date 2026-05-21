# -*- coding: utf-8 -*-
"""
Notebook 02 — Data Mining: CART Feature Selection
===================================================
Tương đương với 02_data_mining.ipynb nhưng chạy được trực tiếp bằng:
    python notebooks/run_02_data_mining.py

Cấu trúc theo Workflow Rule:
  Bước 1: Setup imports + config (seed, paths)
  Bước 2: Load data, kiểm tra shape và dtypes
  Bước 3: Implement chức năng chính (CART Feature Selection)
  Bước 4: Visualize kết quả (lưu vào report/figures/)
  Bước 5: Summary — findings + next steps

Để chạy production trên Kaggle: đặt DEBUG_MODE = False.
"""

import os
import sys
import json
import time

# ── Đường dẫn chuẩn hoá để import từ src/ bất kể chạy từ đâu ──────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Fix Windows terminal encoding (cp1252 -> utf-8)
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend — an toàn trên mọi môi trường
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.kdd.feature_selection import (
    load_and_merge,
    prepare_for_cart,
    run_cart_feature_selection,
    save_results,
)
from src.config import (
    RANDOM_SEED,
    DATA_DIR       as CFG_DATA_DIR,
    PROCESSED_DIR  as CFG_PROCESSED_DIR,
    CART_TOP_K,
    CART_MAX_DEPTH,
    CART_MIN_SAMPLES,
    CART_MAX_BINS,
    CART_CLASS_WEIGHT,
    DEBUG_N_SAMPLES,
)

# ===========================================================================
# BƯỚC 1: Setup — CONFIG
# ===========================================================================
# Thay đổi DUY NHẤT ở đây khi chuyển local → Kaggle production:
# ⚡ Local testing : DEBUG_MODE = True   (50k rows, ~5 phút)
# Kaggle full  : DEBUG_MODE = False  (590k rows, ~30–60 phút)
DEBUG_MODE = True

# Mọi hyperparameter lấy từ src/config.py — KHÔNG hard-code ở đây
N_SAMPLES_DEBUG   = DEBUG_N_SAMPLES     # 50_000
TOP_K             = CART_TOP_K          # 50
MAX_DEPTH         = CART_MAX_DEPTH      # 5
MIN_SAMPLES_SPLIT = CART_MIN_SAMPLES    # 20
MAX_BINS          = CART_MAX_BINS       # 256
CLASS_WEIGHT      = CART_CLASS_WEIGHT   # 27.6

# Paths từ config (đã resolve thành absolute path)
DATA_DIR    = CFG_DATA_DIR
OUTPUT_DIR  = CFG_PROCESSED_DIR
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'report', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  Notebook 02 — CART Feature Selection")
print("=" * 65)
print(f"  DEBUG_MODE  : {DEBUG_MODE}")
print(f"  TOP_K       : {TOP_K}")
print(f"  RANDOM_SEED : {RANDOM_SEED}")
print(f"  DATA_DIR    : {DATA_DIR}")
print()

# ===========================================================================
# ██ BƯỚC 2: Load Data + EDA nhanh
# ===========================================================================
t0 = time.time()
df, y = load_and_merge(
    data_dir    = DATA_DIR,
    debug_mode  = DEBUG_MODE,
    n_samples   = N_SAMPLES_DEBUG,
    random_state= RANDOM_SEED,
)
print(f"  >> Load time: {time.time() - t0:.1f}s\n")

# --- EDA nhanh ---
print("── EDA: Tổng quan dữ liệu " + "─" * 38)
print(f"  Shape     : {df.shape}")
print(f"  Fraud     : {y.sum():,} / {len(y):,} ({y.mean()*100:.2f}%)")
print(f"  Legit     : {(1-y).sum():,}")
print()

# Phân loại cột
numeric_cols     = df.select_dtypes(include='number').columns.drop('isFraud', errors='ignore')
categorical_cols = df.select_dtypes(include='object').columns

print(f"  Numeric cols (excl. isFraud) : {len(numeric_cols)}")
print(f"  Categorical cols (object)    : {len(categorical_cols)}")
print(f"  Categorical list             : {categorical_cols.tolist()}")
print()

# Missing value summary (Top 10 cột missing nhiều nhất)
missing = df.isnull().mean().sort_values(ascending=False).head(10)
print("── EDA: Top 10 cột missing nhiều nhất " + "─" * 25)
for col, rate in missing.items():
    bar = '█' * int(rate * 30)
    print(f"  {col:<30s} {rate*100:5.1f}%  {bar}")
print()

# ===========================================================================
# ██ BƯỚC 3: Preprocessing + Chạy CART Feature Selection
# ===========================================================================
print("── Preprocessing for CART " + "─" * 38)
t1 = time.time()
X, feature_names = prepare_for_cart(df, y, random_state=RANDOM_SEED)
print(f"  >> Preprocessing time: {time.time() - t1:.1f}s\n")

print("── Running CART " + "─" * 48)
t2 = time.time()
selector, results = run_cart_feature_selection(
    X, y, feature_names,
    top_k            = TOP_K,
    max_depth        = MAX_DEPTH,
    min_samples_split= MIN_SAMPLES_SPLIT,
    max_bins         = MAX_BINS,
    class_weight     = CLASS_WEIGHT,
)
elapsed_cart = time.time() - t2
print(f"  >> CART training time: {elapsed_cart:.1f}s")

# ===========================================================================
# BƯỚC 4: Visualize + Save
# ===========================================================================
print("\n── Visualizing " + "─" * 50)

top_names = results['top50_names']
top_imps  = results['top50_importances']

# --- Biểu đồ 1: Top-50 Feature Importance (horizontal bar chart) -----------
fig, ax = plt.subplots(figsize=(10, 14))

colors = ['#E74C3C' if i < 10 else '#3498DB' if i < 25 else '#95A5A6'
          for i in range(len(top_names))]

y_pos = range(len(top_names))
bars  = ax.barh(list(y_pos), top_imps, color=colors, edgecolor='white', linewidth=0.5)

ax.set_yticks(list(y_pos))
ax.set_yticklabels([f"{i+1:2d}. {n}" for i, n in enumerate(top_names)],
                   fontsize=8, fontfamily='monospace')
ax.invert_yaxis()
ax.set_xlabel('Weighted Information Gain (normalized)', fontsize=10)
ax.set_title(
    f'Top {TOP_K} Features — CART Feature Selection\n'
    f'({"DEBUG: 50k rows" if DEBUG_MODE else "PRODUCTION: 590k rows"}, '
    f'max_depth={MAX_DEPTH}, class_weight={CLASS_WEIGHT})',
    fontsize=11, fontweight='bold', pad=12
)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E74C3C', label='Top 10'),
    Patch(facecolor='#3498DB', label='Rank 11–25'),
    Patch(facecolor='#95A5A6', label='Rank 26–50'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()

suffix = '_debug' if DEBUG_MODE else ''
fig1_path = os.path.join(FIGURES_DIR, f'cart_top50_importance{suffix}.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved figure → {fig1_path}")

# --- Biểu đồ 2: Cumulative importance curve ---------------------------------
sorted_all = sorted(results['all_importances'].values(), reverse=True)
cumsum     = np.cumsum(sorted_all)
if cumsum[-1] > 0:
    cumsum = cumsum / cumsum[-1] * 100  # normalize to %

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, len(cumsum) + 1), cumsum, color='#2ECC71', linewidth=2)
ax.axvline(x=TOP_K, color='#E74C3C', linestyle='--', linewidth=1.5,
           label=f'Top {TOP_K} threshold')
ax.axhline(y=cumsum[TOP_K - 1] if len(cumsum) >= TOP_K else 100,
           color='#E74C3C', linestyle=':', linewidth=1, alpha=0.7)
ax.set_xlabel('Number of features (sorted by importance)', fontsize=10)
ax.set_ylabel('Cumulative importance (%)', fontsize=10)
ax.set_title('Cumulative Feature Importance — CART', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(linestyle='--', alpha=0.4)
ax.set_xlim(1, min(len(cumsum), 200))
plt.tight_layout()

fig2_path = os.path.join(FIGURES_DIR, f'cart_cumulative_importance{suffix}.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved figure → {fig2_path}")

# --- Lưu JSON ---------------------------------------------------------------
output_path = save_results(results, OUTPUT_DIR, debug_mode=DEBUG_MODE)

# ===========================================================================
# BƯỚC 5: Summary
# ===========================================================================
total_time = time.time() - t0
cumulative_pct = cumsum[TOP_K - 1] if len(cumsum) >= TOP_K else 100.0

print("\n" + "=" * 65)
print(f"  [OK] SUMMARY -- Notebook 02 Data Mining")
print("=" * 65)
print(f"  Mode           : {'DEBUG (50k rows)' if DEBUG_MODE else 'PRODUCTION (full dataset)'}")
print(f"  Dataset shape  : {df.shape}")
print(f"  Feature matrix : {X.shape} (after encoding)")
print(f"  CART time      : {elapsed_cart:.1f}s")
print(f"  Total time     : {total_time:.1f}s")
print()
print(f"  TOP {TOP_K} features cover {cumulative_pct:.1f}% of total importance")
print()
print(f"  Top 10 features selected:")
for i, (name, imp) in enumerate(zip(top_names[:10], top_imps[:10]), 1):
    print(f"    {i:2d}. {name:<40s} {imp:.6f}")
print()
print(f"  Output JSON  : {output_path}")
print(f"  Figures      : {FIGURES_DIR}")
print()
print("  Next steps:")
print("  [Tuần 5] Đưa Top 50 features vào train LightGBM Baseline")
print("  [Tuần 5] So sánh benchmark: LightGBM vs Custom MLP (AUC-PR)")
if DEBUG_MODE:
    print()
    print("  [!] Day la ket qua DEBUG (50k rows).")
    print("     Đặt DEBUG_MODE = False và chạy trên Kaggle để lấy")
    print("     top50_features.json 'chính hãng' (full 590k rows).")
print("=" * 65)
