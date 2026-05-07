"""
KDD Feature Selection Pipeline
===============================
Orchestrates: Load → Encode (Target Enc + M-flags) → CART → Save Top-K

Thiết kế:
- Mọi preprocessing cho CART dùng CustomTargetEncoder (không dùng Label Encode)
  → Ánh xạ category → xác suất rủi ro liên tục → CART tìm split point có ý nghĩa
- M-flags (M1–M9): T→1, F→0, NaN→-1 (ordinal rõ nghĩa)
- NaN trong numeric: CART xử lý nội bộ qua median_bin (quantile_binning)
- Output: top50_features.json (debug) hoặc top50_features.json (production)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# --- Local imports ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from src.preprocessing.pipeline_a_lgbm import CustomTargetEncoder
from src.kdd.cart_tree import DecisionTreeFeatureSelector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# M-flag mapping: T=match, F=mismatch, NaN=unknown
M_FLAG_COLS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
M_FLAG_MAP  = {'T': 1.0, 'F': 0.0}  # NaN → -1.0 handled separately

# Columns that are not features (IDs + target)
DROP_COLS = ['TransactionID', 'isFraud']


# ---------------------------------------------------------------------------
# Step 1: Load & merge
# ---------------------------------------------------------------------------
def load_and_merge(data_dir, debug_mode=False, n_samples=50_000, random_state=RANDOM_SEED):
    """
    Load train_transaction.csv + train_identity.csv và LEFT JOIN qua TransactionID.

    Args:
        data_dir    : Path đến thư mục chứa raw CSV.
        debug_mode  : True → stratified-sample n_samples rows.
        n_samples   : Số dòng sample khi debug_mode=True.
        random_state: Seed để đảm bảo reproducibility.

    Returns:
        df (pd.DataFrame): Merged dataframe, ĐÃ bao gồm cột isFraud.
        y  (np.ndarray)  : Target array int32.
    """
    print("[1/4] Loading data...")

    trans_path = os.path.join(data_dir, 'train_transaction.csv')
    id_path    = os.path.join(data_dir, 'train_identity.csv')

    df_trans = pd.read_csv(trans_path)
    df_id    = pd.read_csv(id_path)
    print(f"  train_transaction : {df_trans.shape}")
    print(f"  train_identity    : {df_id.shape}")

    # LEFT JOIN — giữ lại tất cả transactions (kể cả không có identity record)
    df = df_trans.merge(df_id, on='TransactionID', how='left')
    print(f"  After JOIN        : {df.shape}")

    y = df['isFraud'].values.astype(np.int32)
    fraud_rate = y.mean()
    print(f"  Fraud rate        : {fraud_rate:.4f} ({y.sum()} / {len(y)})")

    if debug_mode:
        print(f"  [DEBUG] Stratified sampling {n_samples} rows...")
        np.random.seed(random_state)

        fraud_idx = np.where(y == 1)[0]
        legit_idx = np.where(y == 0)[0]

        # Giữ đúng tỷ lệ fraud trong sample
        n_fraud = max(1, int(round(n_samples * fraud_rate)))
        n_legit = n_samples - n_fraud

        sampled_fraud = np.random.choice(fraud_idx, size=min(n_fraud, len(fraud_idx)), replace=False)
        sampled_legit = np.random.choice(legit_idx, size=min(n_legit, len(legit_idx)), replace=False)

        sampled_idx = np.sort(np.concatenate([sampled_fraud, sampled_legit]))
        df = df.iloc[sampled_idx].reset_index(drop=True)
        y  = y[sampled_idx]

        print(f"  Sampled shape     : {df.shape}")
        print(f"  Sampled fraud rate: {y.mean():.4f}")

    return df, y


# ---------------------------------------------------------------------------
# Step 2: Encode M-flags
# ---------------------------------------------------------------------------
def encode_m_flags(df):
    """
    Encode M-flag columns (M1–M9): T→1.0, F→0.0, NaN→-1.0.

    Lý do dùng -1 cho NaN thay vì median:
    - NaN ở M-flags mang ngữ nghĩa "unknown match status" — khác với missing random
    - Giá trị -1 nằm ngoài {0, 1}, CART có thể tạo split riêng cho nhóm này
    """
    existing = [c for c in M_FLAG_COLS if c in df.columns]
    for col in existing:
        df[col] = df[col].map(M_FLAG_MAP).fillna(-1.0).astype(np.float64)
    return df


# ---------------------------------------------------------------------------
# Step 3: Prepare features for CART
# ---------------------------------------------------------------------------
def prepare_for_cart(df, y, random_state=RANDOM_SEED):
    """
    Tiền xử lý đầy đủ để đưa dữ liệu vào CART:
      1. Drop non-feature columns (ID, target)
      2. Encode M-flags (T/F/NaN → 1/0/-1)
      3. Target Encode categorical (object dtype) qua CustomTargetEncoder (K-Fold, anti-leakage)
         → Biến category thành xác suất rủi ro liên tục
         → CART tìm split point có ý nghĩa (thay vì label int ngẫu nhiên)
      4. Numeric NaN: giữ nguyên → CART xử lý qua median_bin trong quantile_binning

    Returns:
        X            (np.ndarray) : Feature matrix float64, shape (n_samples, n_features)
        feature_names(list[str])  : Tên từng cột (dùng để tra cứu sau)
    """
    print("[2/4] Preparing features for CART...")
    df = df.copy()

    # 1. Drop ID + target
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 2. Encode M-flags
    df = encode_m_flags(df)

    # 3. Target-encode remaining categorical columns
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"  Categorical cols to Target Encode ({len(categorical_cols)}): {categorical_cols}")

    if categorical_cols:
        encoder = CustomTargetEncoder(
            cols=categorical_cols,
            k_folds=5,
            smoothing_factor=10,
            random_state=random_state
        )
        # fit_transform dùng K-Fold → output đã có _target_enc cols, đã drop cột thô
        df = encoder.fit_transform(df, y)

    feature_names = df.columns.tolist()
    X = df.values.astype(np.float64)

    n_nan = int(np.isnan(X).sum())
    print(f"  Feature matrix    : {X.shape}")
    print(f"  NaN count (numeric, handled by CART median_bin): {n_nan:,}")

    return X, feature_names


# ---------------------------------------------------------------------------
# Step 4: Run CART Feature Selection
# ---------------------------------------------------------------------------
def run_cart_feature_selection(
    X, y, feature_names,
    top_k=50,
    max_depth=5,
    min_samples_split=20,
    max_bins=256,
    class_weight=27.6
):
    """
    Khởi chạy DecisionTreeFeatureSelector (CART tự code numpy).

    class_weight=27.6 = (1 - 0.035) / 0.035 ≈ pos_weight cho Fraud.
    Weighted Information Gain đảm bảo CART không bỏ qua lớp thiểu số.

    Returns:
        selector (DecisionTreeFeatureSelector): đã fit(), dùng để transform() sau.
        results  (dict): top50_names, top50_importances, all_importances.
    """
    print(f"[3/4] Running CART Feature Selection "
          f"(top_k={top_k}, max_depth={max_depth}, max_bins={max_bins})...")
    print(f"  Input shape   : {X.shape}")
    print(f"  class_weight  : {class_weight} (1/fraud_rate ≈ 27.6)")

    selector = DecisionTreeFeatureSelector(
        top_k=top_k,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_bins=max_bins,
        class_weight=class_weight
    )
    selector.fit(X, y)

    feature_names_arr = np.array(feature_names)
    top_idx          = selector.selected_features_          # sorted by importance desc
    top_k_names      = feature_names_arr[top_idx].tolist()
    top_k_imps       = selector.feature_importances_[top_idx].tolist()
    all_imps         = dict(zip(feature_names, selector.feature_importances_.tolist()))

    print(f"\n  {'Rank':<5} {'Feature':<45} {'Importance':>10}")
    print(f"  {'-'*65}")
    for rank, (name, imp) in enumerate(zip(top_k_names, top_k_imps), 1):
        print(f"  {rank:<5} {name:<45} {imp:>10.6f}")

    results = {
        'top_k'            : top_k,
        'top50_names'      : top_k_names,
        'top50_importances': top_k_imps,
        'all_importances'  : all_imps,
    }
    return selector, results


# ---------------------------------------------------------------------------
# Step 5: Save results
# ---------------------------------------------------------------------------
def save_results(results, output_dir, debug_mode=False):
    """
    Lưu kết quả Feature Selection ra JSON.

    Output:
        data/processed/top50_features_debug.json  (debug_mode=True)
        data/processed/top50_features.json         (debug_mode=False, production)
    """
    print("[4/4] Saving results...")
    os.makedirs(output_dir, exist_ok=True)

    suffix      = '_debug' if debug_mode else ''
    output_path = os.path.join(output_dir, f'top50_features{suffix}.json')

    output_data = {
        'metadata': {
            'debug_mode' : debug_mode,
            'top_k'      : results['top_k'],
            'description': (
                'Top-K features selected by custom CART (numpy) with weighted Information Gain. '
                'debug=True: 50k stratified sample. debug=False: full 590k rows (production).'
            )
        },
        'top50_feature_names': results['top50_names'],
        'top50_importances'  : dict(zip(results['top50_names'], results['top50_importances'])),
        'all_importances'    : results['all_importances'],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Convenience: run full pipeline in one call
# ---------------------------------------------------------------------------
def run_pipeline(data_dir, output_dir, debug_mode=False,
                 n_samples=50_000, top_k=50,
                 max_depth=5, min_samples_split=20,
                 max_bins=256, class_weight=27.6,
                 random_state=RANDOM_SEED):
    """
    Pipeline đầy đủ: Load → Encode → CART → Save.
    Dùng được từ notebook hoặc script.

    Returns:
        selector     : Fitted DecisionTreeFeatureSelector
        results      : dict với top50_names, importances
        output_path  : Đường dẫn file JSON đã lưu
    """
    df, y              = load_and_merge(data_dir, debug_mode, n_samples, random_state)
    X, feature_names   = prepare_for_cart(df, y, random_state)
    selector, results  = run_cart_feature_selection(
        X, y, feature_names,
        top_k=top_k, max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_bins=max_bins, class_weight=class_weight
    )
    output_path = save_results(results, output_dir, debug_mode)
    return selector, results, output_path
