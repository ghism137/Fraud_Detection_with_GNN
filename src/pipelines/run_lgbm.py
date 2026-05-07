"""
run_lgbm.py — Orchestrator: Pipeline A → CART → LightGBM Baseline
==================================================================
Vai trò: "Nhạc trưởng" — gọi đúng thứ tự các module đã có.
Không chứa business logic: mọi tính toán nằm trong module riêng.

Luồng:
  [1] Load & Merge raw CSV
  [2] Chuẩn bị feature matrix (Target Encode + M-flags)
  [3] CART Feature Selection — hoặc load từ cache (--use_cache)
  [4] Lọc Top50 features
  [5] Stratified Train/Val split
  [6] Train LightGBM Baseline
  [7] Evaluate (AUC-PR + Cost Matrix)
  [8] Lưu y_prob.npy → phục vụ Late Fusion Ensemble (Tuần 8)

Sử dụng:
  python src/pipelines/run_lgbm.py --mode debug
  python src/pipelines/run_lgbm.py --mode full --use_cache
"""

import os
import sys
import json
import argparse
import numpy as np

# ── Setup project root path ───────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from src import config as cfg
from src.kdd.feature_selection import (
    load_and_merge,
    prepare_for_cart,
    run_cart_feature_selection,
    save_results,
)
from src.baseline.lgbm_trainer import LGBMFraudTrainer


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="LightGBM Baseline Orchestrator — Fraud Detection"
    )
    parser.add_argument(
        "--mode", choices=["full", "debug"], default="debug",
        help="'debug': stratified sample 50k dòng. 'full': toàn bộ 590k dòng."
    )
    parser.add_argument(
        "--use_cache", action="store_true",
        help="Bỏ qua bước CART nếu file JSON top50 đã tồn tại."
    )
    return parser.parse_args()


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _get_cache_path(debug_mode: bool) -> str:
    """Đường dẫn file JSON cache tương ứng với mode."""
    fname = cfg.CART_CACHE_FILE_DEBUG if debug_mode else cfg.CART_CACHE_FILE
    return os.path.join(cfg.PROCESSED_DIR, fname)


def _load_top50_from_cache(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["top50_feature_names"]


# ── Stratified split (manual, không dùng sklearn để tuân thủ numpy-first) ─────
def _stratified_split(X, y, val_size=0.2, seed=42):
    """Tách Train/Val với tỷ lệ fraud được bảo toàn."""
    np.random.seed(seed)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    np.random.shuffle(fraud_idx)
    np.random.shuffle(legit_idx)

    n_val_fraud = max(1, int(len(fraud_idx) * val_size))
    n_val_legit = max(1, int(len(legit_idx) * val_size))

    val_idx   = np.concatenate([fraud_idx[:n_val_fraud],  legit_idx[:n_val_legit]])
    train_idx = np.concatenate([fraud_idx[n_val_fraud:],  legit_idx[n_val_legit:]])
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    debug_mode = (args.mode == "debug")

    print("=" * 60)
    print(f"  LGBM Pipeline  |  mode={args.mode}  |  use_cache={args.use_cache}")
    print("=" * 60)

    # [1] Load & Merge
    df, y = load_and_merge(
        data_dir=cfg.DATA_DIR,
        debug_mode=debug_mode,
        n_samples=cfg.DEBUG_N_SAMPLES,
        random_state=cfg.RANDOM_SEED,
    )

    # [2] Chuẩn bị feature matrix (Target Encode categoricals, encode M-flags)
    #     Luôn chạy vì cần X_all để train LGBM dù có cache hay không.
    X_all, feature_names = prepare_for_cart(df, y, random_state=cfg.RANDOM_SEED)

    # [3] CART hoặc cache
    cache_json = _get_cache_path(debug_mode)
    if args.use_cache and os.path.exists(cache_json):
        print(f"\n[CACHE] Load top50 từ: {cache_json}")
        top50_names = _load_top50_from_cache(cache_json)
    else:
        if args.use_cache:
            print(f"[CACHE] Không tìm thấy cache. Chạy CART từ đầu...")
        _, results = run_cart_feature_selection(
            X_all, y, feature_names,
            top_k=cfg.CART_TOP_K,
            max_depth=cfg.CART_MAX_DEPTH,
            min_samples_split=cfg.CART_MIN_SAMPLES,
            max_bins=cfg.CART_MAX_BINS,
            class_weight=cfg.CART_CLASS_WEIGHT,
        )
        save_results(results, cfg.PROCESSED_DIR, debug_mode)
        top50_names = results["top50_names"]

    # [4] Lọc Top50 features
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    top50_idx   = [name_to_idx[n] for n in top50_names if n in name_to_idx]
    X_top50     = X_all[:, top50_idx]
    print(f"\n[4] Feature matrix sau lọc Top50: {X_top50.shape}")

    # [5] Stratified Train/Val split
    X_train, X_val, y_train, y_val = _stratified_split(
        X_top50, y, val_size=cfg.VAL_SIZE, seed=cfg.RANDOM_SEED
    )
    print(f"[5] Train: {X_train.shape} | Val: {X_val.shape}")
    print(f"    Fraud rate — Train: {y_train.mean():.4f} | Val: {y_val.mean():.4f}")

    # [6] Train LightGBM
    trainer = LGBMFraudTrainer(params=cfg.LGBM_PARAMS)
    trainer.fit(X_train, y_train, X_val, y_val)

    # [7] Evaluate
    metrics = trainer.evaluate_full(X_val, y_val, split_name="Validation")

    # [8] Lưu y_prob.npy và y_val.npy cho Late Fusion Ensemble (Tuần 8)
    os.makedirs(cfg.PROCESSED_DIR, exist_ok=True)
    suffix     = "_debug" if debug_mode else ""
    proba_path = os.path.join(cfg.PROCESSED_DIR, f"lgbm_proba{suffix}.npy")
    yval_path  = os.path.join(cfg.PROCESSED_DIR, f"lgbm_y_val{suffix}.npy")

    np.save(proba_path, trainer.predict_proba(X_val))
    np.save(yval_path,  y_val)

    print(f"\n[8] Đã lưu artifacts:")
    print(f"    {proba_path}")
    print(f"    {yval_path}")
    print(f"\n>>> AUC-PR = {metrics['AUC-PR']:.4f}  ← cập nhật vào session_state.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
