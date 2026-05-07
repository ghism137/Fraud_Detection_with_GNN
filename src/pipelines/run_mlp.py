"""
run_mlp.py — Orchestrator: Pipeline B → CART → Custom MLP (Numpy)
==================================================================
Vai trò: "Nhạc trưởng" cho luồng MLP thuần Numpy.

LƯU Ý THIẾT KẾ QUAN TRỌNG:
  MLPTrainer.fit() đã tích hợp MLPPipelinePreprocessor (Pipeline B) bên trong.
  → Orchestrator chỉ truyền raw Top50 features, Pipeline B áp dụng tự động.
  → Khác với run_lgbm.py (LGBM không cần Z-Score normalization).

Luồng:
  [1] Load & Merge raw CSV
  [2] Chuẩn bị feature matrix (Target Encode + M-flags, giữ NaN numeric)
  [3] CART Feature Selection — hoặc load từ cache (--use_cache)
  [4] Lọc Top50 features (raw, NaN còn đó — Pipeline B sẽ xử lý)
  [5] Stratified Train/Val split
  [6] Train Custom MLP (MLPTrainer tự chạy Pipeline B + He Init + Cosine Warmup)
  [7] Evaluate (AUC-PR + Cost Matrix threshold)
  [8] Lưu y_prob.npy → phục vụ Late Fusion Ensemble (Tuần 8)

Sử dụng:
  python src/pipelines/run_mlp.py --mode debug
  python src/pipelines/run_mlp.py --mode full --use_cache
"""

import os
import sys
import json
import argparse
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_recall_curve

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
from src.mlp_numpy.mlp_trainer import MLPTrainer


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Custom MLP (Numpy) Orchestrator — Fraud Detection"
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


# ── Cache helpers (pattern giống run_lgbm.py) ─────────────────────────────────
def _get_cache_path(debug_mode: bool) -> str:
    fname = cfg.CART_CACHE_FILE_DEBUG if debug_mode else cfg.CART_CACHE_FILE
    return os.path.join(cfg.PROCESSED_DIR, fname)


def _load_top50_from_cache(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["top50_feature_names"]


# ── Stratified split ───────────────────────────────────────────────────────────
def _stratified_split(X, y, val_size=0.2, seed=42):
    np.random.seed(seed)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    np.random.shuffle(fraud_idx)
    np.random.shuffle(legit_idx)
    n_val_fraud = max(1, int(len(fraud_idx) * val_size))
    n_val_legit = max(1, int(len(legit_idx) * val_size))
    val_idx   = np.concatenate([fraud_idx[:n_val_fraud], legit_idx[:n_val_legit]])
    train_idx = np.concatenate([fraud_idx[n_val_fraud:], legit_idx[n_val_legit:]])
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


# ── Evaluate wrapper (MLPTrainer chưa có evaluate(), viết tại đây) ─────────────
def _evaluate_mlp(y_true: np.ndarray, y_prob: np.ndarray, split_name: str = "Val") -> dict:
    """
    Tính AUC-PR, AUC-ROC và Cost Matrix threshold.
    Tái sử dụng logic quét threshold tương tự lgbm_trainer.find_optimal_threshold().
    """
    auc_pr  = average_precision_score(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)

    # Quét toàn bộ threshold trên PR Curve để tìm τ* tối ưu Cost
    _, _, thresholds = precision_recall_curve(y_true, y_prob)
    min_cost = np.inf
    tau_star = cfg.COST_FP / (cfg.COST_FP + cfg.COST_FN)   # Bayes threshold = 0.04

    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        cost = cfg.COST_FP * fp + cfg.COST_FN * fn
        if cost < min_cost:
            min_cost = cost
            tau_star = tau

    y_pred = (y_prob >= tau_star).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    print(f"\n{'='*55}")
    print(f"  Kết quả MLP [{split_name}] tại τ* = {tau_star:.4f}")
    print(f"{'='*55}")
    print(f"  AUC-PR   : {auc_pr:.4f}   ← Metric chính")
    print(f"  AUC-ROC  : {auc_roc:.4f}")
    print(f"  TP={tp} | FP={fp} | FN={fn} | TN={tn}")
    print(f"  Tổng chi phí ước tính: ${cfg.COST_FP*fp + cfg.COST_FN*fn:,.0f}")
    print(f"{'='*55}\n")

    return {
        "AUC-PR": auc_pr, "AUC-ROC": auc_roc,
        "tau_star": tau_star, "Total_Cost": min_cost,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    debug_mode = (args.mode == "debug")

    print("=" * 60)
    print(f"  MLP Pipeline  |  mode={args.mode}  |  use_cache={args.use_cache}")
    print("=" * 60)

    # [1] Load & Merge
    df, y = load_and_merge(
        data_dir=cfg.DATA_DIR,
        debug_mode=debug_mode,
        n_samples=cfg.DEBUG_N_SAMPLES,
        random_state=cfg.RANDOM_SEED,
    )

    # [2] Chuẩn bị feature matrix (Target Encode + M-flags, giữ NaN numeric)
    #     NaN trong numeric được giữ nguyên — MLPTrainer sẽ Impute qua Pipeline B.
    X_all, feature_names = prepare_for_cart(df, y, random_state=cfg.RANDOM_SEED)

    # [3] CART hoặc cache
    cache_json = _get_cache_path(debug_mode)
    if args.use_cache and os.path.exists(cache_json):
        print(f"\n[CACHE] Load top50 từ: {cache_json}")
        top50_names = _load_top50_from_cache(cache_json)
    else:
        if args.use_cache:
            print("[CACHE] Không tìm thấy cache. Chạy CART từ đầu...")
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
    print(f"    NaN count (sẽ được Pipeline B xử lý): {np.isnan(X_top50).sum():,}")

    # [5] Stratified Train/Val split
    X_train, X_val, y_train, y_val = _stratified_split(
        X_top50, y, val_size=cfg.VAL_SIZE, seed=cfg.RANDOM_SEED
    )
    print(f"[5] Train: {X_train.shape} | Val: {X_val.shape}")
    print(f"    Fraud rate — Train: {y_train.mean():.4f} | Val: {y_val.mean():.4f}")

    # [6] Train MLP
    #     MLPTrainer.fit() tự chạy Pipeline B bên trong:
    #       fit_transform(X_train) → Median Impute + Indicators + Z-Score
    #       transform(X_val)       → áp dụng params đã học từ Train
    trainer = MLPTrainer(
        hidden_sizes=cfg.MLP_HIDDEN_SIZES,
        batch_size=cfg.MLP_BATCH_SIZE,
        max_epochs=cfg.MLP_EPOCHS,
        lr_max=cfg.MLP_LR_MAX,
        lr_min=cfg.MLP_LR_MIN,
        pos_weight=cfg.MLP_POS_WEIGHT,
        patience=cfg.MLP_PATIENCE,
        weight_decay=cfg.MLP_WEIGHT_DECAY,
    )
    trainer.fit(X_train, y_train, X_val, y_val)

    # [7] Evaluate
    y_val_prob = trainer.predict_proba(X_val)
    metrics    = _evaluate_mlp(y_val, y_val_prob, split_name="Validation")

    # [8] Lưu artifacts cho Late Fusion Ensemble (Tuần 8)
    os.makedirs(cfg.PROCESSED_DIR, exist_ok=True)
    suffix     = "_debug" if debug_mode else ""
    proba_path = os.path.join(cfg.PROCESSED_DIR, f"mlp_proba{suffix}.npy")
    yval_path  = os.path.join(cfg.PROCESSED_DIR, f"mlp_y_val{suffix}.npy")

    np.save(proba_path, y_val_prob)
    np.save(yval_path,  y_val)

    print(f"[8] Đã lưu artifacts:")
    print(f"    {proba_path}")
    print(f"    {yval_path}")
    print(f"\n>>> AUC-PR = {metrics['AUC-PR']:.4f}  ← cập nhật vào session_state.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
