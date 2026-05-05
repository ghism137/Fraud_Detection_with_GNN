"""
LightGBM Baseline Trainer
=========================
Module huấn luyện mô hình LightGBM làm baseline, tích hợp:
  - Hyperparameters cốt lõi cho dữ liệu Imbalanced (scale_pos_weight)
  - Tìm ngưỡng phân loại tối ưu τ* theo Cost Matrix (FN=$120, FP=$5)
  - Metric chính: AUC-PR (average_precision)
  - Reproducibility: random_state=42

Tham khảo: Ke et al. 2017 — LightGBM (NeurIPS)
"""

import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

# ─────────────────────────────────────────────
# 1. Hằng số Cost Matrix (Business Analysis)
# ─────────────────────────────────────────────
COST_FN = 120   # Chi phí bỏ lọt 1 giao dịch Fraud ($)
COST_FP = 5     # Chi phí chặn nhầm 1 giao dịch hợp lệ ($)

# ─────────────────────────────────────────────
# 2. Hyperparameters cốt lõi
# ─────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":           "binary",
    "metric":              "average_precision",   # AUC-PR — metric chính của đồ án
    "scale_pos_weight":    27.6,                  # N_legit / N_fraud ≈ 569877 / 20663
    "n_estimators":        3000,
    "learning_rate":       0.05,
    "num_leaves":          63,
    "min_child_samples":   50,                    # Tránh overfit trên nhóm Fraud nhỏ
    "subsample":           0.8,                   # Row subsampling
    "subsample_freq":      1,
    "colsample_bytree":    0.8,                   # Feature subsampling (trong 50 features)
    "reg_lambda":          2.0,                   # L2 regularization
    "reg_alpha":           0.1,                   # L1 regularization
    "random_state":        42,                    # Reproducibility — bắt buộc theo CLAUDE.md
    "n_jobs":              -1,
    "verbose":             -1,
}

# ─────────────────────────────────────────────
# 3. Hàm tìm ngưỡng τ* theo Cost Matrix
# ─────────────────────────────────────────────
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Tìm ngưỡng phân loại τ* tối thiểu hóa tổng chi phí kinh doanh.

    Công thức lý thuyết (phân tích ngưỡng tối ưu):
      Block khi: P(fraud) > τ*
      τ* = COST_FP / (COST_FP + COST_FN) = 5 / (5 + 120) = 0.04

    Tuy nhiên, trong thực tế phân phối dự đoán không hoàn hảo, ta quét
    toàn bộ thresholds trên Precision-Recall Curve để tìm τ* thực nghiệm.

    Returns:
        tau_star (float): Ngưỡng tối ưu
        min_cost (float): Tổng chi phí tối thiểu tại ngưỡng đó
    """
    # Lý thuyết: τ* = C_FP / (C_FP + C_FN) — điểm bàng quan Bayes
    tau_bayes = COST_FP / (COST_FP + COST_FN)   # = 0.04

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    n_total = len(y_true)
    n_fraud = np.sum(y_true == 1)
    n_legit = n_total - n_fraud

    min_cost = np.inf
    tau_star = tau_bayes   # Khởi đầu từ ngưỡng lý thuyết

    for i, tau in enumerate(thresholds):
        y_pred = (y_prob >= tau).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        total_cost = COST_FP * fp + COST_FN * fn

        if total_cost < min_cost:
            min_cost = total_cost
            tau_star = tau

    print(f"  [Cost Matrix] Ngưỡng Bayes lý thuyết: τ_bayes = {tau_bayes:.4f}")
    print(f"  [Cost Matrix] Ngưỡng tối ưu thực nghiệm: τ* = {tau_star:.4f}")
    print(f"  [Cost Matrix] Tổng chi phí tối thiểu: ${min_cost:,.0f}")
    return tau_star, min_cost


# ─────────────────────────────────────────────
# 4. Hàm đánh giá tổng thể
# ─────────────────────────────────────────────
def evaluate(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> dict:
    """
    Tính toán bộ metrics đầy đủ tại ngưỡng τ cho trước.
    """
    y_pred = (y_prob >= tau).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "AUC-PR":    average_precision_score(y_true, y_prob),
        "AUC-ROC":   roc_auc_score(y_true, y_prob),
        "Precision": precision,
        "Recall":    recall,
        "F1":        f1,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Total_Cost": COST_FP * fp + COST_FN * fn,
    }


# ─────────────────────────────────────────────
# 5. Class Trainer chính
# ─────────────────────────────────────────────
class LGBMFraudTrainer:
    """
    Wrapper huấn luyện và đánh giá LightGBM cho bài toán Fraud Detection.
    Tích hợp Early Stopping, AUC-PR callback, và Cost Matrix thresholding.
    """

    def __init__(self, params: dict = None):
        self.params = params or LGBM_PARAMS
        self.model: lgb.LGBMClassifier = None
        self.tau_star: float = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray,   y_val: np.ndarray) -> "LGBMFraudTrainer":
        """
        Huấn luyện với Early Stopping trên tập Validation.
        Callbacks theo dõi average_precision để dừng đúng lúc.
        """
        print("[LGBM] Bắt đầu huấn luyện Baseline LightGBM...")
        print(f"  Train: {X_train.shape} | Val: {X_val.shape}")
        print(f"  scale_pos_weight = {self.params['scale_pos_weight']:.1f}")

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100),
            ],
        )

        # Tìm τ* trên tập Validation ngay sau khi fit
        y_val_prob = self.model.predict_proba(X_val)[:, 1]
        self.tau_star, _ = find_optimal_threshold(y_val, y_val_prob)

        print(f"[LGBM] Huấn luyện hoàn tất. Best iteration: {self.model.best_iteration_}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Trả về xác suất Fraud (cột 1) cho tập đầu vào."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, tau: float = None) -> np.ndarray:
        """Phân loại nhị phân với ngưỡng τ (mặc định dùng τ* từ Cost Matrix)."""
        threshold = tau if tau is not None else self.tau_star
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate_full(self, X: np.ndarray, y_true: np.ndarray,
                      split_name: str = "Test") -> dict:
        """In bảng metrics đầy đủ và trả về dict kết quả."""
        y_prob = self.predict_proba(X)
        metrics = evaluate(y_true, y_prob, self.tau_star)

        print(f"\n{'='*50}")
        print(f"  Kết quả [{split_name}] tại τ* = {self.tau_star:.4f}")
        print(f"{'='*50}")
        print(f"  AUC-PR  : {metrics['AUC-PR']:.4f}   ← Metric chính")
        print(f"  AUC-ROC : {metrics['AUC-ROC']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall   : {metrics['Recall']:.4f}")
        print(f"  F1       : {metrics['F1']:.4f}")
        print(f"  TP={metrics['TP']} | FP={metrics['FP']} | FN={metrics['FN']} | TN={metrics['TN']}")
        print(f"  Tổng chi phí ước tính: ${metrics['Total_Cost']:,.0f}")
        print(f"{'='*50}\n")
        return metrics
