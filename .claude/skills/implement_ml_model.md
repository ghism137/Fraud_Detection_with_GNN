# Skill: Implement ML Model (Baseline Sequence)

> Dùng khi thêm một model mới vào baseline comparison

## Checklist bắt buộc

```
□ 1. Train/val/test split TRƯỚC — không fit bất kỳ thứ gì trên test set
□ 2. Fit scaler/encoder CHỈ trên train set, transform val và test
□ 3. Set random_seed = 42
□ 4. Train model
□ 5. Predict probabilities (predict_proba), không chỉ class
□ 6. Tính đủ 4 metrics: AUC-PR, AUC-ROC, F1, P/R
□ 7. Lưu kết quả vào bảng so sánh trong session_state.md
□ 8. Plot Precision-Recall curve
□ 9. Lưu model checkpoint nếu là PyTorch
```

## Template code chuẩn

```python
# 1. Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Scale (chỉ fit trên train)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # transform only

# 3. Train + Evaluate
from sklearn.metrics import roc_auc_score, average_precision_score
y_prob = model.predict_proba(X_test)[:, 1]

print(f"AUC-PR:  {average_precision_score(y_test, y_prob):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
```

## Sau khi xong

Cập nhật bảng kết quả trong `session_state.md` với AUC-PR và AUC-ROC.
