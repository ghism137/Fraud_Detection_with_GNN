# Imbalanced Learning

> SMOTE, Focal Loss, class weighting, threshold optimization, AUC-PR.

## TL;DR

- IEEE-CIS: ~3.5% fraud — imbalanced nhưng không cực đoan
- Accuracy vô nghĩa (predict all legit → 96.5% accuracy)
- Metric chính: AUC-PR (robust với imbalanced), không phải AUC-ROC
- Strategies: SMOTE (data-level), Focal Loss (loss-level), class weighting (model-level)
- Threshold: optimize theo cost (5·FP + 120·FN), không theo F1
- *(bổ sung thêm khi implement)*

---

## Chi tiết

### 1. Tại sao Accuracy vô nghĩa?

*(Điền: ví dụ số, confusion matrix của "predict all 0")*

### 2. Metrics cho Imbalanced Data

#### AUC-PR (Precision-Recall AUC) — Metric chính

*(Điền: motivation → công thức → tại sao tốt hơn AUC-ROC cho imbalanced)*

#### AUC-ROC — Tham khảo

*(Điền: ưu nhược, khi nào bị inflate)*

#### F1 Score

*(Điền: harmonic mean, threshold dependency)*

### 3. Data-Level: SMOTE

*(Điền: cách hoạt động, ưu nhược, dùng khi nào)*

### 4. Loss-Level: Focal Loss

*(Điền: motivation → công thức → so sánh với BCE → ví dụ số)*

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

### 5. Model-Level: Class Weighting

*(Điền: pos_weight trong PyTorch, cách tính)*

### 6. Threshold Optimization

*(Điền: tại sao default 0.5 không tối ưu, cost-sensitive threshold)*

```
Optimal threshold = argmin(5·FP + 120·FN)
```

---

## Kết nối với project

- Dùng trong: tuần 5 (MLP optimization), tuần 7 (GNN training), tuần 9 (BA)
- File implement: `src/mlp_numpy/losses.py`, training notebooks

## Tài liệu tham khảo

- Davis & Goadrich 2006 — [Precision-Recall vs ROC](https://dl.acm.org/doi/10.1145/1143844.1143874)
- Lin et al. 2017 — [Focal Loss](https://arxiv.org/abs/1708.02002)
