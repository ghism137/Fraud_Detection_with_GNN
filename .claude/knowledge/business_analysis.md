# Business Analysis

> Cost-benefit analysis, ROI, threshold optimization, deployment considerations.

## TL;DR

- Cost matrix: FN = $120 (missed fraud), FP = $5 (false alarm friction)
- Threshold tối ưu: minimize 5·FP + 120·FN, không maximize F1
- ROI = (cost saved by model - cost of running model) / cost of running model
- Deploy: HuggingFace Spaces (Streamlit) — demo, không production
- *(bổ sung thêm khi phân tích)*

---

## Chi tiết

### 1. Cost-Benefit Framework

*(Điền: confusion matrix with costs, total cost formula)*

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | TN = $0 | FP = $5 |
| **Actual Fraud** | FN = $120 | TP = $0 |

### 2. Threshold Optimization

*(Điền: sweep threshold 0→1, plot total cost vs threshold, find minimum)*

### 3. ROI Calculation

*(Điền: công thức ROI, so sánh "có model" vs "không có model")*

### 4. Model Comparison — Business Perspective

*(Điền: không chỉ so AUC-PR, mà so total cost ở optimal threshold)*

### 5. Deployment Considerations

*(Điền: latency, real-time vs batch, concept drift, retraining schedule)*

### 6. Limitations & Risks

*(Điền: model decay, adversarial fraud, fairness/bias concerns)*

---

## Kết nối với project

- Dùng trong: tuần 9
- File implement: `streamlit_app/app.py`, final report
- Notebook: *(tạo notebook riêng nếu cần)*

## Tài liệu tham khảo

*(Thêm link khi tìm được)*
