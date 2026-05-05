# Explainability (XAI)

> GNNExplainer, SHAP — giải thích tại sao model dự đoán fraud.

## TL;DR

- GNNExplainer: tìm subgraph + features quan trọng nhất cho 1 prediction
- SHAP: feature importance dựa trên Shapley values — model-agnostic
- Kết hợp: GNNExplainer cho graph structure, SHAP cho feature weights
- Case study: chọn 2–3 fraud transactions, show visual explanation
- *(bổ sung thêm khi implement)*

---

## Chi tiết

### 1. Motivation — Tại sao cần giải thích model?

*(Điền: black box → trust, regulatory compliance, academic requirement)*

### 2. GNNExplainer (Ying et al. 2019)

*(Điền: cách hoạt động → output → interpretation)*

**Output:**
- Edge mask: edges nào quan trọng nhất cho dự đoán
- Feature mask: features nào của node ảnh hưởng nhất

### 3. SHAP (SHapley Additive exPlanations)

*(Điền: Shapley values → SHAP summary plot → waterfall plot)*

### 4. Case Study Template

*(Điền template cho case study trong báo cáo)*

```
Transaction ID: ___
Predicted: Fraud (probability: ___%)
Actual: Fraud/Legit

GNNExplainer:
- Subgraph: [screenshot]
- Key edges: ___

SHAP:
- Top 5 features: ___
- Waterfall plot: [screenshot]

Interpretation: ___
```

---

## Kết nối với project

- Dùng trong: tuần 8
- File implement: `notebooks/07_xai_analysis.ipynb`
- Output: `report/figures/` (screenshots cho báo cáo)

## Tài liệu tham khảo

- Ying et al. 2019 — [GNNExplainer](https://arxiv.org/abs/1903.03894)
- PyG — [Explainability module](https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html)
- SHAP — [Documentation](https://shap.readthedocs.io/)
