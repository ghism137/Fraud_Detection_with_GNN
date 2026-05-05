# Tech Defaults

## Stack cố định (không thay đổi)

| Tầng | Thư viện | Ghi chú |
|------|----------|---------|
| Language | Python 3.10+ | |
| Data | pandas + numpy | |
| ML baseline | LightGBM | Trục xương sống |
| Imbalanced | imbalanced-learn | SMOTE |
| Preprocess | category_encoders | Target Encoding (K-Fold) |
| KDD / MLP tay | numpy THUẦN | Tự code CART và MLP (bắt buộc) |
| DL framework | PyTorch 2.x | |
| GNN | PyTorch Geometric | |
| XAI | SHAP + PyG GNNExplainer | |
| Visualization | matplotlib + seaborn + plotly | |
| Demo | Streamlit | |
| Deploy | HuggingFace Spaces | |

## Khi không chỉ định thư viện nào dùng

- Classification model mới → scikit-learn trước, PyTorch nếu cần custom
- Visualization → matplotlib cho báo cáo, plotly cho Streamlit demo
- Data loading → pandas, không dùng Spark (không cần scale lớn)

## Compute constraints

- **Kaggle**: 30h GPU/tuần — dùng cho GNN training
- **Colab free**: T4, session tối đa ~4h — dùng cho experiment nhỏ
- **Local**: Không có GPU → chỉ chạy data processing và MLP numpy
- Batch size mặc định: 1024 (giảm nếu OOM)

## Versions cụ thể (để reproducibility)

```
torch==2.1.0
torch-geometric==2.4.0
scikit-learn==1.3.0
pandas==2.1.0
numpy==1.24.0
lightgbm==4.1.0
category_encoders==2.6.2
shap==0.43.0
streamlit==1.28.0
```

## KHÔNG dùng (dù hấp dẫn)

- TensorFlow / Keras → đã chọn PyTorch, không mix
- DGL → đã chọn PyG
- XGBoost → Đã chọn LightGBM làm baseline xương sống do tốc độ vượt trội
- Ray / Dask → không cần distributed computing ở scale này
