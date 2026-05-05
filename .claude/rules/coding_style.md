# Coding Style

## Python chung

```python
# Naming
variable_name   = snake_case       # biến
RANDOM_SEED     = 42               # hằng số → UPPER_SNAKE
class MyModel   = PascalCase       # class
def train_model = snake_case       # function, bắt đầu bằng động từ

# Imports — thứ tự bắt buộc
import os, sys                     # 1. stdlib
import numpy as np                 # 2. third-party
import pandas as pd
import torch
from sklearn.metrics import ...
from src.mlp_numpy.layers import * # 3. local

# Max line length: 100 ký tự
# Dùng f-string, không dùng .format() hay %
```

## OOP & Custom Estimators (Scikit-Learn API)

Khi code tay các class tiền xử lý (Preprocessing) hoặc thuật toán Data Mining (như Custom CART), BẮT BUỘC tuân thủ chuẩn OOP của Scikit-Learn để dễ dàng tích hợp vào Pipeline:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    """
    [Docstring mô tả rõ ràng chức năng của class]
    """
    def __init__(self, param1=10):
        # 1. Chỉ dùng __init__ để gán tham số khởi tạo (hyperparameters)
        # 2. KHÔNG thực hiện tính toán hay khởi tạo dữ liệu state ở đây
        self.param1 = param1
        
    def fit(self, X, y=None):
        # 1. Thực hiện logic tính toán state trên tập Train (lưu vào biến có hậu tố _, ví dụ: self.means_)
        # 2. Xử lý bẫy Index Alignment của Pandas nếu X, y lệch index (ép về numpy array)
        self.is_fitted_ = True
        return self # Bắt buộc return self
        
    def transform(self, X):
        # 1. Chỉ sử dụng state đã fit để biến đổi tập X
        # 2. Tuyệt đối không fit lại trên tập Validation/Test
        pass
        
    # Không cần định nghĩa fit_transform vì TransformerMixin đã tự động cung cấp
```

## Numpy MLP (strict — nộp thầy)

```python
# TUYỆT ĐỐI không import
# import torch, tensorflow, keras, autograd, jax

# Mọi operation phải là numpy thuần
# Shape phải được comment rõ ràng

def forward(self, X):
    # X: (batch_size, n_features)
    self.Z1 = X @ self.W1 + self.b1   # (batch_size, hidden_size)
    self.A1 = self.relu(self.Z1)       # (batch_size, hidden_size)
    self.Z2 = self.A1 @ self.W2 + self.b2  # (batch_size, 1)
    return self.sigmoid(self.Z2)

# Mỗi bước backprop phải có comment giải thích gradient
def backward(self, X, y, y_hat):
    # dL/dZ2 = y_hat - y  (đạo hàm BCE qua sigmoid)
    dZ2 = y_hat - y                    # (batch_size, 1)
    dW2 = self.A1.T @ dZ2 / m         # (hidden_size, 1)
    ...
```

## PyTorch / PyG

```python
# Luôn set seed ngay đầu file
torch.manual_seed(42)
np.random.seed(42)

# Device handling
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop — template chuẩn
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
```

## KDD & Data Processing (Chống rò rỉ dữ liệu)

```python
# MỌI thao tác Target Encoding BẮT BUỘC phải dùng K-Fold hoặc Smoothing
# KHÔNG ĐƯỢC dùng pandas .groupby().mean() trực tiếp trên tập train vì sẽ gây data leakage
import category_encoders as ce
target_enc = ce.TargetEncoder(smoothing=10) # Hoặc dùng K-Fold custom

# Train LightGBM BẮT BUỘC phải dùng Early Stopping
callbacks = [lgb.early_stopping(stopping_rounds=50)]
```

## Notebooks

```python
# Cell đầu tiên của mọi notebook
import sys
sys.path.append('../')  # để import từ src/

RANDOM_SEED = 42
DATA_DIR = '../data/raw/'
OUTPUT_DIR = '../report/figures/'

# Mỗi section có markdown header
# ## 1. Load Data
# ## 2. EDA
# ## 3. ...

# Cell cuối cùng: Summary
# - Findings chính
# - Metrics đạt được
# - Next steps
```

## Metrics — bắt buộc report đủ bộ

```python
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,  # AUC-PR
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

# Luôn report theo thứ tự: AUC-PR → AUC-ROC → F1 → P/R
```
