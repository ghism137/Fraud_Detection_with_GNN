# Backpropagation & Neural Network Fundamentals

> Chain rule, gradient flow, backprop qua nhiều layer — nền tảng cho MLP numpy.

## TL;DR

- Forward: Z = WX + b → A = σ(Z) — tính output layer by layer
- Loss: BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)] — đo sai lệch prediction vs truth
- Backward: chain rule dL/dW = dL/dA · dA/dZ · dZ/dW — truyền gradient ngược
- Update: W -= lr · dW — gradient descent
- MLP numpy: implement 100% bằng numpy, KHÔNG autograd
- *(bổ sung thêm khi học)*

---

## Chi tiết

### 1. Motivation — Tại sao cần backpropagation?

*(Điền: giải thích tại sao không thể tìm W tối ưu bằng giải tích thông thường)*

### 2. Forward Pass

*(Điền: giải thích trực quan → công thức → ví dụ số)*

```
Input X → [W1, b1] → Z1 → ReLU → A1 → [W2, b2] → Z2 → Sigmoid → ŷ
```

### 3. Loss Function — Binary Cross Entropy

*(Điền: motivation → công thức → ví dụ số → tại sao không dùng MSE)*

### 4. Chain Rule — Cốt lõi của Backprop

*(Điền: giải thích chain rule → trace qua 2-layer MLP → ví dụ tính tay)*

**Gradient flow:**
```
dL/dŷ → dL/dZ2 → dL/dW2, dL/db2
                → dL/dA1 → dL/dZ1 → dL/dW1, dL/db1
```

### 5. Gradient Descent & Learning Rate

*(Điền: ý nghĩa lr, lr quá lớn vs quá nhỏ)*

### 6. Activation Functions

*(Điền: Sigmoid, ReLU, Tanh — đạo hàm, ưu nhược điểm)*

| Function | Formula | Derivative | Dùng khi |
|----------|---------|------------|----------|
| Sigmoid | 1/(1+e^-z) | σ(1-σ) | Output layer (binary) |
| ReLU | max(0,z) | 0 or 1 | Hidden layers |
| Tanh | (e^z-e^-z)/(e^z+e^-z) | 1-tanh²(z) | *(ít dùng trong project này)* |

### 7. Weight Initialization

*(Điền: tại sao không khởi tạo W=0, He init, Xavier init)*

---

## Kết nối với project

- Dùng trong: tuần 1–3, MLP numpy (nộp thầy)
- File implement: `src/mlp_numpy/layers.py`, `src/mlp_numpy/losses.py`, `src/mlp_numpy/trainer.py`
- Notebook: `notebooks/03_mlp_numpy.ipynb`

## Tài liệu tham khảo

- Karpathy — [micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Nielsen — [Neural Networks and Deep Learning, ch.1–2](http://neuralnetworksanddeeplearning.com)
- CS231n — [Backpropagation Notes](https://cs231n.github.io/optimization-2/)
- 3Blue1Brown — [Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
