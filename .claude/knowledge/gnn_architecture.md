# GNN Architecture

> GCN, GraphSAGE, message passing paradigm, propagation rule.

## TL;DR

- GNN = neural network trên graph — node "hỏi" neighbors để cập nhật representation
- Message passing: h_v = UPDATE(h_v, AGGREGATE({h_u : u ∈ N(v)}))
- GCN propagation: H' = σ(D̃⁻¹/²ÃD̃⁻¹/²HW) — Ã = A + I (self-loop)
- GraphSAGE: sample neighbors → aggregate (mean/LSTM/pool) — scalable hơn GCN
- Trong project: GCN là target chính, GraphSAGE là stretch goal
- *(bổ sung thêm khi học)*

---

## Chi tiết

### 1. Motivation — Tại sao cần GNN?

*(Điền: tabular ML bỏ qua quan hệ giữa transactions, GNN capture được)*

**So sánh MLP vs GNN cho fraud detection:**

| Khía cạnh | MLP | GNN |
|-----------|-----|-----|
| Input | Feature vector riêng lẻ | Feature + graph structure |
| Fraud signal | Chỉ từ transaction đó | Từ cả neighborhood |
| Phát hiện fraud ring | Không | Có |

### 2. Graph Basics cho GNN

*(Điền: adjacency matrix, degree matrix, node features)*

### 3. Message Passing Paradigm

*(Điền: intuition → formal definition → ví dụ)*

```
Mỗi layer GNN:
1. AGGREGATE: mỗi node thu thập messages từ neighbors
2. UPDATE: kết hợp aggregated message với representation hiện tại
3. Repeat for L layers → mỗi node "nhìn thấy" L-hop neighborhood
```

### 4. GCN — Graph Convolutional Network (Kipf & Welling 2017)

*(Điền: motivation → propagation rule → ý nghĩa normalization → ví dụ)*

**Propagation rule:**
$$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-\frac{1}{2}}\,\tilde{A}\,\tilde{D}^{-\frac{1}{2}}\,H^{(l)}\,W^{(l)}\right)$$

**Ý nghĩa từng thành phần:**
- $\tilde{A} = A + I$: *(điền)*
- $\tilde{D}$: *(điền)*
- $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$: *(điền — symmetric normalization)*
- $W^{(l)}$: *(điền)*

### 5. GraphSAGE (Hamilton et al. 2017)

*(Điền: khác biệt với GCN, sampling strategy, aggregator types)*

### 6. Model Architecture trong project

```
Input features → GCNConv(in, hidden) → ReLU → Dropout
              → GCNConv(hidden, 1) → Sigmoid → fraud probability
```

*(Điền: lý do chọn kiến trúc này, hyperparameters)*

---

## Kết nối với project

- Dùng trong: tuần 6–7, GNN implementation
- File implement: `src/models/gcn.py`, `src/models/graphsage.py`
- Notebook: `notebooks/06_gnn_training.ipynb`

## Tài liệu tham khảo

- distill.pub — [A Gentle Introduction to GNN](https://distill.pub/2021/gnn-intro/)
- Kipf & Welling 2017 — [GCN paper](https://arxiv.org/abs/1609.02907)
- Hamilton et al. 2017 — [GraphSAGE](https://arxiv.org/abs/1706.02216)
- PyG — [Introduction by Example](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html)
