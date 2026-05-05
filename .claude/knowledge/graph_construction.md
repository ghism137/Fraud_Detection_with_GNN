# Graph Construction from Tabular Data

> Cách xây graph từ IEEE-CIS tabular data: thiết kế node, edge, features.

## TL;DR

- Node = account fingerprint: `uid = card1 + '_' + card4 + '_' + addr1`
- Edge = 2 transactions cùng uid → cùng account → connected
- Node features = aggregated transaction features (mean, count, etc.)
- Đây là bipartite-like graph: account nodes liên kết qua shared attributes
- *(bổ sung thêm khi implement)*

---

## Chi tiết

### 1. Motivation — Tại sao cần xây graph?

*(Điền: tabular data → graph, fraud ring detection)*

### 2. Node Design — Account Fingerprint

*(Điền: tại sao card1+card4+addr1, alternatives đã cân nhắc)*

```python
df['uid'] = df['card1'].astype(str) + '_' + \
            df['card4'].astype(str) + '_' + \
            df['addr1'].astype(str)
```

### 3. Edge Design

*(Điền: khi nào 2 nodes có edge, edge types, edge weights)*

**Các loại edge có thể xây:**
- Cùng account (uid) → edge mạnh
- Cùng email domain → edge yếu
- Cùng device → edge trung bình
- *(bổ sung...)*

### 4. Node Feature Aggregation

*(Điền: từ nhiều transactions → 1 node feature vector, aggregation methods)*

### 5. Graph Statistics

*(Điền sau khi xây graph: số nodes, edges, avg degree, connected components)*

---

## Kết nối với project

- Dùng trong: tuần 5–6
- File implement: `src/graph/builder.py`, `src/graph/dataset.py`
- Notebook: `notebooks/05_graph_construction.ipynb`

## Tài liệu tham khảo

*(Thêm link khi tìm được)*
