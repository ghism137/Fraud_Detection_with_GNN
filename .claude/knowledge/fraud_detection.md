# Fraud Detection — Domain Knowledge

> Kiến thức về gian lận tín dụng: patterns, cách định danh, red flags, nghiệp vụ.

## TL;DR

- Fraud chiếm ~3.5% transactions (IEEE-CIS) — imbalanced nặng
- Fraud không ngẫu nhiên — có patterns: cùng card, cùng device, cùng email domain
- Card linkage (card1 + card4 + addr1) tạo "account fingerprint" — cơ sở xây graph
- FN cost >> FP cost: bỏ lọt fraud ($120) đắt hơn báo nhầm ($5)
- *(bổ sung thêm khi học)*

---

## Chi tiết

### 1. Fraud là gì? Cách định danh

*(Điền: định nghĩa fraud trong context credit card, các loại fraud phổ biến)*

**Các loại fraud phổ biến:**
- Card-not-present (CNP) fraud
- Account takeover
- Synthetic identity fraud
- *(bổ sung...)*

**Cách hệ thống thực tế phát hiện fraud:**
- Rule-based systems (if-else thủ công)
- ML-based scoring (XGBoost, neural network)
- Graph-based detection (phát hiện mạng lưới gian lận)
- *(bổ sung...)*

### 2. Fraud Patterns trong IEEE-CIS

*(Điền: những patterns đặc trưng tìm được từ EDA)*

**Red flags kỳ vọng:**
- Cùng card1 nhưng khác addr1 → có thể card bị đánh cắp
- TransactionAmt bất thường so với mean của card đó
- Email domain lạ (không phải gmail, yahoo, hotmail)
- *(bổ sung sau EDA...)*

### 3. Card Linkage — Tại sao card1 + card4 + addr1?

*(Điền: giải thích logic tạo account fingerprint)*

```
uid = card1 + '_' + card4 + '_' + addr1
```

- `card1`: mã số card (hashed)
- `card4`: loại card (visa, mastercard...)
- `addr1`: billing address code

**Ý nghĩa**: Cùng bộ 3 này ≈ cùng một "account". Nhiều transactions từ cùng account có thể chia sẻ fraud signal.

### 4. Cost Matrix — Tại sao FN đắt hơn FP?

*(Điền: giải thích nghiệp vụ)*

| Scenario | Cost | Giải thích |
|----------|------|------------|
| FN (bỏ lọt fraud) | $120 | Ngân hàng phải hoàn tiền + chi phí điều tra |
| FP (báo nhầm legit) | $5 | Khách bị khóa tạm → gọi support → friction nhỏ |

---

## Kết nối với project

- Dùng trong: EDA (notebook 01), Feature Engineering, Business Analysis (notebook cuối)
- File implement: `notebooks/01_eda.ipynb`, `streamlit_app/app.py`

## Tài liệu tham khảo

*(Thêm link paper, blog, video khi tìm được)*
