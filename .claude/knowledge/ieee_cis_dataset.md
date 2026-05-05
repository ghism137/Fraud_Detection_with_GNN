# IEEE-CIS Fraud Detection Dataset

> Chi tiết từng nhóm feature, ý nghĩa, missing patterns, và cách xử lý.

## TL;DR

- 2 bảng: transaction (~590k rows, ~394 cols) + identity (~144k rows, ~41 cols)
- JOIN qua `TransactionID` (left join — identity chỉ có ~25% rows)
- Target: `isFraud` (~3.5% positive)
- V columns: 70–90% missing, group theo pattern → median impute per group
- M columns: T/F/NaN → encode 3 class (NaN là signal riêng)
- *(bổ sung thêm khi EDA)*

---

## Chi tiết

### 1. Transaction Features

#### TransactionDT & TransactionAmt

*(Điền: phân phối, outliers, cách normalize)*

#### ProductCD (W/H/C/S/R)

*(Điền: phân phối mỗi loại, fraud rate theo ProductCD)*

#### Card Features (card1–card6)

*(Điền: ý nghĩa, missing rate, vai trò trong graph construction)*

#### Address (addr1, addr2) & Distance (dist1, dist2)

*(Điền: phân phối, missing pattern, mối liên hệ với fraud)*

#### Email Domains (P_emaildomain, R_emaildomain)

*(Điền: top domains, fraud rate theo domain)*

#### Count Features (C1–C14)

*(Điền: ý nghĩa phỏng đoán, correlation với isFraud)*

#### Timedelta Features (D1–D15)

*(Điền: ý nghĩa, missing pattern)*

#### Match Flags (M1–M9)

*(Điền: encoding strategy T/F/NaN → 1/0/-1)*

#### Vesta Features (V1–V339)

*(Điền: missing pattern analysis, grouping strategy)*

### 2. Identity Features

#### Device Info

*(Điền: DeviceType distribution, DeviceInfo cleaning)*

#### ID Features (id_01–id_38)

*(Điền: ý nghĩa phỏng đoán, missing rates)*

### 3. Missing Value Strategy (đã chốt)

| Nhóm | Strategy | Lý do |
|------|----------|-------|
| V1–V339 | Group theo missing pattern → median per group | Missing theo cụm, không random |
| dist2 | Median by card1 group | Missing phụ thuộc loại card |
| M1–M9 | Encode T/F/NaN → 1/0/-1 | NaN là signal riêng |
| id_* columns | Left JOIN, giữ NaN làm feature | Chỉ 25% rows có identity data |

---

## Kết nối với project

- Dùng trong: notebook 01 (EDA), notebook 02 (data mining), mọi model
- File implement: `notebooks/01_eda.ipynb`

## Tài liệu tham khảo

- [Kaggle competition page](https://www.kaggle.com/c/ieee-fraud-detection)
- *(thêm EDA notebooks hay từ Kaggle community)*
