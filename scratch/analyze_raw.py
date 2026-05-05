import pandas as pd
import numpy as np

# Load train files
print("=== LOADING FILES ===")
train_trans = pd.read_csv(r"C:\Users\Admin\Project\Fraud_Detection_w_GNN\data\raw\train_transaction.csv")
train_id = pd.read_csv(r"C:\Users\Admin\Project\Fraud_Detection_w_GNN\data\raw\train_identity.csv")

print(f"train_transaction: {train_trans.shape}")
print(f"train_identity:    {train_id.shape}")

# ---- isFraud ----
print("\n=== TARGET: isFraud ===")
vc = train_trans["isFraud"].value_counts()
print(vc)
rate = train_trans["isFraud"].mean()
print(f"Fraud rate: {rate:.4%}")

# ---- Column groups ----
cols = list(train_trans.columns)
print("\n=== COLUMN GROUPS ===")
print("Categorical-like:", [c for c in cols if c.startswith(("card","addr","P_","R_","M","email","Dev","id"))][:30])
print("V-columns count:", len([c for c in cols if c.startswith("V")]))
print("C-columns count:", len([c for c in cols if c.startswith("C")]))
print("D-columns count:", len([c for c in cols if c.startswith("D")]))
print("M-columns count:", len([c for c in cols if c.startswith("M")]))

# ---- Missing values train_transaction ----
print("\n=== MISSING VALUES (train_transaction) ===")
miss = train_trans.isnull().sum()
miss_pct = (miss / len(train_trans) * 100).round(2)
miss_df = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct})
miss_df = miss_df[miss_df["missing_count"] > 0].sort_values("missing_pct", ascending=False)
print(f"Columns with missing: {len(miss_df)} / {len(train_trans.columns)}")
print(miss_df.head(40).to_string())

# ---- Missing by group ----
print("\n=== MISSING BY COLUMN GROUP ===")
for prefix in ["V","C","D","M","card","addr"]:
    grp = [c for c in cols if c.startswith(prefix)]
    if grp:
        grp_miss = miss_pct[grp]
        print(f"{prefix}-cols: total={len(grp)}, "
              f"avg_miss={grp_miss.mean():.1f}%, "
              f"max_miss={grp_miss.max():.1f}%, "
              f"cols_>80%_miss={int((grp_miss > 80).sum())}")

# ---- Data types ----
print("\n=== DTYPES SUMMARY ===")
dtype_counts = train_trans.dtypes.value_counts()
print(dtype_counts)

# ---- Categorical columns ----
cat_cols = train_trans.select_dtypes(include="object").columns.tolist()
print(f"\nObject columns ({len(cat_cols)}): {cat_cols}")
for c in cat_cols:
    n_unique = train_trans[c].nunique()
    top5 = train_trans[c].value_counts().head(5).to_dict()
    print(f"  {c}: nunique={n_unique}, top5={top5}")

# ---- TransactionAmt ----
print("\n=== TransactionAmt stats ===")
print(train_trans["TransactionAmt"].describe())

# ---- Identity table ----
print("\n=== TRAIN IDENTITY ===")
print("Columns:", list(train_id.columns))
miss_id = train_id.isnull().sum()
miss_id_pct = (miss_id / len(train_id) * 100).round(2)
id_miss_df = pd.DataFrame({"missing_count": miss_id, "missing_pct": miss_id_pct})
id_miss_df = id_miss_df[id_miss_df["missing_count"] > 0].sort_values("missing_pct", ascending=False)
print(f"\nIdentity missing values ({len(id_miss_df)} cols):")
print(id_miss_df.to_string())

# ---- Join coverage ----
print("\n=== JOIN COVERAGE ===")
merged = train_trans.merge(train_id, on="TransactionID", how="left")
has_identity = merged["id_01"].notna().sum()
print(f"Transactions with identity info: {has_identity} / {len(merged)} ({has_identity/len(merged):.2%})")
