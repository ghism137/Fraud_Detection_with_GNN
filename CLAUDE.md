# Fraud Detection with Graph Neural Network
>
> Đồ án tích hợp 3 môn: Data Mining · Deep Learning · Business Analysis

---

## Quick Reference

| Mục | Nội dung |
|-----|----------|
| Dataset | IEEE-CIS Fraud Detection (Kaggle) |
| Compute | Kaggle Notebooks (GPU, 30h/tuần) · Colab Free T4 |
| Deploy | HuggingFace Spaces (Streamlit) |
| Deadline | 10 tuần, nộp cuối tuần 10 |

## Lệnh thường dùng

```bash
# Cài đặt môi trường
pip install -r requirements.txt

# Chạy notebook theo thứ tự
jupyter notebook notebooks/

# Chạy Streamlit demo local
streamlit run streamlit_app/app.py

# Kiểm tra code style
flake8 src/ --max-line-length=100

# Chạy tests
pytest tests/ -v
```

## Cấu trúc thư mục

```
fraud-detection-gnn/
├── CLAUDE.md             ← bạn đang ở đây
├── .claude/              ← config, rules, agents, skills
├── data/raw/             ← KHÔNG COMMIT, KHÔNG SỬA
├── notebooks/            ← 01→07, theo thứ tự
├── src/
│   ├── preprocessing/      ← Dual-Pipeline (A: LGBM, B: MLP/GNN)
│   ├── kdd/              ← Code tay CART
│   ├── baseline/         ← LightGBM Baseline
│   ├── mlp_numpy/        ← MLP viết tay (Layers, Optimizers, Trainer)
│   ├── graph/            ← Graph construction (3-Tier Heuristic)
│   └── models/           ← PyG GAT (Graph Attention Network)
├── streamlit_app/
└── report/
    ├── figures/          ← chứa biểu đồ (SHAP, ROC, Loss curves)
    └── sections/         ← NƠI ĐẶT BÁO CÁO CỦA TỪNG PHẦN
```

## Rules (đọc đầy đủ trước khi làm)

- [Core Principles](.claude/rules/core_principles.md)
- [Workflow](.claude/rules/workflow.md)
- [Coding Style](.claude/rules/coding_style.md)
- [Tech Defaults](.claude/rules/tech_defaults.md)
- [Response Format](.claude/rules/response_format.md)

## Context hiện tại

- [Session State](.claude/project/session_state.md) — trạng thái session + quyết định đã chốt + kết quả thực nghiệm
- [Issue Tracker / ADR](.claude/project/non_solve_problem.md) — danh sách nợ kỹ thuật, câu hỏi phản biện, và lịch sử sửa lỗi kiến trúc

## Tài liệu dự án

- [Project Overview](.claude/project/project.md) — kiến trúc, dataset, roadmap, tech stack

## Knowledge Base

- [Index](.claude/knowledge/_index.md) — đọc index trước để biết load file nào
- Chủ đề: fraud detection, dataset, backpropagation, GNN, graph construction, imbalanced learning, explainability, business analysis

## Skills có sẵn

- [Explain Math](.claude/skills/explain_math.md) — giải thích công thức toán theo thứ tự chuẩn
- [Implement ML Model](.claude/skills/implement_ml_model.md) — checklist khi thêm model vào baseline
- [Manage Issue Tracker](.claude/skills/manage-issue-tracker.md) — quy trình ghi nhận và giải quyết nợ kỹ thuật/sai lầm kiến trúc
- [Update Session State](.claude/skills/update-session-state.md) — quy trình cập nhật task, log state và chốt phiên hàng ngày

## Agents có sẵn

- [Researcher](.claude/agents/researcher.md) — research thư viện, paper, approach mới
- [Reviewer](.claude/agents/reviewer.md) — review code, kiểm tra logic toán học

## Guard Rails quan trọng (Antigravity)

> Các rule dưới đây được viết bằng ngôn ngữ tự nhiên để Antigravity agent hiểu và tuân thủ.

1. **KHÔNG được sửa file trong `data/raw/`** — dữ liệu gốc bất khả xâm phạm
2. **KHÔNG được sửa `.env` hoặc file chứa secrets**
3. **MLP numpy: TUYỆT ĐỐI không import torch, tensorflow, keras, autograd, jax** — yêu cầu bắt buộc của thầy
4. **CẤM DÙNG RNN/LSTM** — đã bị loại bỏ hoàn toàn khỏi scope để tránh Spatio-Temporal Explosion.
5. **Mọi experiment phải có `random_seed = 42`** — reproducibility bắt buộc
6. **Metric chính là AUC-PR** — không dùng Accuracy vì dataset imbalanced
7. **Trước khi implement, luôn giải thích toán trước** — ý nghĩa trực quan → công thức → ví dụ số → code
8. **Giao tiếp bằng tiếng Việt**, code comments bằng tiếng Anh
