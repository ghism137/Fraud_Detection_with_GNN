# Workflow

## Định nghĩa "session"

- **Bắt đầu**: khi mở chat mới với AI
- **Kết thúc**: khi tắt chat hoặc chuyển sang task khác hoàn toàn
- **Khi kết thúc session**: nói "Tóm tắt session và cập nhật session_state.md"

## Thứ tự ưu tiên khi bắt đầu task

```
1. Đọc session_state.md → xác định tuần hiện tại + trạng thái
2. Đọc notebook liên quan nếu đã có
3. Implement → test nhỏ → mở rộng
4. Cập nhật session_state.md sau khi xong
```

## Quy trình cho mỗi notebook

```
Bước 1: Setup imports + config (seed, paths)
Bước 2: Load data, kiểm tra shape và dtypes
Bước 3: Implement chức năng chính
Bước 4: Visualize kết quả (bắt buộc — dùng trong báo cáo)
Bước 5: Summary cell cuối notebook — ghi lại findings
```

## Khi gặp lỗi

```
1. Đọc error message ĐẦY ĐỦ trước khi hỏi
2. Kiểm tra shape của tensor/array — 90% lỗi DL là shape mismatch
3. Thử với subset nhỏ (1000 rows) trước khi chạy full dataset
4. Nếu Kaggle/Colab OOM: giảm batch_size trước, sau đó giảm feature
```

## Quy trình thêm model mới (Roadmap V2)

```
Tự code CART (Lọc feature) → Train LightGBM (Baseline) 
→ Custom MLP (ReLU) → GCN (PyG) → Late Fusion Ensemble
```
Không skip — mỗi bước là một mắt xích chuẩn bị data hoặc cung cấp prediction cho bước tiếp theo. LightGBM là baseline để đánh giá GNN.

## Git workflow

```bash
# Sau mỗi notebook hoàn thành
git add notebooks/0X_*.ipynb
git commit -m "feat: complete notebook 0X - [tên task]"

# Sau mỗi src file
git add src/
git commit -m "feat: implement [tên component]"

# KHÔNG commit data
# KHÔNG commit checkpoints lớn (>50MB)
```

## Khi context window sắp đầy

Nói: "Tóm tắt session này và cập nhật session_state.md"
Paste tóm tắt vào đầu session tiếp theo.
