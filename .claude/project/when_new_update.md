Bước 1: Lưu trữ Research vào knowledge/ (Không ghi đè)
Nếu bạn vừa đọc một paper mới (ví dụ: phát hiện ra GraphSAGE tốt hơn GCN cho tập dữ liệu lớn):

Mở file .claude/knowledge/gnn_architecture.md.
Không xóa phần viết về GCN.
Thêm một heading mới có gắn tag thời gian/phiên bản: ## [Update Tuần 5] Chuyển đổi: Tại sao GraphSAGE ưu việt hơn GCN.
Lý do: Agent (và chính bạn) khi đọc lại sẽ hiểu được Lịch sử tiến hóa (Evolution history) — tại sao ngày xưa dùng GCN mà nay lại bỏ.
Bước 2: Cập nhật Trạng thái tại session_state.md
Mở .claude/project/session_state.md, đi tới mục Quyết định đã chốt.

Nguyên tắc của mục này là (chỉ append, không xóa).
Bạn gạch ngang phần cũ (dùng ~~text~~) và thêm quyết định mới ở ngay dưới:
markdown

### Kiến trúc (Roadmap V2 -> V3)

- ~~GNN dùng PyTorch + PyG (GCN) trên Top 50 features.~~
- [Update Tuần 5] Đổi sang GraphSAGE với Neighbor Sampling vì ma trận kề quá lớn gây Out-of-Memory.
Bước 3: Cách ly rủi ro bằng Git Branch (Về mặt Code)
Đừng bao giờ code research mới trực tiếp lên branch main.

Nhắc tôi: "Tạo branch mới tên experiment/graphsage-update".
Thử nghiệm thuật toán mới trên branch này.
Nếu fail -> Bỏ branch, kiến trúc cũ trên main vẫn nguyên vẹn.
Nếu thành công -> Merge vào main và cập nhật lại file .claude/project/project.md (Roadmap tổng).
