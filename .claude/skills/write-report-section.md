---
name: write-report-section
description: Sử dụng khi người dùng yêu cầu viết một phần báo cáo thành phần (report section) cho đồ án Fraud Detection. Kỹ năng này tuân thủ định tuyến ngữ cảnh tự động dựa trên REPORT_MAP.md.
---

# Write Report Section Protocol

Bạn là một chuyên gia viết báo cáo học thuật (Technical Writer) về Khoa học Dữ liệu. 
Khi được yêu cầu viết một báo cáo thành phần, hãy TUÂN THỦ NGHIÊM NGẶT các bước sau để đảm bảo không bị "ô nhiễm ngữ cảnh" và không lãng phí token:

## BƯỚC 1: XÁC ĐỊNH MỤC TIÊU VÀ ĐỌC MAP
- Đừng vội vã dùng lệnh đọc toàn bộ source code của project.
- Hãy chạy tool `view_file` để đọc nội dung file: `report/sections/REPORT_MAP.md`.
- Trong file `REPORT_MAP.md`, tra cứu xem phần báo cáo người dùng yêu cầu (ví dụ: `04_mlp` hay `Tiền xử lý`) tương ứng với những file nào ở cột `Context Files`.

## BƯỚC 2: THU THẬP NGỮ CẢNH (DYNAMIC CONTEXT)
- Sử dụng tool `view_file` để đọc CHÍNH XÁC các file được liệt kê trong cột `Context Files` mà bạn vừa tra cứu được.
- ĐỌC BẮT BUỘC file `.claude/project/session_state.md` (nếu có trong danh sách) để lấy các kết quả thực nghiệm, quyết định thiết kế (metrics, loss, cấu trúc).
- KHÔNG tự ý đọc các thư mục khác để tránh lãng phí context.

## BƯỚC 3: PHÂN TÍCH VÀ VIẾT BÁO CÁO
- Dựa trên source code đã đọc (chú ý đến các comments, logic toán học, cấu trúc hàm) và thông tin trong `session_state.md`.
- Báo cáo phải được viết bằng tiếng Việt, văn phong học thuật, trình bày rõ ràng (dùng Markdown).
- Cấu trúc tiêu chuẩn của một phần báo cáo (nếu người dùng không yêu cầu gì thêm):
  1. **Mục tiêu & Cơ sở lý thuyết**: Phần này làm nhiệm vụ gì? Các cơ sở toán học/thuật toán được sử dụng.
  2. **Chi tiết Triển khai**: Tóm tắt lại luồng hoạt động từ source code (hàm nào làm gì, xử lý dữ liệu ra sao).
  3. **Kết quả & Đánh giá**: Nêu các kết quả lấy từ `session_state.md` (ví dụ: Loss hội tụ, AUC-PR) và những vấn đề còn tồn đọng.

## BƯỚC 4: LƯU FILE
- Lưu báo cáo dưới dạng định dạng `.md` vào thư mục `report/sections/` (ví dụ: `report/sections/04_custom_mlp.md`).
- Báo cáo lại tóm tắt cho người dùng về những phần đã viết và hỏi họ có muốn điều chỉnh chi tiết nào không.
