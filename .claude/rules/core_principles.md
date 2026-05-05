# Core Principles

## Triết lý làm việc

1. **Hiểu trước, implement sau**
   Với mọi công thức toán học (backprop, GCN propagation, loss function),
   giải thích ý nghĩa trực quan TRƯỚC khi viết code.
   Không viết code mà bạn không hiểu tại sao nó hoạt động.

2. **Simple > Clever**
   Code đơn giản, dễ đọc luôn được ưu tiên hơn code "xịn" khó hiểu.
   Đây là đồ án học thuật — clarity quan trọng hơn performance micro.

3. **Working > Perfect**
   Ưu tiên pipeline chạy được end-to-end trước khi tối ưu từng bước.
   Milestone quan trọng nhất: mỗi notebook phải CHẠY ĐƯỢC hoàn chỉnh.

4. **Ranh giới numpy vs framework là bất khả xâm phạm**
   MLP numpy: TUYỆT ĐỐI không import torch, tensorflow, hoặc bất kỳ autograd nào.
   Đây là yêu cầu của thầy — vi phạm = mất điểm toàn bộ phần đó.

5. **Reproducibility là bắt buộc**
   Mọi experiment phải có `random_seed = 42` (hoặc số cố định).
   Kết quả phải reproduce được khi chạy lại.

6. **Document lý do, không chỉ document code**
   Comment giải thích WHY, không phải WHAT.
   Đặc biệt quan trọng cho phần toán (backprop, GCN) — thầy sẽ hỏi.

7. **Nếu tôi sai, hãy phản biện lại tôi một cách lịch sự và đưa ra lý do**
   Đừng chỉ làm theo yêu cầu một cách máy móc.
   Luôn đặt câu hỏi: "Tại sao chúng ta làm điều này?", "Có cách nào tốt hơn không?"

8. **Hãy luôn đưa ra các giả định, và các câu hỏi mở**
   Giải thích tại sao bạn đưa ra chúng.
   Tìm kiếm các câu hỏi mở để khám phá sâu hơn về vấn đề.
