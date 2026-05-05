# Skill: Explain Math Concept

> Dùng khi cần giải thích công thức toán học trong project

## Thứ tự bắt buộc

```
1. Động lực (Motivation)
   → Tại sao cần thứ này? Nó giải quyết vấn đề gì?

2. Trực quan (Intuition)
   → Giải thích bằng ngôn ngữ tự nhiên, không có ký hiệu

3. Công thức (Formula)
   → LaTeX, định nghĩa mọi ký hiệu

4. Ví dụ số (Numerical Example)
   → Tính tay với số nhỏ, cụ thể

5. Code (Implementation)
   → Numpy trước nếu liên quan đến MLP tay
   → PyTorch sau nếu dùng framework

6. Kết nối với project (Connection)
   → Thứ này được dùng ở đâu trong fraud detection pipeline?
```

## Các concept quan trọng cần nắm vững (cho bảo vệ)

- Chain rule và backpropagation qua nhiều layer
- GCN propagation rule: ý nghĩa của normalization D^{-1/2} A D^{-1/2}
- Tại sao AUC-PR tốt hơn Accuracy cho imbalanced data
- Focal Loss vs BCE: khi nào dùng cái nào
- Graph construction: tại sao card1+card4+addr1 là node ID tốt

## Ví dụ cách gọi skill

```
"Dùng skill explain_math để giải thích GCN propagation rule cho tôi"
"Giải thích chain rule theo thứ tự trong skill explain_math"
```
