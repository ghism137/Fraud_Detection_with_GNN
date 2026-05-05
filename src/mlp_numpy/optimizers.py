import numpy as np

def get_batches(X, y, batch_size=2048, shuffle=True):
    """
    Trình sinh (Generator) mini-batch an toàn về bộ nhớ.
    Sử dụng để tránh nạp toàn bộ 600,000 dòng dữ liệu vào RAM tính toán cùng lúc (OOM).
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
        
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]


class SGDCosineWarmup:
    """
    Bộ tối ưu hóa (Optimizer) SGD với Momentum, tích hợp:
    1. Linear Warmup: Tăng dần Learning Rate ở các epoch đầu để tránh sốc gradient.
    2. Cosine Annealing: Hạ dần Learning Rate theo đường cong Cosine để hội tụ mịn màng ở các epoch cuối.
    3. Weight Decay (L2 Regularization): Phạt trọng số W phình to, đặc trị Overfitting trên Tabular Data.
    """
    def __init__(self, layers, lr_max=0.01, lr_min=1e-5, warmup_epochs=5, total_epochs=50, momentum=0.9, weight_decay=1e-4):
        # Lọc ra các lớp có tham số (Linear)
        self.layers = [layer for layer in layers if hasattr(layer, 'W')]
        
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.current_epoch = 0
        
        # Khởi tạo vận tốc (velocity) cho kỹ thuật Momentum
        self.vW = [np.zeros_like(layer.W) for layer in self.layers]
        self.vb = [np.zeros_like(layer.b) for layer in self.layers]
        
    def get_lr(self):
        """Tính toán Learning Rate cho epoch hiện tại."""
        if self.current_epoch < self.warmup_epochs:
            # Giai đoạn Warmup: Tuyến tính từ 0 đến lr_max
            return self.lr_max * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Giai đoạn Cosine Annealing Decay
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            # Ép tiến độ trong khoảng [0, 1] để an toàn
            progress = min(max(progress, 0.0), 1.0)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return self.lr_min + (self.lr_max - self.lr_min) * cosine_decay
            
    def step(self):
        """Cập nhật trọng số của mạng nơ-ron."""
        lr = self.get_lr()
        for i, layer in enumerate(self.layers):
            # Tích hợp Weight Decay (L2) vào đạo hàm của W
            dW_with_decay = layer.dW + self.weight_decay * layer.W
            
            # Cập nhật vận tốc (Momentum)
            self.vW[i] = self.momentum * self.vW[i] + lr * dW_with_decay
            self.vb[i] = self.momentum * self.vb[i] + lr * layer.db
            
            # Cập nhật tham số (Weight & Bias)
            layer.W -= self.vW[i]
            layer.b -= self.vb[i]
            
    def zero_grad(self):
        """Xóa gradient thừa từ mini-batch trước."""
        for layer in self.layers:
            layer.dW.fill(0)
            layer.db.fill(0)
            
    def update_epoch(self):
        """Báo hiệu hoàn thành một epoch để điều chỉnh Learning Rate."""
        self.current_epoch += 1
