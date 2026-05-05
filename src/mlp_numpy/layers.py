import numpy as np

class Linear:
    """
    Lớp Fully Connected (Linear) cơ bản bằng Numpy.
    Sử dụng He Initialization để tối ưu cho hàm kích hoạt ReLU,
    tránh hiện tượng vanishing/exploding gradients.
    """
    def __init__(self, in_features, out_features):
        # He Initialization: std = sqrt(2 / in_features)
        std = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * std
        self.b = np.zeros((1, out_features))
        
        # Cache dùng cho quá trình backward
        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
    def forward(self, x):
        """Truyền xuôi (Forward pass)"""
        self.x = x
        return np.dot(x, self.W) + self.b
        
    def backward(self, dout):
        """Lan truyền ngược (Backward pass)"""
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = np.dot(dout, self.W.T)
        return dx

class ReLU:
    """
    Hàm kích hoạt ReLU.
    Giải quyết bài toán gradient triệt tiêu của Sigmoid ở các lớp ẩn.
    """
    def __init__(self):
        self.x = None
        
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
        
    def backward(self, dout):
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx

class Sigmoid:
    """
    Hàm kích hoạt Sigmoid (Thường dùng cho output layer).
    
    [CẢNH BÁO KIẾN TRÚC TỬ HUYỆT]: 
    Lớp này CHỈ dùng trong hàm predict() (Inference) để đổi Logits ra xác suất (0, 1).
    TUYỆT ĐỐI KHÔNG ĐƯỢC ném vào luồng forward() khi huấn luyện (Training), 
    vì BCEWithLogitsLoss đã tự động tích hợp Sigmoid. Việc gọi Sigmoid 2 lần 
    sẽ làm là phẳng toàn bộ Gradient (Vanishing Gradient), giết chết mô hình ngay epoch 1.
    """
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        # Cắt giá trị x để tránh overflow (np.exp với x cực lớn hoặc cực bé)
        x_safe = np.clip(x, -500, 500)
        self.out = 1.0 / (1.0 + np.exp(-x_safe))
        return self.out
        
    def backward(self, dout):
        return dout * self.out * (1.0 - self.out)

class BCEWithLogitsLoss:
    """
    Binary Cross Entropy Loss tính trực tiếp từ Logits thay vì xác suất.
    Hợp nhất Sigmoid và BCELoss giúp tăng tính ổn định toán học (Numerical Stability).
    Tích hợp pos_weight để đặc trị Dữ liệu Mất cân bằng (Imbalanced Data).
    """
    def __init__(self, pos_weight=1.0):
        self.pos_weight = pos_weight
        self.logits = None
        self.y_true = None
        self.probs = None
        
    def forward(self, logits, y_true):
        self.logits = logits
        self.y_true = y_true.reshape(logits.shape)
        
        # Stable Sigmoid
        logits_safe = np.clip(logits, -500, 500)
        self.probs = 1.0 / (1.0 + np.exp(-logits_safe))
        probs_safe = np.clip(self.probs, 1e-7, 1 - 1e-7)
        
        # Tính Loss với trọng số lớp (pos_weight) để phạt nặng lỗi lọt lưới Fraud
        loss = - (self.pos_weight * self.y_true * np.log(probs_safe) + (1 - self.y_true) * np.log(1 - probs_safe))
        return np.mean(loss)
        
    def backward(self):
        # Đạo hàm giải tích đã tích hợp pos_weight cực kỳ thanh lịch
        # L = - w*y*log(p) - (1-y)*log(1-p)
        # dL/dx = - w*y*(1-p) + (1-y)*p
        N = self.y_true.shape[0]
        dx = (- self.pos_weight * self.y_true * (1 - self.probs) + (1 - self.y_true) * self.probs) / N
        return dx
