import numpy as np
from sklearn.metrics import average_precision_score

from ..preprocessing.pipeline_b_mlp import MLPPipelinePreprocessor
from .layers import Linear, ReLU, Sigmoid, BCEWithLogitsLoss
from .optimizers import SGDCosineWarmup, get_batches

class NumpyMLP:
    """
    Ráp nối các cấu phần Layers, Loss, Optimizer thành một mạng Nơ-ron hoàn chỉnh.
    Kiến trúc mặc định: Input -> Linear -> ReLU -> Linear -> ReLU -> Linear(1) -> BCEWithLogitsLoss
    """
    def __init__(self, in_features, hidden_sizes=[64, 32], pos_weight=27.6):
        self.layers = []
        
        # Xây dựng các lớp ẩn
        prev_size = in_features
        for h_size in hidden_sizes:
            self.layers.append(Linear(prev_size, h_size))
            self.layers.append(ReLU())
            prev_size = h_size
            
        # Lớp Output (1 node, Binary Classification)
        self.layers.append(Linear(prev_size, 1))
        
        # Hàm Loss đã tích hợp tham số Imbalance
        self.criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Sigmoid: Đã niêm phong, CHỈ DÙNG CHO INFERENCE
        self.sigmoid = Sigmoid() 
        
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
        
    def backward(self):
        # Tính đạo hàm từ hàm Loss ngược về đầu
        dout = self.criterion.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            
    def predict_proba(self, X):
        """Inference an toàn với Sigmoid để nhả ra xác suất."""
        logits = self.forward(X)
        return self.sigmoid.forward(logits).flatten()


class MLPTrainer:
    """
    Quản lý toàn bộ luồng học máy: Tiền xử lý (Pipeline B) -> Huấn luyện (Mini-batch) -> Early Stopping.
    """
    def __init__(self, hidden_sizes=[64, 32], batch_size=2048, max_epochs=50, 
                 lr_max=0.01, lr_min=1e-5, pos_weight=27.6, patience=5, weight_decay=1e-4):
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.pos_weight = pos_weight
        self.patience = patience
        self.weight_decay = weight_decay
        
        self.preprocessor = MLPPipelinePreprocessor()
        self.model = None
        self.optimizer = None
        self.best_val_auc = -1.0
        
    def fit(self, X_train, y_train, X_val, y_val):
        print("[MLP] Tiền xử lý dữ liệu với Pipeline B...")
        # Sử dụng fit_transform thừa kế từ TransformerMixin
        X_train_scaled = self.preprocessor.fit_transform(X_train, y_train)
        X_val_scaled = self.preprocessor.transform(X_val)
        
        in_features = X_train_scaled.shape[1]
        self.model = NumpyMLP(in_features, self.hidden_sizes, self.pos_weight)
        
        # Nạp các lớp có tham số (Linear) vào Optimizer
        self.optimizer = SGDCosineWarmup(
            self.model.layers, 
            lr_max=self.lr_max, 
            lr_min=self.lr_min,
            total_epochs=self.max_epochs,
            weight_decay=self.weight_decay
        )
        
        patience_counter = 0
        print(f"[MLP] Khởi chạy Gradient Descent (Batch={self.batch_size}, L2={self.weight_decay}, Pos_W={self.pos_weight})")
        
        for epoch in range(self.max_epochs):
            # --- PHASE 1: TRAINING ---
            train_losses = []
            
            for X_batch, y_batch in get_batches(X_train_scaled, y_train, self.batch_size, shuffle=True):
                # 1. Forward pass (Không đi qua Sigmoid)
                logits = self.model.forward(X_batch)
                
                # 2. Tính Loss (BCEWithLogitsLoss xử lý toàn bộ Toán học Log-Sum-Exp và Pos_Weight)
                loss = self.model.criterion.forward(logits, y_batch)
                train_losses.append(loss)
                
                # 3. Backward pass
                self.model.backward()
                
                # 4. Step & Zero Grad
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            self.optimizer.update_epoch()
            
            # --- PHASE 2: VALIDATION ---
            # Batching Validation để chống OOM
            y_val_preds = []
            for i in range(0, X_val_scaled.shape[0], self.batch_size):
                end_idx = min(i + self.batch_size, X_val_scaled.shape[0])
                batch_X = X_val_scaled[i:end_idx]
                
                # Inference CÓ đi qua Sigmoid để lấy Xác suất
                batch_probs = self.model.predict_proba(batch_X)
                y_val_preds.extend(batch_probs)
                
            val_probs = np.array(y_val_preds)
            val_auc_pr = average_precision_score(y_val, val_probs)
            
            print(f"Epoch {epoch+1:02d}/{self.max_epochs} | Loss: {np.mean(train_losses):.4f} | Val AUC-PR: {val_auc_pr:.4f} | LR: {self.optimizer.get_lr():.6f}")
            
            # --- PHASE 3: EARLY STOPPING ---
            if val_auc_pr > self.best_val_auc:
                self.best_val_auc = val_auc_pr
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"[MLP] Kích hoạt Early Stopping tại epoch {epoch+1}. Best AUC-PR: {self.best_val_auc:.4f}")
                break
                
        return self
        
    def predict_proba(self, X):
        """Hàm API suy luận cho tập dữ liệu mới (Test)."""
        if self.model is None:
            raise RuntimeError("Mô hình chưa được fit().")
            
        # Áp dụng Pipeline B
        X_scaled = self.preprocessor.transform(X)
        
        # Suy luận chia lô để tránh RAM Overflow
        preds = []
        for i in range(0, X_scaled.shape[0], self.batch_size):
            end_idx = min(i + self.batch_size, X_scaled.shape[0])
            batch_X = X_scaled[i:end_idx]
            batch_probs = self.model.predict_proba(batch_X)
            preds.extend(batch_probs)
            
        return np.array(preds)
