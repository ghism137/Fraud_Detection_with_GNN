import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MLPPipelinePreprocessor(BaseEstimator, TransformerMixin):
    """
    Tiền xử lý chuyên biệt (Pipeline B) cho mô hình Vector-based (MLP / GNN).
    Giải quyết triệt để "Nghịch lý Missing Indicators và Variance Scale Mismatch":
    Bằng cách áp dụng Z-Score Normalization lên CẢ CÁC BIẾN INDICATORS, 
    ta phá bỏ định dạng thô {0, 1} của chúng để ép Variance về 1, 
    giúp đồng bộ hoàn toàn Gradient Landscape với các biến Continuous.
    """
    def __init__(self):
        self.medians_ = {}
        self.means_ = None
        self.stds_ = None
        self.indicator_cols_ = []  # Danh sách index các cột cần tạo Indicator
        
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_cols = X.shape[1]
        
        # BƯỚC 1: Học Median (cho Imputation) và Ghi nhận cột có NaN (cho Indicators)
        for i in range(n_cols):
            col_data = X[:, i]
            is_nan = np.isnan(col_data)
            
            valid_mask = ~is_nan
            if np.any(valid_mask):
                self.medians_[i] = np.median(col_data[valid_mask])
            else:
                self.medians_[i] = 0.0
                
            if np.any(is_nan):
                self.indicator_cols_.append(i)
                
        # BƯỚC 2: Giả lập transform() trên tập Train để lấy ma trận nháp
        X_combined = self._impute_and_concat(X)
        
        # BƯỚC 3: Học Mean và Std trên TOÀN BỘ MA TRẬN MỞ RỘNG (Kể cả Indicators)
        self.means_ = np.mean(X_combined, axis=0)
        self.stds_ = np.std(X_combined, axis=0)
        
        # Tránh lỗi chia cho 0 nếu một cột hoàn toàn là hằng số
        self.stds_[self.stds_ == 0] = 1e-6
        
        return self
        
    def _impute_and_concat(self, X):
        """Logic cốt lõi: Điền Median và nối thêm cột Indicator (tương thích OOV)."""
        n_samples, n_cols = X.shape
        X_imputed = np.zeros_like(X)
        
        for i in range(n_cols):
            col_data = X[:, i]
            is_nan = np.isnan(col_data)
            
            imputed_col = col_data.copy()
            # Điền Median đã học từ tập Train (Tránh Data Leakage)
            imputed_col[is_nan] = self.medians_.get(i, 0.0)
            X_imputed[:, i] = imputed_col
            
        # Ráp thêm các cột Indicator DỰA TRÊN CẤU TRÚC ĐÃ HỌC TỪ TRAIN
        if len(self.indicator_cols_) > 0:
            indicators = []
            for i in self.indicator_cols_:
                # Nếu tập Test có NaN ở cột này -> ra 1. Không có -> ra 0.
                indicators.append(np.isnan(X[:, i]).astype(np.float64))
            X_indicators = np.column_stack(indicators)
            return np.hstack([X_imputed, X_indicators])
        else:
            return X_imputed
            
    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        
        # 1. Ráp ma trận hỗn hợp (Continuous imputed + Indicators {0, 1})
        X_combined = self._impute_and_concat(X)
        
        # 2. Phá vỡ định kiến 0/1 của Indicator bằng Z-Score Normalization
        # Ép TẤT CẢ các cột (Bao gồm Indicator) về phân phối N(0, 1)
        X_scaled = (X_combined - self.means_) / self.stds_
        
        return X_scaled
