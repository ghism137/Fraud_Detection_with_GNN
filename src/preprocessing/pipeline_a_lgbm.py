import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    """
    K-Fold Mean Target Encoding tích hợp Smoothing/Shrinkage.
    Được xây dựng bằng Pandas thuần để đảm bảo tiêu chí "Code tay",
    ngăn ngừa rò rỉ dữ liệu (Data Leakage) qua K-Fold, và xử lý Unseen Categories.
    """
    def __init__(self, cols, k_folds=5, smoothing_factor=10, random_state=42):
        self.cols = cols
        self.k_folds = k_folds
        self.smoothing_factor = smoothing_factor  # Hệ số m (trọng số làm mượt)
        self.random_state = random_state
        
        # Lưu trữ mapping category -> smoothed mean để dùng cho tập Test
        self.global_means_ = {}
        self.category_mappings_ = {}

    def fit(self, X, y):
        """
        Xây dựng bảng mapping (từ 100% tập Train) để chuẩn bị cho quá trình transform() trên tập Test.
        Đồng thời thỏa mãn chuẩn API của Scikit-learn.
        """
        df = X.copy()
        # [FIX] Tránh bẫy Index Alignment của Pandas khi X và y khác index
        df['target'] = np.array(y)
        
        global_mean_overall = df['target'].mean()

        for col in self.cols:
            self.global_means_[col] = global_mean_overall
            
            # Tính trên 100% dữ liệu Train để có mapping hoàn chỉnh nhất
            stats = df.groupby(col)['target'].agg(['count', 'mean'])
            n = stats['count']
            local_mean = stats['mean']
            
            # Công thức Shrinkage: lambda * Local_Mean + (1 - lambda) * Global_Mean
            lambda_val = n / (n + self.smoothing_factor)
            smoothed_mean = lambda_val * local_mean + (1 - lambda_val) * global_mean_overall
            self.category_mappings_[col] = smoothed_mean.to_dict()
            
        return self

    def fit_transform(self, X, y=None):
        """
        Sử dụng cơ chế K-Fold để encode tập Train nhằm ngăn Data Leakage.
        Dòng dữ liệu ở Fold i chỉ được encode bằng thông tin từ K-1 folds còn lại.
        """
        if y is None:
            raise ValueError("fit_transform yêu cầu biến mục tiêu y")
            
        # Gọi fit để lưu mapping cho tập Validation/Test
        self.fit(X, y)
        
        X_encoded = X.copy()
        df = X.copy()
        # [FIX] Tránh bẫy Index Alignment
        df['target'] = np.array(y)

        # Chia tập Train thành Stratified K-Folds (bảo toàn tỷ lệ 3.5% Fraud)
        np.random.seed(self.random_state)
        fraud_idx = np.where(df['target'] == 1)[0]
        legit_idx = np.where(df['target'] == 0)[0]
        np.random.shuffle(fraud_idx)
        np.random.shuffle(legit_idx)
        
        fraud_folds = np.array_split(fraud_idx, self.k_folds)
        legit_folds = np.array_split(legit_idx, self.k_folds)
        folds = [np.concatenate([fraud_folds[i], legit_folds[i]]) for i in range(self.k_folds)]

        for col in self.cols:
            encoded_col = np.zeros(len(df))
            for i in range(self.k_folds):
                val_idx = folds[i]
                # Lấy K-1 Folds còn lại làm nền tảng học
                train_idx = np.concatenate([folds[j] for j in range(self.k_folds) if j != i])
                
                train_fold = df.iloc[train_idx]
                val_fold = df.iloc[val_idx]

                # Global mean cục bộ của K-1 Folds này
                fold_global_mean = train_fold['target'].mean()

                # Tính Count và Mean trên K-1 Folds
                fold_stats = train_fold.groupby(col)['target'].agg(['count', 'mean'])
                n_fold = fold_stats['count']
                local_mean_fold = fold_stats['mean']

                # Áp dụng Smoothing cho K-1 Folds
                lambda_fold = n_fold / (n_fold + self.smoothing_factor)
                smoothed_fold = lambda_fold * local_mean_fold + (1 - lambda_fold) * fold_global_mean

                # Map giá trị đã học vào Fold đang bị giữ lại (Fold thứ K)
                # Nếu có unseen category trong nội bộ Fold thứ K, fill bằng fold_global_mean
                encoded_col[val_idx] = val_fold[col].map(smoothed_fold).fillna(fold_global_mean)

            # [FIX 2] Gán bằng pd.Series kèm index gốc để tránh Silent Bug lệch dòng
            X_encoded[col + '_target_enc'] = pd.Series(encoded_col, index=X_encoded.index)

        # [FIX 3] Drop các cột categorical thô để tránh lỗi cho mạng MLP
        X_encoded = X_encoded.drop(columns=self.cols)
        return X_encoded

    def transform(self, X):
        """
        Encode tập Validation/Test bằng dictionary đã được học từ 100% tập Train.
        Fallback về Global Mean nếu gặp Unseen Categories.
        """
        X_encoded = X.copy()
        for col in self.cols:
            mapping = self.category_mappings_[col]
            global_mean = self.global_means_[col]
            
            # O(1) Lookup: Map cực nhanh qua dictionary và an toàn với Unseen Categories
            X_encoded[col + '_target_enc'] = X_encoded[col].map(mapping).fillna(global_mean)
            
        # [FIX 3] Drop các cột categorical thô
        X_encoded = X_encoded.drop(columns=self.cols)
        return X_encoded
