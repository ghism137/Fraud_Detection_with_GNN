import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def compute_entropy(y, class_weight=None):
    """
    Tính Entropy với tùy chọn class_weight để trị Imbalanced Data (vd: 3.5% Fraud).
    Nếu class_weight = 27.6, Fraud sẽ bị phạt nặng hơn, kéo Entropy thay đổi rõ rệt.
    """
    if len(y) == 0:
        return 0.0
        
    p1 = np.sum(y == 1) / len(y)
    p0 = 1.0 - p1
    
    if class_weight is not None:
        # Áp dụng trọng số
        w1 = class_weight
        w0 = 1.0
        total_w = p1 * w1 + p0 * w0
        if total_w == 0: return 0.0
        p1 = (p1 * w1) / total_w
        p0 = (p0 * w0) / total_w
        
    if p1 == 0 or p0 == 0:
        return 0.0
        
    return - (p1 * np.log2(p1) + p0 * np.log2(p0))

def quantile_binning(X, max_bins=256):
    """
    Rời rạc hóa (Discretization) ma trận liên tục thành các bucket nguyên.
    Phá vỡ giới hạn O(N log N) của CART -> đưa độ phức tạp về O(N * max_bins).
    """
    X_binned = np.zeros_like(X, dtype=np.int32)
    thresholds = []
    
    for j in range(X.shape[1]):
        col = X[:, j]
        # Bỏ qua NaN khi tính bins (Cơ chế Native Missing mô phỏng)
        valid_mask = ~np.isnan(col)
        valid_col = col[valid_mask]
        
        if len(valid_col) == 0:
            thresholds.append([])
            continue
            
        # Tìm unique values để tối ưu số bin nếu data đã là discrete
        unique_vals = np.unique(valid_col)
        if len(unique_vals) <= max_bins:
            bins = np.sort(unique_vals)
        else:
            percentiles = np.linspace(0, 100, max_bins)
            bins = np.unique(np.percentile(valid_col, percentiles))
            
        thresholds.append(bins)
        # [FIX 3] Bỏ định kiến NaN đi về nhánh Trái: Map NaN vào Median Bin
        X_binned[valid_mask, j] = np.searchsorted(bins, valid_col)
        median_bin = len(bins) // 2
        X_binned[~valid_mask, j] = median_bin
        
    return X_binned, thresholds

class DecisionTreeFeatureSelector(BaseEstimator, TransformerMixin):
    """
    CART tự code bằng Numpy thuần để phục vụ trích xuất Top-K Features.
    Thu thập Information Gain có đánh trọng số theo "Khối lượng" node (W_t / W_total).
    Tuân thủ chuẩn API Scikit-learn: fit() → transform() → tích hợp được với Pipeline.
    """
    def __init__(self, top_k=50, max_depth=5, min_samples_split=20, max_bins=256, class_weight=27.6):
        self.top_k = top_k
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_bins = max_bins
        self.class_weight = class_weight
        
        self.feature_importances_ = None
        self.selected_features_ = None  # Index của Top-K features
        self.thresholds_ = None
        self.w_total_ = 0.0
        
    def _calc_weight(self, y_subset):
        if len(y_subset) == 0: return 0.0
        w1_weight = self.class_weight if self.class_weight is not None else 1.0
        n1 = np.sum(y_subset == 1)
        n0 = len(y_subset) - n1
        return n1 * w1_weight + n0 * 1.0

    def _information_gain(self, y, y_left, y_right):
        parent_entropy = compute_entropy(y, self.class_weight)
        
        # [FIX 1] Sử dụng Tổng Trọng Số (W) thay vì số lượng mẫu (N)
        w_parent = self._calc_weight(y)
        w_left = self._calc_weight(y_left)
        w_right = self._calc_weight(y_right)
        
        if w_left == 0 or w_right == 0:
            return 0.0
            
        e_l = compute_entropy(y_left, self.class_weight)
        e_r = compute_entropy(y_right, self.class_weight)
        
        child_entropy = (w_left / w_parent) * e_l + (w_right / w_parent) * e_r
        return parent_entropy - child_entropy
        
    def fit(self, X, y):
        # [FIX] Chấp nhận cả Pandas DataFrame lẫn Numpy ndarray.
        # np.asarray() không copy nếu X đã là ndarray — zero overhead.
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.w_total_ = self._calc_weight(y)
        
        # 1. Rời rạc hóa X (Quantile Binning)
        X_binned, self.thresholds_ = quantile_binning(X, max_bins=self.max_bins)
        
        # 2. Xây dựng cây và tích lũy Feature Importance
        self._build_tree(X_binned, y, depth=0)
        
        # 3. Chuẩn hóa tổng importance về 1.0 (như sklearn API)
        sum_importance = np.sum(self.feature_importances_)
        if sum_importance > 0:
            self.feature_importances_ /= sum_importance
        
        # 4. Xác định và lưu Top-K features (sắp xếp giảm dần theo importance)
        top_k = min(self.top_k, self.n_features_)
        self.selected_features_ = np.argsort(self.feature_importances_)[::-1][:top_k]
            
        return self

    def transform(self, X):
        """
        Lọc ma trận X, chỉ giữ lại Top-K cột quan trọng nhất.
        Trả về ma trận X_reduced để nạp vào LightGBM / Custom MLP / GNN.
        """
        if self.selected_features_ is None:
            raise RuntimeError("DecisionTreeFeatureSelector chưa được fit(). Gọi fit(X, y) trước.")
        # [FIX] Ép kiểu để hỗ trợ cả DataFrame đầu vào
        X = np.asarray(X, dtype=np.float64)
        return X[:, self.selected_features_]
        
    def _build_tree(self, X_binned, y, depth):
        n_samples = len(y)
        
        # Điều kiện dừng
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return
            
        best_gain = 0.0
        best_feature = None
        best_split_val = None
        best_left_idx = None
        best_right_idx = None
        
        # Duyệt qua từng đặc trưng
        for j in range(self.n_features_):
            col_binned = X_binned[:, j]

            # [FIX] Dùng np.bincount để lấy ĐÚNG những bin đang tồn tại thực sự trong node.
            # - O(N) thay vì O(N log N) của np.unique
            # - Bỏ qua hoàn toàn các bin rỗng (không có dữ liệu tại node này)
            # - Sửa lỗi Off-by-one: split candidate là mọi bin có dữ liệu, bao gồm cả bin lớn nhất
            max_bin_node = int(np.max(col_binned))
            bin_counts = np.bincount(col_binned, minlength=max_bin_node + 1)
            occupied_bins = np.where(bin_counts > 0)[0]  # Các bin thực sự có dữ liệu

            # Split point hợp lệ là ranh giới giữa 2 bin liên tiếp đang có dữ liệu
            # (không cần thử split tại bin cuối cùng vì nhánh phải sẽ rỗng)
            valid_split_vals = occupied_bins[:-1]

            for split_val in valid_split_vals:
                left_idx = col_binned <= split_val
                right_idx = ~left_idx

                gain = self._information_gain(y, y[left_idx], y[right_idx])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_split_val = split_val
                    best_left_idx = left_idx
                    best_right_idx = right_idx
                    
        # Nếu tìm được điểm cắt tối ưu
        if best_feature is not None and best_gain > 0:
            # Tích lũy Feature Importance CÓ TRỌNG SỐ (Tỷ lệ trọng số rẽ nhánh)
            w_node = self._calc_weight(y)
            weighted_gain = (w_node / self.w_total_) * best_gain
            self.feature_importances_[best_feature] += weighted_gain
            
            # Đệ quy 2 nhánh
            self._build_tree(X_binned[best_left_idx, :], y[best_left_idx], depth + 1)
            self._build_tree(X_binned[best_right_idx, :], y[best_right_idx], depth + 1)
