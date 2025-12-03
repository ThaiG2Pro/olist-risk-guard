"""
=============================================================================
PHẦN 3: PREPROCESSING & DATA PREPARATION
=============================================================================
Các hàm xử lý missing values, encoding, scaling, imbalance, train-test split
"""

"""
Triết lý: "Learn from Train, Apply to Test"
1. Xử lý Missing: Học Median/Mode từ Train -> Fill vào Test.
2. Xử lý Outlier: Học Upper/Lower bound từ Train -> Clip giá trị ở Test.
3. Scaling: Fit Scaler/Log từ Train -> Transform Test.
4. Feature Selection: Drop high correlation cols dựa trên Train.
"""

import joblib
import pandas as pd
import numpy as np
import re


from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
)


# ============================================================================
# 1. MISSING VALUE HANDLING
# ============================================================================

def fit_imputer(df, numeric_cols=None, categorical_cols=None):
    """
    Học giá trị thay thế (Median/Mode) từ tập Train.

    Returns:
    --------
    dict : Dictionary chứa giá trị fill cho từng cột.
    """
    impute_values = {}

    # 1. Numeric: Dùng Median (ít nhạy cảm với outlier hơn Mean)
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                impute_values[col] = df[col].median()

    # 2. Categorical: Dùng Mode (Giá trị xuất hiện nhiều nhất)
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                # Lấy mode đầu tiên, nếu không có (toàn NaN) thì điền "Missing"
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    impute_values[col] = mode_val[0]
                else:
                    impute_values[col] = "Missing"

    return impute_values

def transform_imputer(df, impute_values):
    """
    Điền Missing values dựa trên giá trị đã học.
    """
    df = df.copy()
    for col, val in impute_values.items():
        if col in df.columns:
            if pd.isna(val): # Trường hợp train toàn NaN
                val = 0 if df[col].dtype in [np.float64, np.int64] else "Missing"
            df[col] = df[col].fillna(val)
    return df

# ============================================================================
# 2. OUTLIER HANDLING (CAPPING/WINSORIZATION)
# ============================================================================

def fit_outlier_capper(df, cols, multiplier=1.5):
    """
    Học giới hạn (Upper Bound) từ Train theo IQR.
    Chỉ xử lý cận trên (Upper) vì dữ liệu Ecommerce thường bị lệch phải (giá quá cao, ship quá lâu).
    """
    capper_params = {}
    for col in cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + multiplier * IQR
            lower_bound = Q1 - multiplier * IQR

            capper_params[col] = {
                'upper': upper_bound,
                'lower': lower_bound
            }
    return capper_params

def transform_outlier_capper(df, capper_params):
    """
    Thay thế outlier bằng giá trị biên (Clipping).
    Không drop dòng để đảm bảo dự đoán được cho mọi input.
    """
    df = df.copy()
    for col, bounds in capper_params.items():
        if col in df.columns:
            # Clip values: nhỏ hơn lower -> lower, lớn hơn upper -> upper
            # Tuy nhiên với E-commerce, thường chỉ cần chặn trên (giá tiền, cân nặng)
            # Chặn dưới có thể giữ nguyên nếu không âm
            df[col] = np.clip(df[col], a_min=None, a_max=bounds['upper'])
    return df

# ============================================================================
# 3. TRANSFORMATION (LOG & SCALING)
# ============================================================================

def transform_log_skewed(df, cols):
    """
    Log Transformation cho các biến lệch (Skewed).
    Sử dụng np.log1p (log(x+1)) để xử lý số 0.
    Hàm này stateless (không cần fit).
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            # Đảm bảo không log số âm
            df[col] = np.log1p(np.maximum(df[col], 0))
    return df

def fit_scaler(df, cols, method='standard'):
    """
    Fit Scaler trên Train.

    method: 'standard' (Z-score) hoặc 'minmax' (0-1) hoặc 'robust' (Outlier resistant)
    """
    if not cols:
        return None

    df_sub = df[cols]

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        return None

    scaler.fit(df_sub)
    return scaler

def transform_scaler(df, cols, scaler):
    """
    Transform data sử dụng scaler đã fit.
    """
    if scaler is None or not cols:
        return df

    df = df.copy()
    # Chỉ transform các cột có trong list và trong scaler
    # Cần đảm bảo thứ tự cột khớp với lúc fit
    try:
        df[cols] = scaler.transform(df[cols])
    except Exception as e:
        print(f"⚠️ Scaling Warning: {str(e)}. Skipping scaling.")

    return df

# ============================================================================
# 4. FEATURE SELECTION (MULTICOLLINEARITY)
# ============================================================================
# thực hiện sau log-transform, vì log có thể giảm correlation.
def fit_collinearity_remover(df, numeric_cols, threshold=0.95):
    """
    Tìm các cặp features tương quan cao (> threshold) trên tập Train.
    Trả về danh sách các cột cần loại bỏ.
    """
    df_sub = df[numeric_cols].select_dtypes(include=[np.number])
    if df_sub.empty:
        return []

    corr_matrix = df_sub.corr().abs()

    # Chỉ lấy tam giác trên của ma trận
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Tìm các cột có correlation > threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if len(to_drop) > 0:
        print(f"  -> Detected multicollinearity. Dropping: {to_drop}")

    return to_drop

# ============================================================================
# 5. CATEGORICAL ENCODING
# ============================================================================
def fit_label_encoder(df, cols, top_n=20):
    """
    Fit LabelEncoder với chiến lược Top-N.
    - Giữ lại top_n giá trị phổ biến nhất.
    - Các giá trị còn lại gộp thành 'Other'.
    """
    encoders = {}

    for col in cols:
        if col in df.columns:
            # 1. Đếm tần suất
            counts = df[col].value_counts()

            # 2. Lấy danh sách Top N (Học từ Train)
            # Nếu số lượng unique < top_n thì lấy hết, ngược lại cắt top_n
            if len(counts) > top_n:
                valid_labels = counts.head(top_n).index.tolist()
                print(f"  -> Grouping '{col}': Reduced {len(counts)} categories to {top_n} + 'Other'")
            else:
                valid_labels = counts.index.tolist()

            # 3. Tạo Encoder
            encoders[col] = {
                'valid_labels': set(valid_labels),
                'encoder': LabelEncoder()
            }

            # 4. Fit trên dữ liệu giả lập đã gộp
            # Tạo series tạm để fit, gán 'Other' cho các label ngoài top N
            temp_series = df[col].apply(lambda x: x if x in valid_labels else 'Other').astype(str)

            # Đảm bảo 'Other' luôn được fit để transform không lỗi
            if 'Other' not in list(temp_series.unique()):
                # fit thêm chữ Other vào để encoder biết mặt nó
                fit_data = list(temp_series.unique()) + ['Other']
                encoders[col]['encoder'].fit(fit_data)
            else:
                encoders[col]['encoder'].fit(temp_series)

    return encoders

def transform_label_encoder(df, cols, encoders_dict):
    """
    Transform Categorical sang số.
    Xử lý Unseen labels bằng cách gán về 'Other' hoặc mode.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns and col in encoders_dict:
            config = encoders_dict[col]
            valid_labels = config['valid_labels']
            le = config['encoder']

            # Map giá trị lạ về 'Other'
            df[col] = df[col].apply(lambda x: x if x in valid_labels else 'Other').astype(str)

            # Transform
            df[col] = le.transform(df[col])

    return df
# ============================================================================
# 5.5. POST-PROCESSING SAFETY NET (Xử lý các cột bị bỏ quên)
# ============================================================================

def fill_forgotten_nans(df, df_name="Train"):
    """
    Tìm và điền giá trị mặc định cho các cột còn sót NaN
    (do chưa được định nghĩa trong feature groups).
    """
    df = df.copy()
    # Tìm cột còn NaN
    cols_with_nan = df.columns[df.isna().any()].tolist()

    if len(cols_with_nan) > 0:
        print(f"\n⚠️  Found {len(cols_with_nan)} columns with NaNs in {df_name} (Missed by Imputer):")
        print(f"   List: {cols_with_nan}")

        # Xử lý: Fill 0 cho số, 'Unknown' cho chữ
        for col in cols_with_nan:
            # Kiểm tra kiểu dữ liệu
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                fill_val = 0
            else:
                fill_val = "Unknown"

            # Fill
            df[col] = df[col].fillna(fill_val)
            print(f"   -> Filled '{col}' with {fill_val}")
    return df


# ============================================================================
# 6. MASTER PIPELINE
# ============================================================================

def train_preparation_pipeline(X_train,
                             skewed_cols=[],
                             normal_cols=[],
                             categorical_cols=[],
                             top_n_categories=20):
    """
    Chạy toàn bộ quy trình Fit trên tập Train.
    Trả về: X_train đã xử lý và dictionary chứa tất cả các transformers (artifacts).
    """
    X_train = X_train.copy()
    artifacts = {}

    # 1. Identify Numeric Cols (Skewed + Normal)
    numeric_cols = skewed_cols + normal_cols

    # 2. Imputation (Học Median/Mode)
    print("  [1/5] Fitting Imputer...")
    imputer_params = fit_imputer(X_train, numeric_cols, categorical_cols)
    X_train = transform_imputer(X_train, imputer_params)
    artifacts['imputer'] = imputer_params

    # 3. Outlier Capping (Chỉ cho Skewed & Normal numeric)
    print("  [2/5] Fitting Outlier Capper...")
    capper_params = fit_outlier_capper(X_train, numeric_cols, multiplier=3.0) # 3.0 cho safety
    X_train = transform_outlier_capper(X_train, capper_params)
    artifacts['capper'] = capper_params

    # 4. Log Transform (Chỉ Skewed)
    print("  [3/5] Applying Log Transform...")
    X_train = transform_log_skewed(X_train, skewed_cols)
    # Log transform không có params để fit

    # 5. Collinearity Removal
    print("  [4/5] Checking Multicollinearity...")
    # Lưu ý: Check collinearity sau khi đã log transform
    drop_cols = fit_collinearity_remover(X_train, numeric_cols)
    X_train = X_train.drop(columns=drop_cols)
    artifacts['drop_cols'] = drop_cols

    # Cập nhật lại list cols sau khi drop
    final_normal_cols = [c for c in normal_cols if c not in drop_cols]
    # Skewed cols đã log rồi thì coi như normal cols để scale (tùy chọn, thường log xong thì phân phối đã đẹp)
    final_skewed_cols = [c for c in skewed_cols if c not in drop_cols]
    scale_cols = final_normal_cols + final_skewed_cols

    # 6. Scaling (StandardScaler)
    print("  [5/5] Fitting Scaler...")
    scaler = fit_scaler(X_train, scale_cols, method='standard')
    X_train = transform_scaler(X_train, scale_cols, scaler)
    artifacts['scaler'] = scaler
    artifacts['scale_cols'] = scale_cols # Lưu lại thứ tự cột

    # 7. Categorical Encoding
    print("  [6/6] Encoding Categoricals...")
    encoders = fit_label_encoder(X_train, categorical_cols, top_n=top_n_categories)
    X_train = transform_label_encoder(X_train, categorical_cols, encoders)
    artifacts['encoders'] = encoders

    #  Convert Boolean to Int (True/False -> 1/0)
    for col in X_train.select_dtypes(include='bool').columns:
        X_train[col] = X_train[col].astype(int)

    # 8. Final Safety Net
    X_train = fill_forgotten_nans(X_train, "Train")
    X_train = clean_column_names(X_train)


    return X_train, artifacts

def test_preparation_pipeline(X_test, artifacts, skewed_cols=[], categorical_cols=[]):
    """
    Áp dụng artifacts đã học từ Train lên Test.
    Đảm bảo KHÔNG CÓ DATA LEAKAGE.
    """
    X_test = X_test.copy()

    # 1. Imputation
    if 'imputer' in artifacts:
        X_test = transform_imputer(X_test, artifacts['imputer'])

    # 2. Outlier Capping
    if 'capper' in artifacts:
        X_test = transform_outlier_capper(X_test, artifacts['capper'])

    # 3. Log Transform
    X_test = transform_log_skewed(X_test, skewed_cols)

    # 4. Drop Collinear Columns
    if 'drop_cols' in artifacts:
        X_test = X_test.drop(columns=artifacts['drop_cols'], errors='ignore')

    # 5. Scaling
    if 'scaler' in artifacts:
        X_test = transform_scaler(X_test, artifacts['scale_cols'], artifacts['scaler'])

    # 6. Encoding
    if 'encoders' in artifacts:
        X_test = transform_label_encoder(X_test, categorical_cols, artifacts['encoders'])

    #  Convert Boolean to Int (True/False -> 1/0)
    for col in X_test.select_dtypes(include='bool').columns:
        X_test[col] = X_test[col].astype(int)
    # 7. Final Safety Net
    X_test = fill_forgotten_nans(X_test, "Test")
    X_test = clean_column_names(X_test)

    return X_test

def clean_column_names(df):
    """Làm sạch tên cột để tránh lỗi JSON của LightGBM"""
    new_cols = []
    for col in df.columns:
        # Thay thế ký tự đặc biệt bằng underscore
        new_col = re.sub(r'[^\w]', '_', col)
        new_cols.append(new_col)
    df.columns = new_cols
    return df
