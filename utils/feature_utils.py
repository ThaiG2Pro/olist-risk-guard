"""
=============================================================================
 FEATURE ENGINEERING FUNCTIONS
=============================================================================
Các hàm tạo features cho logistics, products, reviews, geography
"""
"""
Triết lý:
1. Stateless Transformations: Tính toán trên từng dòng độc lập (Safe).
2. Stateful Transformations: Học từ dữ liệu quá khứ (Train) -> Áp dụng cho Test/Future.
   -> Đảm bảo không Data Leakage.
   -> Chỉ sử dụng thông tin có sẵn tại thời điểm 'order_purchase_timestamp'.
"""
from datetime import datetime
from scipy.spatial.distance import cdist
import re
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# GROUP 1: STATELESS FEATURES (An toàn, chạy độc lập trên từng DF)
# ============================================================================

def process_datetime_features(df, date_col='order_purchase_timestamp'):
    """
    Tách các đặc trưng thời gian từ timestamp mua hàng.
    """
    df = df.copy()
    if date_col not in df.columns:
        return df

    df[date_col] = pd.to_datetime(df[date_col])

    # Cyclic features (Tốt hơn one-hot cho tháng/giờ)
    df['purchase_month'] = df[date_col].dt.month
    df['purchase_day'] = df[date_col].dt.day
    df['purchase_hour'] = df[date_col].dt.hour
    df['purchase_dayofweek'] = df[date_col].dt.dayofweek

    # Business logic
    df['is_weekend'] = (df['purchase_dayofweek'] >= 5).astype(int)
    df['is_month_end'] = (df[date_col].dt.is_month_end).astype(int)

    return df

def calculate_distance_haversine(lat1, lon1, lat2, lon2):
    """Hàm phụ trợ tính khoảng cách"""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Earth radius km
    return c * r

def process_geo_features(df, geo_df):
    """
    Tính khoảng cách giữa Customer và Seller.
    Input: df (đơn hàng), geo_df (bảng geolocation gốc)
    """
    df = df.copy()

    # 1. Aggregate Geo Data (Lấy trung bình tọa độ cho mỗi Zip)
    if 'geolocation_zip_code_prefix' in geo_df.columns:
        geo_agg = geo_df.groupby('geolocation_zip_code_prefix').agg({
            'geolocation_lat': 'mean',
            'geolocation_lng': 'mean'
        }).reset_index()
    else:
        return df

    # 2. Merge Customer Coords
    df = df.merge(geo_agg, left_on='customer_zip_code_prefix',
                 right_on='geolocation_zip_code_prefix', how='left')
    df.rename(columns={'geolocation_lat': 'cust_lat', 'geolocation_lng': 'cust_lng'}, inplace=True)
    df.drop(columns=['geolocation_zip_code_prefix'], inplace=True, errors='ignore')

    # 3. Merge Seller Coords
    df = df.merge(geo_agg, left_on='seller_zip_code_prefix',
                 right_on='geolocation_zip_code_prefix', how='left')
    df.rename(columns={'geolocation_lat': 'sell_lat', 'geolocation_lng': 'sell_lng'}, inplace=True)
    df.drop(columns=['geolocation_zip_code_prefix'], inplace=True, errors='ignore')

    # 4. Calculate Distance
    # Fill NA tọa độ bằng 0 hoặc mean (để tránh lỗi code, logic xử lý NA nên ở prep pipeline)
    cols = ['cust_lat', 'cust_lng', 'sell_lat', 'sell_lng']
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mean())

    df['distance_km'] = calculate_distance_haversine(
        df['cust_lat'], df['cust_lng'], df['sell_lat'], df['sell_lng']
    )

    # Clean up coords if not needed
    df.drop(columns=cols, inplace=True)

    return df

def process_product_pricing(df):
    """
    Tạo features về giá và sản phẩm (Không dùng thông tin tương lai)
    """
    df = df.copy()

    # 1. Freight Ratio (Phí ship chiếm bao nhiêu % tổng đơn)
    if 'total_freight' in df.columns and 'total_price' in df.columns:
        df['freight_ratio'] = df['total_freight'] / (df['total_price'] + 0.01)
        df['is_high_freight'] = (df['freight_ratio'] > 0.3).astype(int)

    # 2. Product Features
    if 'product_description_lenght' in df.columns:
        df['desc_len_log'] = np.log1p(df['product_description_lenght'])

    if 'product_photos_qty' in df.columns:
        df['has_enough_photos'] = (df['product_photos_qty'] >= 3).astype(int)

    # 3. Size/Weight (Nếu có)
    if 'product_weight_g' in df.columns:
        df['weight_log'] = np.log1p(df['product_weight_g'])

    return df

# ============================================================================
# GROUP 2: HISTORICAL / STATEFUL FEATURES (Core Anti-Leakage Logic)
# ============================================================================

def fit_seller_history(train_df, target_col='target', date_col='order_purchase_timestamp'):
    """
    Học chỉ số risk của seller từ tập train bằng smoothing (Bayesian).
    Trả về DataFrame: seller_id, seller_risk_score, seller_orders
    """
    # Chỉ lấy cột cần thiết
    data = train_df[['seller_id', target_col]].copy()

    # Prior toàn cục
    global_mean = data[target_col].mean()
    C = 10  # trọng số giả định

    # Aggregation
    seller_stats = data.groupby('seller_id')[target_col].agg(['count', 'mean', 'sum'])
    seller_stats.columns = ['seller_orders', 'raw_defect_rate', 'defects']

    # Bayesian smoothing
    seller_stats['seller_risk_score'] = (
        (seller_stats['defects'] + C * global_mean) /
        (seller_stats['seller_orders'] + C)
    )

    # Debug info
    print(f"[fit_seller_history] rows={len(data)}, sellers={seller_stats.shape[0]}, global_mean={global_mean:.4f}, C={C}")

    return seller_stats.reset_index()[['seller_id', 'seller_risk_score', 'seller_orders']]


def transform_seller_history(df, seller_stats_df, global_risk_mean=0.15):
    """
    Áp dụng seller risk đã học lên df mới (merge + fillna).
    """
    df = df.copy()

    # Merge left
    before = len(df)
    df = df.merge(seller_stats_df, on='seller_id', how='left')

    # Debug: số seller không khớp
    n_missing = df['seller_risk_score'].isna().sum()
    print(f"[transform_seller_history] merged={before} rows, missing_seller_risk={n_missing}")

    # Fill NA cho seller mới
    df['seller_risk_score'] = df['seller_risk_score'].fillna(global_risk_mean)
    df['seller_orders'] = df['seller_orders'].fillna(0)

    # Feature đơn giản
    df['is_new_seller'] = (df['seller_orders'] < 5).astype(int)

    return df


def fit_route_stats(train_df, date_col='order_purchase_timestamp'):
    """
    Tính thống kê thời gian giao hàng theo cặp (seller_state, customer_state)
    Dùng đơn delivered (có order_delivered_customer_date).
    Trả về: seller_state, customer_state, route_median_days, route_std_days, route_volume
    """
    required_cols = ['order_delivered_customer_date', date_col, 'seller_state', 'customer_state']

    # Lọc data hợp lệ
    valid_data = train_df.dropna(subset=required_cols).copy()
    print(f"[fit_route_stats] valid_rows={len(valid_data)}")

    # Tính actual days
    valid_data['actual_days'] = (
        pd.to_datetime(valid_data['order_delivered_customer_date']) -
        pd.to_datetime(valid_data[date_col])
    ).dt.total_seconds() / 86400

    # Aggregate theo route
    route_stats = valid_data.groupby(['seller_state', 'customer_state'])['actual_days'].agg(
        ['median', 'std', 'count']
    ).reset_index()

    route_stats.columns = ['seller_state', 'customer_state', 'route_median_days', 'route_std_days', 'route_volume']
    print(f"[fit_route_stats] routes_computed={len(route_stats)}")

    return route_stats


def transform_route_stats(df, route_stats_df, global_median_days=7):
    """
    Áp dụng route stats vào df: merge và tính gap giữa estimate và lịch sử.
    """
    df = df.copy()

    # Merge
    before = len(df)
    df = df.merge(route_stats_df, on=['seller_state', 'customer_state'], how='left')
    merged_missing = df['route_median_days'].isna().sum()
    print(f"[transform_route_stats] merged={before} rows, missing_route_stats={merged_missing}")

    # Fill cho route mới
    df['route_median_days'] = df['route_median_days'].fillna(global_median_days)
    df['route_volume'] = df['route_volume'].fillna(0)

    # So sánh estimate vs history nếu có ngày estimate
    if 'order_estimated_delivery_date' in df.columns and 'order_purchase_timestamp' in df.columns:
        df['days_estimated'] = (
            pd.to_datetime(df['order_estimated_delivery_date']) -
            pd.to_datetime(df['order_purchase_timestamp'])
        ).dt.total_seconds() / 86400

        df['gap_promise_vs_reality'] = df['days_estimated'] - df['route_median_days']
        df['is_optimistic_promise'] = (df['gap_promise_vs_reality'] < -2).astype(int)

    return df

# ============================================================================
# GROUP 3: TARGET CREATION (Chỉ dùng 1 lần đầu tiên)
# ============================================================================

def create_target(df, threshold=3):
    """
    Tạo target binary.
    1 = Negative Review (Score <= threshold) -> RISK
    0 = Positive Review
    """
    df = df.copy()
    if 'review_score' in df.columns:
        df['target'] = (df['review_score'] <= threshold).astype(int)
    return df

# ============================================================================
# GROUP 4: DYNAMIC / POINT-IN-TIME FEATURES (Advanced Anti-Leakage)
# ============================================================================

def process_dynamic_seller_features(orders_df, reviews_df, window_days=90):
    """
    Tính rate khiếu nại của seller trong window trước thời điểm mua (point-in-time).
    """
    print(f"Generating Dynamic Seller Features (Window: {window_days} days)...")

    # Prepare reviews with seller mapping
    rev_df = reviews_df[['order_id', 'review_score', 'review_creation_date']].copy()
    rev_df['review_creation_date'] = pd.to_datetime(rev_df['review_creation_date'])
    mapping = orders_df[['order_id', 'seller_id']].drop_duplicates()
    rev_df = rev_df.merge(mapping, on='order_id', how='inner')
    rev_df['is_bad_review'] = (rev_df['review_score'] <= 3).astype(int)

    # Daily aggregates per seller
    rev_df = rev_df.sort_values('review_creation_date')
    daily_stats = (
        rev_df.set_index('review_creation_date')
        .groupby('seller_id')
        .resample('D')['is_bad_review']
        .agg(['sum', 'count'])
        .reset_index()
    )
    daily_stats.rename(columns={'sum': 'daily_bad', 'count': 'daily_total'}, inplace=True)
    daily_stats = daily_stats.sort_values(['seller_id', 'review_creation_date'])

    # Rolling sums over window_days
    indexer = daily_stats.groupby('seller_id').rolling(f'{window_days}D', on='review_creation_date')
    daily_stats['rolling_bad'] = indexer['daily_bad'].sum().values
    daily_stats['rolling_total'] = indexer['daily_total'].sum().values

    # Smoothed recent defect rate
    daily_stats['seller_recent_defect_rate'] = (daily_stats['rolling_bad'] + 1) / (daily_stats['rolling_total'] + 5)
    seller_history = daily_stats[['seller_id', 'review_creation_date', 'seller_recent_defect_rate']].sort_values('review_creation_date')

    # Point-in-time merge into orders
    orders_df = orders_df.copy()
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    orders_df = orders_df.sort_values('order_purchase_timestamp')

    merged_df = pd.merge_asof(
        orders_df,
        seller_history,
        left_on='order_purchase_timestamp',
        right_on='review_creation_date',
        by='seller_id',
        direction='backward',
        tolerance=pd.Timedelta(days=window_days)
    )

    global_mean = daily_stats['seller_recent_defect_rate'].mean()
    merged_df['seller_recent_defect_rate'] = merged_df['seller_recent_defect_rate'].fillna(global_mean)

    # Trend vs lifetime risk if available
    if 'seller_risk_score' in merged_df.columns:
        merged_df['seller_risk_trend'] = merged_df['seller_recent_defect_rate'] - merged_df['seller_risk_score']

    return merged_df

# ============================================================================
# GROUP 5: HYBRID MODELING FEATURES (PHASE 3)
# ============================================================================

def prepare_hybrid_data(X_train, X_test, kmeans_model, features_list):
    """
    Tích hợp thông tin Cluster vào tập dữ liệu và đảm bảo đồng bộ cột (Alignment).

    Parameters:
    -----------
    X_train, X_test : DataFrame
        Dữ liệu đã qua xử lý (Scaled)
    kmeans_model : KMeans
        Mô hình clustering đã train
    features_list : list
        Danh sách các feature dùng để cluster / clustering_features

    Returns:
    --------
    X_train_final, X_test_final : DataFrame
        Dữ liệu đã có thêm cột Cluster (One-hot)
    """
    print(f"Integrating Cluster Features (Input Train: {X_train.shape})...")

    def _add_cluster(df_in):
        df = df_in.copy()
        # Predict cluster
        X_cluster_input = df[features_list]
        clusters = kmeans_model.predict(X_cluster_input)

        # One-hot encoding
        df['Cluster'] = clusters
        df = pd.get_dummies(df, columns=['Cluster'], prefix='Cluster')
        return df

    # 1. Transform cả 2 tập
    X_train_final = _add_cluster(X_train)
    X_test_final = _add_cluster(X_test)

    # 2. Alignment (Đồng bộ cột)
    missing_cols = set(X_train_final.columns) - set(X_test_final.columns)
    for c in missing_cols:
        X_test_final[c] = 0

    # Loại bỏ các cột thừa ở Test (nếu có - hiếm gặp) và sắp xếp lại theo thứ tự Train
    X_test_final = X_test_final[X_train_final.columns]

    print(f"✓ Hybrid Data Ready - Train: {X_train_final.shape}, Test: {X_test_final.shape}")
    return X_train_final, X_test_final
