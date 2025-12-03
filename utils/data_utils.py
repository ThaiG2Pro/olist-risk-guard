"""
=============================================================================
DATA LOADING & EXPLORATION UTILITIES
=============================================================================
Các hàm cơ bản để load data, merge tables, và khám phá dữ liệu nhanh
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1.1 DATA LOADING FUNCTIONS
# ============================================================================

def load_olist_tables(data_path, tables_to_load=None, verbose=True):
    """
    Load các bảng Olist dataset một cách linh hoạt

    Parameters:
    -----------
    data_path : str or Path
        Đường dẫn đến folder chứa data
    tables_to_load : list or None
        Danh sách các bảng cần load. Nếu None, load tất cả
    verbose : bool
        In thông tin về các bảng được load

    Returns:
    --------
    dict : Dictionary chứa các DataFrame với key là tên bảng
    """
    data_path = Path(data_path)

    # Danh sách các bảng standard của Olist
    available_tables = {
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'order_payments': 'olist_order_payments_dataset.csv',
        'order_reviews': 'olist_order_reviews_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'geolocation': 'olist_geolocation_dataset.csv',
        'product_category': 'product_category_name_translation.csv'
    }

    # Nếu không chỉ định, load tất cả
    if tables_to_load is None:
        tables_to_load = list(available_tables.keys())

    tables = {}
    for table_name in tables_to_load:
        if table_name in available_tables:
            file_path = data_path / available_tables[table_name]
            if file_path.exists():
                tables[table_name] = pd.read_csv(file_path)
                if verbose:
                    print(f"✓ Loaded {table_name}: {tables[table_name].shape}")
            else:
                if verbose:
                    print(f"✗ File not found: {file_path}")
        else:
            if verbose:
                print(f"✗ Unknown table: {table_name}")

    return tables


def quick_summary(df, name="DataFrame"):
    """
    In summary nhanh về DataFrame
    """
    print(f"\n{'='*60}")
    print(f"SUMMARY: {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Missing': missing[missing > 0],
        'Percent': missing_pct[missing > 0]
    }).sort_values('Missing', ascending=False)
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values!")
    print(f"\nData types:")
    print(df.dtypes.value_counts())


def batch_quick_summary(tables_dict):
    """
    In summary cho nhiều bảng cùng lúc
    """
    for name, df in tables_dict.items():
        quick_summary(df, name)


# ============================================================================
# 1.2 DATA MERGING FUNCTIONS
# ============================================================================

def create_base_dataset(tables, merge_strategy='left'):
    """
    Merge các bảng cơ bản để tạo dataset chính

    Parameters:
    -----------
    tables : dict
        Dictionary chứa các DataFrame
    merge_strategy : str
        Chiến lược merge: 'left', 'inner', 'outer'

    Returns:
    --------
    DataFrame : Base dataset đã merge
    """
    print("Creating base dataset...")

    # Bắt đầu từ orders
    df = tables['orders'].copy()
    print(f"Starting with orders: {df.shape}")

    # Merge với order_reviews
    if 'order_reviews' in tables:
        df = df.merge(
            tables['order_reviews'][['order_id', 'review_score', 'review_comment_message', 'review_creation_date']],
            on='order_id',
            how=merge_strategy
        )
        print(f"After merge reviews: {df.shape}")

    # Merge với order_items (aggregate trước)
    if 'order_items' in tables:
        order_items_agg = tables['order_items'].groupby('order_id').agg({
            'order_item_id': 'count',
            'product_id': 'first',  # Lấy product đầu tiên
            'seller_id': 'first',   # Lấy seller đầu tiên
            'price': ['sum'],
            'freight_value': ['sum']
        }).reset_index()

        # Flatten column names
        order_items_agg.columns = ['order_id', 'num_items', 'product_id', 'seller_id',
                                    'total_price',
                                    'total_freight']

        df = df.merge(order_items_agg, on='order_id', how=merge_strategy)
        print(f"After merge order_items: {df.shape}")

    # Merge với customers
    if 'customers' in tables:
        df = df.merge(
            tables['customers'][['customer_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state']],
            on='customer_id',
            how=merge_strategy
        )
        print(f"After merge customers: {df.shape}")

    # Merge với sellers
    if 'sellers' in tables and 'seller_id' in df.columns:
        sellers = tables['sellers'].copy()
        sellers.columns = ['seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state']
        df = df.merge(sellers, on='seller_id', how=merge_strategy)
        print(f"After merge sellers: {df.shape}")

    # Merge với products
    if 'products' in tables and 'product_id' in df.columns:
        products = tables['products'][['product_id', 'product_category_name',
                                       'product_weight_g', 'product_length_cm',
                                       'product_height_cm', 'product_width_cm',
                                       'product_photos_qty', 'product_description_lenght']].copy()
        df = df.merge(products, on='product_id', how=merge_strategy)
        print(f"After merge products: {df.shape}")
        df['product_photos_qty'] = df['product_photos_qty'].fillna(0)
        df['product_description_lenght'] = df['product_description_lenght'].fillna(0)

    # Merge với product category translation
    if 'product_category' in tables and 'product_category_name' in df.columns:
        df = df.merge(
            tables['product_category'],
            on='product_category_name',
            how='left'
        )
        print(f"After merge category translation: {df.shape}")

    print(f"\n✓ Base dataset created: {df.shape}")
    return df


def merge_custom_tables(df_base, tables_list, merge_keys_list, how='left'):
    """
    Merge linh hoạt các bảng tùy chỉnh

    Parameters:
    -----------
    df_base : DataFrame
        DataFrame cơ sở
    tables_list : list of DataFrames
        Danh sách các bảng cần merge
    merge_keys_list : list of str or list
        Danh sách các key để merge tương ứng
    how : str or list
        Phương pháp merge ('left', 'inner', 'outer')

    Returns:
    --------
    DataFrame : DataFrame đã merge
    """
    df = df_base.copy()

    # Nếu how là string, convert thành list
    if isinstance(how, str):
        how = [how] * len(tables_list)

    for i, (table, keys, merge_how) in enumerate(zip(tables_list, merge_keys_list, how)):
        print(f"Merging table {i+1}/{len(tables_list)}: {table.shape}")
        df = df.merge(table, on=keys, how=merge_how)
        print(f"  Result shape: {df.shape}")

    return df


# ============================================================================
# 1.3 DATA FILTERING & SAMPLING FUNCTIONS
# ============================================================================

def filter_by_conditions(df, conditions_dict, verbose=True):
    """
    Filter DataFrame theo nhiều điều kiện

    Parameters:
    -----------
    df : DataFrame
    conditions_dict : dict
        Dictionary với key là tên cột, value là điều kiện
        VD: {'order_status': ['delivered'],
             'review_score': [1,2,3,4,5]}
    verbose : bool

    Returns:
    --------
    DataFrame : DataFrame đã filter
    """
    df_filtered = df.copy()
    initial_shape = df_filtered.shape

    for col, values in conditions_dict.items():
        if col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[col].isin(values)]
            if verbose:
                print(f"Filter by {col} in {values}: {df_filtered.shape}")

    if verbose:
        removed = initial_shape[0] - df_filtered.shape[0]
        pct = 100 * removed / initial_shape[0]
        print(f"\n✓ Filtered: removed {removed} rows ({pct:.2f}%)")

    return df_filtered


def temporal_split(df, date_col, split_date=None, test_size=0.2):
    """
    Chia train/test theo thời gian

    Parameters:
    -----------
    df : DataFrame
    date_col : str
        Tên cột chứa timestamp
    split_date : str or None
        Ngày chia train/test. Nếu None, tự động tính theo test_size
    test_size : float
        Tỷ lệ test set (0-1)

    Returns:
    --------
    tuple : (train_df, test_df, split_date_used)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    if split_date is None:
        split_idx = int(len(df) * (1 - test_size))
        split_date = df.iloc[split_idx][date_col]
    else:
        split_date = pd.to_datetime(split_date)

    train = df[df[date_col] < split_date].copy()
    test = df[df[date_col] >= split_date].copy()

    print(f"Temporal split at: {split_date}")
    print(f"Train: {train.shape} ({train[date_col].min()} to {train[date_col].max()})")
    print(f"Test:  {test.shape} ({test[date_col].min()} to {test[date_col].max()})")

    return train, test, split_date

