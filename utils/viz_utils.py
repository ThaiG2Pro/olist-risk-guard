import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from wordcloud import WordCloud, STOPWORDS

# --- 1. GLOBAL CONFIGURATION ---
COLORS = {
    'primary': '#2E86C1',    # Xanh dương (Safe/Positive)
    'danger': '#E74C3C',     # Đỏ (Risk/Negative)
    'warning': '#F1C40F',    # Vàng (Warning)
    'neutral': '#95A5A6',    # Xám
    'text': '#2C3E50',       # Đen xanh
    'palette_div': 'RdYlBu', # Diverging palette
    'palette_seq': 'Blues'   # Sequential palette
}

def set_style():
    """Thiết lập style chuẩn cho toàn bộ dự án"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.titlesize'] = 15
    plt.rcParams['text.color'] = COLORS['text']
    plt.rcParams['axes.labelcolor'] = COLORS['text']
    plt.rcParams['xtick.color'] = COLORS['text']
    plt.rcParams['ytick.color'] = COLORS['text']

# --- 2. DESCRIPTIVE PLOTS (PHASE 0 - EDA) ---

def plot_numeric_distribution(df, col, title=None, figsize=(10, 5)):
    """Vẽ phân phối của biến số (Histogram + Boxplot)"""
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    
    sns.boxplot(x=df[col], ax=ax_box, color=COLORS['primary'])
    sns.histplot(df[col], ax=ax_hist, kde=True, color=COLORS['primary'], alpha=0.6)
    
    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    
    if title: ax_box.set_title(title)
    plt.xlabel(col)
    return fig

def plot_categorical_count(df, col, top_n=10, title=None, figsize=(10, 6), horizontal=True):
    """Vẽ biểu đồ thanh cho biến phân loại (Top N)
    """
    data = df[col].value_counts().head(top_n).reset_index()
    data.columns = [col, 'count']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if horizontal:
        sns.barplot(data=data, y=col, x='count', palette='viridis', ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
    else:
        sns.barplot(data=data, x=col, y='count', palette='viridis', ax=ax)
        ax.set_ylabel("Count")
        ax.set_xlabel(col)
    
    ax.set_title(title if title else f"Top {top_n} {col}")
    return fig


def plot_orders_trend(df, date_col='order_purchase_timestamp', rule='M', title="Order Trend"):
    """Vẽ xu hướng số lượng đơn hàng theo thời gian"""
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])
    trend = temp.set_index(date_col).resample(rule).size()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trend.index, trend.values, marker='o', linestyle='-', color=COLORS['primary'], linewidth=2)
    ax.fill_between(trend.index, trend.values, color=COLORS['primary'], alpha=0.1)
    
    ax.set_title(title)
    ax.set_ylabel("Number of Orders")
    return fig

def plot_correlation_heatmap(df, cols, figsize=(10, 8)):
    """Vẽ ma trận tương quan"""
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=COLORS['palette_div'], 
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    ax.set_title("Correlation Matrix")
    return fig

def plot_geolocation_map(geo_df, lat_col='geolocation_lat', lng_col='geolocation_lng'):
    """Vẽ bản đồ phân bố (Scatter plot mô phỏng bản đồ)"""
    fig, ax = plt.subplots(figsize=(8, 8))
    # Lấy mẫu để vẽ nhanh hơn nếu dữ liệu lớn
    data = geo_df.sample(min(50000, len(geo_df)), random_state=42)
    
    sns.scatterplot(data=data, x=lng_col, y=lat_col, alpha=0.3, s=5, color=COLORS['primary'], ax=ax)
    ax.set_title("Customer/Order Geolocation Distribution")
    ax.axis('off')
    return fig

def generate_wordcloud(text_series, title="Word Cloud", max_words=100, figsize=(10, 6)):
    """
    Tạo và vẽ WordCloud từ series text.
    Tự động dịch các từ khóa phổ biến từ Bồ Đào Nha sang Anh để dễ hiểu.
    """
    # 1. Từ điển dịch thuật (Portuguese -> English) cho ngữ cảnh E-commerce
    pt_to_en = {
        # Negative Keywords
        'nao': 'not', 'não': 'not',
        'entrega': 'delivery',
        'atraso': 'delay', 'atrasado': 'delayed',
        'produto': 'product',
        'chegou': 'arrived', 'recebi': 'received', # Thường đi với "not" (not arrived)
        'pessimo': 'terrible', 'ruim': 'bad',
        'defeito': 'defect', 'quebrado': 'broken',
        'dinheiro': 'money', 'reembolso': 'refund',
        'loja': 'store', 'vendedor': 'seller',
        'prazo': 'deadline', 'dia': 'day', 'dias': 'days',
        'veio': 'came', 'compra': 'purchase',
        'ainda': 'yet', 'nada': 'nothing',
        'aguardando': 'waiting',
        
        # Positive Keywords
        'bom': 'good', 'boa': 'good',
        'otimo': 'great', 'excelente': 'excellent',
        'recomendo': 'recommend',
        'rapido': 'fast', 'rápido': 'fast', 'rapida': 'fast',
        'antes': 'before', # Arrived before deadline
        'bem': 'well',
        'tudo': 'everything', 'certo': 'right',
        'lindo': 'beautiful', 'perfeito': 'perfect',
        'gostei': 'liked', 'adorei': 'loved',
        'parabens': 'congrats',
        'super': 'super', 'muito': 'very'
    }
    
    # 2. Gộp text và xử lý
    # Chuyển về chữ thường và bỏ dòng trống
    text_data = text_series.dropna().astype(str).str.lower()
    
    # Gộp thành 1 chuỗi lớn
    full_text = " ".join(text_data)
    
    # 3. Thực hiện thay thế từ (Translation)
    # Cách này nhanh hơn dịch từng dòng
    for pt, en in pt_to_en.items():
        # Thêm khoảng trắng để tránh thay thế một phần của từ khác
        full_text = full_text.replace(f" {pt} ", f" {en} ")
    
    # 4. Stopwords (Loại bỏ các từ vô nghĩa tiếng Bồ Đào Nha còn lại)
    # Các từ nối như: o, a, e, de, da, do, em, que...
    stopwords = set(STOPWORDS)
    stopwords.update([
        "o", "a", "e", "de", "da", "do", "em", "que", "um", "uma", "para", "com", "na", "no", 
        "se", "foi", "mas", "minha", "meu", "por", "as", "os", "ja", "isso", "estou", "era"
    ])
    
    # 5. Vẽ WordCloud
    wc = WordCloud(
        background_color="white", 
        max_words=max_words, 
        width=800, height=400,
        stopwords=stopwords,
        colormap='viridis',
        collocations=False # Tắt tính năng ghép từ để tránh lặp từ
    ).generate(full_text)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis("off")
    
    return fig

# --- 3. MODELING PLOTS (PHASE 1, 2, 3 - REUSED) ---

def plot_pca_scatter(X_pca, figsize=(10, 6)):
    """Vẽ biểu đồ phân tán PCA (Dùng cho Phase 3)"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=10, c='gray')
    ax.set_title('PCA Visualization of Orders (2 Components)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)
    return fig

def plot_elbow_curve(k_range, inertia_values, figsize=(10, 6)):
    """Vẽ đường Elbow để chọn K (Dùng cho Phase 3)"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_range, inertia_values, 'bx-', linewidth=2, markersize=8)
    ax.set_xlabel('k (Number of clusters)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method For Optimal k')
    ax.grid(True, alpha=0.3)
    return fig

def plot_cluster_heatmap(profile_summary, figsize=(12, 6)):
    """Vẽ Heatmap đặc tính cụm (Dùng cho Phase 3)"""
    # Drop cột không cần visual màu
    viz_data = profile_summary.drop(columns=['Order_Count', 'Population_%'], errors='ignore')
    # Min-Max Scaling cho visual
    viz_data_norm = viz_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        viz_data_norm, 
        annot=profile_summary.drop(columns=['Order_Count', 'Population_%'], errors='ignore'),
        cmap='RdYlGn_r', 
        fmt='.2f', ax=ax
    )
    ax.set_title('Cluster Characteristics Heatmap (Red=Risk/High, Green=Safe/Low)')
    return fig

def plot_business_impact(df_results, tier_summary, low_th, high_th, figsize=(14, 6)):
    """
    Vẽ biểu đồ tác động kinh doanh: Phân phối điểm & Lift Chart (Dùng cho Phase 4)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 1. Distribution
    sns.histplot(data=df_results, x='risk_score', hue='y_true', bins=50, kde=True, element="step", ax=ax1)
    ax1.axvline(low_th, color=COLORS['warning'], linestyle='--', label='Yellow Thresh')
    ax1.axvline(high_th, color=COLORS['danger'], linestyle='--', label='Red Thresh')
    ax1.set_title('Risk Score Distribution')
    ax1.legend()

    # 2. Lift Chart
    if 'Lift' not in tier_summary.columns and 'Precision' in tier_summary.columns:
         avg_rate = df_results['y_true'].mean()
         tier_summary['Lift'] = tier_summary['Precision'] / avg_rate

    sns.barplot(data=tier_summary, x='Tier', y='Lift', palette=[COLORS['danger'], COLORS['warning'], COLORS['neutral']], ax=ax2)
    ax2.axhline(1.0, color='black', linestyle='--', label='Baseline')
    
    for p in ax2.patches:
        value = p.get_height()
        if not pd.isna(value):
            ax2.text(p.get_x() + p.get_width()/2, value, f'{value:.2f}x', ha='center', va='bottom')
            
    ax2.set_title('Lift Chart (Efficiency vs Random)')
    
    plt.tight_layout()
    return fig