import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt

# Import các bộ xử lý từ utils
from utils import preparation_utils as pu
from utils import feature_utils as fu
from utils import viz_utils as vu
from utils import llm_utils as lu

# --- 1. CẤU HÌNH TRANG & SESSION STATE ---
st.set_page_config(
    page_title="Olist Risk Guard AI",
    page_icon="🛡️",
    layout="wide"
)

# Khởi tạo Session State
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'ai_email_content' not in st.session_state:
    st.session_state.ai_email_content = ""

# --- 2. LOAD HỆ THỐNG ---
@st.cache_resource
def load_prediction_system():
    try:
        with open('models/model_config.json', 'r') as f:
            config = json.load(f)
        loaded_kmeans = joblib.load('models/kmeans_cluster_model.pkl')
        kmeans_model = loaded_kmeans['model'] if isinstance(loaded_kmeans, dict) else loaded_kmeans

        loaded_hybrid = joblib.load('models/final_hybrid_model.pkl')
        hybrid_model = loaded_hybrid['model'] if isinstance(loaded_hybrid, dict) else loaded_hybrid

        prep_artifacts = joblib.load('models/preprocessing_artifacts.pkl')
        return config, kmeans_model, hybrid_model, prep_artifacts
    except Exception as e:
        st.error(f"Lỗi khởi động hệ thống: {e}")
        return None, None, None, None

config, kmeans_model, hybrid_model, prep_artifacts = load_prediction_system()

# Dùng @st.cache_data để không tính toán lại khi đổi filter
@st.cache_data
def run_full_pipeline(df_raw, _config, _prep_artifacts, _kmeans_model, _hybrid_model):
    """
    Hàm xử lý dữ liệu và dự báo. Được Cache lại nếu đầu vào không đổi.
    Dấu _ trước tên biến là để báo Streamlit không cần hash các object phức tạp này.
    """
    # 1. Display Data: Giữ nguyên gốc
    df_display = df_raw.copy()

    if 'order_id' not in df_display.columns: df_display['order_id'] = df_display.index
    if 'seller_id' not in df_display.columns: df_display['seller_id'] = 'Unknown'

    # 2. Preprocessing
    all_features = _config['features']['all_features']
    skewed_cols = _config['features']['skewed_cols']
    cat_cols = _config['features']['categorical_cols']

    X_processed = pu.test_preparation_pipeline(
        df_raw,
        artifacts=_prep_artifacts,
        skewed_cols=skewed_cols,
        categorical_cols=cat_cols
    )

    # Align columns & Deduplicate
    for col in all_features:
        if col not in X_processed.columns:
            X_processed[col] = 0

    X_processed = X_processed.loc[:, ~X_processed.columns.duplicated()]
    X_processed = X_processed.reindex(columns=all_features, fill_value=0)

    # 3. Clustering
    cluster_feats = _config['features']['clustering_features']
    X_cluster = X_processed[cluster_feats].copy()
    clusters = _kmeans_model.predict(X_cluster)

    # 4. Risk Prediction
    X_hybrid = X_processed.copy()
    X_hybrid['Cluster'] = clusters
    X_hybrid = pd.get_dummies(X_hybrid, columns=['Cluster'], prefix='Cluster')

    # Realign again for hybrid model
    X_hybrid = X_hybrid.loc[:, ~X_hybrid.columns.duplicated()]
    X_hybrid = X_hybrid.reindex(columns=all_features, fill_value=0)

    risk_scores = _hybrid_model.predict_proba(X_hybrid)[:, 1]

    # 5. Kết quả cuối
    df_display['risk_score'] = risk_scores
    df_display['Cluster'] = clusters

    return df_display

def get_cluster_name_ui(cluster_id):
    """[FIX LỖI 5] Map số cluster thành tên hiển thị"""
    mapping = {
        2: "💎 VIP",
        1: "💰 Price Sensitive",
        0: "📦 Standard",
        3: "📍 Local / Others"
    }
    return mapping.get(cluster_id, f"Cluster {cluster_id}")

# --- 3. GIAO DIỆN ---
st.title("🛡️ Olist Risk Guard - Interactive Mode")

st.sidebar.header("⚙️ Configuration")
user_api_key = st.sidebar.text_input(
    "🔑 Gemini API Key (Optional)",
    type="password",
    help="Nhập key của bạn để không bị giới hạn quota. Nếu để trống, hệ thống sẽ dùng Key Demo (có giới hạn)."
)
# Khởi tạo session_state khi load page
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if 'is_configured' not in st.session_state:
    st.session_state.is_configured = False

if st.sidebar.button("Comfirm key (Phải confirm để dùng key)"):
    st.session_state.api_key = user_api_key
    st.session_state.is_configured = lu.init_gemini(user_api_key)

# Xác định final_api_key để dùng trong app
final_api_key = None
if user_api_key:
    final_api_key = user_api_key
    st.sidebar.success("Đang dùng Key cá nhân của bạn!")
elif "GEMINI_API_KEY" in st.secrets:
    final_api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.info("Đang dùng Key Demo của hệ thống.") # Có thể ẩn dòng này nếu muốn
else:
    st.sidebar.warning("⚠️ Chưa có API Key. Tính năng AI sẽ không hoạt động.")

# INPUT
st.sidebar.divider()
st.sidebar.header("1. Input Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV (Raw)", type=['csv'])
use_demo = st.sidebar.checkbox("Dùng dữ liệu mẫu")

# Logic load dữ liệu
df_input = None
if use_demo:
    try:
        df_input = pd.read_csv('data/sample_data.csv')
    except:
        st.sidebar.error("Chưa có file mẫu. Hãy upload file.")
elif uploaded_file:
    df_input = pd.read_csv(uploaded_file)

# PROCESS & SAVE STATE
if df_input is not None:
    # Chỉ chạy dự báo nếu data thay đổi
    if st.session_state.processed_data is None or not df_input.equals(st.session_state.get('last_input')):
        with st.spinner("AI đang phân tích..."):
            try:
                results = run_full_pipeline(
                    df_input, config, prep_artifacts, kmeans_model, hybrid_model
                )
                st.session_state.processed_data = results
                st.session_state.last_input = df_input # Lưu lại để so sánh
                st.toast("Dự báo hoàn tất!", icon="✅")
            except Exception as e:
                st.error(f"Lỗi Pipeline: {e}")
                st.stop()

#  HIỂN THỊ (Chỉ chạy khi đã có data trong session_state) ---
if st.session_state.processed_data is not None:
    df_result = st.session_state.processed_data

    if 'last_selected_idx' not in st.session_state:
        st.session_state.last_selected_idx = None


    # Filter
    st.sidebar.header("2. Filter & Select")
    filter_risk = st.sidebar.radio("Lọc rủi ro:", ["Tất cả", "🔴 Rủi ro cao", "🟡 Cảnh báo"])

    if "🔴" in filter_risk:
        mask = df_result['risk_score'] >= config['thresholds']['high_risk']
    elif "🟡" in filter_risk:
        mask = (df_result['risk_score'] >= config['thresholds']['low_risk']) & \
               (df_result['risk_score'] < config['thresholds']['high_risk'])
    else:
        mask = [True] * len(df_result)

    filtered_df = df_result[mask]

    if len(filtered_df) > 0:
        selected_idx = st.sidebar.selectbox("Chọn đơn hàng:", filtered_df.index)

        #  Kiểm tra xem có đổi đơn hàng không?
        if selected_idx != st.session_state.last_selected_idx:
            st.session_state.ai_email_content = ""     # Xóa nội dung cũ
            st.session_state.ai_content_type = ""      # Xóa loại hành động cũ
            st.session_state.last_selected_idx = selected_idx # Cập nhật index mới
            st.rerun() # (Tuỳ chọn: Rerun để refresh UI mượt hơn)

        order = df_result.loc[selected_idx]

        # --- DASHBOARD ---
        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        # Hiển thị thông tin gốc
        c1.metric("Giá trị đơn", f"${order.get('total_price', 0):,.2f}")
        c1.metric("Phí vận chuyển", f"${order.get('total_freight', 0):,.2f}")

        # Hiển thị thông tin AI
        cluster_label = get_cluster_name_ui(order['Cluster'])
        c2.metric("Rủi ro (AI)", f"{order['risk_score']:.1%}")
        c2.metric("Phân khúc", cluster_label)

        # Hiển thị ID
        #c3.info(f"Seller ID: {order['seller_id']}")

        c3.info(f"Order ID: {order['order_id']}")

        # --- ACTION & GENAI ---
        st.markdown("---")
        st.subheader("⚡ Action Center (Trung tâm hành động)")

        col_left, col_right = st.columns([1, 2])

        risk = order['risk_score']
        high_th = config['thresholds']['high_risk']
        low_th = config['thresholds']['low_risk']


        show_ai_button = False # Biến cờ để điều khiển hiển thị
        action_label = ""
        ai_task_type = ""  # "Email" hoặc "Kịch bản gọi điện"

        with col_left:
            if risk >= high_th:
                st.error("🔥 **GỌI ĐIỆN KHẨN CẤP**")
                st.caption("Khách hàng có nguy cơ rất cao. Cần tương tác trực tiếp.")
                # Cấu hình cho AI
                show_ai_button = True
                action_label = "📞 Soạn Kịch bản Gọi điện"
                ai_task_type = "Kịch bản gọi điện"

            elif risk >= low_th:
                st.warning("⚠️ **GỬI EMAIL THEO DÕI**")
                st.caption("Đơn hàng cần được chăm sóc để tránh rủi ro.")
                # Cấu hình cho AI
                show_ai_button = True
                action_label = "✉️ Soạn Email Hỗ trợ"
                ai_task_type = "Email"

            else:
                # Rủi ro thấp -> Không hiện nút AI, xóa nội dung cũ (đã xử lý ở trên)
                st.success("✅ **KHÔNG CẦN HÀNH ĐỘNG**")
                st.caption("Đơn hàng an toàn. Tiết kiệm nguồn lực.")

        with col_right:
            # [SỬA] Kiểm tra final_api_key thay vì st.secrets
            if final_api_key:
                if 'api_key' not in st.session_state or st.session_state.api_key != final_api_key:
                    st.session_state.api_key = final_api_key
                    st.session_state.is_configured = lu.init_gemini(final_api_key)

                if st.session_state.is_configured:
                    if st.button(action_label, type="primary"):
                        with st.spinner("Gemini đang viết..."):
                            content = lu.generate_prescriptive_content(
                                order, risk , order['Cluster'], ai_task_type
                            )
                            st.session_state.ai_email_content = content
                            st.session_state.ai_content_type = ai_task_type
                else:
                    st.session_state.ai_email_content = None  # Xóa nội dung cũ nếu key lỗi
                    st.error("API Key không hợp lệ. Vui lòng kiểm tra lại.")
            else:
                st.info("Vui lòng nhập Gemini API Key ở thanh bên trái để dùng tính năng này.")

            #  Hiển thị nội dung AI (Markdown Render)
            if st.session_state.ai_email_content:

                st.markdown("---")

                # 1. Chọn màu sắc giao diện
                if st.session_state.ai_content_type == "Kịch bản gọi điện":
                    border_color = "#D32F2F" # Đỏ đậm
                    bg_color = "#FFEBEE"     # Đỏ nhạt
                    icon = "📞"
                    title = "Kịch Bản Gọi Điện (Proactive Call)"
                else:
                    border_color = "#FFA000" # Vàng đậm
                    bg_color = "#FFF8E1"     # Vàng nhạt
                    icon = "✉️"
                    title = "Nội Dung Email Chăm Sóc (Proactive Email)"

                # 2. Vẽ khung tiêu đề (Header)
                st.markdown(f"""
                <div style="
                    background-color: {bg_color};
                    border-left: 5px solid {border_color};
                    padding: 10px 15px;
                    border-radius: 5px 5px 0 0;
                    margin-bottom: 0px;">
                    <h4 style="margin:0; color: #333;">{icon} {title}</h4>
                </div>
                """, unsafe_allow_html=True)

                # 3. Vẽ nội dung (Body)
                container = st.container()
                with container:
                    # Tạo một khối style tiệp màu với header
                    st.markdown(f"""
                    <div style="
                        background-color: {bg_color};
                        border-left: 5px solid {border_color};
                        padding: 15px;
                        border-radius: 0 0 5px 5px;
                        margin-top: -5px;">
                        """, unsafe_allow_html=True)

                    # Render nội dung chính (Markdown thuần sẽ không bị lỗi thẻ div)
                    st.markdown(st.session_state.ai_email_content)

                    st.markdown("</div>", unsafe_allow_html=True) # Đóng thẻ div của body

                # 4. Nút Copy/Action phụ
                c_copy, c_send = st.columns([1, 5])
                with c_send:
                    if st.button("🚀 Gửi ngay (Giả lập)", key="btn_send"):
                        st.toast("Đã gửi tin nhắn thành công!", icon="✅")
