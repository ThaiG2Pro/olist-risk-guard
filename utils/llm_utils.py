import google.generativeai as genai
import time

def init_gemini(api_key):
    if not api_key: return False
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        _ = model.generate_content("ping")  # test API
        return True
    except Exception as e:
        return False

def get_proactive_strategy(cluster_id):
    """
    Chiến lược 'Phủ đầu' dựa trên 4 Personas mới:
    0: Standard
    1: Shipping Pain (Nhạy cảm phí ship - Cần Voucher)
    2: VIP Bulky (Hàng to tiền lớn - Cần hẹn giờ/An toàn)
    3: Local Goldmine (Khách gần - Cần Upsell)
    """
    strategies = {
        2: { # VIP Bulky
            "tone": "Trang trọng, Đẳng cấp nhưng nhấn mạnh sự Cẩn thận (Care).",
            "action": "Thông báo đã dặn dò Shipper xử lý nhẹ tay vì hàng giá trị cao & cồng kềnh. Đề xuất hẹn giờ giao.",
            "value_add": "Cam kết bảo hiểm hàng hóa 100%."
        },
        1: { # Shipping Pain
            "tone": "Thấu hiểu, Chia sẻ gánh nặng tài chính.",
            "action": "Thừa nhận phí vận chuyển tuyến này cao và cảm ơn khách đã tin tưởng.",
            "value_add": "TẶNG NGAY Voucher hỗ trợ phí ship cho đơn sau (như một sự bù đắp)."
        },
        3: { # Local Goldmine (Khách gần - Ít rủi ro)
            "tone": "Thân thiện, Gần gũi (Hàng xóm).",
            "action": "Thông báo hàng đang ở kho rất gần và sẽ đến cực nhanh.",
            "value_add": "Mời đánh giá 5 sao để nhận ưu đãi thành viên thân thiết."
        },
        0: { # Standard
            "tone": "Chuyên nghiệp, Nhanh gọn.",
            "action": "Xác nhận đơn hàng đang đi đúng tiến độ.",
            "value_add": "Cung cấp link theo dõi đơn hàng trực tiếp."
        }
    }
    return strategies.get(cluster_id, strategies[0])

def generate_prescriptive_content(order_data, risk_score, cluster_id, action_type="Email"):
    """
    Tạo nội dung Chăm sóc chủ động (Proactive Care).
    """
    strategy = get_proactive_strategy(cluster_id)

    # 1. Lấy thông tin ngữ cảnh
    product_cat = order_data.get('product_category_name_english', 'Sản phẩm').replace('_', ' ').title()
    price = order_data.get('total_price', 0)

    # [FIX VẤN ĐỀ 2] Xác định "Mối quan tâm tiềm ẩn" thay vì khẳng định lỗi
    # Logic: Giá cao -> lo về tiền. Ship xa/Risk cao -> lo về thời gian.
    if risk_score > 0.6:
        potential_concern = "thời gian vận chuyển có thể biến động vào giờ cao điểm"
    elif order_data.get('freight_ratio', 0) > 0.3: # Giả định có cột này hoặc logic tương tự
        potential_concern = "chi phí vận chuyển"
    else:
        potential_concern = "trải nghiệm nhận hàng suôn sẻ"

    # [FIX VẤN ĐỀ 3] PROMPT MỚI - PROACTIVE STYLE
    prompt = f"""
    [VAI TRÒ]: Bạn là Chuyên viên Chăm sóc Khách hàng (Customer Success) của Olist.

    [NGỮ CẢNH]: Khách hàng vừa đặt đơn hàng trị giá ${price} ({product_cat}).
    Hệ thống AI dự báo có rủi ro tiềm ẩn về: "{potential_concern}".
    LƯU Ý QUAN TRỌNG: Sự cố CHƯA xảy ra. Đừng xin lỗi. Đừng báo tin xấu.

    [MỤC TIÊU]: Viết một {action_type} (Tiếng Việt) để:
    1. Xác nhận đơn hàng đã được tiếp nhận và đang được chăm sóc đặc biệt.
    2. Trấn an khách hàng rằng Olist đang chủ động theo dõi đơn hàng này ("Proactive Monitoring").
    3. Tạo cảm giác tin cậy và chuyên nghiệp.

    [CHIẾN LƯỢC THEO PHÂN KHÚC KHÁCH HÀNG]:
    - Giọng điệu: {strategy['tone']}
    - Hành động cam kết: {strategy['action']}
    - Giá trị gia tăng (Quà/Voucher): {strategy['value_add']}

    [QUY TẮC BẮT BUỘC]:
    - KHÔNG dùng từ "Xin lỗi", "Sự cố", "Trục trặc", "Rủi ro", "Delay".
    - KHÔNG nhắc đến thông số kỹ thuật (Risk Score, Cluster).
    - Hãy dùng từ ngữ tích cực: "Ưu tiên", "Đảm bảo", "Theo dõi", "Quà tặng".
    - Ngắn gọn (dưới 150 từ). Định dạng Markdown đẹp.
    """

    # Gọi API (Giữ nguyên logic fallback model cũ)
    priority_models = ['gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-3-pro-preview','gemini-2.5-pro']
    generation_config = genai.types.GenerationConfig(temperature=0.7, max_output_tokens=400)

    for model_name in priority_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, generation_config=generation_config)
            return response.text.strip() # .strip() để loại bỏ khoảng trắng thừa đầu cuối
        except:
            continue

    return f"⚠️ Lỗi kết nối AI. Gợi ý hành động thủ công: {strategy['action']}"
