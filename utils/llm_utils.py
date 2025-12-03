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
    Chiến lược 'Phủ đầu' (Pre-emptive):
    Tập trung vào việc trấn an, tạo niềm tin và giá trị gia tăng thay vì xin lỗi.
    """
    strategies = {
        2: { # VIP
            "tone": "Trang trọng, Đẳng cấp, Cá nhân hóa cao.",
            "action": "Thông báo đã kích hoạt chế độ 'Theo dõi ưu tiên' (Priority Monitoring).",
            "value_add": "Tặng voucher đặc quyền cho lần sau (như một lời tri ân, không phải đền bù)."
        },
        1: { # Price Sensitive
            "tone": "Thân thiện, Nhấn mạnh vào sự tiết kiệm/giá trị.",
            "action": "Xác nhận đơn hàng thành công và cam kết không phát sinh chi phí.",
            "value_add": "Tặng mã Freeship hoặc giảm 5% cho đơn tiếp theo."
        },
        0: { # Standard
            "tone": "Chuyên nghiệp, Rõ ràng, Tin cậy.",
            "action": "Thông báo quy trình chuẩn bị hàng đang diễn ra suôn sẻ.",
            "value_add": "Cung cấp kênh hỗ trợ nhanh nếu cần."
        },
        3: { # Local/Remote
            "tone": "Gần gũi, Chu đáo.",
            "action": "Lưu ý về tuyến vận chuyển xa và cam kết theo dõi sát sao.",
            "value_add": "Cam kết cập nhật lộ trình thường xuyên."
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
