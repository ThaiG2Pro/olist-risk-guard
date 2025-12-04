
# 🛡️ Olist Risk Guard AI: Proactive Customer Support System

> **"Turning Reactive Support into Proactive Prevention"**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/) 
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Model](https://img.shields.io/badge/Model-Hybrid%20(KMeans%20%2B%20LGBM)-orange)
![Status](https://img.shields.io/badge/Status-Prototype-success)

<div align="center"> <h3> <a href="https://olist-risk-guard.streamlit.app/"> 🚀 CLICK HERE TO LAUNCH LIVE DEMO </a> </h3> <p><em>⚠️ Lưu ý: App chạy trên Free Tier Cloud nên có thể mất <strong>30-60 giây</strong> để khởi động (Wake up) sau khi bấm. Vui lòng kiên nhẫn!</em></p> </div>

## 📋 Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. System Architecture](#2-system-architecture)
- [3. Model Performance & Card](#3-model-performance--card)
- [4. GenAI Strategy](#4-genai-strategy)
- [5. Installation & Setup](#5-installation--setup)
- [6. Project Structure](#6-project-structure)
- [7. Limitations & Future Work](#7-limitations--future-work)

---

## 1. Executive Summary

**Olist Risk Guard AI** giải quyết bài toán cốt lõi của E-commerce: **Làm sao phát hiện khách hàng sắp không hài lòng trước khi họ viết đánh giá 1 sao?**

Thay vì quy trình truyền thống (Khách complain $\rightarrow$ CS xử lý), hệ thống này tạo ra quy trình mới:
1.  **Predict:** Dự báo xác suất rủi ro (Risk Score) ngay khi đơn hàng đang vận chuyển.
2.  **Segment:** Phân nhóm khách hàng (VIP, Price-sensitive, Standar,..) để có kịch bản xử lý phù hợp.
3.  **Prevent:** Sử dụng **GenAI (Google Gemini)** tự động soạn thảo email "phủ đầu rủi ro" (Pre-emptive action).

**Business Impact:**
* 🎯 **Precision (High Risk Tier):** ~31% (Gấp gần 2 lần so với chọn ngẫu nhiên).
* ⏱️ **Efficiency:** Giúp đội CSKH chỉ cần tập trung vào **top 5%** đơn hàng rủi ro nhất thay vì dàn trải.

## 📊 Demo



> _Giao diện Dashboard hiển thị danh sách đơn hàng rủi ro cao và tính năng AI soạn email tự động._

![App Demo](demo-olist-risk-guard.gif)


---

## 2. System Architecture

Dưới đây là luồng dữ liệu và quy trình xử lý của hệ thống (End-to-End Pipeline):

```mermaid
graph TD
    subgraph Data_Source
        A["Olist Database (Orders, Reviews, Sellers...)"] --> B["Data Cleaning & Merge"]
    end

    subgraph "Analytics Core"
        B --> C{"Feature Engineering"}
        C -->|Stateless| D["Time/Distance Features"]
        C -->|Stateful| E["Seller Risk Score / Route History"]
        D --> F["K-Means Clustering"]
        E --> F
        D --> G["Model (Logistic + Calibration)"]
        E --> G
    end

    subgraph "Application & Action"
        G -->|Risk Score| H["Decision Engine (Thresholding)"]
        F -->|Cluster ID| H
        H --> I{"High Risk?"}
        I -->|Yes| J["GenAI Agent (Gemini API)"]
        I -->|No| K["Standard Process"]
        J --> L["Drafted Proactive Email"]
        L --> M["Streamlit Dashboard"]
    end

````

### Key Components:

1. **Input Processor:** Xử lý dữ liệu thô, tính toán khoảng cách (Haversine), lịch sử người bán,v.v
    
2. **Hybrid Model Core:** Kết hợp Unsupervised (hiểu hành vi) và Supervised (dự báo rủi ro).
    
3. **GenAI Agent:** "Nhân viên ảo" soạn nội dung dựa trên ngữ cảnh (Context-aware generation).

---

## 3. Model Performance & Card

### Model Card

| **Attribute**           | **Description**                                                                 |
| ----------------------- | ------------------------------------------------------------------------------- |
| **Model Type**          | Hybrid: K-Means (k=4) + Logistic Classifier (w/ Isotonic Calibration)           |
| **Input Features**      | 40 features (Delivery delay, Seller history, Price, Freight ratio, Distance...) |
| **Target Variable**     | Binary: `1` (Review Score $\le$ 3), `0` (Review Score > 3)                      |
| **Training Data**       | Olist E-commerce Dataset (100k orders, 2016-2018)                               |
| **Evaluation Strategy** | Temporal Split (Train on Past, Test on Future) to avoid data leakage.           |

### Performance Metrics (Test Set)

Chúng tôi **không** sử dụng Accuracy hay AUC làm thước đo chính, vì dữ liệu hành vi review chứa nhiều nhiễu cảm xúc khó dự đoán. 
**Business Metrics (Kết quả thực tế)** Chúng tôi chia đơn hàng thành các tầng rủi ro. Chiến lược là tập trung nguồn lực CSKH vào **Top 5% (Red Zone)**. 

| Metric                   | Giá trị   | Ý nghĩa kinh doanh (Business Value)                                                                                      |
| :----------------------- | :-------- | :----------------------------------------------------------------------------------------------------------------------- |
| **Precision (Red Zone)** | **31%**   | Cứ **3 cuộc gọi** chăm sóc khách hàng, nhân viên sẽ tiếp cận đúng **1 khách hàng** đang thực sự gặp vấn đề nghiêm trọng. |
| **Baseline Precision**   | ~16.4%    | Nếu chọn ngẫu nhiên, nhân viên phải gọi 6 cuộc mới trúng 1 người (lãng phí nguồn lực).                                   |
| **Lift Score**           | **1.88x** | **Hiệu quả gấp gần 2 lần** so với quy trình ngẫu nhiên hiện tại.                                                         |


---

## 4. GenAI Strategy

Hệ thống không dùng template tĩnh. Chúng tôi sử dụng **Prompt Engineering** với kỹ thuật **Persona & Context Injection**:

- **Input Context:** `Risk Score`, `Cluster Type` (e.g., VIP), `Delay Days`, `Customer History`.
    
- **Prompt Strategy:**
    
    - _Role:_ Senior Customer Success Manager.
        
    - _Constraint:_ Không xin lỗi suông, không dùng từ ngữ tiêu cực ("Lỗi", "Hỏng"), tập trung vào giải pháp ("Theo dõi ưu tiên").
        
    - _Adaptation:_
        
        - _VIP Cluster:_ Giọng văn trang trọng, tặng quyền lợi đặc biệt.
            
        - _Standard Cluster:_ Giọng văn thân thiện, rõ ràng, tặng Voucher Freeship.
            

---

## 5. Installation & Setup  

### Hướng dẫn cài đặt

1. **Clone Repository:**

   ```
    git clone https://github.com/ThaiG2Pro/olist-risk-guard.git
    cd olist-risk-guard
    ```

1. **Cài đặt thư viện:**

   ```
    pip install -r requirements.txt
    ```

2. Cấu hình API Key (Bắt buộc):

    Tạo file .streamlit/secrets.toml và dán Google Gemini API Key của bạn vào:

   ```
    GEMINI_API_KEY = "AIzaSyDxxxx..."
    ```

3. **Khởi chạy ứng dụng:**

   ```
    streamlit run app.py
    ```


---
## 6. Project Structure

```
Olist-Risk-Guard/
├── app.py                  # Streamlit Dashboard (Main App)
├── requirements.txt        # Các thư viện cần thiết
├── models/                 # Chứa các model đã train (.pkl)
├── notebooks/              # Jupyter Notebooks theo từng Phase
│   ├── phase0.ipynb        # EDA
│   ├── phase1.ipynb        # Diagnosis & SHAP
│   └── phase2.ipynb        # Model Training & Evaluation
└── utils/                  # Bộ thư viện tiện ích 
```

---
## 7. Limitations & Future Work

### Hạn chế hiện tại (Limitations)

- **Cold Start:** Chưa xử lý tốt các Seller mới hoặc Sản phẩm mới (thiếu lịch sử).
    
- **Static Data:** Dữ liệu đang dừng ở 2018 (offline training), chưa có pipeline update realtime.
    
- **Latency:** Phụ thuộc vào tốc độ phản hồi của Gemini API (miễn phí).
    

### Kế hoạch phát triển (Roadmap)

- [ ] **MLOps:** Xây dựng pipeline tự động retrain model hàng tháng với Airflow/Github Actions.
    
- [ ] **Feedback Loop:** Cho phép nhân viên CSKH đánh giá chất lượng email do AI viết để finetune lại prompt.
    
- [ ] **Multi-channel:** Mở rộng tích hợp gửi tin nhắn qua WhatsApp/Zalo OA.

## 8. Đóng góp (Contributing)

Mọi đóng góp đều được hoan nghênh. Vui lòng mở Pull Request hoặc Issue để thảo luận.