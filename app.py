import streamlit as st
import pandas as pd
import joblib
import plotly.express as px  # เพิ่มการทำกราฟที่สวยกว่าเดิม

# 1. Page Configuration
st.set_page_config(
    page_title="SuperStore Profit Intelligence",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. Load Model & Data
@st.cache_resource
def load_assets():
    model = joblib.load('profit_model.pkl')
    # ลองโหลดไฟล์ csv มาโชว์สถิติ (ถ้าคุณมีไฟล์ใน GitHub)
    try:
        data = pd.read_csv('SuperStoreOrders - SuperStoreOrders.csv')
        data['sales'] = data['sales'].replace(',', '', regex=True).astype(float)
    except:
        data = None
    return model, data


model, df_sample = load_assets()

# --- Custom CSS เพื่อความสวยงาม ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #007bff; color: white; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3222/3222672.png", width=100)
    st.title("Settings")
    st.info("กรอกข้อมูลการขายในช่องด้านล่างเพื่อพยากรณ์กำไรสุทธิ")

    st.header("📥 Input Details")
    sales = st.number_input("ยอดขาย (Sales $)", min_value=0.0, value=500.0)
    quantity = st.number_input("จำนวนชิ้น (Quantity)", min_value=1, value=5)
    discount = st.slider("ส่วนลด (Discount %)", 0.0, 1.0, 0.1)
    shipping = st.number_input("ค่าขนส่ง (Shipping Cost $)", min_value=0.0, value=25.0)

    predict_btn = st.button("🔍 วิเคราะห์กำไร")

# --- Main Content ---
st.title("💎 SuperStore Profit Intelligence Dashboard")
st.markdown("---")

# สร้าง Tabs: หน้าพยากรณ์ และ หน้าดูสถิติรวม
tab1, tab2 = st.tabs(["🚀 Profit Prediction", "📊 Data Insights"])

with tab1:
    if predict_btn:
        # เตรียมข้อมูลพยากรณ์
        input_data = pd.DataFrame([[sales, quantity, discount, shipping]],
                                  columns=['sales', 'quantity', 'discount', 'shipping_cost'])
        prediction = model.predict(input_data)[0]

        # แสดง Metrics สำคัญ
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Predicted Profit", f"${prediction:,.2f}", delta=f"{(prediction / sales) * 100:.1f}% Margin")
        with m2:
            st.metric("Total Sales", f"${sales:,.2f}")
        with m3:
            st.metric("Cost Ratio", f"{((sales - prediction) / sales) * 100:.1f}%")

        # แสดงผลลัพธ์เป็น Card
        if prediction > 0:
            st.success(f"### ผลกำไรคาดการณ์: ${prediction:,.2f}")
            st.balloons()
        else:
            st.error(f"### ผลขาดทุนคาดการณ์: ${prediction:,.2f}")

        # กราฟเปรียบเทียบสัดส่วน
        fig = px.pie(values=[sales - prediction, prediction], names=['Costs', 'Profit'],
                     hole=.4, title="Profit vs Cost Breakdown", color_discrete_sequence=['#ff4b4b', '#00cc96'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("### ⬅️ กรุณาระบุข้อมูลที่แถบด้านซ้ายเพื่อเริ่มต้น")
        st.image("https://img.freepik.com/free-vector/data-analysis-concept-illustration_114360-1594.jpg", width=500)

with tab2:
    st.subheader("📌 สรุปภาพรวมจากฐานข้อมูลจริง")
    if df_sample is not None:
        c1, c2 = st.columns(2)
        # กราฟความสัมพันธ์ยอดขายกับกำไร
        fig_scatter = px.scatter(df_sample.sample(1000), x="sales", y="profit", color="discount",
                                 title="Sales vs Profit Relationship (Sample 1000 orders)")
        c1.plotly_chart(fig_scatter, use_container_width=True)

        # กราฟกำไรรายประเทศ/ภูมิภาค
        fig_bar = px.bar(df_sample.groupby('market')['profit'].sum().reset_index(),
                         x='market', y='profit', title="Total Profit by Market")
        c2.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("กรุณาอัปโหลดไฟล์ CSV เข้า GitHub เพื่อดูสถิติส่วนนี้")

# Footer
st.markdown("---")
st.caption("AI Model: Random Forest Regressor | Last Update: March 2024")