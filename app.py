import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="SuperStore Profit Intelligence", page_icon="💰", layout="wide")


# 2. Load Model
@st.cache_resource
def load_assets():
    model = joblib.load('profit_model.pkl')
    return model


model = load_assets()

# --- Header ---
st.markdown("""
    <div style='background-color:#1e293b;padding:20px;border-radius:10px;margin-bottom:25px'>
        <h1 style='color:white;text-align:center;margin:0;'>📊 SuperStore Business Intelligence</h1>
        <p style='color:#94a3b8;text-align:center;margin:5px 0 0 0;'>Profit Prediction and Data Analysis System</p>
    </div>
    """, unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2 = st.tabs(["🚀 Prediction System", "📈 Data Insights (EDA)"])

with tab1:
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("📥 Input Data")
        with st.container(border=True):
            sales = st.number_input("Sales ($)", min_value=0.0, value=250.0)
            quantity = st.number_input("Quantity", min_value=1, value=5)
            discount = st.slider("Discount Rate", 0.0, 1.0, 0.1)
            shipping = st.number_input("Shipping Cost ($)", min_value=0.0, value=20.0)

            predict_btn = st.button("Calculate Profit", use_container_width=True, type="primary")

    with col2:
        st.subheader("🎯 Result")
        if predict_btn:
            input_df = pd.DataFrame([[sales, quantity, discount, shipping]],
                                    columns=['sales', 'quantity', 'discount', 'shipping_cost'])
            prediction = model.predict(input_df)[0]

            # Metrics Row
            m1, m2 = st.columns(2)
            margin = (prediction / sales) * 100
            m1.metric("Predicted Profit", f"${prediction:,.2f}")
            m2.metric("Profit Margin", f"{margin:.2f}%")

            # แสดงผลลัพธ์
            if prediction > 0:
                st.success(f"**Analysis:** รายการนี้คาดการณ์ว่าจะมีกำไรสุทธิ ${prediction:,.2f}")
            else:
                st.error(f"**Analysis:** รายการนี้มีความเสี่ยงที่จะขาดทุน ${abs(prediction):,.2f}")

            # Indicator Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                gauge={'axis': {'range': [-100, 500]},
                       'bar': {'color': "#3b82f6"},
                       'steps': [
                           {'range': [-100, 0], 'color': "#fee2e2"},
                           {'range': [0, 500], 'color': "#dcfce7"}]}))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("กรุณาระบุข้อมูลและกดปุ่ม Calculate เพื่อเริ่มการพยากรณ์")

with tab2:
    st.subheader("🔍 สรุปผลการวิเคราะห์ข้อมูล (EDA)")

    # ส่วนนี้แนะนำให้คุณอัปโหลดรูป Heatmap ที่คุณแคปจาก Colab ขึ้น GitHub ด้วยนะครับ
    # สมมติว่าชื่อไฟล์ heatmap.png และ distribution.png

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Correlation Matrix**")
        # st.image("heatmap.png", use_container_width=True) # ปลดคอมเมนต์บรรทัดนี้ถ้าอัปโหลดรูปแล้ว
        st.write("จากการวิเคราะห์ Heatmap พบว่ายอดขาย (Sales) สัมพันธ์กับกำไรมากที่สุด")

    with c2:
        st.markdown("**2. Profit Distribution**")
        # st.image("distribution.png", use_container_width=True) # ปลดคอมเมนต์บรรทัดนี้ถ้าอัปโหลดรูปแล้ว
        st.write("ข้อมูลกำไรมีการกระจายตัวแบบกระจุกตัวที่ค่าบวกเล็กน้อย (Skewed Distribution)")

# Footer
st.divider()
st.caption("Developed for ML Deployment Course Project")