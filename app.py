import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# 1. ตั้งค่าหน้าเว็บให้ดูเป็น Dashboard มืออาชีพ
st.set_page_config(page_title="SuperStore Profit Intelligence", page_icon="💰", layout="wide")


# 2. ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_model():
    return joblib.load('profit_model.pkl')


model = load_model()

# --- ส่วนหัว Dashboard ---
st.markdown("""
    <div style='background-color:#007bff;padding:20px;border-radius:10px;margin-bottom:25px'>
        <h1 style='color:white;text-align:center;'>📊 SuperStore Business Intelligence Dashboard</h1>
        <p style='color:white;text-align:center;'>ระบบวิเคราะห์และพยากรณ์กำไรอัจฉริยะด้วย Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# --- สร้าง Tabs ---
tab1, tab2 = st.tabs(["🚀 ระบบพยากรณ์กำไร (Prediction)", "📈 บทวิเคราะห์ข้อมูล (Data Insights)"])

with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📥 ป้อนข้อมูลการขาย")
        with st.container(border=True):
            sales = st.number_input("ยอดขาย (Sales $)", min_value=0.0, value=250.0, step=10.0)
            quantity = st.slider("จำนวนสินค้า (Quantity)", 1, 100, 5)
            discount = st.slider("ส่วนลด (Discount Rate)", 0.0, 1.0, 0.1, 0.05)
            shipping_cost = st.number_input("ค่าขนส่ง (Shipping Cost $)", min_value=0.0, value=20.0, step=1.0)

            predict_btn = st.button("คำนวณกำไรคาดการณ์", use_container_width=True, type="primary")

    with col2:
        st.subheader("🎯 ผลการวิเคราะห์")
        if predict_btn:
            # เตรียมข้อมูลและพยากรณ์
            input_df = pd.DataFrame([[sales, quantity, discount, shipping_cost]],
                                    columns=['sales', 'quantity', 'discount', 'shipping_cost'])
            prediction = model.predict(input_df)[0]

            # แสดง Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("กำไรคาดการณ์", f"${prediction:,.2f}", delta=f"{(prediction / sales) * 100:.1f}% Margin")
            m2.metric("ต้นทุนรวม (ประมาณการ)", f"${sales - prediction:,.2f}")
            m3.metric("จุดคุ้มทุน", "ผ่าน" if prediction > 0 else "ไม่ผ่าน", delta_color="normal")

            # แสดง Gauges Chart (เกจวัดความเสี่ยง)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Profitability Gauge"},
                gauge={
                    'axis': {'range': [None, sales * 0.5]},
                    'bar': {'color': "#007bff"},
                    'steps': [
                        {'range': [-sales, 0], 'color': "#ff4b4b"},
                        {'range': [0, sales * 0.5], 'color': "#00cc96"}]}))
            st.plotly_chart(fig_gauge, use_container_width=True)

            if prediction > 0:
                st.success(f"✅ รายการนี้มีแนวโน้มได้รับกำไรประมาณ ${prediction:,.2f}")
                st.balloons()
            else:
                st.error(f"⚠️ รายการนี้มีความเสี่ยงที่จะขาดทุนประมาณ ${abs(prediction):,.2f}")
        else:
            st.info("กรุณากรอกข้อมูลที่ด้านซ้ายเพื่อดูการพยากรณ์")

with tab2:
    st.subheader("🔍 สรุป Insights จากการทำ EDA")
    c1, c2 = st.columns(2)

    with c1:
        st.write("**1. ความสัมพันธ์ของตัวแปร (Feature Correlation)**")
        st.info("💡 จากการวิเคราะห์ Heatmap: 'Sales' และ 'Shipping Cost' มีผลกระทบสูงสุดต่อกำไร")
        # ตรงนี้ถ้าคุณมีรูป Heatmap ให้ใช้ st.image('heatmap.png')
        st.markdown("---")
        st.write("**2. การกระจายตัวของกำไร (Profit Distribution)**")
        st.write("กำไรส่วนใหญ่ของร้านค้าเกาะกลุ่มอยู่ที่ 0 ถึง 50 ดอลลาร์")

    with c2:
        # ตัวอย่างกราฟจำลองที่โชว์ว่า Random Forest วิเคราะห์ยังไง
        st.write("**3. ปัจจัยสำคัญที่มีผลต่อโมเดล (Feature Importance)**")
        feat_importance = pd.DataFrame({
            'Feature': ['Sales', 'Discount', 'Shipping Cost', 'Quantity'],
            'Importance': [0.45, 0.25, 0.20, 0.10]
        })
        fig_feat = px.bar(feat_importance, x='Importance', y='Feature', orientation='h', color='Importance')
        st.plotly_chart(fig_feat, use_container_width=True)

# --- Footer ---
st.divider()
st.caption("© 2024 SuperStore Analytics Project | Built with Streamlit & Scikit-Learn")