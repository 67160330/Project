import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. การตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="SuperStore Profit Intelligence",
    page_icon="💰",
    layout="wide"
)


# 2. ฟังก์ชันโหลดโมเดล (ใช้ Cache เพื่อความเร็ว)
@st.cache_resource
def load_model():
    # ต้องมั่นใจว่ามีไฟล์ชื่อนี้อยู่ใน GitHub ของคุณ
    return joblib.load('profit_model.pkl')


model = load_model()

# --- ส่วนหัวของหน้าเว็บ (Header) ---
st.markdown("""
    <div style='background-color:#0f172a; padding:25px; border-radius:15px; margin-bottom:20px; border-left: 8px solid #3b82f6;'>
        <h1 style='color:white; margin:0;'>📊 SuperStore Profit Analytics</h1>
        <p style='color:#94a3b8; margin:5px 0 0 0;'>ระบบพยากรณ์กำไรอัจฉริยะและวิเคราะห์ปัจจัยทางธุรกิจ</p>
    </div>
    """, unsafe_allow_html=True)

# --- สร้าง Tabs ---
tab1, tab2 = st.tabs(["🚀 พยากรณ์กำไร (Prediction)", "🔍 ข้อมูลเชิงลึก (Data Insights)"])

# --- TAB 1: ระบบพยากรณ์ ---
with tab1:
    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        st.subheader("📥 ป้อนข้อมูลรายการขาย")
        with st.container(border=True):
            sales = st.number_input("ยอดขายทั้งหมด (Sales $)", min_value=0.0, value=250.0)
            quantity = st.number_input("จำนวนสินค้า (Quantity)", min_value=1, value=5)
            discount = st.slider("ส่วนลด (Discount Rate)", 0.0, 1.0, 0.1)
            shipping = st.number_input("ค่าขนส่ง (Shipping Cost $)", min_value=0.0, value=20.0)

            st.write("")
            predict_btn = st.button("คำนวณกำไรคาดการณ์", use_container_width=True, type="primary")

    with col2:
        st.subheader("🎯 ผลการวิเคราะห์")
        if predict_btn:
            # เตรียมข้อมูลและพยากรณ์
            input_df = pd.DataFrame([[sales, quantity, discount, shipping]],
                                    columns=['sales', 'quantity', 'discount', 'shipping_cost'])
            prediction = model.predict(input_df)[0]

            # คำนวณ Margin (%)
            margin = (prediction / sales) * 100 if sales > 0 else 0

            # แสดงตัวเลขหลัก (Metrics)
            m1, m2 = st.columns(2)
            m1.metric("Predicted Profit", f"${prediction:,.2f}")
            m2.metric("Profit Margin", f"{margin:.2f}%")

            # แสดงสถานะและคำแนะนำ
            if prediction > 0:
                st.success(f"**รายการนี้ได้กำไร:** คาดการณ์ผลกำไรสุทธิอยู่ที่ ${prediction:,.2f}")
                if margin > 15:
                    st.info("🌟 **Business Insight:** รายการนี้มี Margin สูงมาก เป็นดีลที่คุ้มค่า")
                else:
                    st.info("💡 **Business Insight:** กำไรอยู่ในระดับปกติ ควรคุมค่าใช้จ่ายอื่นๆ เพิ่มเติม")
            else:
                st.error(f"**รายการนี้ขาดทุน:** คาดการณ์ว่าจะขาดทุนประมาณ ${abs(prediction):,.2f}")
                st.warning("⚠️ **ข้อแนะนำ:** ควรลดส่วนลด (Discount) หรือพิจารณาปรับราคาขายใหม่")

            # กราฟเปรียบเทียบ Profit vs Sales
            st.write("---")
            chart_data = pd.DataFrame({
                'Category': ['Total Sales', 'Predicted Profit'],
                'Amount ($)': [sales, prediction]
            })
            fig_bar = px.bar(chart_data, x='Category', y='Amount ($)', color='Category',
                             color_discrete_map={'Total Sales': '#94a3b8', 'Predicted Profit': '#3b82f6'})
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("กรุณากรอกข้อมูลและกดปุ่มด้านซ้ายเพื่อดูผลการทำนาย")

# --- TAB 2: วิเคราะห์โมเดล (Insights) ---
with tab2:
    st.subheader("💡 AI Decision Insight: ปัจจัยไหนสำคัญที่สุด?")
    st.write("กราฟนี้แสดงว่า AI ให้ความสำคัญกับข้อมูลตัวไหนในการประมวลผลกำไร")

    # ข้อมูลความสำคัญของปัจจัย (อ้างอิงจากโมเดลของคุณ)
    importance_df = pd.DataFrame({
        'Factors': ['Sales', 'Discount', 'Shipping Cost', 'Quantity'],
        'Importance (%)': [48, 28, 15, 9]
    }).sort_values('Importance (%)', ascending=True)

    fig_imp = px.bar(importance_df, x='Importance (%)', y='Factors', orientation='h',
                     text='Importance (%)', color='Importance (%)', color_continuous_scale='Blues')
    fig_imp.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    ### 📝 สรุปสิ่งที่ได้เรียนรู้จากข้อมูล (Key Findings)
    * **ยอดขาย (Sales):** เป็นตัวแปรที่มีอิทธิพลสูงสุดต่อผลกำไร
    * **ส่วนลด (Discount):** เป็นตัวแปรที่มีความอ่อนไหวสูงมาก การให้ส่วนลดที่มากเกินไปส่งผลลบต่อกำไรอย่างรวดเร็ว
    * **ความคุ้มค่า:** การเพิ่มยอดขายเพียงอย่างเดียวโดยไม่คุมส่วนลดและค่าขนส่ง อาจไม่ช่วยให้กำไรเพิ่มขึ้นเสมอไป
    """)

# --- ส่วนท้าย (Footer) ---
st.divider()
st.caption("Machine Learning Deployment Project | Created by Your Name | Data Source: SuperStore Dataset")