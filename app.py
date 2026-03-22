import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. Page Config - ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="SuperStore Profit Intelligence", page_icon="💰", layout="wide")


# 2. Load Model
@st.cache_resource
def load_assets():
    # ใช้ joblib โหลดโมเดลที่เราเทรนมา
    model = joblib.load('profit_model.pkl')
    return model


model = load_assets()

# --- ส่วนหัว (Header) แบบ Professional ---
st.markdown("""
    <div style='background-color:#0f172a;padding:30px;border-radius:15px;margin-bottom:25px;border-left: 10px solid #3b82f6;'>
        <h1 style='color:white;margin:0;font-family:sans-serif;'>📊 SuperStore Business Intelligence</h1>
        <p style='color:#94a3b8;margin:5px 0 0 0;font-size:1.1em;'>ระบบพยากรณ์กำไรและวิเคราะห์ปัจจัยสำคัญด้วย Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# --- แบ่งส่วนการทำงานด้วย Tabs ---
tab1, tab2 = st.tabs(["🚀 ระบบพยากรณ์กำไร", "🔍 บทวิเคราะห์โมเดล (Insights)"])

with tab1:
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("📥 ข้อมูลรายการขาย")
        with st.container(border=True):
            sales = st.number_input("ยอดขาย (Sales $)", min_value=0.0, value=250.0,
                                    help="ราคาสินค้าทั้งหมดก่อนหักค่าใช้จ่าย")
            quantity = st.number_input("จำนวนสินค้า (Quantity)", min_value=1, value=5)
            discount = st.slider("ส่วนลด (Discount Rate)", 0.0, 1.0, 0.1, help="0.1 คือ ลด 10%")
            shipping = st.number_input("ค่าขนส่ง (Shipping Cost $)", min_value=0.0, value=20.0)

            st.markdown("---")
            predict_btn = st.button("คำนวณกำไรคาดการณ์", use_container_width=True, type="primary")

    with col2:
        st.subheader("🎯 ผลการพยากรณ์")
        if predict_btn:
            # เตรียมข้อมูลส่งให้ AI
            input_df = pd.DataFrame([[sales, quantity, discount, shipping]],
                                    columns=['sales', 'quantity', 'discount', 'shipping_cost'])
            prediction = model.predict(input_df)[0]

            # คำนวณ Margin
            margin = (prediction / sales) * 100 if sales > 0 else 0

            # แสดงผลแบบ Metrics
            c1, c2 = st.columns(2)
            c1.metric("กำไรจากการคาดการณ์ (Predicted Profit)", f"${prediction:,.2f}")
            c2.metric("อัตรากำไรสุทธิ (Profit Margin)", f"{margin:.2f}%")

            if prediction > 0:
                st.success(f"**สรุป:** รายการนี้มีแนวโน้มได้รับผลกำไรประมาณ **${prediction:,.2f}**")
            else:
                st.error(f"**สรุป:** รายการนี้มีความเสี่ยงที่จะขาดทุนประมาณ **${abs(prediction):,.2f}**")

            # สร้างกราฟแท่งเปรียบเทียบ Profit vs Sales
            st.bar_chart(pd.DataFrame({'Values': [sales, prediction]}, index=['Total Sales', 'Predicted Profit']))
        else:
            st.info("กรุณาระบุข้อมูลที่แถบด้านซ้าย แล้วกดปุ่มเพื่อดูผลลัพธ์")

with tab2:
    st.subheader("💡 AI Decision Making: ตัวแปรไหนสำคัญที่สุด?")

    # ดึงข้อมูลจากรูปที่คุณแคปมาใส่เป็นกราฟในแอปเลย
    importance_data = pd.DataFrame({
        'Features': ['Sales', 'Discount', 'Shipping Cost', 'Quantity'],
        'Importance Score': [0.48, 0.28, 0.15, 0.09]  # ตัวเลขโดยประมาณจากรูปของคุณ
    }).sort_values('Importance Score', ascending=True)

    fig = px.bar(importance_data, x='Importance Score', y='Features', orientation='h',
                 title="Feature Importance (Random Forest Regressor)",
                 color='Importance Score', color_continuous_scale='Blues')

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **คำอธิบายกราฟ:**
    * **ยอดขาย (Sales):** เป็นปัจจัยหลักที่โมเดลใช้คำนวณกำไร (มีอิทธิพลสูงสุด)
    * **ส่วนลด (Discount):** เป็นตัวแปรที่ส่งผลกระทบอย่างมีนัยสำคัญต่อการลดลงของกำไร
    * **ค่าขนส่ง (Shipping Cost):** มีผลในลำดับถัดมา ซึ่งโมเดลนำมาหักลบเพื่อให้ได้กำไรที่แม่นยำขึ้น
    """)

# --- Footer ---
st.divider()
st.caption("Machine Learning Deployment Project | Algorithm: Random Forest Regressor")