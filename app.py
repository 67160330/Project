import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

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
    <div style='background-color:#0f172a;padding:30px;border-radius:15px;margin-bottom:25px;border-left: 10px solid #3b82f6;'>
        <h1 style='color:white;margin:0;'>📊 SuperStore Business Intelligence</h1>
        <p style='color:#94a3b8;margin:5px 0 0 0;font-size:1.1em;'>ระบบวิเคราะห์และพยากรณ์กำไรด้วย Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2 = st.tabs(["🚀 พยากรณ์กำไร (Prediction)", "🔍 เจาะลึกการตัดสินใจของ AI (Insights)"])

with tab1:
    col1, col2 = st.columns([1, 1.2], gap="large")
    with col1:
        st.subheader("📥 ข้อมูลรายการขาย")
        with st.container(border=True):
            sales = st.number_input("ยอดขาย (Sales $)", min_value=0.0, value=250.0)
            quantity = st.number_input("จำนวนสินค้า (Quantity)", min_value=1, value=5)
            discount = st.slider("ส่วนลด (Discount Rate)", 0.0, 1.0, 0.1)
            shipping = st.number_input("ค่าขนส่ง (Shipping Cost $)", min_value=0.0, value=20.0)
            predict_btn = st.button("คำนวณกำไร", use_container_width=True, type="primary")

    with col2:
        st.subheader("🎯 ผลการทำนาย")
        if predict_btn:
            input_df = pd.DataFrame([[sales, quantity, discount, shipping]],
                                    columns=['sales', 'quantity', 'discount', 'shipping_cost'])
            prediction = model.predict(input_df)[0]
            margin = (prediction / sales) * 100 if sales > 0 else 0

            c1, c2 = st.columns(2)
            c1.metric("Predicted Profit", f"${prediction:,.2f}")
            c2.metric("Profit Margin", f"{margin:.2f}%")

            if prediction > 0:
                st.success(f"✅ รายการนี้มีแนวโน้มได้กำไรประมาณ ${prediction:,.2f}")
            else:
                st.error(f"⚠️ รายการนี้มีความเสี่ยงขาดทุนประมาณ ${abs(prediction):,.2f}")
        else:
            st.info("รอผลการพยากรณ์... (กรุณากรอกข้อมูลด้านซ้ายแล้วกดปุ่ม)")

with tab2:
    st.subheader("💡 ทำไม AI ถึงทำนายผลแบบนั้น?")
    st.write("กราฟด้านล่างแสดงให้เห็นว่า **ตัวแปรไหนมีผลต่อการคำนวณกำไรมากที่สุด**")

    # ข้อมูล Feature Importance (อ้างอิงจากกราฟที่คุณส่งมา)
    importance_df = pd.DataFrame({
        'ปัจจัย (Features)': ['Sales', 'Discount', 'Shipping Cost', 'Quantity'],
        'น้ำหนักความสำคัญ (%)': [48, 28, 15, 9]
    }).sort_values('น้ำหนักความสำคัญ (%)', ascending=True)

    # สร้างกราฟสวยๆ ด้วย Plotly
    fig = px.bar(importance_df, x='น้ำหนักความสำคัญ (%)', y='ปัจจัย (Features)',
                 orientation='h', text='น้ำหนักความสำคัญ (%)',
                 color='น้ำหนักความสำคัญ (%)', color_continuous_scale='Blues')

    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(showlegend=False, height=400)

    st.plotly_chart(fig, use_container_width=True)

    # ส่วนอธิบาย Insight แบบเข้าใจง่าย (Bullet Points)
    st.markdown("---")
    st.subheader("📝 สรุป Insight สำหรับนักธุรกิจ")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **1. ยอดขาย (Sales) คือหัวใจหลัก**
        * AI ให้ความสำคัญกับยอดขายเกือบ 50% เพราะเป็นตัวกำหนดขนาดกำไรเบื้องต้น

        **2. ส่วนลด (Discount) คือตัวแปรอันตราย**
        * มีผลต่อการทำนายสูงถึง 28% หมายความว่าการลดราคาเพียงเล็กน้อย ส่งผลกระทบต่อกำไรอย่างรวดเร็ว
        """)

    with c2:
        st.markdown("""
        **3. ค่าขนส่ง (Shipping) ต้องระวัง**
        * เป็นต้นทุนแฝงที่ AI นำมาคำนวณเพื่อหักลบผลกำไรสุทธิ

        **4. ระบบนี้ช่วยอะไร?**
        * ช่วยให้เจ้าของร้านรู้ว่า 'ไม่ควรลดราคาจนเกินไป' แม้ยอดขายจะเยอะ แต่ถ้า Discount สูง AI จะเตือนทันทีว่าจะขาดทุน
        """)

st.divider()
st.caption("AI Model: Random Forest Regressor | Created by [ชื่อของคุณ]")