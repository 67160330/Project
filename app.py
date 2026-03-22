import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. การตั้งค่าหน้าเว็บ (Web Configuration)
st.set_page_config(
    page_title="SuperStore Profit Intelligence",
    page_icon="💰",
    layout="wide"
)


# 2. ฟังก์ชันโหลดตัวแบบพยากรณ์ (Load Model)
@st.cache_resource
def load_model():
    # ตรวจสอบว่ามีไฟล์ profit_model.pkl อยู่ใน Repository
    return joblib.load('profit_model.pkl')


model = load_model()

# --- ส่วนหัวของแอปพลิเคชัน (Header Section) ---
st.markdown("""
    <div style='background-color:#1e293b; padding:30px; border-radius:15px; margin-bottom:25px; border-left: 10px solid #3b82f6;'>
        <h1 style='color:white; margin:0; font-family:sans-serif;'>📊 SuperStore Business Intelligence Dashboard</h1>
        <p style='color:#cbd5e1; margin:5px 0 0 0; font-size:1.1em;'>ระบบวิเคราะห์และพยากรณ์กำไรอัจฉริยะด้วยเทคโนโลยี Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# --- การแบ่งส่วนการทำงานด้วย Tabs ---
tab1, tab2 = st.tabs(["🚀 การพยากรณ์กำไร (Profit Prediction)", "🔍 ข้อมูลเชิงลึกของโมเดล (Model Insights)"])

# --- TAB 1: ระบบพยากรณ์ผลตอบแทน ---
with tab1:
    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        st.subheader("📥 ป้อนข้อมูลปัจจัยการขาย")
        with st.container(border=True):
            sales = st.number_input("ยอดขายรวม (Sales $)", min_value=0.0, value=250.0,
                                    help="ราคาสินค้ารวมก่อนหักค่าใช้จ่าย")
            quantity = st.number_input("จำนวนสินค้า (Quantity)", min_value=1, value=5)
            discount = st.slider("อัตราส่วนลด (Discount Rate)", 0.0, 1.0, 0.1, help="0.10 หมายถึง ส่วนลด 10%")
            shipping = st.number_input("ต้นทุนการจัดส่ง (Shipping Cost $)", min_value=0.0, value=20.0)

            st.markdown("---")
            predict_btn = st.button("ประมวลผลการพยากรณ์", use_container_width=True, type="primary")

    with col2:
        st.subheader("🎯 ผลการวิเคราะห์และประมาณการ")
        if predict_btn:
            # การเตรียมข้อมูลนำเข้า (Data Preprocessing)
            input_df = pd.DataFrame([[sales, quantity, discount, shipping]],
                                    columns=['sales', 'quantity', 'discount', 'shipping_cost'])

            # การพยากรณ์ด้วยโมเดล
            prediction = model.predict(input_df)[0]

            # การคำนวณอัตรากำไรสุทธิ (Profit Margin)
            margin = (prediction / sales) * 100 if sales > 0 else 0

            # การแสดงตัวชี้วัดหลัก (Key Metrics)
            m1, m2 = st.columns(2)
            m1.metric("กำไรคาดการณ์ (Predicted Profit)", f"${prediction:,.2f}")
            m2.metric("อัตรากำไรสุทธิ (Profit Margin)", f"{margin:.2f}%")

            # การวิเคราะห์สถานะทางธุรกิจ
            if prediction > 0:
                st.success(f"**สถานะ:** รายการนี้มีแนวโน้มได้รับผลกำไรสุทธิประมาณ **${prediction:,.2f}**")
                if margin > 15:
                    st.info(
                        "💡 **กลยุทธ์แนะนำ:** รายการนี้มีอัตรากำไรสูง (High Margin) ควรส่งเสริมกิจกรรมทางการขายในลักษณะนี้")
                else:
                    st.info(
                        "💡 **กลยุทธ์แนะนำ:** กำไรอยู่ในเกณฑ์มาตรฐาน ควรบริหารจัดการต้นทุนผันแปรเพื่อเพิ่มประสิทธิภาพ")
            else:
                st.error(f"**สถานะ:** รายการนี้มีความเสี่ยงต่อการขาดทุนสุทธิประมาณ **${abs(prediction):,.2f}**")
                st.warning(
                    "⚠️ **คำเตือนเชิงกลยุทธ์:** พิจารณาปรับลดอัตราส่วนลดหรือปรับโครงสร้างราคาสินค้าเพื่อรักษาเสถียรภาพของกำไร")

            # กราฟเปรียบเทียบสัดส่วน (Comparison Chart)
            st.write("---")
            chart_data = pd.DataFrame({
                'ประเภท': ['ยอดขายรวม', 'กำไรที่คาดการณ์'],
                'มูลค่า ($)': [sales, prediction]
            })
            fig_bar = px.bar(chart_data, x='ประเภท', y='มูลค่า ($)', color='ประเภท',
                             color_discrete_map={'ยอดขายรวม': '#94a3b8', 'กำไรที่คาดการณ์': '#3b82f6'},
                             text_auto='.2s')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("ระบบพร้อมใช้งาน: กรุณาระบุข้อมูลปัจจัยการขายที่แถบด้านซ้ายและกดปุ่มเพื่อประมวลผล")

# --- TAB 2: บทวิเคราะห์ความสำคัญของปัจจัย (Model Insights) ---
with tab2:
    st.subheader("💡 การวิเคราะห์ลำดับความสำคัญของปัจจัย (Feature Importance)")
    st.write("การแสดงผลน้ำหนักความสำคัญที่ตัวแบบ Machine Learning ใช้ในการตัดสินใจพยากรณ์ผลกำไร")

    # ข้อมูล Feature Importance จากผลการเทรนโมเดล
    importance_df = pd.DataFrame({
        'ปัจจัยการขาย (Features)': ['ยอดขาย (Sales)', 'ส่วนลด (Discount)', 'ค่าขนส่ง (Shipping Cost)',
                                    'จำนวนสินค้า (Quantity)'],
        'น้ำหนักความสำคัญ (%)': [48, 28, 15, 9]
    }).sort_values('น้ำหนักความสำคัญ (%)', ascending=True)

    fig_imp = px.bar(importance_df, x='น้ำหนักความสำคัญ (%)', y='ปัจจัยการขาย (Features)',
                     orientation='h', text='น้ำหนักความสำคัญ (%)',
                     color='น้ำหนักความสำคัญ (%)', color_continuous_scale='Blues')

    fig_imp.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.subheader("📝 บทสรุปเชิงวิเคราะห์ (Analytical Insights)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **1. อิทธิพลของยอดขาย (Revenue Impact)**
        * ยอดขายเป็นปัจจัยหลักที่มีผลกระทบต่อกำไรสูงที่สุด (48%) สะท้อนให้เห็นว่าขนาดของรายได้เป็นรากฐานสำคัญของผลประกอบการ
        """)

    with c2:
        st.markdown("""
        **2. ความอ่อนไหวต่อส่วนลด (Discount Sensitivity)**
        * ส่วนลดมีอิทธิพลสูงถึง 28% ชี้ให้เห็นว่าการปรับเปลี่ยนนโยบายส่วนลดเพียงเล็กน้อย ส่งผลกระทบเชิงลบต่อกำไรอย่างรวดเร็ว
        """)

# --- ส่วนท้ายของหน้าเว็บ (Footer) ---
st.divider()
st.caption("© 2024 SuperStore Business Intelligence Project | พัฒนาด้วย Scikit-Learn และ Streamlit")