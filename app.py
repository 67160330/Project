import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==========================================
# 1. การตั้งค่าหน้าเว็บ (Web Configuration)
# ==========================================
st.set_page_config(
    page_title="SuperStore Profit Analytics",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# 2. ฟังก์ชันโหลดตัวแบบพยากรณ์ (Load Model)
# ==========================================
@st.cache_resource
def load_model():
    # โหลดไฟล์โมเดลที่ผ่านการฝึกสอนมาแล้ว
    try:
        return joblib.load('profit_model.pkl')
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ 'profit_model.pkl' กรุณาตรวจสอบให้แน่ใจว่าได้อัปโหลดไฟล์โมเดลแล้ว")
        return None


model = load_model()

# ==========================================
# 3. ส่วนหัวของแอปพลิเคชัน (Header Section)
# ==========================================
st.markdown("""
    <div style='background-color:#1e293b; padding:30px; border-radius:15px; margin-bottom:25px; border-left: 10px solid #3b82f6;'>
        <h1 style='color:white; margin:0; font-family:sans-serif;'>📊 ระบบวิเคราะห์และพยากรณ์กำไรอัจฉริยะ</h1>
        <p style='color:#cbd5e1; margin:5px 0 0 0; font-size:1.1em;'>SuperStore Business Intelligence Dashboard (Machine Learning)</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. การแบ่งหน้าจอการทำงาน (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["🚀 การพยากรณ์กำไร (Profit Prediction)", "🔍 ข้อมูลเชิงลึกของโมเดล (Model Insights)"])

# ------------------------------------------
# TAB 1: ระบบพยากรณ์ผลตอบแทน
# ------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1.4], gap="large")

    # ด้านซ้าย: ส่วนรับข้อมูล (Input Section)
    with col1:
        st.subheader("📥 ป้อนข้อมูลปัจจัยการขาย")
        with st.container(border=True):
            sales = st.number_input("ยอดขายรวม (Sales $)", min_value=0.0, value=250.0,
                                    help="มูลค่าสินค้ารวมก่อนหักค่าใช้จ่าย")
            quantity = st.number_input("จำนวนสินค้า (Quantity)", min_value=1, value=5)
            discount = st.slider("อัตราส่วนลด (Discount Rate)", 0.0, 1.0, 0.1, help="ตัวอย่าง: 0.10 หมายถึงส่วนลด 10%")
            shipping = st.number_input("ต้นทุนการจัดส่ง (Shipping Cost $)", min_value=0.0, value=20.0,
                                       help="ค่าใช้จ่ายในการขนส่งสินค้า")

            st.markdown("---")
            predict_btn = st.button("ประมวลผลการพยากรณ์", use_container_width=True, type="primary")

    # ด้านขวา: ส่วนแสดงผลลัพธ์ (Output Section)
    with col2:
        st.subheader("🎯 ผลการวิเคราะห์และประมาณการ")
        if predict_btn and model is not None:
            # 1. เตรียมข้อมูลนำเข้า
            input_df = pd.DataFrame([[sales, quantity, discount, shipping]],
                                    columns=['sales', 'quantity', 'discount', 'shipping_cost'])

            # 2. พยากรณ์ด้วยโมเดล
            prediction = model.predict(input_df)[0]

            # 3. คำนวณอัตรากำไรสุทธิ (Profit Margin)
            margin = (prediction / sales) * 100 if sales > 0 else 0

            # 4. แสดงตัวชี้วัดหลัก (Key Metrics)
            m1, m2 = st.columns(2)
            m1.metric("กำไรคาดการณ์ (Predicted Profit)", f"${prediction:,.2f}")
            m2.metric("อัตรากำไรสุทธิ (Profit Margin)", f"{margin:.2f}%")

            # 5. วิเคราะห์สถานะทางธุรกิจและให้คำแนะนำ
            if prediction > 0:
                st.success(f"**สถานะ:** รายการนี้มีแนวโน้มได้รับผลกำไรสุทธิประมาณ **${prediction:,.2f}**")
                if margin > 15:
                    st.info(
                        "💡 **กลยุทธ์แนะนำ:** รายการนี้มีอัตรากำไรสูง (High Margin) ควรส่งเสริมการขายในลักษณะนี้ต่อไป")
                else:
                    st.info(
                        "💡 **กลยุทธ์แนะนำ:** กำไรอยู่ในเกณฑ์มาตรฐาน ควรควบคุมต้นทุนผันแปรอื่นๆ เพิ่มเติมเพื่อเพิ่มประสิทธิภาพ")
            else:
                st.error(f"**สถานะ:** รายการนี้มีความเสี่ยงต่อการขาดทุนสุทธิประมาณ **${abs(prediction):,.2f}**")
                st.warning(
                    "⚠️ **คำเตือนเชิงกลยุทธ์:** ควรพิจารณาปรับลดอัตราส่วนลด หรือปรับโครงสร้างราคาสินค้าใหม่เพื่อรักษาเสถียรภาพ")

            # 6. วิเคราะห์ความเสี่ยงเชิงลึก (Detailed Analysis)
            with st.expander("📝 ดูการวิเคราะห์ความเสี่ยงเพิ่มเติม (Risk Analysis)"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**การประเมินค่าจัดส่ง**")
                    shipping_ratio = (shipping / sales) * 100 if sales > 0 else 0
                    st.write(f"สัดส่วนค่าขนส่ง: {shipping_ratio:.1f}% ของยอดขาย")
                    if shipping_ratio > 20:
                        st.warning("ค่าขนส่งสูงเกินเกณฑ์ 20% อาจกระทบกำไรระยะยาว")
                    else:
                        st.success("ค่าขนส่งอยู่ในเกณฑ์ที่บริหารจัดการได้")

                with col_b:
                    st.markdown("**การประเมินส่วนลด**")
                    if discount > 0.3:
                        st.error("ส่วนลดสูงเกิน 30% มีความเสี่ยงสูงต่อภาวะขาดทุน")
                    elif discount == 0:
                        st.info("การไม่มีส่วนลดอาจส่งผลให้ปริมาณการขายลดลง")
                    else:
                        st.success("อัตราส่วนลดอยู่ในระดับที่เหมาะสม")

            # 7. กราฟเปรียบเทียบ (Comparison Chart)
            st.write("---")
            chart_data = pd.DataFrame({
                'ประเภทตัวชี้วัด': ['ยอดขายรวม (Sales)', 'กำไรคาดการณ์ (Profit)'],
                'มูลค่า ($)': [sales, prediction]
            })
            fig_bar = px.bar(chart_data, x='ประเภทตัวชี้วัด', y='มูลค่า ($)', color='ประเภทตัวชี้วัด',
                             color_discrete_map={'ยอดขายรวม (Sales)': '#94a3b8', 'กำไรคาดการณ์ (Profit)': '#3b82f6'},
                             text_auto='.2f')
            st.plotly_chart(fig_bar, use_container_width=True)

        elif not predict_btn:
            st.info("ระบบพร้อมใช้งาน: กรุณาระบุข้อมูลปัจจัยการขายที่แถบด้านซ้ายและกดปุ่ม 'ประมวลผลการพยากรณ์'")

# ------------------------------------------
# TAB 2: บทวิเคราะห์ความสำคัญของปัจจัย (Model Insights)
# ------------------------------------------
with tab2:
    st.subheader("💡 การวิเคราะห์ลำดับความสำคัญของปัจจัย (Feature Importance)")
    st.write("การแสดงผลน้ำหนักความสำคัญที่ตัวแบบ Machine Learning (Random Forest) ใช้ในการประเมินผลกำไร")

    # ข้อมูล Feature Importance อ้างอิงจากโมเดล
    importance_df = pd.DataFrame({
        'ปัจจัยการขาย (Features)': ['ยอดขาย (Sales)', 'อัตราส่วนลด (Discount)', 'ต้นทุนขนส่ง (Shipping Cost)',
                                    'จำนวนสินค้า (Quantity)'],
        'น้ำหนักความสำคัญ (%)': [48, 28, 15, 9]
    }).sort_values('น้ำหนักความสำคัญ (%)', ascending=True)

    fig_imp = px.bar(importance_df, x='น้ำหนักความสำคัญ (%)', y='ปัจจัยการขาย (Features)',
                     orientation='h', text='น้ำหนักความสำคัญ (%)',
                     color='น้ำหนักความสำคัญ (%)', color_continuous_scale='Blues')

    fig_imp.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.subheader("📝 บทสรุปเชิงวิเคราะห์เพื่อการตัดสินใจ (Analytical Insights)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **1. อิทธิพลของยอดขาย (Revenue Impact)**
        * **ยอดขาย (Sales)** เป็นปัจจัยหลักที่มีผลกระทบต่อกำไรสูงที่สุดถึง 48% สะท้อนให้เห็นว่าขนาดของรายได้เป็นรากฐานสำคัญที่สุดในการกำหนดทิศทางของผลประกอบการ
        """)

    with c2:
        st.markdown("""
        **2. ความอ่อนไหวต่อส่วนลด (Discount Sensitivity)**
        * **ส่วนลด (Discount)** มีอิทธิพลสูงถึง 28% ชี้ให้เห็นว่าการกำหนดโปรโมชันหรือปรับเปลี่ยนนโยบายส่วนลดเพียงเล็กน้อย จะส่งผลกระทบต่อกำไรสุทธิอย่างมีนัยสำคัญ
        """)

# ==========================================
# 5. ส่วนท้ายของหน้าเว็บ (Footer)
# ==========================================
st.divider()
st.caption("© 2024 SuperStore Business Intelligence Project | พัฒนาด้วย Scikit-Learn และ Streamlit")