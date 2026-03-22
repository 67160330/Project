import streamlit as st
import pandas as pd
import joblib

# 1. ตั้งค่าหน้าเว็บ (ชื่อและไอคอนบนแท็บเบราว์เซอร์)
st.set_page_config(page_title="SuperStore Profit Predictor", page_icon="💰", layout="wide")


# 2. โหลดโมเดล
@st.cache_resource  # ช่วยให้โหลดเร็วขึ้น
def load_model():
    return joblib.load('profit_model.pkl')


model = load_model()

# --- ส่วนหัวของหน้าเว็บ ---
st.title("📊 SuperStore Profit Analysis & Prediction")
st.markdown("ระบบพยากรณ์กำไรจากยอดขายและค่าขนส่ง โดยใช้ Machine Learning (Random Forest)")
st.divider()  # เส้นคั่น

# --- ส่วนของการรับข้อมูล (Sidebar หรือ Columns) ---
col1, col2 = st.columns([1, 2])  # แบ่งเป็น 2 คอลัมน์ (ซ้ายมือป้อนข้อมูล / ขวามือแสดงผล)

with col1:
    st.subheader("🛠️ Input Features")
    with st.expander("กรอกข้อมูลการขายตรงนี้", expanded=True):
        sales = st.number_input("Sales ($)", min_value=0.0, value=100.0, step=10.0)
        quantity = st.slider("Quantity (ชิ้น)", min_value=1, max_value=100, value=5)
        discount = st.slider("Discount (ส่วนลด)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        shipping_cost = st.number_input("Shipping Cost ($)", min_value=0.0, value=10.0, step=1.0)

    predict_btn = st.button("🚀 Predict Profit", use_container_width=True, type="primary")

# --- ส่วนของการแสดงผลการพยากรณ์ ---
with col2:
    st.subheader("📈 Prediction Result")
    if predict_btn:
        # เตรียมข้อมูล
        input_data = pd.DataFrame([[sales, quantity, discount, shipping_cost]],
                                  columns=['sales', 'quantity', 'discount', 'shipping_cost'])

        # ทำนายผล
        prediction = model.predict(input_data)[0]

        # แสดงผลแบบ Card สวยๆ
        st.write("---")
        if prediction >= 0:
            st.success(f"### Estimated Profit: ${prediction:,.2f}")
            st.balloons()  # ใส่ Effect ดีใจถ้าได้กำไร
        else:
            st.error(f"### Estimated Loss: ${prediction:,.2f}")
            st.warning("คำเตือน: รายการนี้มีแนวโน้มจะขาดทุน!")

        # เพิ่มกราฟจำลองเพื่อให้ดูเป็น Dashboard
        chart_data = pd.DataFrame({"Category": ["Sales", "Shipping Cost", "Profit"],
                                   "Value": [sales, shipping_cost, prediction]})
        st.bar_chart(data=chart_data, x="Category", y="Value")
    else:
        st.info("👈 กรุณากรอกข้อมูลและกดปุ่ม Predict เพื่อดูผลลัพธ์")

# --- ส่วนล่างสุด (Footer) ---
st.divider()
st.caption("Developed for ML Deployment Project | Dataset: Global SuperStore")