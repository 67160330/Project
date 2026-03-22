import streamlit as st
import pandas as pd
import joblib  # ใช้ joblib แทน pickle ทั้งหมด

# 1. โหลดโมเดล (ตรวจสอบชื่อไฟล์ให้ตรงกับใน GitHub)
model = joblib.load('profit_model.pkl')

st.title("💰 SuperStore Profit Predictor")
st.write("เครื่องมือพยากรณ์กำไรจากข้อมูลยอดขาย")

# 2. สร้างช่องรับข้อมูล (Input)
sales = st.number_input("ยอดขาย (Sales $)", value=0.0)
quantity = st.number_input("จำนวนชิ้น (Quantity)", value=1, step=1)
discount = st.slider("ส่วนลด (Discount)", 0.0, 0.8, 0.0)
shipping = st.number_input("ค่าขนส่ง (Shipping Cost)", value=0.0)

# 3. ปุ่มกดทำนายผล
if st.button("พยากรณ์กำไร"):
    # สร้าง DataFrame ให้มีชื่อคอลัมน์และลำดับเหมือนตอนที่เทรนใน Colab
    input_data = pd.DataFrame([[sales, quantity, discount, shipping]],
                              columns=['sales', 'quantity', 'discount', 'shipping_cost'])

    # พยากรณ์ผล
    prediction = model.predict(input_data)

    # แสดงผลลัพธ์
    st.success(f"กำไรที่คาดการณ์ได้คือ: ${prediction[0]:,.2f}")