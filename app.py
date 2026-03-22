import streamlit as st
import pandas as pd
import joblib  # แก้จุดที่ 1: เปลี่ยนจาก import pickle เป็น joblib

# แก้จุดที่ 2: การโหลดโมเดล (ใช้ joblib.load และไม่ต้องมี open(...) )
model = joblib.load('profit_model.pkl')

st.title("💰 SuperStore Profit Predictor")

# สร้างช่องรับข้อมูล
sales = st.number_input("Sales ($)", value=0.0)
quantity = st.number_input("Quantity", value=1, step=1) # เพิ่มอันนี้เพื่อให้ตรงกับโมเดล
discount = st.slider("Discount", 0.0, 0.8, 0.1)
shipping = st.number_input("Shipping Cost", value=0.0)

if st.button("Predict Profit"):
    # ทำข้อมูลที่รับมาให้เป็น DataFrame (เรียงคอลัมน์ให้ตรงกับที่เทรนมา)
    input_data = pd.DataFrame([[sales, quantity, discount, shipping]],
                              columns=['sales', 'quantity', 'discount', 'shipping_cost'])

    # พยากรณ์
    prediction = model.predict(input_data)
    st.success(f"Estimated Profit: ${prediction[0]:,.2f}")