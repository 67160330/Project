import streamlit as st
import pandas as pd
import pickle

# โหลดโมเดล
model = pickle.load(open('profit_model.pkl', 'rb'))
cols = pickle.load(open('model_columns.pkl', 'rb'))

st.title("💰 SuperStore Profit Predictor")

# สร้างช่องรับข้อมูล (ตัวอย่าง)
sales = st.number_input("Sales ($)", value=0.0)
discount = st.slider("Discount", 0.0, 0.8, 0.1)
shipping = st.number_input("Shipping Cost", value=0.0)

if st.button("Predict Profit"):
    # ทำข้อมูลที่รับมาให้เป็น DataFrame เหมือนตอน Train
    # (ต้องเขียนโค้ดจัดการ One-hot encoding ให้ตรงกับตอนเทรนด้วย)
    input_data = pd.DataFrame([[sales, discount, shipping]], columns=['sales', 'discount', 'shipping_cost'])

    # พยากรณ์
    prediction = model.predict(input_data)
    st.success(f"Estimated Profit: ${prediction[0]:,.2f}")