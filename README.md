# 📊 SuperStore Profit Analytics (ระบบวิเคราะห์และพยากรณ์กำไรอัจฉริยะ)

**SuperStore Profit Analytics** คือ Web Application สำหรับสนับสนุนการตัดสินใจทางธุรกิจ (Decision Support System) ที่ขับเคลื่อนด้วยเทคโนโลยี Machine Learning ระบบนี้ถูกพัฒนาขึ้นเพื่อช่วยให้ผู้บริหารหรือเจ้าของกิจการสามารถพยากรณ์ผลกำไร/ขาดทุนล่วงหน้า จากการกำหนดตัวแปรต่างๆ เช่น ยอดขาย ส่วนลด และค่าขนส่ง เพื่อลดความเสี่ยงในการจัดโปรโมชันที่อาจทำให้เกิดภาวะขาดทุน

🌐 **Live Demo:** [ใส่ลิงก์ Streamlit App ของคุณที่นี่]

---

## 🌟 ฟีเจอร์หลัก (Key Features)

แอปพลิเคชันแบ่งออกเป็น 3 ส่วนหลัก ดังนี้:

1. **🚀 Profit Prediction (ระบบพยากรณ์กำไร):** - รับค่าตัวแปรนำเข้า (Sales, Quantity, Discount, Shipping Cost)
   - พยากรณ์ผลกำไรสุทธิ (Predicted Profit) และอัตรากำไร (Profit Margin)
   - มีระบบแจ้งเตือนความเสี่ยง (Risk Analysis) หากกำหนดส่วนลดหรือค่าขนส่งสูงเกินไป
2. **🔍 Feature Insights (ข้อมูลเชิงลึกของปัจจัย):** - แสดงผลการวิเคราะห์ Feature Importance จากโมเดล
   - อธิบายว่าปัจจัยใดมีอิทธิพลต่อผลกำไรมากที่สุด (Explainable AI)
3. **📈 Model Evaluation (ประสิทธิภาพโมเดล):** - แสดงผลการประเมินความแม่นยำของตัวแบบด้วยข้อมูล Test Set
   - เปรียบเทียบประสิทธิภาพระหว่างโมเดล Linear Regression และ Random Forest

---

## 🧠 ข้อมูลเชิงลึกของโมเดล (Machine Learning Model)

จากการทดสอบและเปรียบเทียบอัลกอริทึม โปรเจคนี้ได้ตัดสินใจเลือกใช้ **Random Forest Regressor** แทนที่ Linear Regression ด้วยเหตุผลดังนี้:

| อัลกอริทึม (Algorithm) | MAE (ความคลาดเคลื่อนเฉลี่ย) | R-Squared (ความอธิบายข้อมูล) |
| ---------------------- | --------------------------- | ----------------------------- |
| Linear Regression      | $59.76                      | 38.26%                        |
| **Random Forest** | **$37.09** | **67.94%** |

**เหตุผลทางสถิติ:**
ข้อมูลทางธุรกิจชุดนี้มีความสัมพันธ์ที่ **ไม่เป็นเส้นตรง (Non-linear Relationship)** โดยเฉพาะอย่างยิ่งตัวแปร **ส่วนลด (Discount)** ที่มีผลกระทบต่อกำไรอย่างรวดเร็วเมื่อเกินจุดคุ้มทุน โมเดล Random Forest ที่ใช้หลักการ Decision Trees จึงสามารถเรียนรู้รูปแบบที่ซับซ้อนนี้ได้ดีกว่า ส่งผลให้ความแม่นยำเพิ่มขึ้นอย่างมีนัยสำคัญ

---

## 🛠️ เทคโนโลยีที่ใช้ (Tech Stack)

* **Language:** Python
* **Machine Learning:** Scikit-Learn (`RandomForestRegressor`)
* **Web Framework:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Plotly Express

---
