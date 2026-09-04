import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "random_forest.joblib")
REPORT_PATH = os.path.join(PROJECT_ROOT, "outputs", "reports", "evaluation_report.csv")

# Custom Title
st.title("🎓 Student Performance Prediction & Data Mining Dashboard")
st.markdown("Hệ thống phân tích hành vi học tập và dự đoán khả năng **Đậu / Rớt** môn học của học sinh.")

# Load Model & Scaler Reference
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_data():
    clean_file = os.path.join(DATA_DIR, "student_clean.csv")
    if os.path.exists(clean_file):
        return pd.read_csv(clean_file)
    return None

@st.cache_data
def load_train_features():
    xtrain_file = os.path.join(DATA_DIR, "X_train.csv")
    if os.path.exists(xtrain_file):
        return pd.read_csv(xtrain_file)
    return None

model = load_model()
df_clean = load_data()
X_train = load_train_features()

# Top Metrics Row
if df_clean is not None:
    total_students = len(df_clean)
    pass_count = (df_clean["pass"] == 1).sum()
    fail_count = (df_clean["pass"] == 0).sum()
    pass_rate = (pass_count / total_students) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số học sinh", f"{total_students}")
    col2.metric("Số lượng Đậu", f"{pass_count} ({pass_rate:.1f}%)")
    col3.metric("Số lượng Rớt", f"{fail_count} ({100 - pass_rate:.1f}%)")
    col4.metric("Mô hình tốt nhất", "Random Forest (AUC 0.72)")

st.divider()

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Dự đoán kết quả học tập",
    "📊 Biểu đồ phân tích (EDA & Mining)",
    "📈 Bảng xếp hạng mô hình",
    "📁 Khám phá tập dữ liệu"
])

# ----------------- TAB 1: PREDICTION -----------------
with tab1:
    st.subheader("Nhập thông tin học sinh để dự đoán nguy cơ Đậu/Rớt")
    st.markdown("Mô hình **Random Forest** sẽ tính toán dựa trên các đặc trưng nhân khẩu học, thói quen học tập và lịch sử kết quả.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.slider("Tuổi (age)", 15, 22, 17)
            failures = st.selectbox("Số lần trượt môn trước đó (failures)", [0, 1, 2, 3], index=0)
            studytime = st.selectbox("Thời gian tự học hàng tuần (studytime)", 
                                     options=[(1, "< 2 giờ"), (2, "2 - 5 giờ"), (3, "5 - 10 giờ"), (4, "> 10 giờ")],
                                     format_func=lambda x: x[1], index=1)[0]
            absences = st.slider("Số buổi vắng học (absences)", 0, 50, 4)
            higher = st.selectbox("Có dự định học lên cao không? (higher)", ["yes", "no"], index=0)
            
        with c2:
            school = st.selectbox("Trường học (school)", ["GP (Gabriel Pereira)", "MS (Mousinho da Silveira)"], index=0)
            sex = st.selectbox("Giới tính (sex)", ["F (Nữ)", "M (Nam)"], index=0)
            address = st.selectbox("Nơi ở (address)", ["U (Thành thị)", "R (Nông thôn)"], index=0)
            freetime = st.slider("Thời gian rảnh sau giờ học (1: rất ít -> 5: rất nhiều)", 1, 5, 3)
            goout = st.slider("Đi chơi với bạn bè (1: rất ít -> 5: rất nhiều)", 1, 5, 3)
            
        with c3:
            Medu = st.slider("Học vấn của mẹ (0: không -> 4: đại học)", 0, 4, 3)
            Fedu = st.slider("Học vấn của cha (0: không -> 4: đại học)", 0, 4, 3)
            health = st.slider("Tình trạng sức khỏe (1: rất kém -> 5: rất tốt)", 1, 5, 4)
            Dalc = st.slider("Uống rượu bia ngày thường (1: rất ít -> 5: nhiều)", 1, 5, 1)
            internet = st.selectbox("Nhà có kết nối Internet không?", ["yes", "no"], index=0)

        submit_btn = st.form_submit_button("🚀 Tiến hành Dự đoán", use_container_width=True)

    if submit_btn:
        if model is not None and X_train is not None:
            # Prepare single sample template matching X_train columns
            input_dict = {col: 0.0 for col in X_train.columns}
            
            # Numeric values
            input_dict["age"] = float(age)
            input_dict["failures"] = float(failures)
            input_dict["studytime"] = float(studytime)
            input_dict["absences"] = float(absences)
            input_dict["freetime"] = float(freetime)
            input_dict["goout"] = float(goout)
            input_dict["Medu"] = float(Medu)
            input_dict["Fedu"] = float(Fedu)
            input_dict["health"] = float(health)
            input_dict["Dalc"] = float(Dalc)
            input_dict["Walc"] = 1.0
            input_dict["famrel"] = 4.0
            input_dict["traveltime"] = 1.0

            # One-hot encoded categorical values
            if school.startswith("MS") and "school_MS" in input_dict:
                input_dict["school_MS"] = 1.0
            if sex.startswith("M") and "sex_M" in input_dict:
                input_dict["sex_M"] = 1.0
            if address.startswith("U") and "address_U" in input_dict:
                input_dict["address_U"] = 1.0
            if higher == "yes" and "higher_yes" in input_dict:
                input_dict["higher_yes"] = 1.0
            if internet == "yes" and "internet_yes" in input_dict:
                input_dict["internet_yes"] = 1.0

            sample_df = pd.DataFrame([input_dict])
            
            # Predict
            prob = model.predict_proba(sample_df)[0]
            pred = model.predict(sample_df)[0]
            
            pass_prob = prob[1] * 100
            fail_prob = prob[0] * 100

            st.write("### Kết quả dự đoán:")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if pred == 1:
                    st.success(f"### 🎉 KẾT QUẢ: ĐẬU\nXác suất đỗ: **{pass_prob:.1f}%**")
                else:
                    st.error(f"### ⚠️ KẾT QUẢ: NGUY CƠ RỚT\nXác suất rớt: **{fail_prob:.1f}%**")
                    
            with res_col2:
                st.write("**Khuyến nghị & Can thiệp sớm:**")
                if failures > 0:
                    st.warning(f"📌 Học sinh đã từng trượt {failures} lần môn học trước đây: Đây là dấu hiệu cảnh báo mạnh nhất. Cần gia sư hỗ trợ kèm riêng.")
                if absences > 10:
                    st.warning(f"📌 Vắng học {absences} buổi là mức cao: Nhà trường và gia đình cần liên lạc để kiểm soát chuyên cần.")
                if studytime <= 1:
                    st.info("📌 Thời gian tự học dưới 2 giờ/tuần: Khuyến khích tăng thời gian tự học lên tối thiểu 2-5 giờ/tuần.")
                if pred == 1:
                    st.info("✅ Học sinh có nền tảng và thói quen học tập tương đối tốt. Cần duy trì phong độ hiện tại.")
        else:
            st.error("Chưa tìm thấy file mô hình `random_forest.joblib`. Vui lòng chạy pipeline trước!")

# ----------------- TAB 2: VISUAL ANALYTICS -----------------
with tab2:
    st.subheader("Biểu đồ phân tích dữ liệu & Khai phá tri thức")
    
    col_a, col_b = st.columns(2)
    with col_a:
        img_feat = os.path.join(FIG_DIR, "feature_importance.png")
        if os.path.exists(img_feat):
            st.image(img_feat, caption="Top 10 đặc trưng quan trọng nhất đối với kết quả học tập", use_container_width=True)
            
        img_factors = os.path.join(FIG_DIR, "key_factors_analysis.png")
        if os.path.exists(img_factors):
            st.image(img_factors, caption="Ảnh hưởng của số lần trượt, thời gian học và số buổi vắng đến tỷ lệ đỗ", use_container_width=True)

    with col_b:
        img_cm = os.path.join(FIG_DIR, "confusion_matrix_rf.png")
        if os.path.exists(img_cm):
            st.image(img_cm, caption="Ma trận nhầm lẫn (Confusion Matrix) của Random Forest", use_container_width=True)

        img_roc = os.path.join(FIG_DIR, "roc_curve_rf.png")
        if os.path.exists(img_roc):
            st.image(img_roc, caption="Đường cong ROC và AUC của Random Forest", use_container_width=True)

    st.divider()
    img_corr = os.path.join(FIG_DIR, "correlation_heatmap.png")
    if os.path.exists(img_corr):
        st.image(img_corr, caption="Ma trận tương quan giữa các biến định lượng", use_container_width=True)

# ----------------- TAB 3: MODEL COMPARISON -----------------
with tab3:
    st.subheader("Bảng so sánh hiệu năng các mô hình (Tập kiểm thử)")
    if os.path.exists(REPORT_PATH):
        report_df = pd.read_csv(REPORT_PATH)
        
        # Format percentage
        display_df = report_df.copy()
        for col in ["accuracy", "precision", "recall", "f1_score"]:
            display_df[col] = (display_df[col] * 100).map("{:.2f}%".format)
            
        display_df.rename(columns={
            "model": "Mô hình",
            "accuracy": "Accuracy (Độ chính xác)",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-Score"
        }, inplace=True)
        
        st.dataframe(display_df, use_container_width=True)

        # Bar chart comparison
        chart_data = report_df.set_index("model")[["accuracy", "precision", "recall", "f1_score"]]
        st.bar_chart(chart_data)
    else:
        st.info("Chưa có file báo cáo mô hình.")

# ----------------- TAB 4: DATASET EXPLORER -----------------
with tab4:
    st.subheader("Dữ liệu học sinh (Student Performance Dataset)")
    if df_clean is not None:
        st.write(f"Hiển thị dữ liệu mẫu ({len(df_clean)} dòng):")
        st.dataframe(df_clean, use_container_width=True)
    else:
        st.info("Chưa tìm thấy dataset đã xử lý.")
