#  Student Performance Data Mining Project

## 1. Giới thiệu

Dự án nhằm phân tích dữ liệu học sinh và xây dựng các mô hình học máy để dự đoán khả năng đậu / rớt môn học.

Ngoài dự đoán, dự án còn tập trung vào:

-  Khai phá tri thức từ dữ liệu (data mining)
-  Phân tích hành vi học tập
-  Đưa ra insight có thể áp dụng trong thực tế giáo dục

---

## 2. Bài toán

Bài toán được xây dựng dưới dạng **classification (phân loại)**:

 Input: thông tin học sinh (studytime, absences, failures, ...)
 Output:

   `pass = 1` → học sinh đậu (G3 ≥ 10)
   `pass = 0` → học sinh rớt (G3 < 10)

---

## 3. Dataset

- Nguồn: UCI Machine Learning Repository
- Tên: Student Performance Dataset

### Dữ liệu bao gồm:

 - Thông tin cá nhân: age, sex, address
 - Hành vi học tập: studytime, absences
 - Kết quả học tập: G1, G2, G3

### Lưu ý:

 -  Biến G1, G2, G3 được loại bỏ khi training để tránh **data leakage**
 -  Biến mục tiêu `pass` được tạo từ G3


## 4. Các bước thực hiện

### 1. Exploratory Data Analysis (EDA)

 -  Phân tích phân phối dữ liệu
 -  Xác định yếu tố ảnh hưởng đến kết quả học tập
 -  Phát hiện vấn đề: missing, duplicate, imbalance


### 2. Data Preprocessing

 -  Xử lý duplicate
 -  Encoding biến categorical
 -  Scaling dữ liệu số
 -  Train/Test split (stratified)


### 3. Pattern Mining (Association Rules)

Sử dụng thuật toán Apriori
Khám phá luật kết hợp giữa:

   - studytime
   - absences
   -  failures
Đánh giá bằng: support, confidence, lift


### 4. Clustering (KMeans)

 - Phân nhóm học sinh
 - Sử dụng Elbow method để chọn số cụm
 - Đánh giá bằng Silhouette Score


### 5. Classification (Supervised Learning)

 -  Baseline: Logistic Regression
 -  Model chính: Random Forest
 -  Hyperparameter tuning (GridSearchCV)
 -  Cross-validation


### 6. Semi-supervised Learning

 -  Sử dụng LabelPropagation
 -  Mô phỏng thiếu nhãn (10% → 70%)
 -  So sánh với supervised learning
 -  Vẽ learning curve


### 7. Model Evaluation & Error Analysis

 -  Metrics: Accuracy, Precision, Recall, F1-score
 - Confusion Matrix
 - ROC-AUC
 - Phân tích lỗi (False Positive, False Negative)
 - Rút ra actionable insights


## 5. Project Structure

```
configs/            cấu hình

data/
  raw/              dữ liệu gốc
  processed/        dữ liệu đã xử lý

notebooks/
  01_eda.ipynb
  02_preprocessing.ipynb
  03_pattern_mining.ipynb
  04_clustering.ipynb
  05_classification.ipynb
  06_semi_supervised.ipynb
  07_evaluation.ipynb

src/                source code (hàm xử lý, model)

scripts/
  run_pipeline.py   chạy toàn bộ pipeline

outputs/
  figures/          biểu đồ
  models/           model đã train
  tables/           kết quả
```

---

## 6. Cách chạy project (Reproducible)

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```


### 2. Chạy pipeline

```bash
python scripts/run_pipeline.py
```

Pipeline sẽ thực hiện:

 - Load dữ liệu
 - Tiền xử lý
 - Train model
 - Đánh giá
 - Lưu kết quả vào thư mục `outputs/`


## 7. Kết quả

Dự án tạo ra:

 -  Biểu đồ EDA
 -  Luật kết hợp (association rules)
 -  Kết quả phân cụm (clustering)
 -  Kết quả classification (Accuracy, F1-score)
 -  Confusion Matrix
 -  ROC Curve
 -  Feature Importance


## 8. Insight chính

 - Học sinh có nhiều lần trượt (failures) có nguy cơ rớt cao
 -  Nghỉ học nhiều (absences) ảnh hưởng tiêu cực đến kết quả
 -  Tăng thời gian học (studytime) giúp cải thiện điểm
 - Có thể xây dựng hệ thống cảnh báo sớm cho học sinh yếu


## 9. Ghi chú kỹ thuật

 -  Đã xử lý data leakage
 -  Sử dụng stratified split
 -  Có cross-validation
 -  Có phân tích lỗi chi tiết
 - Mô hình có thể tái lập (reproducible)

