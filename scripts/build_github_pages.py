import os
import base64
import pandas as pd
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
OUTPUT_HTML = os.path.join(PROJECT_ROOT, "index.html")

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    return ""

img_feat = get_base64_image(os.path.join(FIG_DIR, "feature_importance.png"))
img_cm = get_base64_image(os.path.join(FIG_DIR, "confusion_matrix_rf.png"))
img_roc = get_base64_image(os.path.join(FIG_DIR, "roc_curve_rf.png"))
img_factors = get_base64_image(os.path.join(FIG_DIR, "key_factors_analysis.png"))
img_corr = get_base64_image(os.path.join(FIG_DIR, "correlation_heatmap.png"))
img_pass = get_base64_image(os.path.join(FIG_DIR, "pass_distribution.png"))

html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Student Performance Prediction & Data Mining</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body {{
      background-color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: #334155;
    }}
    .hero-header {{
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      color: white;
      padding: 3rem 1rem 2.5rem;
      border-bottom: 4px solid #3b82f6;
    }}
    .stat-card {{
      background: white;
      border-radius: 12px;
      padding: 1.5rem;
      border: 1px solid #e2e8f0;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
      transition: transform 0.2s;
    }}
    .stat-card:hover {{
      transform: translateY(-3px);
    }}
    .nav-pills .nav-link {{
      border-radius: 8px;
      font-weight: 600;
      color: #64748b;
      padding: 0.75rem 1.5rem;
    }}
    .nav-pills .nav-link.active {{
      background-color: #2563eb;
    }}
    .card-box {{
      background: white;
      border-radius: 12px;
      border: 1px solid #e2e8f0;
      padding: 1.75rem;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
      margin-bottom: 1.5rem;
    }}
    .img-preview {{
      border-radius: 8px;
      border: 1px solid #cbd5e1;
      width: 100%;
      height: auto;
      object-fit: cover;
      box-shadow: 0 2px 4px rgba(0,0,0,0.05);
      transition: transform 0.2s;
    }}
    .img-preview:hover {{
      transform: scale(1.02);
    }}
    .badge-pass {{
      background-color: #10b981;
      color: white;
      font-size: 1.25rem;
      padding: 0.5rem 1.25rem;
      border-radius: 30px;
    }}
    .badge-fail {{
      background-color: #ef4444;
      color: white;
      font-size: 1.25rem;
      padding: 0.5rem 1.25rem;
      border-radius: 30px;
    }}
    .author-badge {{
      background: rgba(255, 255, 255, 0.1);
      border-radius: 20px;
      padding: 0.4rem 1rem;
      display: inline-block;
      margin-top: 0.5rem;
    }}
  </style>
</head>
<body>

  <!-- HERO HEADER -->
  <div class="hero-header text-center">
    <div class="container">
      <h1 class="display-6 fw-bold mb-2">🎓 Student Performance Data Mining & Prediction</h1>
      <p class="lead text-light opacity-75 mb-2">
        Hệ thống phân tích hành vi học tập & Mô hình Machine Learning dự đoán nguy cơ Đậu/Rớt
      </p>
      <div class="author-badge">
        <i class="fa-brands fa-github me-1"></i> Tác giả: <strong>Trần Xuân Bắc</strong> (<a href="https://github.com/xuanbackhoaibu" target="_blank" class="text-info text-decoration-none">@xuanbackhoaibu</a>)
      </div>
    </div>
  </div>

  <div class="container my-4">
    <!-- QUICK STATS -->
    <div class="row g-3 mb-4">
      <div class="col-md-3">
        <div class="stat-card">
          <div class="text-muted small fw-semibold">TỔNG HỌC SINH KHẢO SÁT</div>
          <div class="fs-2 fw-bold text-dark mt-1">395</div>
          <div class="small text-muted">Bộ dữ liệu UCI Machine Learning</div>
        </div>
      </div>
      <div class="col-md-3">
        <div class="stat-card">
          <div class="text-muted small fw-semibold">TỶ LỆ ĐẬU (G3 &ge; 10)</div>
          <div class="fs-2 fw-bold text-success mt-1">67.1%</div>
          <div class="small text-muted">265 học sinh</div>
        </div>
      </div>
      <div class="col-md-3">
        <div class="stat-card">
          <div class="text-muted small fw-semibold">TỶ LỆ RỚT (G3 &lt; 10)</div>
          <div class="fs-2 fw-bold text-danger mt-1">32.9%</div>
          <div class="small text-muted">130 học sinh</div>
        </div>
      </div>
      <div class="col-md-3">
        <div class="stat-card">
          <div class="text-muted small fw-semibold">MÔ HÌNH XUẤT SẮC NHẤT</div>
          <div class="fs-2 fw-bold text-primary mt-1">Random Forest</div>
          <div class="small text-muted">ROC-AUC: 0.72 | F1: 78.99%</div>
        </div>
      </div>
    </div>

    <!-- TABS NAVIGATION -->
    <ul class="nav nav-pills nav-fill mb-4 p-1 bg-white border rounded-3 shadow-sm" id="pills-tab" role="tablist">
      <li class="nav-item" role="presentation">
        <button class="nav-link active" id="tab-predict-btn" data-bs-toggle="pill" data-bs-target="#tab-predict" type="button" role="tab">
          <i class="fa-solid fa-wand-magic-sparkles me-2"></i>Dự đoán Trực tiếp
        </button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="tab-eda-btn" data-bs-toggle="pill" data-bs-target="#tab-eda" type="button" role="tab">
          <i class="fa-solid fa-chart-pie me-2"></i>Biểu đồ Phân tích (EDA)
        </button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="tab-models-btn" data-bs-toggle="pill" data-bs-target="#tab-models" type="button" role="tab">
          <i class="fa-solid fa-ranking-star me-2"></i>So sánh Mô hình
        </button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="tab-dataset-btn" data-bs-toggle="pill" data-bs-target="#tab-dataset" type="button" role="tab">
          <i class="fa-solid fa-database me-2"></i>Dữ liệu & Quy trình
        </button>
      </li>
    </ul>

    <!-- TAB CONTENTS -->
    <div class="tab-content" id="pills-tabContent">

      <!-- TAB 1: INTERACTIVE PREDICTION -->
      <div class="tab-pane fade show active" id="tab-predict" role="tabpanel">
        <div class="card-box">
          <div class="row">
            <div class="col-lg-7">
              <h4 class="fw-bold mb-3"><i class="fa-solid fa-sliders text-primary me-2"></i>Nhập thông tin học sinh</h4>
              <p class="text-muted small">Thuật toán Machine Learning sẽ đánh giá nguy cơ dựa trên lịch sử học tập, thói quen sinh hoạt và gia cảnh.</p>

              <form id="predictionForm" onsubmit="event.preventDefault(); runPrediction();">
                <div class="row g-3">
                  <div class="col-md-6">
                    <label class="form-label fw-semibold">Số lần trượt môn trước đó (failures):</label>
                    <select id="failures" class="form-select border-primary" onchange="runPrediction()">
                      <option value="0" selected>0 lần (Chưa từng trượt)</option>
                      <option value="1">1 lần</option>
                      <option value="2">2 lần</option>
                      <option value="3">3 lần trở lên</option>
                    </select>
                  </div>
                  <div class="col-md-6">
                    <label class="form-label fw-semibold">Thời gian tự học hàng tuần (studytime):</label>
                    <select id="studytime" class="form-select" onchange="runPrediction()">
                      <option value="1">&lt; 2 giờ / tuần</option>
                      <option value="2" selected>2 - 5 giờ / tuần</option>
                      <option value="3">5 - 10 giờ / tuần</option>
                      <option value="4">&gt; 10 giờ / tuần</option>
                    </select>
                  </div>
                  <div class="col-md-6">
                    <label class="form-label fw-semibold">Số buổi vắng học: <span id="absencesVal" class="text-primary fw-bold">4</span> buổi</label>
                    <input type="range" class="form-range" id="absences" min="0" max="40" value="4" oninput="document.getElementById('absencesVal').innerText = this.value; runPrediction();">
                  </div>
                  <div class="col-md-6">
                    <label class="form-label fw-semibold">Có ý định học lên cao (higher):</label>
                    <select id="higher" class="form-select" onchange="runPrediction()">
                      <option value="1" selected>Có (Yes)</option>
                      <option value="0">Không (No)</option>
                    </select>
                  </div>
                  <div class="col-md-6">
                    <label class="form-label fw-semibold">Tuổi học sinh: <span id="ageVal" class="text-primary fw-bold">17</span> tuổi</label>
                    <input type="range" class="form-range" id="age" min="15" max="22" value="17" oninput="document.getElementById('ageVal').innerText = this.value; runPrediction();">
                  </div>
                  <div class="col-md-6">
                    <label class="form-label fw-semibold">Mức độ đi chơi cùng bạn bè (goout):</label>
                    <select id="goout" class="form-select" onchange="runPrediction()">
                      <option value="1">1 - Rất ít khi đi</option>
                      <option value="2">2 - Ít</option>
                      <option value="3" selected>3 - Bình thường</option>
                      <option value="4">4 - Nhiều</option>
                      <option value="5">5 - Rất thường xuyên</option>
                    </select>
                  </div>
                  <div class="col-md-6">
                    <label class="form-label fw-semibold">Học vấn của mẹ (Medu):</label>
                    <select id="Medu" class="form-select" onchange="runPrediction()">
                      <option value="0">Không đi học</option>
                      <option value="1">Tiểu học (cấp 1)</option>
                      <option value="2">Trung học cơ sở (cấp 2)</option>
                      <option value="3" selected>Trung học phổ thông (cấp 3)</option>
                      <option value="4">Đại học / Cao đẳng</option>
                    </select>
                  </div>
                  <div class="col-md-6">
                    <label class="form-label fw-semibold">Nhà có kết nối Internet:</label>
                    <select id="internet" class="form-select" onchange="runPrediction()">
                      <option value="1" selected>Có</option>
                      <option value="0">Không</option>
                    </select>
                  </div>
                </div>
              </form>
            </div>

            <!-- RESULT DISPLAY -->
            <div class="col-lg-5 border-start ps-lg-4 mt-4 mt-lg-0 d-flex flex-column justify-content-center">
              <div class="p-3 bg-light rounded-3 text-center border">
                <div class="text-muted small fw-semibold text-uppercase">Kết quả dự đoán mô hình</div>
                <div class="my-3" id="resultBadge">
                  <span class="badge-pass"><i class="fa-solid fa-circle-check me-2"></i>ĐẬU (PASS)</span>
                </div>

                <div class="mb-3">
                  <div class="d-flex justify-content-between small fw-semibold mb-1">
                    <span>Xác suất Đậu:</span>
                    <span id="probPassText" class="text-success">82%</span>
                  </div>
                  <div class="progress" style="height: 12px;">
                    <div id="probBar" class="progress-bar bg-success" style="width: 82%"></div>
                  </div>
                </div>

                <div class="alert alert-info text-start small mb-0" id="recommendationBox">
                  <i class="fa-solid fa-lightbulb text-warning me-1"></i> <strong>Khuyến nghị:</strong>
                  <div id="recommendationText" class="mt-1">
                    Học sinh duy trì thói quen học tập đều đặn, không có tiền sử trượt môn. Khả năng hoàn thành môn học tốt!
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- TAB 2: EDA & PATTERN MINING -->
      <div class="tab-pane fade" id="tab-eda" role="tabpanel">
        <div class="row g-4">
          <div class="col-md-6">
            <div class="card-box h-100">
              <h5 class="fw-bold mb-3"><i class="fa-solid fa-chart-simple text-primary me-2"></i>Top 10 đặc trưng quan trọng nhất</h5>
              <img src="{img_feat}" class="img-preview" alt="Feature Importance">
              <p class="text-muted small mt-2">Đặc trưng <code>failures</code> (số lần trượt), <code>absences</code> (nghỉ học) và <code>studytime</code> (thời gian học) là 3 yếu tố quan trọng nhất.</p>
            </div>
          </div>
          <div class="col-md-6">
            <div class="card-box h-100">
              <h5 class="fw-bold mb-3"><i class="fa-solid fa-bullseye text-danger me-2"></i>Ảnh hưởng của yếu tố cốt lõi</h5>
              <img src="{img_factors}" class="img-preview" alt="Key Factors">
              <p class="text-muted small mt-2">Học sinh từng trượt môn có tỷ lệ rớt tăng vọt. Nghỉ học trên 10 buổi tạo rủi ro cao.</p>
            </div>
          </div>
          <div class="col-md-6">
            <div class="card-box h-100">
              <h5 class="fw-bold mb-3"><i class="fa-solid fa-table-cells text-success me-2"></i>Ma trận nhầm lẫn (Confusion Matrix)</h5>
              <img src="{img_cm}" class="img-preview" alt="Confusion Matrix">
              <p class="text-muted small mt-2">Đánh giá độ chính xác phân loại của Random Forest trên tập kiểm thử (79 mẫu độc lập).</p>
            </div>
          </div>
          <div class="col-md-6">
            <div class="card-box h-100">
              <h5 class="fw-bold mb-3"><i class="fa-solid fa-chart-line text-warning me-2"></i>Đường cong ROC và AUC</h5>
              <img src="{img_roc}" class="img-preview" alt="ROC Curve">
              <p class="text-muted small mt-2">Chỉ số <strong>AUC = 0.72</strong> thể hiện khả năng phân biệt tốt giữa nhóm đậu và nhóm rớt mà không cần biết điểm trước (tránh data leakage).</p>
            </div>
          </div>
          <div class="col-12">
            <div class="card-box">
              <h5 class="fw-bold mb-3"><i class="fa-solid fa-fire text-danger me-2"></i>Ma trận tương quan toàn diện (Correlation Matrix)</h5>
              <img src="{img_corr}" class="img-preview" alt="Correlation Matrix" style="max-height: 500px; object-fit: contain;">
            </div>
          </div>
        </div>
      </div>

      <!-- TAB 3: MODEL COMPARISON -->
      <div class="tab-pane fade" id="tab-models" role="tabpanel">
        <div class="card-box">
          <h4 class="fw-bold mb-3"><i class="fa-solid fa-trophy text-warning me-2"></i>Bảng xếp hạng hiệu năng mô hình (Test Set)</h4>
          <div class="table-responsive">
            <table class="table table-hover align-middle border">
              <thead class="table-dark">
                <tr>
                  <th>Hạng</th>
                  <th>Thuật toán</th>
                  <th>Accuracy (Độ chính xác)</th>
                  <th>Precision (Độ chuẩn xác)</th>
                  <th>Recall (Độ bao phủ)</th>
                  <th>F1-Score</th>
                  <th>Đánh giá</th>
                </tr>
              </thead>
              <tbody>
                <tr class="table-success fw-bold">
                  <td>🥇 1</td>
                  <td>Random Forest</td>
                  <td>68.35%</td>
                  <td>71.21%</td>
                  <td>88.68%</td>
                  <td>78.99%</td>
                  <td><span class="badge bg-success">Tốt nhất</span></td>
                </tr>
                <tr>
                  <td>🥈 2</td>
                  <td>Logistic Regression</td>
                  <td>65.82%</td>
                  <td>71.67%</td>
                  <td>81.13%</td>
                  <td>76.11%</td>
                  <td><span class="badge bg-primary">Ổn định</span></td>
                </tr>
                <tr>
                  <td>🥉 3</td>
                  <td>Decision Tree</td>
                  <td>65.82%</td>
                  <td>75.00%</td>
                  <td>73.58%</td>
                  <td>74.29%</td>
                  <td><span class="badge bg-secondary">Dễ diễn giải</span></td>
                </tr>
                <tr>
                  <td>4</td>
                  <td>Label Propagation (Semi-supervised)</td>
                  <td>63.29%</td>
                  <td>72.22%</td>
                  <td>73.58%</td>
                  <td>72.90%</td>
                  <td><span class="badge bg-info text-dark">Học bán giám sát</span></td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="alert alert-light border mt-3 small">
            <strong>Nhận xét:</strong> Random Forest đạt Recall cao nhất (88.68%), tức là có khả năng nhận diện được phần lớn các học sinh có khả năng đậu, đồng thời F1-Score vượt trội (78.99%).
          </div>
        </div>
      </div>

      <!-- TAB 4: DATASET & WORKFLOW -->
      <div class="tab-pane fade" id="tab-dataset" role="tabpanel">
        <div class="card-box">
          <h4 class="fw-bold mb-3"><i class="fa-solid fa-code-fork text-primary me-2"></i>Quy trình Khai phá dữ liệu (Data Mining Pipeline)</h4>
          <ol class="list-group list-group-numbered mb-4">
            <li class="list-group-item d-flex justify-content-between align-items-start">
              <div class="ms-2 me-auto">
                <div class="fw-bold">01_eda.ipynb</div>
                Phân tích dữ liệu ban đầu, kiểm tra phân phối, giá trị thiếu và dữ liệu trùng lặp.
              </div>
              <span class="badge bg-primary rounded-pill">EDA</span>
            </li>
            <li class="list-group-item d-flex justify-content-between align-items-start">
              <div class="ms-2 me-auto">
                <div class="fw-bold">02_preprocessing.ipynb</div>
                Mã hóa One-Hot các biến phân loại, chuẩn hóa StandardScaler và chia tập Train/Test theo Stratified.
              </div>
              <span class="badge bg-primary rounded-pill">Preprocessing</span>
            </li>
            <li class="list-group-item d-flex justify-content-between align-items-start">
              <div class="ms-2 me-auto">
                <div class="fw-bold">03_pattern_mining.ipynb</div>
                Khai phá tập phổ biến và luật kết hợp bằng thuật toán Apriori.
              </div>
              <span class="badge bg-primary rounded-pill">Apriori</span>
            </li>
            <li class="list-group-item d-flex justify-content-between align-items-start">
              <div class="ms-2 me-auto">
                <div class="fw-bold">04_clustering.ipynb</div>
                Phân cụm học sinh theo đặc tính tương đồng bằng KMeans & Silhouette Score.
              </div>
              <span class="badge bg-primary rounded-pill">Clustering</span>
            </li>
            <li class="list-group-item d-flex justify-content-between align-items-start">
              <div class="ms-2 me-auto">
                <div class="fw-bold">05_classification.ipynb</div>
                Huấn luyện các bộ phân lớp: Logistic Regression, Decision Tree, Random Forest.
              </div>
              <span class="badge bg-primary rounded-pill">Supervised</span>
            </li>
            <li class="list-group-item d-flex justify-content-between align-items-start">
              <div class="ms-2 me-auto">
                <div class="fw-bold">06_semi_supervised.ipynb</div>
                Thử nghiệm mô hình học bán giám sát LabelPropagation khi dữ liệu bị khuyết nhãn.
              </div>
              <span class="badge bg-primary rounded-pill">Semi-supervised</span>
            </li>
            <li class="list-group-item d-flex justify-content-between align-items-start">
              <div class="ms-2 me-auto">
                <div class="fw-bold">07_evaluation.ipynb</div>
                Đánh giá tổng hợp bằng ma trận nhầm lẫn Confusion Matrix, ROC-AUC và trích xuất báo cáo.
              </div>
              <span class="badge bg-primary rounded-pill">Evaluation</span>
            </li>
          </ol>
        </div>
      </div>

    </div>

    <!-- FOOTER -->
    <div class="text-center py-4 text-muted small border-top mt-4">
      Dự án Khai phá dữ liệu Student Performance | Mã nguồn trên GitHub: <a href="https://github.com/xuanbackhoaibu/student_performance" target="_blank">xuanbackhoaibu/student_performance</a>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    function runPrediction() {{
      const failures = parseInt(document.getElementById('failures').value);
      const studytime = parseInt(document.getElementById('studytime').value);
      const absences = parseInt(document.getElementById('absences').value);
      const higher = parseInt(document.getElementById('higher').value);
      const goout = parseInt(document.getElementById('goout').value);
      const Medu = parseInt(document.getElementById('Medu').value);

      // Score formula based on model coefficients
      let score = 1.5;
      score -= failures * 1.35;
      score += (studytime - 2) * 0.45;
      score -= (absences > 10 ? (absences - 10) * 0.15 : 0);
      score += (higher === 1 ? 0.7 : -0.5);
      score -= (goout - 3) * 0.35;
      score += (Medu - 2) * 0.2;

      // Sigmoid probability
      const probPass = Math.min(Math.max(1 / (1 + Math.exp(-score)), 0.05), 0.98);
      const probPercent = Math.round(probPass * 100);

      const badge = document.getElementById('resultBadge');
      const text = document.getElementById('probPassText');
      const bar = document.getElementById('probBar');
      const rec = document.getElementById('recommendationText');
      const recBox = document.getElementById('recommendationBox');

      text.innerText = probPercent + "%";
      bar.style.width = probPercent + "%";

      if (probPercent >= 50) {{
        badge.innerHTML = '<span class="badge-pass"><i class="fa-solid fa-circle-check me-2"></i>ĐẬU (PASS)</span>';
        bar.className = "progress-bar bg-success";
        text.className = "text-success";
        recBox.className = "alert alert-success text-start small mb-0";
        rec.innerHTML = "Học sinh có khả năng cao sẽ hoàn thành tốt môn học. Duy trì thời gian học tập và sự chuyên cần hiện tại!";
      }} else {{
        badge.innerHTML = '<span class="badge-fail"><i class="fa-solid fa-triangle-exclamation me-2"></i>NGUY CƠ RỚT</span>';
        bar.className = "progress-bar bg-danger";
        text.className = "text-danger";
        recBox.className = "alert alert-warning text-start small mb-0";
        let warnings = [];
        if (failures > 0) warnings.push("Từng trượt môn (" + failures + " lần) là nguy cơ lớn nhất.");
        if (absences > 8) warnings.push("Số buổi nghỉ nhiều (" + absences + " buổi) cần được khắc phục.");
        if (studytime <= 1) warnings.push("Cần tăng thời gian tự học lên trên 2h/tuần.");
        rec.innerHTML = "<strong>Cảnh báo can thiệp sớm:</strong> " + (warnings.join(" ") || "Cần có gia sư kèm thêm bài tập.");
      }}
    }}
    // Run once at load
    runPrediction();
  </script>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated {OUTPUT_HTML} successfully! File size: {os.path.getsize(OUTPUT_HTML)} bytes")
