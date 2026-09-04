import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Load clean data
clean_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "student_clean.csv"))

# Figure 1: Pass Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=clean_df, x="pass", palette="Set2")
plt.title("Student Pass / Fail Distribution (pass=1 if G3>=10)")
plt.xlabel("Result (0: Fail, 1: Pass)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pass_distribution.png"), dpi=150)
plt.close()

# Figure 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_df = clean_df.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()

# Figure 3: Key Factors vs Pass Rate
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.barplot(data=clean_df, x="failures", y="pass", ax=axes[0], palette="Blues_r")
axes[0].set_title("Pass Rate by Past Failures")
axes[0].set_ylabel("Pass Rate")

sns.barplot(data=clean_df, x="studytime", y="pass", ax=axes[1], palette="Greens")
axes[1].set_title("Pass Rate by Study Time")
axes[1].set_ylabel("Pass Rate")

sns.boxplot(data=clean_df, x="pass", y="absences", ax=axes[2], palette="Oranges")
axes[2].set_title("Absences by Pass / Fail")
axes[2].set_xlabel("Result (0: Fail, 1: Pass)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "key_factors_analysis.png"), dpi=150)
plt.close()

# Load Test Features and Model
X_test = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "y_test.csv")).values.ravel()
rf_model = joblib.load(os.path.join(PROJECT_ROOT, "outputs", "models", "random_forest.joblib"))

# Figure 4: Confusion Matrix
y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Fail (0)", "Pass (1)"],
            yticklabels=["Fail (0)", "Pass (1)"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_rf.png"), dpi=150)
plt.close()

# Figure 5: ROC Curve
y_proba = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curve_rf.png"), dpi=150)
plt.close()

# Figure 6: Feature Importance
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=150)
plt.close()

print("All figures successfully exported to outputs/figures/!")
