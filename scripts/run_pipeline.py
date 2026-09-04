import os
import sys
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_dataset
from src.data.cleaner import check_missing, remove_duplicates, create_pass_label
from src.mining.association import run_apriori, generate_rules
from src.mining.clustering import run_kmeans
from src.models.supervised import (
    train_random_forest,
    train_logistic_regression,
    train_decision_tree,
    predict,
)
from src.models.semi_supervised import train_label_propagation
from src.evaluation.metrics import evaluate_classification
from src.evaluation.report import save_results


def load_config(config_path="configs/params.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print("STARTING STUDENT PERFORMANCE DATA MINING PIPELINE")
    print("=" * 60)

    # 0. Load Configuration
    config_path = os.path.join(PROJECT_ROOT, "configs", "params.yaml")
    config = load_config(config_path)
    seed = config.get("seed", 42)
    np.random.seed(seed)

    # Ensure required directories exist
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "figures"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "reports"), exist_ok=True)

    # 1. Load Data
    raw_path = os.path.join(PROJECT_ROOT, config["paths"]["raw_data_math"])
    print(f"\n[1/7] Loading raw data from: {raw_path}")
    df = load_dataset(raw_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    # 2. Clean & Preprocess
    print("\n[2/7] Cleaning data and creating target variable 'pass'...")
    df = remove_duplicates(df)
    df = create_pass_label(df)
    
    # Save clean dataset
    clean_path = os.path.join(PROJECT_ROOT, config["paths"]["processed_data"])
    df.to_csv(clean_path, index=False)
    print(f"Saved clean dataset to {clean_path}")

    # Remove G1, G2, G3 to prevent data leakage
    df_model = df.drop(columns=["G1", "G2", "G3"])

    # One-hot encoding for categorical columns
    categorical_cols = df_model.select_dtypes(include=["object", "string"]).columns
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
    encoded_path = os.path.join(PROJECT_ROOT, config["paths"]["encoded_data"])
    df_encoded.to_csv(encoded_path, index=False)
    print(f"Saved encoded dataset to {encoded_path}")

    # 3. Features & Train/Test Split
    print("\n[3/7] Splitting data into train and test sets...")
    X = df_encoded.drop("pass", axis=1)
    y = df_encoded["pass"]

    test_size = config.get("preprocessing", {}).get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Save processed splits
    X_train_scaled.to_csv(os.path.join(PROJECT_ROOT, config["paths"]["train_features"]), index=False)
    X_test_scaled.to_csv(os.path.join(PROJECT_ROOT, config["paths"]["test_features"]), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(PROJECT_ROOT, config["paths"]["train_labels"]), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(PROJECT_ROOT, config["paths"]["test_labels"]), index=False)
    print(f"Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")

    # 4. Association Pattern Mining (Apriori)
    print("\n[4/7] Running Apriori pattern mining...")
    try:
        bin_df = pd.DataFrame({
            "high_absence": (df["absences"] > 5).astype(bool),
            "low_studytime": (df["studytime"] <= 2).astype(bool),
            "many_failures": (df["failures"] > 0).astype(bool),
            "pass": (df["pass"] == 1).astype(bool)
        })
        itemsets = run_apriori(bin_df, min_support=0.1)
        rules = generate_rules(itemsets, min_confidence=0.5)
        print(f"Found {len(itemsets)} frequent itemsets and {len(rules)} association rules.")
    except Exception as e:
        print(f"Pattern mining notice: {e}")

    # 5. Clustering (KMeans)
    print("\n[5/7] Running KMeans clustering...")
    n_clusters = config.get("clustering", {}).get("n_clusters", 3)
    kmeans_model, cluster_labels = run_kmeans(X_train_scaled, n_clusters=n_clusters)
    print(f"KMeans (k={n_clusters}) clustered {len(cluster_labels)} train samples.")

    # 6. Model Training (Supervised & Semi-Supervised)
    print("\n[6/7] Training Supervised and Semi-Supervised models...")
    rf_model = train_random_forest(X_train_scaled, y_train)
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    dt_model = train_decision_tree(X_train_scaled, y_train)

    # Semi-supervised (masking 30% labels for demonstration)
    y_semi = np.array(y_train, copy=True)
    mask = np.random.rand(len(y_semi)) < 0.3
    y_semi[mask] = -1
    semi_model = train_label_propagation(X_train_scaled, y_semi)

    # 7. Evaluation & Report
    print("\n[7/7] Evaluating models on test set:")
    models = {
        "Random Forest": rf_model,
        "Logistic Regression": lr_model,
        "Decision Tree": dt_model,
        "Label Propagation": semi_model
    }

    report_list = []
    for name, model in models.items():
        y_pred = predict(model, X_test_scaled)
        metrics = evaluate_classification(y_test, y_pred)
        metrics["model"] = name
        report_list.append(metrics)
        print(f" -> {name:20s}: Accuracy={metrics['accuracy']:.4f} | "
              f"Precision={metrics['precision']:.4f} | "
              f"Recall={metrics['recall']:.4f} | "
              f"F1={metrics['f1_score']:.4f}")

    # Save report
    report_df = pd.DataFrame(report_list)
    report_path = os.path.join(PROJECT_ROOT, "outputs", "reports", "evaluation_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\nEvaluation report saved to: {report_path}")

    # Save best model (Random Forest)
    model_path = os.path.join(PROJECT_ROOT, "outputs", "models", "random_forest.joblib")
    joblib.dump(rf_model, model_path)
    print(f"Saved Random Forest model to: {model_path}")

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
