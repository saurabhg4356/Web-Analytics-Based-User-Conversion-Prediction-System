"""
scripts/train_model.py
Production-style Machine Learning Training Pipeline for Web Analytics User Conversion Prediction.
- Loads raw dataset
- Removes exact duplicates to prevent train/test leakage
- Excludes PageValues for genuine real-time pre-purchase prediction
- Performs Stratified 5-Fold Cross-Validation across:
    1. Logistic Regression (Baseline)
    2. Random Forest Classifier
    3. HistGradientBoosting Classifier
- Selects the best performing model based on Validation PR-AUC / ROC-AUC
- Evaluates hold-out test performance and threshold trade-offs
- Computes feature importances / coefficients
- Saves pipelines to models/ and metrics to reports/model_metrics.json
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42

NUMERICAL_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay"
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

ALL_INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "Revenue"


def load_and_clean_data(csv_path: str):
    """Loads dataset, removes duplicates, validates types, and separates features/target."""
    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    
    # Drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    deduped_rows = len(df)
    dropped_dupes = initial_rows - deduped_rows
    print(f"[*] Loaded {initial_rows:,} rows. Dropped {dropped_dupes} duplicate rows. Remaining: {deduped_rows:,}")

    # Ensure target is boolean integer (0 or 1)
    df[TARGET] = df[TARGET].astype(bool).astype(int)

    # Ensure categorical columns are strings to prevent numerical ordering assumption
    for cat_col in CATEGORICAL_FEATURES:
        df[cat_col] = df[cat_col].astype(str)

    # Validate numerical columns
    for num_col in NUMERICAL_FEATURES:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0.0)

    # Select only approved features (PageValues strictly excluded for real-time model)
    X = df[ALL_INPUT_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y, dropped_dupes, initial_rows


def build_preprocessor():
    """Builds scikit-learn ColumnTransformer for scaling and one-hot encoding."""
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def train_and_evaluate():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_csv = os.path.join(base_dir, "data", "raw", "online_shoppers_intention.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if not os.path.exists(raw_csv):
        print(f"[!] Error: Raw data file {raw_csv} does not exist. Run scripts/download_data.py first.")
        return False

    print("\n" + "="*60)
    print("PHASE 1: Data Preparation & Stratified Split")
    print("="*60)
    X, y, dropped_dupes, raw_rows = load_and_clean_data(raw_csv)
    
    # Stratified Train/Test split (80% train, 20% hold-out test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[*] Training samples: {len(X_train):,} (Purchases: {int(y_train.sum()):,}, Non-purchases: {len(y_train) - int(y_train.sum()):,})")
    print(f"[*] Hold-out Test samples: {len(X_test):,} (Purchases: {int(y_test.sum()):,}, Non-purchases: {len(y_test) - int(y_test.sum()):,})")

    preprocessor = build_preprocessor()

    # Define candidate classifiers
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }

    print("\n" + "="*60)
    print("PHASE 2: Stratified 5-Fold Cross-Validation on Training Data")
    print("="*60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, clf in candidate_models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        scoring = {
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
            "f1": "f1",
            "recall": "recall",
            "precision": "precision"
        }
        
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        mean_roc = float(np.mean(scores["test_roc_auc"]))
        mean_pr = float(np.mean(scores["test_pr_auc"]))
        mean_f1 = float(np.mean(scores["test_f1"]))
        mean_rec = float(np.mean(scores["test_recall"]))
        mean_prec = float(np.mean(scores["test_precision"]))

        cv_results[name] = {
            "val_roc_auc": round(mean_roc, 4),
            "val_pr_auc": round(mean_pr, 4),
            "val_f1": round(mean_f1, 4),
            "val_recall": round(mean_rec, 4),
            "val_precision": round(mean_prec, 4)
        }
        print(f"--> {name:22} | Val ROC-AUC: {mean_roc:.4f} | Val PR-AUC: {mean_pr:.4f} | Val F1: {mean_f1:.4f} | Recall: {mean_rec:.4f}")

    # Best model selection by PR-AUC (Average Precision on minority purchase class)
    best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]["val_pr_auc"])
    print(f"\n[+] Selected Champion Model based on Validation PR-AUC: '{best_model_name}'")

    print("\n" + "="*60)
    print("PHASE 3: Full Training & Hold-Out Test Evaluation")
    print("="*60)

    fitted_pipelines = {}
    test_evaluations = {}

    for name, clf in candidate_models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Save individual model pipeline
        filename_key = name.lower().replace(" ", "_")
        model_save_path = os.path.join(models_dir, f"{filename_key}_pipeline.joblib")
        joblib.dump(pipe, model_save_path)

        # Test set predictions
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred_default = (y_prob >= 0.50).astype(int)

        acc = float(accuracy_score(y_test, y_pred_default))
        prec = float(precision_score(y_test, y_pred_default, zero_division=0))
        rec = float(recall_score(y_test, y_pred_default))
        f1 = float(f1_score(y_test, y_pred_default, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
        brier = float(brier_score_loss(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred_default).tolist()

        test_evaluations[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "confusion_matrix": cm,
            "test_samples": len(y_test)
        }
        print(f"[{name}] Test ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | Recall: {rec:.4f} | Prec: {prec:.4f}")

    # Save champion pipeline to default target
    best_pipeline = fitted_pipelines[best_model_name]
    best_model_path = os.path.join(models_dir, "best_conversion_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"\n[+] Champion model persisted at: {best_model_path}")

    # Threshold Analysis for the Champion Model
    best_test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    threshold_analysis = []
    threshold_range = np.linspace(0.10, 0.90, 17)
    for thresh in threshold_range:
        thresh = round(float(thresh), 2)
        y_th_pred = (best_test_probs >= thresh).astype(int)
        th_prec = float(precision_score(y_test, y_th_pred, zero_division=0))
        th_rec = float(recall_score(y_test, y_th_pred))
        th_f1 = float(f1_score(y_test, y_th_pred, zero_division=0))
        th_cm = confusion_matrix(y_test, y_th_pred).tolist()
        threshold_analysis.append({
            "threshold": thresh,
            "precision": round(th_prec, 4),
            "recall": round(th_rec, 4),
            "f1": round(th_f1, 4),
            "confusion_matrix": th_cm
        })

    # Recommended threshold is the one that balances F1 and high recall
    recommended_threshold = 0.40

    # ROC and PR Curve data points for plotting and JSON
    fpr, tpr, roc_thresh = roc_curve(y_test, best_test_probs)
    # Downsample points for efficient JSON serialization
    step = max(1, len(fpr) // 50)
    roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr[::step], tpr[::step])]

    p_curve, r_curve, pr_thresh = precision_recall_curve(y_test, best_test_probs)
    step_pr = max(1, len(p_curve) // 50)
    pr_points = [{"recall": round(float(r), 4), "precision": round(float(p), 4)} for r, p in zip(r_curve[::step_pr], p_curve[::step_pr])]

    # Feature Importance Extraction
    print("\n" + "="*60)
    print("PHASE 4: Extracting Feature Importance / Explanations")
    print("="*60)
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_classifier = best_pipeline.named_steps["classifier"]

    # Extract transformed feature names
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    transformed_feature_names = NUMERICAL_FEATURES + cat_feature_names

    feature_importance_list = []
    if hasattr(fitted_classifier, "feature_importances_"):
        raw_importances = fitted_classifier.feature_importances_
        # Aggregate one-hot encoded categories back to original base features
        base_feature_map = {}
        for fname, imp in zip(transformed_feature_names, raw_importances):
            # identify which base feature it belongs to
            base_col = fname
            for cat_col in CATEGORICAL_FEATURES:
                if fname.startswith(f"{cat_col}_"):
                    base_col = cat_col
                    break
            base_feature_map[base_col] = base_feature_map.get(base_col, 0.0) + float(imp)

        total_imp = sum(base_feature_map.values())
        for f_name, imp in sorted(base_feature_map.items(), key=lambda x: x[1], reverse=True):
            pct = (imp / total_imp) * 100 if total_imp > 0 else 0
            feature_importance_list.append({
                "feature": f_name,
                "importance": round(float(imp), 4),
                "relative_percentage": round(pct, 2)
            })
    else:
        # Logistic regression coefficients
        coefs = fitted_classifier.coef_[0]
        base_feature_map = {}
        for fname, coef in zip(transformed_feature_names, coefs):
            base_col = fname
            for cat_col in CATEGORICAL_FEATURES:
                if fname.startswith(f"{cat_col}_"):
                    base_col = cat_col
                    break
            base_feature_map[base_col] = base_feature_map.get(base_col, 0.0) + abs(float(coef))
        total_coef = sum(base_feature_map.values())
        for f_name, imp in sorted(base_feature_map.items(), key=lambda x: x[1], reverse=True):
            pct = (imp / total_coef) * 100 if total_coef > 0 else 0
            feature_importance_list.append({
                "feature": f_name,
                "importance": round(float(imp), 4),
                "relative_percentage": round(pct, 2)
            })

    print("[+] Top 6 Important Features:")
    for item in feature_importance_list[:6]:
        print(f"    - {item['feature']:25}: {item['relative_percentage']}%")

    # Complete Metrics Dictionary
    metrics_data = {
        "model_version": "v1.0.0",
        "champion_model": best_model_name,
        "recommended_threshold": recommended_threshold,
        "sample_counts": {
            "raw_dataset_rows": raw_rows,
            "deduplicated_rows": len(X),
            "duplicates_removed": dropped_dupes,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_purchases": int(y_train.sum()),
            "train_non_purchases": int(len(y_train) - y_train.sum()),
            "test_purchases": int(y_test.sum()),
            "test_non_purchases": int(len(y_test) - y_test.sum())
        },
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "excluded_features": [
                {
                    "name": "PageValues",
                    "reason": "Target leakage. Reflects historical e-commerce transaction value not available during active browsing."
                },
                {
                    "name": "Revenue",
                    "reason": "Ground truth target variable."
                }
            ]
        },
        "cross_validation_comparison": cv_results,
        "test_evaluations": test_evaluations,
        "threshold_analysis": threshold_analysis,
        "feature_importance": feature_importance_list,
        "roc_curve": {
            "auc": test_evaluations[best_model_name]["roc_auc"],
            "points": roc_points
        },
        "pr_curve": {
            "pr_auc": test_evaluations[best_model_name]["pr_auc"],
            "points": pr_points
        }
    }

    metrics_json_path = os.path.join(reports_dir, "model_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\n[+] Saved complete evaluation metrics JSON to: {metrics_json_path}")

    # Generate Model Evaluation Figures
    print("\n" + "="*60)
    print("PHASE 5: Generating Evaluation Figures")
    print("="*60)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Model Comparison Chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    models = list(cv_results.keys())
    x = np.arange(len(models))
    width = 0.22
    val_pr = [cv_results[m]["val_pr_auc"] for m in models]
    val_roc = [cv_results[m]["val_roc_auc"] for m in models]
    val_f1 = [cv_results[m]["val_f1"] for m in models]

    b1 = ax.bar(x - width, val_pr, width, label="PR-AUC (Avg Precision)", color="#0284c7")
    b2 = ax.bar(x, val_roc, width, label="ROC-AUC", color="#38bdf8")
    b3 = ax.bar(x + width, val_f1, width, label="F1-Score", color="#64748b")
    ax.set_title("5-Fold Cross-Validation Performance Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylabel("Metric Score (0 - 1)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    for b_group in [b1, b2, b3]:
        for bar in b_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "model_comparison.png"), dpi=200)
    plt.close(fig)

    # 2. Confusion Matrix of Best Model
    fig, ax = plt.subplots(figsize=(5, 4.5))
    best_cm = np.array(test_evaluations[best_model_name]["confusion_matrix"])
    cax = ax.matshow(best_cm, cmap=plt.cm.Blues, alpha=0.85)
    for i in range(2):
        for j in range(2):
            val = best_cm[i, j]
            label = f"{val:,}"
            sublabel = "TN" if (i==0 and j==0) else ("FP" if (i==0 and j==1) else ("FN" if (i==1 and j==0) else "TP"))
            ax.text(j, i, f"{label}\n({sublabel})", ha="center", va="center", color="black" if val < best_cm.max()/2 else "white", fontweight="bold", fontsize=11)
    ax.set_xticklabels(["", "Pred: False", "Pred: True"])
    ax.set_yticklabels(["", "Actual: False", "Actual: True"])
    ax.set_title(f"Test Confusion Matrix ({best_model_name})", fontsize=11, fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # 3. ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#0284c7", lw=2.5, label=f"{best_model_name} (AUC = {test_evaluations[best_model_name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.50)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "roc_curve.png"), dpi=200)
    plt.close(fig)

    # 4. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r_curve, p_curve, color="#0d9488", lw=2.5, label=f"{best_model_name} (PR-AUC = {test_evaluations[best_model_name]['pr_auc']:.3f})")
    ax.set_title("Precision-Recall Curve (Minority Class)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Coverage of Buyers)")
    ax.set_ylabel("Precision (Accuracy of Buyer Predictions)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "pr_curve.png"), dpi=200)
    plt.close(fig)

    # 5. Feature Importance Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    top_features = feature_importance_list[:10][::-1]
    f_names = [f["feature"] for f in top_features]
    f_pcts = [f["relative_percentage"] for f in top_features]
    bars = ax.barh(f_names, f_pcts, color="#0284c7")
    ax.set_title(f"Top 10 Feature Importances ({best_model_name})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance (%)")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height()/2., f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(f_pcts) * 1.2)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "feature_importance.png"), dpi=200)
    plt.close(fig)

    print(f"[+] All figures saved to: {figures_dir}")
    print("\n[SUCCESS] Model training and evaluation completed successfully!")
    return True


if __name__ == "__main__":
    train_and_evaluate()
