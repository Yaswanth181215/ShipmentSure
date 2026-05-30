"""
train.py
--------
Model training, hyperparameter tuning, and evaluation
for ShipmentSure project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

from preprocess import run_preprocessing

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────
# 1. Define Models
# ─────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":             XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }


# ─────────────────────────────────────────────
# 2. Evaluate a Single Model
# ─────────────────────────────────────────────
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "F1-Score":  round(f1_score(y_test, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n📊 {name}")
    print("-" * 40)
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k}: {v}")
    print(classification_report(y_test, y_pred))

    return metrics, y_pred, y_proba


# ─────────────────────────────────────────────
# 3. Hyperparameter Tuning (Best Model)
# ─────────────────────────────────────────────
def tune_xgboost(X_train, y_train):
    print("\n🔧 Tuning XGBoost with GridSearchCV...")
    param_grid = {
        'n_estimators':  [100, 200],
        'max_depth':     [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample':     [0.8, 1.0]
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"✅ Best Params: {grid.best_params_}")
    print(f"✅ Best CV F1:  {grid.best_score_:.4f}")
    return grid.best_estimator_


# ─────────────────────────────────────────────
# 4. Plot Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Delayed', 'On Time'],
                yticklabels=['Delayed', 'On Time'])
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrix_{model_name.replace(' ', '_')}.png", dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved for {model_name}")


# ─────────────────────────────────────────────
# 5. Plot ROC Curves (All Models)
# ─────────────────────────────────────────────
def plot_roc_curves(roc_data):
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=150)
    plt.close()
    print("✅ ROC curves saved.")


# ─────────────────────────────────────────────
# 6. Feature Importance
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Top 15 Feature Importances (XGBoost)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("✅ Feature importance plot saved.")


# ─────────────────────────────────────────────
# 7. Main Training Runner
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  🚚 ShipmentSure — Model Training Pipeline")
    print("=" * 50)

    # Load preprocessed data
    X_train, X_test, y_train, y_test, feature_names = run_preprocessing("data/Train.csv")

    models      = get_models()
    all_metrics = []
    roc_data    = {}
    best_model  = None
    best_f1     = 0

    # Train and evaluate all models
    for name, model in models.items():
        metrics, y_pred, y_proba = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        all_metrics.append(metrics)
        plot_confusion_matrix(y_test, y_pred, name)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, metrics['ROC-AUC'])

        if metrics['F1-Score'] > best_f1:
            best_f1    = metrics['F1-Score']
            best_model = (name, model)

    # ROC curve plot
    plot_roc_curves(roc_data)

    # Tune best model (XGBoost)
    tuned_xgb = tune_xgboost(X_train, y_train)
    tuned_metrics, tuned_pred, tuned_proba = evaluate_model(
        "XGBoost (Tuned)", tuned_xgb, X_train, X_test, y_train, y_test
    )
    all_metrics.append(tuned_metrics)
    plot_confusion_matrix(y_test, tuned_pred, "XGBoost_Tuned")
    plot_feature_importance(tuned_xgb, feature_names)

    # Save best model
    joblib.dump(tuned_xgb, "models/best_model.pkl")
    print("\n✅ Best model (XGBoost Tuned) saved to models/best_model.pkl")

    # Save metrics comparison
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("outputs/model_comparison.csv", index=False)
    print("✅ Model comparison saved to outputs/model_comparison.csv")

    print("\n📊 FINAL MODEL COMPARISON:")
    print(metrics_df.to_string(index=False))
    print("\n🎉 Training pipeline complete!")


if __name__ == "__main__":
    main()
