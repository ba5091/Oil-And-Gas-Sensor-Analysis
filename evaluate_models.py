"""Time-aware evaluation for the synthetic pump failure prediction project."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "synthetic_oil_gas_pump_sensors.csv"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def make_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal features without using future information from a different split."""
    features = frame.copy()
    for column in ["temperature", "pressure", "flow_rate"]:
        features[column] = pd.to_numeric(features[column], errors="coerce").interpolate(method="linear", limit_direction="both")

    for column in ["temperature", "vibration", "flow_rate", "motor_current", "pressure"]:
        features[f"{column}_rm12"] = features[column].rolling(12, min_periods=1).mean()
        features[f"{column}_rs12"] = features[column].rolling(12, min_periods=1).std().fillna(0)

    features["temp_rate"] = features["temperature"].diff(12).fillna(0)
    features["vib_rms12"] = np.sqrt((features["vibration"] ** 2).rolling(12, min_periods=1).mean())
    features["flow_var"] = features["flow_rate"].rolling(36, min_periods=1).std().fillna(0)

    excluded = {"timestamp", "failure", "failure_probability", "equipment_health"}
    feature_columns = [column for column in features.columns if column not in excluded]
    return features[feature_columns]


def evaluate_thresholds(y_true: pd.Series, probabilities: np.ndarray) -> list[dict]:
    rows = []
    for threshold in THRESHOLDS:
        prediction = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, prediction, zero_division=0),
                "recall": recall_score(y_true, prediction, zero_division=0),
                "f1": f1_score(y_true, prediction, zero_division=0),
            }
        )
    return rows


def save_threshold_table(results: dict) -> None:
    table = [
        [
            name,
            *[
                f"{detail['precision']:.4f}",
                f"{detail['recall']:.4f}",
                f"{detail['f1']:.4f}",
            ],
        ]
        for name, detail in results.items()
    ]
    print("\n## Threshold comparison by model")
    print(tabulate(table, headers=["Model", "Precision", "Recall", "F1"], tablefmt="github"))


def plot_roc_curves(models: dict[str, object], y_true: pd.Series, probability_map: dict[str, np.ndarray]) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for name, probabilities in probability_map.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
        roc_auc = roc_auc_score(y_true, probabilities)
        axis.plot(false_positive_rate, true_positive_rate, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random classifier")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC curves for held-out failure detection", pad=12)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "model_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_pr_curves(models: dict[str, object], y_true: pd.Series, probability_map: dict[str, np.ndarray]) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for name, probabilities in probability_map.items():
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        pr_auc = average_precision_score(y_true, probabilities)
        axis.plot(recall, precision, linewidth=2, label=f"{name} (PR-AUC={pr_auc:.3f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall curves for held-out failure detection", pad=12)
    axis.legend(loc="lower left")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "precision_recall.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrices(models: dict[str, object], y_true: pd.Series, probability_map: dict[str, np.ndarray]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, (name, probabilities) in zip(axes, probability_map.items()):
        predictions = (probabilities >= 0.5).astype(int)
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                    xticklabels=["Healthy", "Failure"], yticklabels=["Healthy", "Failure"], ax=axis)
        axis.set_title(name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    df = load_dataset()
    final_episode_start = len(df) - 1300
    train_frame = df.iloc[:final_episode_start].copy().reset_index(drop=True)
    test_frame = df.iloc[final_episode_start:].copy().reset_index(drop=True)

    X_train = make_feature_frame(train_frame)
    X_test = make_feature_frame(test_frame)
    y_train = train_frame["failure"].astype(int)
    y_test = test_frame["failure"].astype(int)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=0,
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
    }

    report = {}
    probabilities_by_model = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        probabilities_by_model[name] = probabilities

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
            "classification_report": classification_report(y_test, predictions, digits=4, output_dict=True),
            "thresholds": evaluate_thresholds(y_test, probabilities),
            "y_true": y_test.astype(int).tolist(),
            "y_score": probabilities.astype(float).tolist(),
            "y_pred": predictions.astype(int).tolist(),
            "timestamps": test_frame["timestamp"].astype(str).tolist(),
        }
        report[name] = metrics

    print("\n## Held-out time-based evaluation")
    print(f"Train rows: {len(train_frame):,} | Test rows: {len(test_frame):,}")
    metric_rows = []
    for model_name, metrics in report.items():
        metric_rows.append(
            [
                model_name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['roc_auc']:.4f}",
                f"{metrics['pr_auc']:.4f}",
            ]
        )
    print(tabulate(metric_rows, headers=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], tablefmt="github"))

    save_threshold_table({name: {"threshold": 0.5, **metrics["thresholds"][2]} for name, metrics in report.items()})
    for name, metrics in report.items():
        threshold_metrics = metrics["thresholds"]
        print(f"\n### {name} threshold scan")
        threshold_rows = [
            [f"{row['threshold']:.2f}", f"{row['precision']:.4f}", f"{row['recall']:.4f}", f"{row['f1']:.4f}"]
            for row in threshold_metrics
        ]
        print(tabulate(threshold_rows, headers=["Threshold", "Precision", "Recall", "F1"], tablefmt="github"))

    plot_roc_curves(models, y_test, probabilities_by_model)
    plot_pr_curves(models, y_test, probabilities_by_model)
    plot_confusion_matrices(models, y_test, probabilities_by_model)

    with (ROOT / "evaluation_results.json").open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2)

    print(f"\nSaved evaluation summary to {ROOT / 'evaluation_results.json'}")
    print(f"Saved curves to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
