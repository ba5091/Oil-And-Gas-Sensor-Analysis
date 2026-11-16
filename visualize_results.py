"""Create publication-ready figures from the synthetic pump-maintenance analysis."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "demo_results.json"
EVALUATION_PATH = ROOT / "evaluation_results.json"
FIGURE_DIR = ROOT / "figures"
LEGACY_FIGURE_DIR = ROOT / "results_figures"
for directory in (FIGURE_DIR, LEGACY_FIGURE_DIR):
    directory.mkdir(exist_ok=True)

with RESULTS_PATH.open(encoding="utf-8") as results_file:
    results = json.load(results_file)

if EVALUATION_PATH.exists():
    with EVALUATION_PATH.open(encoding="utf-8") as evaluation_file:
        evaluations = json.load(evaluation_file)
else:
    evaluations = results.get("test_predictions", {})

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def save_figure(figure, filename):
    for directory in (FIGURE_DIR, LEGACY_FIGURE_DIR):
        figure.savefig(directory / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_degradation_profile():
    series = results["series"]
    hours = np.asarray(series["hour"])
    first_failure = next((index for index, label in enumerate(series["failure"]) if label), None)
    profiles = [
        ("temperature", "Temperature", "Temperature (C)", "#d97732"),
        ("vibration", "Vibration", "Vibration (mm/s RMS)", "#247a78"),
        ("motor_current", "Motor current", "Motor current (A)", "#5470a6"),
        ("health", "Equipment health", "Health (%)", "#4e8052"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis, (key, title, ylabel, color) in zip(axes.flat, profiles):
        axis.plot(hours, series[key], color=color, linewidth=1.8)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Hours into held-out episode")
        if first_failure is not None:
            axis.axvspan(hours[first_failure], hours[-1], color="#c94c4c", alpha=0.09, label="Failure horizon")
            axis.legend(frameon=False, loc="best")
    figure.suptitle("Figure 1 — Degradation profile", fontsize=16, fontweight="bold")
    figure.tight_layout()
    save_figure(figure, "degradation_profile.png")


def plot_sensor_behavior():
    df = pd.read_csv(ROOT / "synthetic_oil_gas_pump_sensors.csv")
    held_out = df.iloc[-1300:].reset_index(drop=True).iloc[::12].reset_index(drop=True)
    hours = np.arange(len(held_out))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, column, label, color in [
        (axes[0, 0], "flow_rate", "Flow rate (L/min)", "#3b82f6"),
        (axes[0, 1], "pressure", "Pressure (psi)", "#f59e0b"),
        (axes[1, 0], "gas_concentration", "Gas concentration (ppm)", "#ef4444"),
        (axes[1, 1], "rpm", "RPM", "#10b981"),
    ]:
        axis.plot(hours, held_out[column], color=color, linewidth=1.8)
        axis.set_title(label, loc="left", fontweight="bold")
        axis.set_xlabel("Hours into held-out episode")
    figure.suptitle("Figure 2 — Sensor behaviour", fontsize=16, fontweight="bold")
    figure.tight_layout()
    save_figure(figure, "sensor_behavior.png")


def plot_model_roc_curves():
    figure, axis = plt.subplots(figsize=(8, 6))
    for name, metrics in evaluations.items():
        y_true = np.asarray(metrics["y_true"], dtype=int)
        y_score = np.asarray(metrics["y_score"], dtype=float)
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        axis.plot(false_positive_rate, true_positive_rate, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random classifier")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Figure 3 — Model ROC curves", pad=12)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    save_figure(figure, "model_roc_curves.png")


def plot_confusion_matrices():
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, (name, metrics) in zip(axes, evaluations.items()):
        matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                    xticklabels=["Healthy", "Failure"], yticklabels=["Healthy", "Failure"], ax=axis)
        axis.set_title(name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
    figure.suptitle("Figure 4 — Confusion matrices", fontsize=15, fontweight="bold")
    figure.tight_layout()
    save_figure(figure, "confusion_matrices.png")


def plot_precision_recall():
    figure, axis = plt.subplots(figsize=(8, 6))
    for name, metrics in evaluations.items():
        y_true = np.asarray(metrics["y_true"], dtype=int)
        y_score = np.asarray(metrics["y_score"], dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = metrics["pr_auc"]
        axis.plot(recall, precision, linewidth=2, label=f"{name} (PR-AUC={pr_auc:.3f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Figure 5 — Precision-Recall curves", pad=12)
    axis.legend(loc="lower left")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    save_figure(figure, "precision_recall.png")


def plot_feature_importance():
    importance = results["importance"]
    ranked = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:10]
    figure, axis = plt.subplots(figsize=(9, 6))
    names = [item[0] for item in ranked]
    values = [item[1] for item in ranked]
    axis.barh(names[::-1], values[::-1], color="#4f46e5")
    axis.set_title("Figure 6 — Random Forest feature importance", pad=12)
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    figure.tight_layout()
    save_figure(figure, "feature_importance.png")


def plot_correlation_heatmap():
    correlations = results["corr_health"]
    ordered = list(correlations.keys())
    values = np.array([[correlations[name] for name in ordered]], dtype=float)
    figure, axis = plt.subplots(figsize=(9, 2.2))
    sns.heatmap(values, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                xticklabels=ordered, yticklabels=["Equipment health"], cbar_kws={"label": "Pearson r"}, ax=axis)
    axis.set_title("Figure 7 — Equipment-health correlation heatmap", pad=12)
    figure.tight_layout()
    save_figure(figure, "correlation_heatmap.png")


def plot_anomaly_detection():
    series = results["series"]
    hours = np.asarray(series["hour"])
    anomaly_flags = np.asarray(series["anomaly"], dtype=int)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.scatter(hours, anomaly_flags, c=np.where(anomaly_flags == 1, "#dc2626", "#9ca3af"), s=26)
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Inlier", "Anomaly"])
    axis.set_xlabel("Hours into held-out episode")
    axis.set_title("Figure 8 — Isolation Forest anomaly detection", pad=12)
    figure.tight_layout()
    save_figure(figure, "anomaly_detection.png")


def plot_failure_probability():
    series = results["series"]
    hours = np.asarray(series["hour"])
    probabilities = np.asarray(series["fprob"]) / 100.0
    actual = np.asarray(series["failure"], dtype=int)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(hours, probabilities, color="#2563eb", linewidth=2, label="Failure probability")
    axis.plot(hours, actual, color="#dc2626", linewidth=1.5, label="Actual failure label")
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Decision threshold")
    axis.set_xlabel("Hours into held-out episode")
    axis.set_ylabel("Probability / label")
    axis.set_title("Figure 9 — Failure probability over time", pad=12)
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    save_figure(figure, "failure_probability.png")


def main():
    plot_degradation_profile()
    plot_sensor_behavior()
    plot_model_roc_curves()
    plot_confusion_matrices()
    plot_precision_recall()
    plot_feature_importance()
    plot_correlation_heatmap()
    plot_anomaly_detection()
    plot_failure_probability()
    print(f"Saved all figures to {FIGURE_DIR} and {LEGACY_FIGURE_DIR}")


if __name__ == "__main__":
    main()
