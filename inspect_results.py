"""Print a compact tabular summary of the predictive-maintenance run."""
import json
from pathlib import Path

from tabulate import tabulate


results_path = Path(__file__).resolve().parent / "demo_results.json"
with results_path.open(encoding="utf-8") as results_file:
    results = json.load(results_file)

metric_rows = []
for model_name, metrics in results["metrics"].items():
    metric_rows.append([
        model_name,
        *(f"{metrics[key]:.4f}" for key in ("accuracy", "precision", "recall", "f1", "roc_auc")),
    ])

print("## Classifier performance (held-out episode)")
print(tabulate(
    metric_rows,
    headers=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    tablefmt="github",
))

print("\n## Top 8 Random Forest feature importances")
importance_rows = [
    [rank, feature, f"{importance:.4f}"]
    for rank, (feature, importance) in enumerate(results["importance"].items(), start=1)
]
print(tabulate(importance_rows, headers=["Rank", "Feature", "Importance"], tablefmt="github"))

print("\n## Dataset health")
health_rows = [
    ["Total samples", f"{results['rows']:,}"],
    ["Failure baseline rate", f"{results['failure_rate']:.2%}"],
    ["Isolation Forest anomaly rate (all rows)", f"{results['anomaly_rate']:.2%}"],
]
print(tabulate(health_rows, headers=["Indicator", "Value"], tablefmt="github"))

print("\n## Feature correlations with equipment health")
correlation_rows = sorted(results["corr_health"].items(), key=lambda item: abs(item[1]), reverse=True)
print(tabulate(
    [[feature, f"{correlation:.4f}"] for feature, correlation in correlation_rows],
    headers=["Feature", "Pearson correlation"],
    tablefmt="github",
))