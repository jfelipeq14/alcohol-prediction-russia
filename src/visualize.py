import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(history: dict, output_dir: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train MSE")
    plt.plot(history["val_loss"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, output_dir: Path, title: str = "Neural Network"):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lims = [
        min(min(y_true), min(y_pred)),
        max(max(y_true), max(y_pred)),
    ]
    plt.plot(lims, lims, "r--", alpha=0.8)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title} — Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()


def plot_ranking(predictions: pd.DataFrame, output_dir: Path, top_n: int = 15):
    region_avg = (
        predictions.groupby("Region")["Predicted_2024"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    region_avg.head(top_n).plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title(f"Top {top_n} Regions — Avg Pure Alcohol 2024")
    axes[0].set_ylabel("Liters per capita")
    axes[0].tick_params(axis="x", rotation=45)

    region_avg.tail(top_n).plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_title(f"Bottom {top_n} Regions — Avg Pure Alcohol 2024")
    axes[1].set_ylabel("Liters per capita")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "ranking.png", dpi=150)
    plt.close()


def plot_metrics_comparison(nn_metrics: dict, baseline_metrics: dict, output_dir: Path):
    labels = list(nn_metrics.keys())
    nn_vals = [nn_metrics[k] for k in labels]
    base_vals = [baseline_metrics[k] for k in labels]

    x = range(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, nn_vals, width=0.35, label="Neural Network", alpha=0.8)
    plt.bar([i + 0.35 for i in x], base_vals, width=0.35, label="Linear Regression", alpha=0.8)
    plt.xticks([i + 0.175 for i in x], labels)
    plt.title("Model Comparison")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()
