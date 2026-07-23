import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_training_history(history: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Train MSE")
    ax.plot(history["val_loss"], label="Validation MSE")
    best_epoch = np.argmin(history["val_loss"]) + 1
    ax.axvline(best_epoch - 1, color="gray", linestyle="--", alpha=0.5, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred_nn, y_pred_lr, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_pred, title in zip(
        axes,
        [y_pred_nn, y_pred_lr],
        ["Neural Network", "Linear Regression"],
    ):
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidth=0.5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", alpha=0.7)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        ax.grid(True)
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
    region_avg.head(top_n).plot(kind="bar", ax=axes[0], color="steelblue", legend=False)
    axes[0].set_title(f"Top {top_n} Regions — Avg Pure Alcohol 2024")
    axes[0].set_ylabel("Liters per capita")
    axes[0].tick_params(axis="x", rotation=45)

    region_avg.tail(top_n).plot(kind="bar", ax=axes[1], color="coral", legend=False)
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
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x, nn_vals, width=0.35, label="Neural Network", color="steelblue", alpha=0.8)
    bars2 = ax.bar([i + 0.35 for i in x], base_vals, width=0.35, label="Linear Regression", color="coral", alpha=0.8)
    ax.set_xticks([i + 0.175 for i in x])
    ax.set_xticklabels(labels)
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(axis="y")
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()


def plot_beverage_ranking(beverage_ranking: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(beverage_ranking)))
    ax.barh(beverage_ranking["Type of alcoholic beverages"],
            beverage_ranking["Avg_Pure_Alcohol_2024"], color=colors)
    ax.set_xlabel("Liters of pure alcohol per capita")
    ax.set_title("Estimated 2024 consumption by beverage type")
    for i, v in enumerate(beverage_ranking["Avg_Pure_Alcohol_2024"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "beverage_ranking.png", dpi=150)
    plt.close()


def plot_comparison_2023_2024(comp: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(comp["alc_2023"], comp["Predicted_2024"],
               alpha=0.4, edgecolors="k", linewidth=0.5)
    lims = [0, comp[["alc_2023", "Predicted_2024"]].max().max() + 0.5]
    ax.plot(lims, lims, "r--", alpha=0.5, label="No change")
    ax.set_xlabel("Actual 2023")
    ax.set_ylabel("Predicted 2024")
    ax.set_title("Comparison 2023 vs 2024")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_2023_2024.png", dpi=150)
    plt.close()


def plot_residuals(y_true, y_pred, residuals, pivoted_test, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(y_true, bins=20, alpha=0.6, label="Actual", color="steelblue", edgecolor="white")
    axes[0].hist(y_pred, bins=20, alpha=0.6, label="Predicted", color="coral", edgecolor="white")
    axes[0].set_xlabel("Liters of pure alcohol per capita")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution: Actual vs Predicted")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual plot")

    beverage_types = pivoted_test["Type of alcoholic beverages"].values
    sns.boxplot(x=beverage_types, y=residuals, ax=axes[2], palette="Set2")
    axes[2].axhline(0, color="red", linestyle="--", alpha=0.4)
    axes[2].set_title("Residuals by beverage type")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output_dir / "residuals.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(pivoted: pd.DataFrame, output_dir: Path):
    year_cols = [f"alc_{y}" for y in range(2017, 2024)]
    corr_data = pivoted[year_cols].dropna()
    corr_matrix = corr_data.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
                vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Correlation between years")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()
