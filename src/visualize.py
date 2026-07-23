import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_training_history(history: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="MSE Entrenamiento")
    ax.plot(history["val_loss"], label="MSE Validación")
    best_epoch = np.argmin(history["val_loss"]) + 1
    ax.axvline(best_epoch - 1, color="gray", linestyle="--", alpha=0.5, label=f"Mejor época ({best_epoch})")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title("Historial de Entrenamiento")
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
        ["Red Neuronal", "Regresión Lineal"],
    ):
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidth=0.5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", alpha=0.7)
        ax.set_xlabel("Real")
        ax.set_ylabel("Predicho")
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
    axes[0].set_title(f"Top {top_n} regiones — Consumo estimado 2024")
    axes[0].set_ylabel("Litros per cápita")
    axes[0].tick_params(axis="x", rotation=45)

    region_avg.tail(top_n).plot(kind="bar", ax=axes[1], color="coral", legend=False)
    axes[1].set_title(f"Bottom {top_n} regiones — Consumo estimado 2024")
    axes[1].set_ylabel("Litros per cápita")
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
    bars1 = ax.bar(x, nn_vals, width=0.35, label="Red Neuronal", color="steelblue", alpha=0.8)
    bars2 = ax.bar([i + 0.35 for i in x], base_vals, width=0.35, label="Regresión Lineal", color="coral", alpha=0.8)
    ax.set_xticks([i + 0.175 for i in x])
    ax.set_xticklabels(labels)
    ax.set_title("Comparación de Modelos")
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
    ax.set_xlabel("Litros de alcohol puro per cápita")
    ax.set_title("Consumo estimado 2024 por tipo de bebida")
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
    ax.plot(lims, lims, "r--", alpha=0.5, label="Sin cambio")
    ax.set_xlabel("Real 2023")
    ax.set_ylabel("Predicho 2024")
    ax.set_title("Comparación 2023 vs 2024")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_2023_2024.png", dpi=150)
    plt.close()


def plot_residuals(y_true, y_pred, residuals, pivoted_test, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(residuals, bins=20, density=True, alpha=0.6, color="steelblue", edgecolor="white")
    mu, std = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
    axes[0].plot(x, pdf, "r--", alpha=0.7, label=f"N({mu:.3f}, {std:.3f})")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Densidad")
    axes[0].set_title("Distribución de residuales")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Predicho")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuales vs Valores Predichos")

    beverage_types = pivoted_test["Type of alcoholic beverages"].values
    sns.boxplot(x=beverage_types, y=residuals, ax=axes[2], palette="Set2")
    axes[2].axhline(0, color="red", linestyle="--", alpha=0.4)
    axes[2].set_title("Residuales por tipo de bebida")
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
    ax.set_title("Correlación entre años")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()
