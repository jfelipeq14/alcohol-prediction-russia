import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import torch

from sklearn.linear_model import LinearRegression
import pandas as pd

from src.config import Config
from src.data.loader import CSVLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter
from src.data.dataset import AlcoholDataset
from src.model.architecture import AlcoholPredictor
from src.training.trainer import Trainer
from src.evaluate.metrics import Evaluator
from src.predict.predictor import Predictor
from src.visualize import (
    plot_training_history,
    plot_actual_vs_predicted,
    plot_ranking,
    plot_metrics_comparison,
)


def main():
    config = Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(" Alcohol Consumption Prediction — Russia 2024")
    print("=" * 55)

    # ── 1. Load ────────────────────────────────────────────────
    print("\n[1/9] Loading data...")
    loader = CSVLoader(config.data_dir / config.raw_data_file)
    df = loader.load()
    print(f"       Rows: {len(df):,} | Columns: {len(df.columns)}")

    # ── 2. Pivot ───────────────────────────────────────────────
    print("[2/9] Pivoting to wide format (features por año)...")
    preprocessor = DataPreprocessor()
    pivoted = preprocessor.pivot_data(df)
    print(f"       Samples (region × beverage): {len(pivoted)}")

    # ── 3. Extract train data ──────────────────────────────────
    print("[3/9] Extracting features & target...")
    X, y = preprocessor.extract_train_data(pivoted)
    print(f"       Features: {X.shape[1]} | Target: {y.name}")

    # ── 4. Split ───────────────────────────────────────────────
    print("[4/9] Splitting train/val/test (70/15/15)...")
    splitter = DataSplitter(
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
    )
    splits = splitter.split(X, y)
    print(f"       Train: {len(splits['X_train'])} | Val: {len(splits['X_val'])} | Test: {len(splits['X_test'])}")

    # ── 5. Encode & Scale ──────────────────────────────────────
    print("[5/9] Encoding categories & scaling features...")
    X_train = preprocessor.fit_transform(splits["X_train"])
    X_val = preprocessor.transform(splits["X_val"])
    X_test = preprocessor.transform(splits["X_test"])
    y_train = splits["y_train"].values
    y_val = splits["y_val"].values
    y_test = splits["y_test"].values

    train_dataset = AlcoholDataset(X_train, y_train)
    val_dataset = AlcoholDataset(X_val, y_val)
    test_dataset = AlcoholDataset(X_test, y_test)

    input_dim = X_train.shape[1]
    print(f"       Input dimension: {input_dim} ({input_dim - 2 - 18} regions OH + 6 beverages OH + 18 numerical)")

    # ── 6. Build Model ─────────────────────────────────────────
    print("[6/9] Building neural network...")
    model = AlcoholPredictor(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout_rate,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Architecture: {input_dim} → {config.hidden_dims} → 1")
    print(f"       Parameters: {total_params:,}")

    # ── 7. Train ───────────────────────────────────────────────
    print("[7/9] Training...")
    trainer = Trainer(model, train_dataset, val_dataset, config)
    history = trainer.train()
    final_val_loss = min(history["val_loss"])
    print(f"       Best validation MSE: {final_val_loss:.6f}")

    # ── 8. Evaluate ────────────────────────────────────────────
    print("[8/9] Evaluating...")
    evaluator = Evaluator()

    model.load_state_dict(torch.load(config.output_dir / "best_model.pth", weights_only=True))
    nn_metrics = evaluator.evaluate(model, test_dataset)
    print("       ── Neural Network ──")
    for name, val in nn_metrics.items():
        print(f"       {name}: {val:.4f}")

    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluator.evaluate_sklearn(baseline, X_test, y_test)
    print("       ── Linear Regression ──")
    for name, val in baseline_metrics.items():
        print(f"       {name}: {val:.4f}")

    # ── 9. Predict 2024 ────────────────────────────────────────
    print("[9/9] Predicting 2024...")
    predictor = Predictor(config, preprocessor, model)
    predictions = predictor.predict_2024(pivoted)
    predictor.save_predictions(predictions, config.output_dir / "predicciones_2024.csv")

    region_ranking = predictor.ranking_by_region(predictions)
    region_ranking.to_csv(config.output_dir / "ranking_regiones.csv", index=False)
    beverage_ranking = predictor.ranking_by_beverage(predictions)
    beverage_ranking.to_csv(config.output_dir / "ranking_bebidas.csv", index=False)

    print(f"       Predictions saved: predicciones_2024.csv ({len(predictions)} rows)")
    print(f"       Region ranking:    ranking_regiones.csv")
    print(f"       Beverage ranking:  ranking_bebidas.csv")

    # ── 10. Generate plots ─────────────────────────────────────
    print("\nGenerating plots...")
    plot_training_history(history, config.output_dir)

    y_test_pred_nn = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().flatten()
    plot_actual_vs_predicted(y_test, y_test_pred_nn, config.output_dir, title="Neural Network")

    y_test_pred_base = baseline.predict(X_test)
    plot_actual_vs_predicted(y_test, y_test_pred_base, config.output_dir, title="Linear Regression (baseline)")

    plot_ranking(predictions, config.output_dir)
    plot_metrics_comparison(nn_metrics, baseline_metrics, config.output_dir)

    # ── Done ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" Done — Results saved in output/")
    print("=" * 55)


if __name__ == "__main__":
    main()
