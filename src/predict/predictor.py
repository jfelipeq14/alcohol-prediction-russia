import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import AlcoholDataset
from src.config import Config
from src.data.preprocessor import DataPreprocessor


class Predictor:
    def __init__(self, config: Config, preprocessor: DataPreprocessor, model: torch.nn.Module):
        self.config = config
        self.preprocessor = preprocessor
        self.model = model

    def predict_2024(self, pivoted: pd.DataFrame) -> pd.DataFrame:
        X_2024 = self.preprocessor.extract_predict_data(pivoted)
        X_processed = self.preprocessor.transform(X_2024)

        dataset = AlcoholDataset(X_processed)
        loader = DataLoader(dataset, batch_size=64)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch in loader:
                predictions.append(self.model(X_batch).numpy())

        y_pred = np.vstack(predictions).flatten()
        y_pred = np.maximum(y_pred, 0)

        result = pivoted[["Region", "Type of alcoholic beverages"]].copy()
        result["Predicted_2024"] = y_pred
        return result

    @staticmethod
    def ranking_by_region(predictions: pd.DataFrame) -> pd.DataFrame:
        return (
            predictions.groupby("Region")["Predicted_2024"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"Predicted_2024": "Avg_Pure_Alcohol_2024"})
        )

    @staticmethod
    def ranking_by_beverage(predictions: pd.DataFrame) -> pd.DataFrame:
        return (
            predictions.groupby("Type of alcoholic beverages")["Predicted_2024"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"Predicted_2024": "Avg_Pure_Alcohol_2024"})
        )

    def save_predictions(self, predictions: pd.DataFrame, path):
        predictions.to_csv(path, index=False)
