import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.dataset import AlcoholDataset


class MSE:
    def __call__(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)


class MAE:
    def __call__(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)


class R2Score:
    def __call__(self, y_true, y_pred):
        return r2_score(y_true, y_pred)


class Evaluator:
    def __init__(self):
        self.metrics = {"MSE": MSE(), "MAE": MAE(), "R²": R2Score()}

    def evaluate(self, model: torch.nn.Module, dataset: AlcoholDataset):
        loader = DataLoader(dataset, batch_size=64)
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                all_preds.append(model(X_batch).numpy())
                all_true.append(y_batch.numpy())
        y_pred = np.vstack(all_preds).flatten()
        y_true = np.vstack(all_true).flatten()
        return {name: metric(y_true, y_pred) for name, metric in self.metrics.items()}

    def evaluate_sklearn(self, model, X_test: np.ndarray, y_test: np.ndarray):
        y_pred = model.predict(X_test)
        return {name: metric(y_test, y_pred) for name, metric in self.metrics.items()}
