import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.training.early_stopping import EarlyStopping
from src.data.dataset import AlcoholDataset
from src.config import Config


class Trainer:
    def __init__(self, model: nn.Module, train_dataset: AlcoholDataset, val_dataset: AlcoholDataset, config: Config):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        self.optimizer = optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.criterion = nn.MSELoss()
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        self.max_epochs = config.max_epochs
        self.checkpoint_path = config.output_dir / "best_model.pth"

    def train(self):
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._run_epoch(training=True)
            val_loss = self._run_epoch(training=False)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

            if val_loss == min(history["val_loss"]):
                torch.save(self.model.state_dict(), self.checkpoint_path)

            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch}")
                break

        self.early_stopping.restore(self.model)
        return history

    def _run_epoch(self, training: bool):
        self.model.train(training)
        loader = self.train_loader if training else self.val_loader
        total_loss = 0.0

        with torch.set_grad_enabled(training):
            for X_batch, y_batch in loader:
                if training:
                    self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                if training:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()

        return total_loss / len(loader)
