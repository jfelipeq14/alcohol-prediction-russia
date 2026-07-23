from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root
    raw_data_file: str = "Consumption of alcoholic beverages 2017-2023 (Pivot table).csv"
    output_dir: Path = project_root / "output"

    # Data
    target_column: str = "Consumption of alcoholic beverages (in liters of pure alcohol per capita)"
    feature_years: List[int] = None
    target_year: int = 2023
    predict_year: int = 2024
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

    # Model
    hidden_dims: List[int] = None
    dropout_rate: float = 0.3

    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 200
    early_stopping_patience: int = 15
    weight_decay: float = 1e-5

    def __post_init__(self):
        if self.feature_years is None:
            self.feature_years = [2017, 2018, 2019, 2020, 2021, 2022]
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]
