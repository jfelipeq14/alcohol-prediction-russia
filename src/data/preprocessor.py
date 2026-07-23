import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    def __init__(self):
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.cat_cols_ = ["Region", "Type of alcoholic beverages"]
        self.num_col_count_ = None
        self.fitted_ = False

    def pivot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={
            "Consumption of alcoholic beverages (thousands of decaliters)": "vol",
            "Consumption of alcoholic beverages (in liters per capita)": "cap",
            "Consumption of alcoholic beverages (in liters of pure alcohol per capita)": "alc",
        })
        pivoted = df.pivot_table(
            index=["Region", "Type of alcoholic beverages"],
            columns="Year",
            values=["vol", "cap", "alc"],
        )
        pivoted.columns = [f"{metric}_{year}" for metric, year in pivoted.columns]
        pivoted = pivoted.reset_index()
        return pivoted

    def extract_train_data(self, pivoted: pd.DataFrame):
        feature_cols = [f"{m}_{y}" for m in ["vol", "cap", "alc"] for y in range(2017, 2023)]
        X = pivoted[self.cat_cols_ + feature_cols]
        y = pivoted["alc_2023"]
        return X, y

    def extract_predict_data(self, pivoted: pd.DataFrame):
        feature_cols = [f"{m}_{y}" for m in ["vol", "cap", "alc"] for y in range(2018, 2024)]
        X = pivoted[self.cat_cols_ + feature_cols]
        return X

    def _get_num_cols(self, X: pd.DataFrame):
        return [c for c in X.columns if c not in self.cat_cols_]

    def fit(self, X: pd.DataFrame):
        num_cols = self._get_num_cols(X)
        self.cat_encoder.fit(X[self.cat_cols_].values)
        self.scaler.fit(X[num_cols].values)
        self.num_col_count_ = len(num_cols)
        self.fitted_ = True

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        cat_encoded = self.cat_encoder.transform(X[self.cat_cols_].values)
        num_cols = self._get_num_cols(X)
        if len(num_cols) != self.num_col_count_:
            raise ValueError(
                f"Expected {self.num_col_count_} numerical columns, got {len(num_cols)}. "
                "Use extract_predict_data() for 2024 features."
            )
        num_scaled = self.scaler.transform(X[num_cols].values)
        return np.hstack([cat_encoded, num_scaled])

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
