import pandas as pd
from constants import COLUMN_NAMES
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

class DataCleaner:

    @staticmethod
    def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
        
        return df.dropna(axis=1)

    @staticmethod
    def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = COLUMN_NAMES
        return df

    @staticmethod
    def sort_by_engine(df: pd.DataFrame) -> pd.DataFrame:

        return df.sort_values(
            by=["id", "cycle"]
        ).reset_index(drop=True)
    
class Normalize:

    EXCLUDED_COLUMNS = [
        "id",
        "cycle",
        "RUL",
        "failure_within_w1",
    ]

    def __init__(self) -> None:
        self.scaler = MinMaxScaler()
        self.columns_to_normalize: list[str] = []

    def fit(self, df: pd.DataFrame) -> None:

        df = df.copy()
        df["cycle_norm"] = df["cycle"]
        self.columns_to_normalize = [
            column
            for column in df.columns
            if column not in self.EXCLUDED_COLUMNS
        ]
        self.scaler.fit(df[self.columns_to_normalize])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df["cycle_norm"] = df["cycle"]
        df[self.columns_to_normalize] = self.scaler.transform(
            df[self.columns_to_normalize]
        )
        return df.reset_index(drop=True)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
