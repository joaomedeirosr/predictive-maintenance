import pandas as pd
import numpy as np
from typing import Tuple

from constants import COLUMN_NAMES


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

    @staticmethod
    def normalize_with_numpy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        df['cycle_norm'] = df['cycle']
        cols_normalize = df.columns.difference(['id', 'cycle', 'RUL', 'failure_within_w1'])
        data = df[cols_normalize].values
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        denom = np.where((data_max - data_min) == 0, 1, data_max - data_min)
        data_norm = (data - data_min) / denom
        norm_train_df = pd.DataFrame(data_norm, columns=cols_normalize, index=df.index)
        join_df = df[['id', 'cycle', 'RUL', 'failure_within_w1']].join(norm_train_df)
        df = join_df.reindex(columns=df.columns)
        return df, data_min, data_max

    @staticmethod
    def normalize_test(df: pd.DataFrame, data_min: np.ndarray, data_max: np.ndarray) -> pd.DataFrame:
        df['cycle_norm'] = df['cycle']
        cols_normalize = df.columns.difference(['id', 'cycle', 'RUL', 'failure_within_w1'])
        denom = np.where((data_max - data_min) == 0, 1, data_max - data_min)
        data = df[cols_normalize].values
        data_norm = (data - data_min) / denom
        norm_test_df = pd.DataFrame(data_norm, columns=cols_normalize, index=df.index)
        test_join_df = df[df.columns.difference(cols_normalize)].join(norm_test_df)
        df = test_join_df.reindex(columns=df.columns).reset_index(drop=True)
        return df