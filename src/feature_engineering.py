import numpy as np
import pandas as pd

class FeatureEngineering:
    @staticmethod
    def rul_with_numpy(train_df: pd.DataFrame) -> pd.DataFrame:
        max_cycles = train_df.groupby('id')['cycle'].transform(np.max)
        train_df['RUL'] = max_cycles.values - train_df['cycle'].values
        return train_df

    @staticmethod
    def rul_test(test_df: pd.DataFrame, truth_df: pd.DataFrame, w1: int = 30) -> pd.DataFrame:
        rul = test_df.groupby('id')['cycle'].max().reset_index()
        rul.columns = ['id', 'max']
        truth_df.columns = ['additional_rul']
        truth_df['id'] = truth_df.index + 1
        truth_df['max'] = rul['max'] + truth_df['additional_rul']
        truth_df.drop('additional_rul', axis=1, inplace=True)
        test_df = test_df.merge(truth_df, on=['id'], how='left')
        test_df['RUL'] = test_df['max'] - test_df['cycle']
        test_df.drop('max', axis=1, inplace=True)
        test_df['failure_within_w1'] = np.where(test_df['RUL'] <= w1, 1, 0)
        return test_df

    @staticmethod
    def generating_target_variable(df: pd.DataFrame, w1: int = 30) -> pd.DataFrame:
        df['failure_within_w1'] = np.where(df['RUL'] <= w1, 1, 0)
        return df
