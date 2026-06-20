import pandas as pd


class DataCleaner:

    @staticmethod
    def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
        
        return df.dropna(axis=1)