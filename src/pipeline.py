from pathlib import Path
from data_loader import DataLoader
from preprocess import DataCleaner, Normalize
from feature_engineering import FeatureEngineering


def prepare_data(data_dir=None):
    project_root = Path(__file__).resolve().parents[1]
    loader = DataLoader(data_dir or project_root / "data")
    cleaner = DataCleaner()

    train_df, test_df, truth_df = loader.load_all()

    # Limpeza
    train_df = cleaner.remove_empty_columns(train_df)
    test_df = cleaner.remove_empty_columns(test_df)
    truth_df = cleaner.remove_empty_columns(truth_df)

    train_df = cleaner.rename_columns(train_df)
    test_df = cleaner.rename_columns(test_df)

    train_df = cleaner.sort_by_engine(train_df)
    test_df = cleaner.sort_by_engine(test_df)

    # Feature Engineering
    train_df = FeatureEngineering.rul_with_numpy(train_df)
    train_df = FeatureEngineering.generating_target_variable(train_df)

    test_df = FeatureEngineering.rul_test(test_df, truth_df)

    # Normalização
    normalizer = Normalize()

    train_df = normalizer.fit_transform(train_df)
    test_df = normalizer.transform(test_df)

    return train_df, test_df, truth_df
