from data_loader import DataLoader
from preprocess import DataCleaner, Normalize
from feature_engineering import FeatureEngineering

def main():
    loader = DataLoader("../data")
    cleaner = DataCleaner()

    train_df, test_df, truth_df = loader.load_all()

    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"Truth: {truth_df.shape}")

    # Limpeza
    train_df = cleaner.remove_empty_columns(train_df)
    test_df = cleaner.remove_empty_columns(test_df)
    truth_df = cleaner.remove_empty_columns(truth_df)

    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"Truth: {truth_df.shape}")

    train_df = cleaner.rename_columns(train_df)
    test_df = cleaner.rename_columns(test_df)

    print(train_df.head())
    print(test_df.head())

    train_df = cleaner.sort_by_engine(train_df)
    test_df = cleaner.sort_by_engine(test_df)

    print(train_df.head())
    print(test_df.head())

    # Feature engineering
    train_df = FeatureEngineering.rul_with_numpy(train_df)
    train_df = FeatureEngineering.generating_target_variable(train_df)

    test_df = FeatureEngineering.rul_test(test_df, truth_df)

    # Normalização
    train_df, data_min, data_max = Normalize.normalize_with_numpy(train_df)
    test_df = Normalize.normalize_test(test_df, data_min, data_max)

    print(train_df.head())
    print(test_df.head())

if __name__ == "__main__":
    main()
