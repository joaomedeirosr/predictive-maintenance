from data_loader import DataLoader
from preprocess import DataCleaner


def main():

    loader = DataLoader("../data")
    cleaner = DataCleaner()

    train_df, test_df, truth_df = loader.load_all()

    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"Truth: {truth_df.shape}")

    train_df = cleaner.remove_empty_columns(train_df)
    test_df = cleaner.remove_empty_columns(test_df)
    truth_df = cleaner.remove_empty_columns(truth_df)


    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"Truth: {truth_df.shape}")


    print(train_df.head())
    
    print(test_df.head())

    print(truth_df.head())


if __name__ == "__main__":
    main()