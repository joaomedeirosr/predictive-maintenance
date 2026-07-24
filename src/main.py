from pipeline import prepare_data
from sequence_generator import SequenceGenerator
import os
import random
import numpy as np

def set_seed(seed: int = 1234) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)


set_seed(1234)

def main():

    train_df, test_df, truth_df = prepare_data()

    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"Truth: {truth_df.shape}")

    print(train_df.head())
    print(test_df.head())

    generator = SequenceGenerator(sequence_length=50)

    X_train, y_train = generator.create_dataset(
        train_df,
        seq_cols=["s2"],
        label=["failure_within_w1"]
    )

    print(X_train.shape)
    print(y_train.shape)

if __name__ == "__main__":
    main()
