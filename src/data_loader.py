from pathlib import Path
import pandas as pd


class DataLoader:

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)

    def load_train_data(self) -> pd.DataFrame:
    
        file_path = self.data_dir / "PM_train.txt"
        return pd.read_csv(file_path, sep=" ", header=None)

    def load_test_data(self) -> pd.DataFrame:

        file_path = self.data_dir / "PM_test.txt"
        return pd.read_csv(file_path, sep=" ", header=None)

    def load_truth_data(self) -> pd.DataFrame:
      
        file_path = self.data_dir / "PM_truth.txt"
        return pd.read_csv(file_path, sep=" ", header=None)

    def load_all(self):
       
        return (
            self.load_train_data(),
            self.load_test_data(),
            self.load_truth_data()
        )