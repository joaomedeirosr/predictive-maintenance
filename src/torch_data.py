from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader


class TensorLoader:

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)

    def _read_as_tensor(self, file_name: str) -> torch.Tensor:
        file_path = self.data_dir / file_name
        linhas = []
        with open(file_path) as arquivo:
            for linha in arquivo:
                valores = [float(v) for v in linha.split()]
                linhas.append(valores)
        return torch.tensor(linhas, dtype=torch.float32)

    def load_train_data(self) -> torch.Tensor:
        return self._read_as_tensor("pm_train.txt")

    def load_test_data(self) -> torch.Tensor:
        return self._read_as_tensor("pm_test.txt")

    def load_truth_data(self) -> torch.Tensor:
        return self._read_as_tensor("pm_truth.txt")

    def load_all(self):
        return (
            self.load_train_data(),
            self.load_test_data(),
            self.load_truth_data(),
        )


def compute_train_rul(data: torch.Tensor) -> torch.Tensor:
    units = data[:, 0]
    cycles = data[:, 1]
    rul = torch.zeros_like(cycles)

    for unit_id in torch.unique(units):
        mask = units == unit_id
        max_cycle = cycles[mask].max()
        rul[mask] = max_cycle - cycles[mask]

    return rul


def compute_test_rul(data: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    units = data[:, 0]
    cycles = data[:, 1]
    rul = torch.zeros_like(cycles)

    for unit_id in torch.unique(units):
        mask = units == unit_id
        max_cycle = cycles[mask].max()
        truth_value = truth[int(unit_id) - 1, 0]
        rul[mask] = truth_value + (max_cycle - cycles[mask])

    return rul


def normalize_sensors(train_data: torch.Tensor, test_data: torch.Tensor):
    sensors_train = train_data[:, 5:]
    min_values = sensors_train.min(dim=0).values
    max_values = sensors_train.max(dim=0).values
    range_values = max_values - min_values + 1e-8

    train_data[:, 5:] = (sensors_train - min_values) / range_values
    test_data[:, 5:] = (test_data[:, 5:] - min_values) / range_values

    return train_data, test_data


def build_dataset(data: torch.Tensor, rul: torch.Tensor) -> TensorDataset:
    features = data[:, 2:]
    return TensorDataset(features, rul)


if __name__ == "__main__":
    loader = TensorLoader("../data")
    train = loader.load_train_data()
    test = loader.load_test_data()
    truth = loader.load_truth_data()

    train_rul = compute_train_rul(train)
    test_rul = compute_test_rul(test, truth)

    train, test = normalize_sensors(train, test)

    train_dataset = build_dataset(train, train_rul)
    test_dataset = build_dataset(test, test_rul)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    features_batch, rul_batch = next(iter(train_loader))
    print("features batch:", features_batch.shape)
    print("rul batch:", rul_batch.shape)