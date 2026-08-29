import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_data import TensorLoader, compute_train_rul, normalize_sensors, build_dataset


class RULModel(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


if __name__ == "__main__":
    loader = TensorLoader("../data")
    train = loader.load_train_data()
    train_rul = compute_train_rul(train)
    train, _ = normalize_sensors(train, train.clone())

    dataset = build_dataset(train, train_rul)
    batches = DataLoader(dataset, batch_size=64)

    features_batch, rul_batch = next(iter(batches))

    model = RULModel(input_dim=24)
    output = model(features_batch)

    print("saída do modelo:", output.shape)
    print("rul esperado:", rul_batch.shape)