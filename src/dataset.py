import numpy as np
import torch
from torch.utils.data import TensorDataset


def create_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1)
    return TensorDataset(X_tensor, y_tensor)
