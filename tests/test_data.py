import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import create_tensor_dataset
from src.sequence_generator import SequenceGenerator


def test_sequence_and_label_alignment():
    frame = pd.DataFrame({
        "id": [1, 1, 1, 1],
        "sensor": [10, 20, 30, 40],
        "target": [0, 0, 1, 1],
    })
    features, labels = SequenceGenerator(3).create_dataset(
        frame, ["sensor"], ["target"]
    )
    np.testing.assert_array_equal(features[:, :, 0], [[10, 20, 30], [20, 30, 40]])
    np.testing.assert_array_equal(labels[:, 0], [1, 1])


def test_engine_with_exact_sequence_length_is_included():
    frame = pd.DataFrame({"id": [1, 1], "x": [1, 2], "y": [0, 1]})
    features, labels = SequenceGenerator(2).create_dataset(frame, ["x"], ["y"])
    assert features.shape == (1, 2, 1)
    assert labels.shape == (1, 1)


def test_numpy_arrays_are_converted_to_tensor_dataset():
    features = np.ones((4, 3, 2), dtype=np.float64)
    labels = np.array([[0], [1], [0], [1]], dtype=np.int64)
    dataset = create_tensor_dataset(features, labels)

    assert isinstance(dataset, TensorDataset)
    x_tensor, y_tensor = dataset.tensors
    assert x_tensor.shape == (4, 3, 2)
    assert y_tensor.shape == (4,)
    assert x_tensor.dtype == torch.float32
    assert y_tensor.dtype == torch.float32

    x_batch, y_batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert x_batch.shape == (2, 3, 2)
    assert y_batch.shape == (2,)
