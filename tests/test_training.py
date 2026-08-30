import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.trainer import run_epoch
from src.evaluator import evaluate


class DummyClassificationModel(nn.Module):
    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


def test_run_epoch_train_and_eval_modes():
    device = torch.device("cpu")
    model = DummyClassificationModel(input_dim=4)
    features = torch.randn(20, 4)
    labels = torch.randint(0, 2, (20,)).float()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=5)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train mode test
    train_metrics = run_epoch(model, loader, criterion, device, optimizer=optimizer)
    assert "loss" in train_metrics
    assert "accuracy" in train_metrics
    assert 0.0 <= train_metrics["accuracy"] <= 1.0

    # Eval mode test (no optimizer)
    eval_metrics = run_epoch(model, loader, criterion, device, optimizer=None)
    assert "loss" in eval_metrics
    assert "accuracy" in eval_metrics
    assert 0.0 <= eval_metrics["accuracy"] <= 1.0


def test_evaluate_metrics_structure():
    device = torch.device("cpu")
    model = DummyClassificationModel(input_dim=4)
    features = torch.randn(10, 4)
    labels = torch.randint(0, 2, (10,))
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=5)

    res = evaluate(model, loader, device, threshold=0.5)

    assert "confusion_matrix" in res
    assert "classification_report" in res
    assert isinstance(res["confusion_matrix"], list)
    assert isinstance(res["classification_report"], dict)
