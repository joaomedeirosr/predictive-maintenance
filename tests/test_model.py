import torch
import torch.nn as nn
from src.model import RULModel


def test_rul_model_forward_shape():
    input_dim = 24
    hidden_dim = 32
    batch_size = 16

    model = RULModel(input_dim=input_dim, hidden_dim=hidden_dim)
    dummy_input = torch.randn(batch_size, input_dim)
    output = model(dummy_input)

    assert output.shape == (batch_size,)


def test_rul_model_single_sample():
    input_dim = 10
    model = RULModel(input_dim=input_dim)
    single_input = torch.randn(1, input_dim)
    output = model(single_input)

    assert output.shape == (1,)


def test_rul_model_gradients():
    input_dim = 5
    model = RULModel(input_dim=input_dim)
    x = torch.randn(4, input_dim)
    target = torch.tensor([10.0, 20.0, 30.0, 40.0])

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    optimizer.zero_grad()
    predictions = model(x)
    loss = criterion(predictions, target)
    loss.backward()

    # Verify that gradients are computed for all model parameters
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
