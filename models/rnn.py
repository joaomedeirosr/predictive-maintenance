"""Modelo RNN simples para classificacao binaria."""

import torch
from torch import nn


class RNNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        rnn_output, _ = self.rnn(X)
        last_output = rnn_output[:, -1, :]
        return self.output_layer(last_output).squeeze(1)
