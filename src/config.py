from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    sequence_length: int = 50
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    threshold: float = 0.5
    seed: int = 1234
    checkpoint_path: str = "saved_models/rnn_best.pt"
