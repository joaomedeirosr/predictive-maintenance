import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.rnn import RNNClassifier
from pipeline import prepare_data
from sequence_generator import SequenceGenerator


SEQUENCE_LENGTH = 50
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001


def train(model, data_loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = loss_function(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def main():
    torch.manual_seed(1234)
    np.random.seed(1234)

    train_df, test_df, _ = prepare_data()

    feature_columns = [
        column
        for column in train_df.columns
        if column not in ["id", "cycle", "RUL", "failure_within_w1"]
    ]

    generator = SequenceGenerator(sequence_length=SEQUENCE_LENGTH)
    X_train, y_train = generator.create_dataset(
        train_df,
        seq_cols=feature_columns,
        label=["failure_within_w1"],
    )
    X_test, y_test = generator.create_dataset(
        test_df,
        seq_cols=feature_columns,
        label=["failure_within_w1"],
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1)

    print("X_train:", X_train_tensor.shape)
    print("y_train:", y_train_tensor.shape)
    print("X_test: ", X_test_tensor.shape)
    print("y_test: ", y_test_tensor.shape)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    x_batch, y_batch = next(iter(train_loader))
    print("Primeiro batch X:", x_batch.shape)
    print("Primeiro batch y:", y_batch.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    model = RNNClassifier(input_size=len(feature_columns)).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, loss_function, optimizer, device)
        print(f"Epoca {epoch + 1}/{EPOCHS} - loss: {loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            probabilities = torch.sigmoid(model(x_batch))
            predictions = (probabilities >= 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += len(y_batch)

    print(f"Acuracia no teste: {correct / total:.2%}")


if __name__ == "__main__":
    main()
