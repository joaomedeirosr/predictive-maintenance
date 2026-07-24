from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_data import (
    TensorLoader,
    compute_train_rul,
    compute_test_rul,
    normalize_sensors,
    build_dataset,
)
from model import RULModel

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_PATH = Path("../models")
MODEL_NAME = "rul_model.pt"


def main() -> None:
    loader = TensorLoader("../data")
    train, test, truth = loader.load_all()

    train_rul = compute_train_rul(train)
    test_rul = compute_test_rul(test, truth)

    train, test = normalize_sensors(train, test)

    train_dataset = build_dataset(train, train_rul)
    test_dataset = build_dataset(test, test_rul)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RULModel(input_dim=24)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_total = 0.0
        for features_batch, rul_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(features_batch)
            loss = loss_fn(predictions, rul_batch)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * len(features_batch)
        train_loss = train_loss_total / len(train_dataset)

        model.eval()
        test_loss_total = 0.0
        with torch.inference_mode():
            for features_batch, rul_batch in test_loader:
                predictions = model(features_batch)
                loss = loss_fn(predictions, rul_batch)
                test_loss_total += loss.item() * len(features_batch)
        test_loss = test_loss_total / len(test_dataset)

        print(f"Época {epoch:02d} | Erro médio de treino (MAE): {train_loss:.2f} | Erro médio de teste (MAE): {test_loss:.2f}")

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH / MODEL_NAME)
    print(f"Modelo salvo em {MODEL_PATH / MODEL_NAME}")


if __name__ == "__main__":
    main()