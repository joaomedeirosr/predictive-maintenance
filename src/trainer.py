from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader


def run_epoch(model, loader: DataLoader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0

    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += ((torch.sigmoid(logits) >= 0.5) == labels.bool()).sum().item()
            count += labels.size(0)
    return {"loss": total_loss / count, "accuracy": correct / count}


def train_model(model, train_loader, val_loader, config, device):
    labels = train_loader.dataset.tensors[1]
    positives = labels.sum().item()
    negatives = len(labels) - positives
    pos_weight = torch.tensor([negatives / positives if positives else 1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint = Path(config.checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"Epoch {epoch:02d} | train loss {train_metrics['loss']:.4f} | "
              f"val loss {val_metrics['loss']:.4f} | val acc {val_metrics['accuracy']:.4f}")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return history
