import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    model.eval()
    probabilities, targets = [], []
    for features, labels in loader:
        probabilities.extend(torch.sigmoid(model(features.to(device))).cpu().numpy())
        targets.extend(labels.numpy())
    y_true = np.asarray(targets, dtype=int)
    y_pred = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }
