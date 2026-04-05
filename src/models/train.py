"""Training helpers for PyTorch models."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def create_data_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """Creates a DataLoader.

    Args:
        x: Input array.
        y: Label array.
        batch_size: Batch size.
        shuffle: Shuffle flag.

    Returns:
        DataLoader instance.
    """
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    """Trains a model.

    Args:
        model: Model instance.
        train_loader: Train DataLoader.
        epochs: Number of epochs.
        learning_rate: Optimizer learning rate.
        device: Torch device.
    """
    model.to(device)
    class_weights = None
    if isinstance(train_loader.dataset, torch.utils.data.TensorDataset):
        labels = train_loader.dataset.tensors[1].detach().cpu().long()
        if labels.numel() > 0:
            counts = torch.bincount(labels)
            valid_counts = counts[counts > 0].float()
            if valid_counts.numel() > 0:
                class_weights = counts.sum().float() / (len(counts) * counts.clamp(min=1).float())
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")


def predict(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Runs prediction.

    Args:
        model: Trained model.
        x: Input array.
        batch_size: Batch size.
        device: Torch device.

    Returns:
        Predicted labels.
    """
    y_dummy = np.zeros((len(x),), dtype=np.int64)
    loader = create_data_loader(x=x, y=y_dummy, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device, dtype=torch.float32)
            logits = model(batch_x)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy().tolist())
    return np.asarray(preds, dtype=np.int64)
