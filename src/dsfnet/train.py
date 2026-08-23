from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochResult:
    """Algorithm 3 (Figure 7), inner batch loop for one epoch."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return EpochResult(loss=total_loss / total, accuracy=correct / total)


@torch.no_grad()
def evaluate_loss_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EpochResult:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return EpochResult(loss=total_loss / total, accuracy=correct / total)
