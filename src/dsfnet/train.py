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
    """Algorithm 3 (Figure 7), inner batch loop for one epoch.

    Uses automatic mixed precision on CUDA devices (a no-op on CPU) — this
    is transparent to callers and existing CPU-based tests, since autocast
    and GradScaler are both disabled when device.type != "cuda"."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    use_amp = device.type == "cuda"

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return EpochResult(loss=total_loss / total, accuracy=correct / total)
