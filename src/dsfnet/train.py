from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def logit_adjustment_vector(
    fake_prior: float,
    tau: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Menon et al. (2021) logit adjustment: add tau * log(class prior) to the
    logits during training.

    This is the corrective counterpart to the prior-shift confound. Training on
    the StyleGAN3-augmented set moves the fake prior from 0.333 to 0.500, which
    by itself shifts the argmax boundary ~0.69 nats toward "fake". Adding the log
    prior during training makes the model learn a prior-free scoring function, so
    two models trained on differently balanced sets stay comparable at a fixed
    threshold.
    """
    if not 0.0 < fake_prior < 1.0:
        raise ValueError(f"fake_prior must be in (0, 1), got {fake_prior}")
    priors = torch.tensor([1.0 - fake_prior, fake_prior], dtype=torch.float32, device=device)
    return tau * torch.log(priors)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logit_adjustment: torch.Tensor | None = None,
) -> EpochResult:
    """Algorithm 3 (Figure 7), inner batch loop for one epoch.

    Uses automatic mixed precision on CUDA devices (a no-op on CPU) — this
    is transparent to callers and existing CPU-based tests, since autocast
    and GradScaler are both disabled when device.type != "cuda".

    `logit_adjustment` is added to the logits for the loss only; reported
    accuracy and any downstream scoring use the unadjusted logits.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    if logit_adjustment is not None:
        logit_adjustment = logit_adjustment.to(device)

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            adjusted = logits if logit_adjustment is None else logits + logit_adjustment
            loss = criterion(adjusted, y)
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
