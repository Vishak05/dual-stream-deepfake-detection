"""Scoring and per-sample score persistence.

Saving scores rather than only summary metrics is what makes the confound
analysis cheap: re-thresholding, prior correction, temperature scaling, matched
operating points and per-generator breakdowns are all recoverable from a stored
score vector, with no retraining.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class PredictionResult:
    y_true: np.ndarray
    y_score: np.ndarray             # P(fake)
    sources: np.ndarray | None = None

    def y_pred(self, threshold: float = 0.5) -> np.ndarray:
        return (self.y_score >= threshold).astype(int)


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> PredictionResult:
    """Scores every sample in `loader`. Handles datasets that yield sources."""
    model.eval()
    y_true: list[int] = []
    y_score: list[float] = []
    sources: list[str] = []

    for batch in loader:
        if len(batch) == 3:
            x, y, batch_sources = batch
            sources.extend(batch_sources)
        else:
            x, y = batch
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1)[:, 1]
        y_true.extend(y.tolist())
        y_score.extend(probs.cpu().tolist())

    return PredictionResult(
        y_true=np.asarray(y_true),
        y_score=np.asarray(y_score),
        sources=np.asarray(sources) if sources else None,
    )


def save_predictions(path: Path, result: PredictionResult, **metadata) -> None:
    """Writes scores plus metadata (notably the training fake prior, which prior
    correction needs) to a .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"y_true": result.y_true, "y_score": result.y_score}
    if result.sources is not None:
        arrays["sources"] = result.sources
    for key, value in metadata.items():
        arrays[f"meta_{key}"] = np.asarray(value)
    np.savez(path, **arrays)


def load_predictions(path: Path) -> tuple[PredictionResult, dict]:
    data = np.load(Path(path), allow_pickle=False)
    result = PredictionResult(
        y_true=data["y_true"],
        y_score=data["y_score"],
        sources=data["sources"] if "sources" in data else None,
    )
    metadata = {k[len("meta_"):]: data[k].item() for k in data.files if k.startswith("meta_")}
    return result, metadata
