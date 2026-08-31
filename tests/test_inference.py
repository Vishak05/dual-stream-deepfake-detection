import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from dsfnet.inference import PredictionResult, load_predictions, predict_scores, save_predictions


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))

    def forward(self, x):
        return self.net(x)


class _SourceDataset(Dataset):
    """Stands in for DeepfakeImageDataset(return_source=True)."""

    def __init__(self, x, y, sources):
        self.x, self.y, self.sources = x, y, sources

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.sources[i]


def _loader(with_sources=False, n=16):
    torch.manual_seed(0)
    x = torch.rand(n, 3, 8, 8)
    y = (x.mean(dim=(1, 2, 3)) > 0.5).long()
    if with_sources:
        sources = ["FFHQ" if label == 0 else "StyleGAN3" for label in y.tolist()]
        return DataLoader(_SourceDataset(x, y, sources), batch_size=4)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_predict_scores_returns_probabilities():
    result = predict_scores(_TinyModel(), _loader(), torch.device("cpu"))
    assert result.y_true.shape == result.y_score.shape == (16,)
    assert ((result.y_score >= 0.0) & (result.y_score <= 1.0)).all()
    assert result.sources is None


def test_predict_scores_collects_sources_when_present():
    result = predict_scores(_TinyModel(), _loader(with_sources=True), torch.device("cpu"))
    assert result.sources is not None
    assert set(result.sources.tolist()) <= {"FFHQ", "StyleGAN3"}


def test_y_pred_applies_the_given_threshold():
    result = PredictionResult(y_true=np.array([0, 1]), y_score=np.array([0.4, 0.6]))
    assert result.y_pred(0.5).tolist() == [0, 1]
    assert result.y_pred(0.7).tolist() == [0, 0]
    assert result.y_pred(0.3).tolist() == [1, 1]


def test_scores_round_trip_with_metadata(tmp_path):
    """The training fake prior has to survive to disk -- prior correction is
    impossible without knowing what prior the model was trained under."""
    result = predict_scores(_TinyModel(), _loader(with_sources=True), torch.device("cpu"))
    path = tmp_path / "scores.npz"
    save_predictions(path, result, train_fake_prior=1 / 3, model="spatial_only")

    loaded, metadata = load_predictions(path)
    assert np.array_equal(loaded.y_true, result.y_true)
    assert np.allclose(loaded.y_score, result.y_score)
    assert np.array_equal(loaded.sources, result.sources)
    assert metadata["train_fake_prior"] == pytest.approx(1 / 3)
    assert metadata["model"] == "spatial_only"
