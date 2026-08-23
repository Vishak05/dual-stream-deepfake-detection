import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dsfnet.train import train_one_epoch, evaluate_loss_accuracy


class _TinyDummyModel(nn.Module):
    """Small stand-in classifier so this test exercises the training loop
    logic itself, independent of the real (slower) DualStreamNet/ResNet-18."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))

    def forward(self, x):
        return self.net(x)


def _make_learnable_loader():
    torch.manual_seed(0)
    x = torch.rand(32, 3, 8, 8)
    # label depends on mean pixel value so the tiny linear model can learn it
    y = (x.mean(dim=(1, 2, 3)) > 0.5).long()
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)


def test_train_one_epoch_returns_finite_loss_and_accuracy():
    model = _TinyDummyModel()
    loader = _make_learnable_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"))
    assert result.loss == result.loss  # not NaN
    assert 0.0 <= result.accuracy <= 1.0


def test_train_one_epoch_reduces_loss_over_several_epochs():
    torch.manual_seed(0)
    model = _TinyDummyModel()
    loader = _make_learnable_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    first = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"))
    for _ in range(9):
        last = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"))

    assert last.loss < first.loss


def test_evaluate_loss_accuracy_does_not_update_weights():
    model = _TinyDummyModel()
    loader = _make_learnable_loader()
    before = [p.clone() for p in model.parameters()]
    evaluate_loss_accuracy(model, loader, device=torch.device("cpu"))
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))
