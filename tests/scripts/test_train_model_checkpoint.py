from pathlib import Path
import importlib.util
import sys

import torch
import torch.nn as nn

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train_model.py"
_spec = importlib.util.spec_from_file_location("train_model", _SCRIPT_PATH)
train_model = importlib.util.module_from_spec(_spec)
sys.modules["train_model"] = train_model
_spec.loader.exec_module(train_model)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def test_save_and_load_checkpoint_round_trips_epoch_and_history(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"
    history = [{"epoch": 1, "train_loss": 0.5}, {"epoch": 2, "train_loss": 0.3}]

    train_model.save_checkpoint(checkpoint_path, model, optimizer, epoch=2, history=history)

    loaded_model = _TinyModel()
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-3)
    resumed_epoch, resumed_history = train_model.load_checkpoint(
        checkpoint_path, loaded_model, loaded_optimizer, device=torch.device("cpu")
    )

    assert resumed_epoch == 2
    assert resumed_history == history
    for original_param, loaded_param in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(original_param, loaded_param)


def test_load_checkpoint_restores_optimizer_state(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(4, 4)
    y = torch.tensor([0, 1, 0, 1])
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()  # gives Adam's internal state (exp_avg, exp_avg_sq) something non-zero to save

    checkpoint_path = tmp_path / "checkpoint.pt"
    train_model.save_checkpoint(checkpoint_path, model, optimizer, epoch=1, history=[])

    loaded_model = _TinyModel()
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-3)
    train_model.load_checkpoint(checkpoint_path, loaded_model, loaded_optimizer, device=torch.device("cpu"))

    original_state = optimizer.state_dict()["state"]
    loaded_state = loaded_optimizer.state_dict()["state"]
    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        assert torch.equal(original_state[key]["exp_avg"], loaded_state[key]["exp_avg"])
