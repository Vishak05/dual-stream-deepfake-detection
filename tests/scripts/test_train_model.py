from pathlib import Path
import importlib.util
import sys

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train_model.py"
_spec = importlib.util.spec_from_file_location("train_model", _SCRIPT_PATH)
train_model = importlib.util.module_from_spec(_spec)
sys.modules["train_model"] = train_model
_spec.loader.exec_module(train_model)


def _parse(tmp_path, *extra):
    return train_model.build_arg_parser().parse_args([
        "--manifest", str(tmp_path / "manifest.csv"),
        "--model", "dual_stream",
        "--out-dir", str(tmp_path / "runs" / "exp1"),
        *extra,
    ])


def test_arg_parser_defaults_match_base_paper_hyperparameters(tmp_path):
    args = _parse(tmp_path)
    assert args.epochs == 20
    assert args.batch_size == 64
    assert args.lr == 1e-4
    assert args.val_split == 0.25
    assert args.model == "dual_stream"


def test_arg_parser_rejects_unknown_model_name(tmp_path):
    parser = train_model.build_arg_parser()
    try:
        parser.parse_args([
            "--manifest", str(tmp_path / "manifest.csv"),
            "--model", "not_a_real_model",
            "--out-dir", str(tmp_path),
        ])
        assert False, "expected SystemExit from argparse choices validation"
    except SystemExit:
        pass


def test_spectral_defaults_select_the_redesigned_representation(tmp_path):
    args = _parse(tmp_path)
    assert args.freq_variant == "radial"
    assert args.n_bins == 16
    assert args.freq_transform == "fft"
    assert args.no_window is False
    assert args.keep_dc is False


def test_scalar_variant_is_available_as_the_ablation_arm(tmp_path):
    assert _parse(tmp_path, "--freq-variant", "scalar").freq_variant == "scalar"


def test_logit_adjustment_is_opt_in(tmp_path):
    assert _parse(tmp_path).logit_adjust is False
    assert _parse(tmp_path, "--logit-adjust").logit_adjust is True


def test_frequency_only_model_is_selectable(tmp_path):
    args = train_model.build_arg_parser().parse_args([
        "--manifest", str(tmp_path / "m.csv"),
        "--model", "frequency_only",
        "--out-dir", str(tmp_path / "run"),
    ])
    assert args.model == "frequency_only"


def test_build_model_passes_the_spectral_configuration(tmp_path):
    args = _parse(tmp_path, "--freq-variant", "radial", "--n-bins", "8",
                  "--freq-transform", "dct", "--keep-dc")
    model = train_model.build_model(args)
    assert model.frequency.n_bins == 8
    assert model.frequency.transform == "dct"
    assert model.frequency.drop_dc is False


def test_build_model_ignores_spectral_flags_for_the_spatial_baseline(tmp_path):
    args = train_model.build_arg_parser().parse_args([
        "--manifest", str(tmp_path / "m.csv"),
        "--model", "spatial_only",
        "--out-dir", str(tmp_path / "run"),
        "--n-bins", "8",
    ])
    model = train_model.build_model(args)  # must not raise on the unused flag
    assert not hasattr(model, "frequency")


# --- stratified split ------------------------------------------------------

class _Entry:
    def __init__(self, label, source):
        self.label = label
        self.source = source


def test_split_is_stratified_by_label():
    """random_split left the class balance to chance; a rare generator could land
    entirely on one side of the split."""
    entries = [_Entry(0, "FFHQ") for _ in range(80)] + [_Entry(1, "StyleGAN2") for _ in range(20)]
    train_idx, val_idx = train_model.stratified_split(entries, val_split=0.25, seed=42)

    assert len(train_idx) == 75 and len(val_idx) == 25
    assert set(train_idx).isdisjoint(val_idx)
    val_fake = sum(1 for i in val_idx if entries[i].label == 1)
    assert val_fake == 5  # 20% of the validation split, matching the overall prior


def test_split_falls_back_when_a_stratum_is_too_small():
    """One-off sources must not crash the split; it degrades to coarser strata."""
    entries = [_Entry(0, "FFHQ") for _ in range(20)] + [_Entry(1, "RareGAN")]
    train_idx, val_idx = train_model.stratified_split(entries, val_split=0.25, seed=0)
    assert len(train_idx) + len(val_idx) == 21


def test_split_is_deterministic_for_a_given_seed():
    entries = [_Entry(i % 2, "FFHQ" if i % 2 == 0 else "StyleGAN2") for i in range(40)]
    first = train_model.stratified_split(entries, 0.25, seed=3)
    second = train_model.stratified_split(entries, 0.25, seed=3)
    assert (first[0] == second[0]).all() and (first[1] == second[1]).all()
