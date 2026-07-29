from pathlib import Path

import pytest

from neurots_net.config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[1]


def test_public_configs_parse_and_select_expected_split_modes() -> None:
    fold = ExperimentConfig.from_yaml(ROOT / "configs" / "neurots_brats26_peds_5fold.yaml")
    single = ExperimentConfig.from_yaml(ROOT / "configs" / "neurots_brats26_peds_single_split.yaml")

    assert fold.data.split_mode == "fold"
    assert fold.data.num_folds == 5
    assert single.data.split_mode == "single"
    assert single.data.split_file
    assert single.data.split_name == "single_seed42"
    assert fold.data.label_mode == single.data.label_mode == "multiclass"


def test_sampling_probabilities_sum_to_one() -> None:
    for filename in ("neurots_brats26_peds_5fold.yaml", "neurots_brats26_peds_single_split.yaml"):
        cfg = ExperimentConfig.from_yaml(ROOT / "configs" / filename)
        assert sum(cfg.data.patch_sampling_probabilities.values()) == pytest.approx(1.0)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("data:\n  batch_size: 2\n  batch_size: 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate YAML key 'batch_size'"):
        ExperimentConfig.from_yaml(path)
