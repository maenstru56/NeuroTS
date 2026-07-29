from pathlib import Path

import numpy as np

from neurots_net.engine.train import _learning_rate_for_epoch
from neurots_net.config import ExperimentConfig
from neurots_net.postprocessing import (
    PostprocessingParameters,
    hard_seg_from_main_probabilities,
    postprocess_main_probabilities,
)
from neurots_net.inference import find_fold_checkpoints


def test_cosine_schedule_warms_up_then_decays() -> None:
    cfg = ExperimentConfig()
    cfg.train.scheduler = "cosine"
    cfg.train.learning_rate = 1.2e-3
    cfg.train.cosine_eta_min = 1e-6
    cfg.train.warmup_epochs = 10
    cfg.train.epochs = 100
    assert _learning_rate_for_epoch(cfg, 1) < _learning_rate_for_epoch(cfg, 10)
    assert _learning_rate_for_epoch(cfg, 10) > _learning_rate_for_epoch(cfg, 100)
    assert _learning_rate_for_epoch(cfg, 100) >= cfg.train.cosine_eta_min


def test_core_postprocessing_preserves_main_ed() -> None:
    probabilities = np.zeros((5, 8, 8, 8), dtype=np.float32)
    probabilities[0] = 0.9
    probabilities[4, 2:6, 2:6, 2:6] = 0.95
    probabilities[0, 2:6, 2:6, 2:6] = 0.01
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    base = hard_seg_from_main_probabilities(probabilities)
    params = PostprocessingParameters(
        cc_prob_threshold=0.99,
        et_prob_threshold=0.99,
        protect_tc_wt=True,
    )
    result, _ = postprocess_main_probabilities(probabilities, params, base_segmentation=base)
    assert np.count_nonzero(result == 4) == 64


def test_rare_present_checkpoint_discovery_supports_single_split(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_rare_present_score.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert find_fold_checkpoints(tmp_path, "rare_present") == [checkpoint]
