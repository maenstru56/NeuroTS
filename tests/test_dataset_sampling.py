from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurots_net.config import AugmentationConfig
from neurots_net.data.brats26 import CachedCaseRecord
from neurots_net.data.dataset import CachedBraTSPatchDataset


def _write_case(path: Path, label: np.ndarray) -> CachedCaseRecord:
    image = np.zeros((4, *label.shape), dtype=np.float32)
    volumes = {str(class_id): int((label == class_id).sum()) for class_id in range(1, 5)}
    metadata = {"label_properties": {"label_volumes": volumes}}
    np.savez_compressed(
        path,
        image=image,
        label=label.astype(np.uint8),
        metadata=np.asarray(json.dumps(metadata)),
    )
    return CachedCaseRecord(path.stem, path, "train", path.stem)


def _dataset(records: list[CachedCaseRecord], probabilities: dict[str, float]) -> CachedBraTSPatchDataset:
    return CachedBraTSPatchDataset(
        records=records,
        patch_size=[16, 16, 16],
        epoch_size=8,
        foreground_patch_prob=None,
        min_foreground_voxels=8,
        label_mode="multiclass",
        patch_sampling_strategy="component_balanced",
        patch_sampling_probabilities=probabilities,
        target_class_jitter_voxels=4,
        targeted_patch_retries=8,
        cache_memory_cases=2,
        augment=False,
        augmentation_config=AugmentationConfig(enabled=False),
        seed=42,
    )


def test_targeted_ed_sampling_chooses_ed_positive_case(tmp_path: Path) -> None:
    negative = np.zeros((32, 32, 32), dtype=np.uint8)
    negative[8:24, 8:24, 8:24] = 2
    positive = negative.copy()
    positive[14:19, 14:19, 14:19] = 4
    records = [
        _write_case(tmp_path / "negative.npz", negative),
        _write_case(tmp_path / "positive.npz", positive),
    ]
    _, label, info = _dataset(records, {"ed": 1.0})[0]
    assert info["requested_target"] == "ed"
    assert info["fallback_target"] == ""
    assert int((label.numpy() == 4).sum()) > 0


def test_hard_negative_sampling_avoids_ed(tmp_path: Path) -> None:
    negative = np.zeros((32, 32, 32), dtype=np.uint8)
    negative[8:24, 8:24, 8:24] = 2
    positive = negative.copy()
    positive[14:19, 14:19, 14:19] = 4
    records = [
        _write_case(tmp_path / "negative.npz", negative),
        _write_case(tmp_path / "positive.npz", positive),
    ]
    _, label, info = _dataset(records, {"hard_negative": 1.0})[0]
    assert info["requested_target"] == "hard_negative"
    assert int((label.numpy() == 4).sum()) == 0
    assert int((label.numpy() > 0).sum()) >= 8
