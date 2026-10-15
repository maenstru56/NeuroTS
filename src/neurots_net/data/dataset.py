
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from neurots_net.config import AugmentationConfig
from neurots_net.data.augmentations import apply_neurots_augmentation
from neurots_net.data.brats26 import CachedCaseRecord, load_cached_case


TARGET_LABELS = {"et": 1, "net": 2, "cc": 3, "ed": 4}
TARGET_NAMES_BY_LABEL = {value: key for key, value in TARGET_LABELS.items()}


def extract_padded_patch(array: np.ndarray, origin: tuple[int, int, int], patch_size: tuple[int, int, int]) -> np.ndarray:
    spatial_shape = array.shape[-3:]
    output_shape = (*array.shape[:-3], *patch_size)
    output = np.zeros(output_shape, dtype=array.dtype)
    source_slices = []
    target_slices = []
    for axis in range(3):
        start = int(origin[axis])
        end = start + int(patch_size[axis])
        src_start = max(0, start)
        src_end = min(int(spatial_shape[axis]), end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        source_slices.append(slice(src_start, src_end))
        target_slices.append(slice(dst_start, dst_end))
    output[(..., *target_slices)] = array[(..., *source_slices)]
    return output


class CachedBraTSPatchDataset(Dataset):
    def __init__(
        self,
        records: list[CachedCaseRecord],
        patch_size: list[int] | tuple[int, int, int],
        epoch_size: int,
        foreground_patch_prob: float | None,
        min_foreground_voxels: int,
        label_mode: str,
        patch_sampling_strategy: str,
        patch_sampling_probabilities: dict[str, float],
        target_class_jitter_voxels: int,
        targeted_patch_retries: int,
        cache_memory_cases: int,
        augment: bool,
        augmentation_config: AugmentationConfig,
        seed: int,
    ) -> None:
        if not records:
            raise ValueError("CachedBraTSPatchDataset requires at least one record.")
        self.records = list(records)
        self.patch_size = tuple(int(item) for item in patch_size)
        self.epoch_size = int(epoch_size)
        self.foreground_patch_prob = 0.0 if foreground_patch_prob is None else float(foreground_patch_prob)
        self.min_foreground_voxels = int(min_foreground_voxels)
        self.label_mode = str(label_mode).strip().lower()
        if self.label_mode != "multiclass":
            raise ValueError("The public NeuroTS-Net dataset supports label_mode='multiclass' only.")
        self.patch_sampling_strategy = str(patch_sampling_strategy).strip().lower()
        self.patch_sampling_probabilities = self._normalize_probabilities(
            patch_sampling_probabilities,
            require_unit_sum=self.patch_sampling_strategy == "component_balanced",
            name="patch_sampling_probabilities",
        )
        self.target_class_jitter_voxels = max(0, int(target_class_jitter_voxels))
        self.targeted_patch_retries = max(1, int(targeted_patch_retries))
        self.cache_memory_cases = max(0, int(cache_memory_cases))
        self.augment = bool(augment)
        self.augmentation_config = augmentation_config
        self.seed = int(seed)
        self.epoch = 0
        self._memory_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._component_cache: OrderedDict[tuple[str, int], list[np.ndarray]] = OrderedDict()
        self._record_positive_classes: dict[str, set[int]] = {}
        self.target_labels = TARGET_LABELS
        self.target_names_by_label = TARGET_NAMES_BY_LABEL
        self.ed_target_class_id = TARGET_LABELS["ed"]
        self._positive_records = self._build_positive_record_index()

    @staticmethod
    def _normalize_probabilities(
        probabilities: dict[str, float] | None,
        *,
        require_unit_sum: bool = False,
        name: str = "probabilities",
    ) -> dict[str, float]:
        if not probabilities:
            return {}
        cleaned = {str(key).strip().lower(): max(0.0, float(value)) for key, value in probabilities.items()}
        total = sum(cleaned.values())
        if total <= 0.0:
            return {}
        if require_unit_sum and abs(total - 1.0) > 1e-4:
            raise ValueError(f"{name} must sum to 1.0 for component_balanced sampling, got {total:.8f}: {cleaned}")
        return {key: value / total for key, value in cleaned.items()}

    @staticmethod
    def _weighted_choice(probabilities: dict[str, float], rng: np.random.Generator, fallback: str) -> str:
        if not probabilities:
            return fallback
        keys = list(probabilities)
        weights = np.asarray([probabilities[key] for key in keys], dtype=np.float64)
        weights = weights / weights.sum().clip(min=1e-12)
        return str(keys[int(rng.choice(len(keys), p=weights))])

    def _build_positive_record_index(self) -> dict[int, list[CachedCaseRecord]]:
        positive: dict[int, list[CachedCaseRecord]] = {class_id: [] for class_id in self.target_labels.values()}
        for record in self.records:
            try:
                metadata = load_cached_case(record.cache_path)["metadata"]
                volumes = metadata.get("label_properties", {}).get("label_volumes", {})
                present_classes: set[int] = set()
                for class_id in positive:
                    if int(volumes.get(str(class_id), 0)) > 0:
                        positive[class_id].append(record)
                        present_classes.add(int(class_id))
                self._record_positive_classes[str(record.cache_path)] = present_classes
            except Exception:
                continue
        return positive

    def _present_classes(self, record: CachedCaseRecord, options: tuple[int, ...] | None = None) -> list[int]:
        if options is None:
            options = tuple(sorted(self.target_labels.values()))
        present = self._record_positive_classes.get(str(record.cache_path), set())
        return [int(class_id) for class_id in options if int(class_id) in present]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size

    def _load_record(self, record: CachedCaseRecord) -> dict[str, Any]:
        key = str(record.cache_path)
        if self.cache_memory_cases > 0 and key in self._memory_cache:
            payload = self._memory_cache.pop(key)
            self._memory_cache[key] = payload
            return payload
        payload = load_cached_case(record.cache_path)
        if "label" not in payload:
            raise ValueError(f"Training cache file has no label: {record.cache_path}")
        if self.cache_memory_cases > 0:
            self._memory_cache[key] = payload
            while len(self._memory_cache) > self.cache_memory_cases:
                self._memory_cache.popitem(last=False)
        return payload

    def _component_voxel_lists(self, record: CachedCaseRecord, label: np.ndarray, class_id: int) -> list[np.ndarray]:
        key = (str(record.cache_path), int(class_id))
        cached = self._component_cache.get(key)
        if cached is not None:
            self._component_cache.move_to_end(key)
            return cached
        structure = np.ones((3, 3, 3), dtype=bool)
        components, count = ndimage.label(label == int(class_id), structure=structure)
        voxel_lists = [np.argwhere(components == component_id) for component_id in range(1, int(count) + 1)]
        voxel_lists = [voxels for voxels in voxel_lists if len(voxels) > 0]
        self._component_cache[key] = voxel_lists
        while len(self._component_cache) > max(16, self.cache_memory_cases * 8):
            self._component_cache.popitem(last=False)
        return voxel_lists

    def _sample_random_origin(self, label: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int]:
        shape = tuple(int(item) for item in label.shape)
        patch = self.patch_size
        origin = [
            int(rng.integers(0, max(1, shape[axis] - patch[axis] + 1))) if shape[axis] > patch[axis] else -(patch[axis] - shape[axis]) // 2
            for axis in range(3)
        ]
        return tuple(origin)

    def _sample_foreground_origin(self, label: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int]:
        shape = tuple(int(item) for item in label.shape)
        patch = self.patch_size
        if int((label > 0).sum()) >= self.min_foreground_voxels:
            coords = np.argwhere(label > 0)
            center = coords[int(rng.integers(0, len(coords)))]
            return tuple(int(center[axis]) - patch[axis] // 2 for axis in range(3))
        return self._sample_random_origin(label, rng)

    def _sample_target_origin(
        self,
        record: CachedCaseRecord,
        label: np.ndarray,
        class_id: int,
        rng: np.random.Generator,
    ) -> tuple[tuple[int, int, int], bool]:
        patch = self.patch_size
        components = self._component_voxel_lists(record, label, class_id)
        if not components:
            return self._sample_foreground_origin(label, rng), True
        for _ in range(self.targeted_patch_retries):
            component = components[int(rng.integers(0, len(components)))]
            center = component[int(rng.integers(0, len(component)))].astype(np.int64, copy=False)
            if self.target_class_jitter_voxels > 0:
                jitter = rng.integers(
                    -self.target_class_jitter_voxels,
                    self.target_class_jitter_voxels + 1,
                    size=3,
                    endpoint=False,
                )
                center = center + jitter
            origin = tuple(int(center[axis]) - patch[axis] // 2 for axis in range(3))
            label_patch = extract_padded_patch(label, origin, patch)
            if int((label_patch == int(class_id)).sum()) > 0:
                return origin, False
        component = components[int(rng.integers(0, len(components)))]
        center = component[int(rng.integers(0, len(component)))]
        return tuple(int(center[axis]) - patch[axis] // 2 for axis in range(3)), True

    def _choose_record_for_class(self, class_id: int, rng: np.random.Generator) -> CachedCaseRecord:
        candidates = self._positive_records.get(int(class_id)) or self.records
        return candidates[int(rng.integers(0, len(candidates)))]

    def _choose_hard_negative_record(self, rng: np.random.Generator) -> CachedCaseRecord:
        ed_positive = set(self._positive_records.get(self.ed_target_class_id) or [])
        ed_negative = [record for record in self.records if record not in ed_positive]
        non_ed_targets = tuple(class_id for class_id in sorted(self.target_labels.values()) if int(class_id) != self.ed_target_class_id)
        tumor_ed_negative = [record for record in ed_negative if self._present_classes(record, non_ed_targets)]
        candidates = tumor_ed_negative or ed_negative or self.records
        return candidates[int(rng.integers(0, len(candidates)))]

    def _sample_origin(self, record: CachedCaseRecord, label: np.ndarray, rng: np.random.Generator) -> tuple[tuple[int, int, int], str, int | None, bool]:
        use_foreground = rng.random() < self.foreground_patch_prob
        sample_type = "foreground" if use_foreground else "random"
        origin = self._sample_foreground_origin(label, rng) if use_foreground else self._sample_random_origin(label, rng)
        return origin, sample_type, None, False

    def _sample_hard_negative_origin(
        self,
        label: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[tuple[int, int, int], bool]:
        non_ed_foreground = np.isin(label, [1, 2, 3])
        generic_foreground = label > 0
        patch = self.patch_size
        for _ in range(self.targeted_patch_retries):
            if bool(non_ed_foreground.any()):
                coords = np.argwhere(non_ed_foreground)
            elif bool(generic_foreground.any()):
                coords = np.argwhere(generic_foreground)
            else:
                coords = np.empty((0, 3), dtype=np.int64)
            if coords.shape[0] > 0:
                center = coords[int(rng.integers(0, len(coords)))].astype(np.int64, copy=False)
                if self.target_class_jitter_voxels > 0:
                    center = center + rng.integers(
                        -self.target_class_jitter_voxels,
                        self.target_class_jitter_voxels + 1,
                        size=3,
                        endpoint=False,
                    )
                origin = tuple(int(center[axis]) - patch[axis] // 2 for axis in range(3))
            else:
                origin = self._sample_random_origin(label, rng)
            label_patch = extract_padded_patch(label, origin, patch)
            if (
                int((label_patch == self.ed_target_class_id).sum()) == 0
                and int((label_patch > 0).sum()) >= self.min_foreground_voxels
            ):
                return origin, False
        for _ in range(self.targeted_patch_retries):
            origin = self._sample_random_origin(label, rng)
            label_patch = extract_padded_patch(label, origin, patch)
            if int((label_patch == self.ed_target_class_id).sum()) == 0:
                return origin, True
        return self._sample_foreground_origin(label, rng), True

    def _class_id_for_name(self, name: str) -> int | None:
        return self.target_labels.get(str(name).strip().lower())

    def _sample_info(self, sample_type: str, target_class_id: int | None, fallback: bool, label_patch: np.ndarray) -> dict[str, Any]:
        target_class_name = "" if target_class_id is None else self.target_names_by_label.get(int(target_class_id), "")
        info: dict[str, Any] = {
            "requested_target": str(sample_type),
            "successful_target": "" if bool(fallback) else str(sample_type),
            "fallback_target": str(sample_type) if bool(fallback) else "",
            "target_class_id": 0 if target_class_id is None else int(target_class_id),
            "target_class_name": target_class_name,
            "target_voxels": 0,
        }
        if target_class_id is not None:
            target_voxels = int((label_patch == int(target_class_id)).sum())
            info["target_voxels"] = target_voxels
            if target_voxels <= 0:
                info["successful_target"] = ""
                info["fallback_target"] = str(sample_type)
        for name in TARGET_LABELS:
            class_id = self._class_id_for_name(name)
            voxels = int((label_patch == int(class_id)).sum()) if class_id is not None else 0
            info[f"actual_contains_{name}"] = int(voxels > 0)
            info[f"actual_voxels_{name}"] = voxels
        if sample_type == "hard_negative" and int(info["actual_contains_ed"]) > 0:
            info["successful_target"] = ""
            info["fallback_target"] = str(sample_type)
        return info

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index))
        if self.patch_sampling_strategy == "component_balanced":
            sample_type = self._weighted_choice(self.patch_sampling_probabilities, rng, fallback="foreground")
            target_class_id = TARGET_LABELS.get(sample_type)
            if sample_type == "hard_negative":
                record = self._choose_hard_negative_record(rng)
            elif target_class_id is not None:
                record = self._choose_record_for_class(int(target_class_id), rng)
            else:
                record = self.records[int(rng.integers(0, len(self.records)))]
        else:
            sample_type = "legacy"
            target_class_id = None
            record = self.records[int(rng.integers(0, len(self.records)))]

        payload = self._load_record(record)
        image = payload["image"].astype(np.float32, copy=False)
        label = payload["label"].astype(np.int64, copy=False)
        fallback = False
        if self.patch_sampling_strategy == "component_balanced":
            if sample_type == "random":
                origin = self._sample_random_origin(label, rng)
            elif sample_type == "foreground":
                origin = self._sample_foreground_origin(label, rng)
            elif sample_type == "hard_negative":
                origin, fallback = self._sample_hard_negative_origin(label, rng)
            elif target_class_id is not None:
                origin, fallback = self._sample_target_origin(record, label, int(target_class_id), rng)
            else:
                origin = self._sample_foreground_origin(label, rng)
                fallback = True
        else:
            origin, sample_type, target_class_id, fallback = self._sample_origin(record, label, rng)

        image_patch = extract_padded_patch(image, origin, self.patch_size).astype(np.float32, copy=False)
        label_patch = extract_padded_patch(label, origin, self.patch_size).astype(np.int64, copy=False)
        sample_info = self._sample_info(sample_type, target_class_id, fallback, label_patch)
        if self.augment:
            image_patch, label_patch = apply_neurots_augmentation(
                image_patch,
                label_patch,
                rng,
                asdict(self.augmentation_config),
                self.patch_size,
            )
        return (
            torch.from_numpy(np.ascontiguousarray(image_patch)),
            torch.from_numpy(np.ascontiguousarray(label_patch)),
            sample_info,
        )
