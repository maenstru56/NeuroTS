from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
from tqdm import tqdm

from neurots_net.config import DataConfig
from neurots_net.data.nifti_io import load_nifti_zyx
from neurots_net.utils.io import ensure_dir, save_json


CASE_PATTERN = re.compile(r"^BraTS-PED-\d{5}-\d{3}$")


@dataclass(frozen=True)
class BraTSCase:
    case_id: str
    case_dir: Path
    modality_paths: dict[str, Path]
    seg_path: Path | None
    split: str


@dataclass(frozen=True)
class CachedCaseRecord:
    case_id: str
    cache_path: Path
    split: str
    group_id: str


def group_id_from_case_id(case_id: str) -> str:
    parts = case_id.split("-")
    if len(parts) >= 4:
        return "-".join(parts[:3])
    return case_id


def _case_sort_key(path: Path) -> str:
    return path.name


def _find_case_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(
        [item for item in root_path.iterdir() if item.is_dir() and CASE_PATTERN.match(item.name)],
        key=_case_sort_key,
    )


def _find_modality_path(case_dir: Path, case_id: str, modality: str) -> Path:
    expected = case_dir / f"{case_id}-{modality}.nii.gz"
    if expected.exists():
        return expected
    matches = sorted(case_dir.glob(f"*-{modality}.nii.gz"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not find modality '{modality}' for {case_id} in {case_dir}.")


def discover_training_cases(cfg: DataConfig) -> list[BraTSCase]:
    cases: dict[str, BraTSCase] = {}
    for root in cfg.train_roots:
        for case_dir in _find_case_dirs(root):
            case_id = case_dir.name
            if case_id in cases:
                raise ValueError(f"Duplicate training case id discovered: {case_id}")
            modality_paths = {modality: _find_modality_path(case_dir, case_id, modality) for modality in cfg.modalities}
            seg_path = case_dir / f"{case_id}-seg.nii.gz"
            if not seg_path.exists():
                raise FileNotFoundError(f"Missing segmentation for training case {case_id}: {seg_path}")
            cases[case_id] = BraTSCase(case_id, case_dir, modality_paths, seg_path, "train")
    return [cases[key] for key in sorted(cases)]


def discover_validation_cases(cfg: DataConfig) -> list[BraTSCase]:
    cases = []
    for case_dir in _find_case_dirs(cfg.validation_root):
        case_id = case_dir.name
        modality_paths = {modality: _find_modality_path(case_dir, case_id, modality) for modality in cfg.modalities}
        cases.append(BraTSCase(case_id, case_dir, modality_paths, None, "validation"))
    return cases


def foreground_bbox(image: np.ndarray, margin: int) -> tuple[list[int], list[int]]:
    if image.ndim != 4:
        raise ValueError(f"Expected image shape [C,D,H,W], got {image.shape}.")
    mask = np.any(np.isfinite(image) & (np.abs(image) > 0), axis=0)
    if not bool(mask.any()):
        start = [0, 0, 0]
        end = [int(item) for item in image.shape[1:]]
        return start, end
    coords = np.where(mask)
    start = [max(0, int(coords[axis].min()) - int(margin)) for axis in range(3)]
    end = [
        min(int(image.shape[axis + 1]), int(coords[axis].max()) + int(margin) + 1)
        for axis in range(3)
    ]
    return start, end


def crop_zyx(array: np.ndarray, start: list[int], end: list[int]) -> np.ndarray:
    slices = tuple(slice(int(start[axis]), int(end[axis])) for axis in range(3))
    if array.ndim == 4:
        return array[(slice(None), *slices)]
    return array[slices]


def _clip_and_normalize_modalities(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    percentiles: list[float],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    normalized = np.empty_like(image, dtype=np.float32)
    stats: list[dict[str, float]] = []
    lo_pct, hi_pct = float(percentiles[0]), float(percentiles[1])
    for channel in range(image.shape[0]):
        values = image[channel]
        valid = foreground_mask & np.isfinite(values)
        if int(valid.sum()) < 16:
            valid = np.isfinite(values)
        valid_values = values[valid].astype(np.float32, copy=False)
        if valid_values.size == 0:
            mean = 0.0
            std = 1.0
            lo = float(np.nanmin(values)) if values.size else 0.0
            hi = float(np.nanmax(values)) if values.size else 1.0
        else:
            lo = float(np.percentile(valid_values, lo_pct))
            hi = float(np.percentile(valid_values, hi_pct))
            if hi <= lo:
                hi = lo + 1.0
            clipped_valid = np.clip(valid_values, lo, hi)
            mean = float(clipped_valid.mean())
            std = float(clipped_valid.std())
            if std < 1e-6:
                std = 1.0
        clipped = np.clip(values.astype(np.float32, copy=False), lo, hi)
        normalized[channel] = (clipped - mean) / std
        stats.append({"clip_low": lo, "clip_high": hi, "mean": mean, "std": std})
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized, stats


def load_case_arrays(case: BraTSCase, modalities: list[str]) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    modality_arrays = []
    spacing = None
    shape_xyz = None
    for modality in modalities:
        loaded = load_nifti_zyx(case.modality_paths[modality], dtype=np.float32)
        if spacing is None:
            spacing = loaded.spacing_zyx
            shape_xyz = loaded.shape_xyz
        elif loaded.data_zyx.shape != modality_arrays[0].shape:
            raise ValueError(f"Shape mismatch in {case.case_id} for modality {modality}.")
        modality_arrays.append(loaded.data_zyx)
    image = np.stack(modality_arrays, axis=0)
    label = None
    if case.seg_path is not None:
        loaded_label = load_nifti_zyx(case.seg_path, dtype=np.uint8)
        if loaded_label.data_zyx.shape != image.shape[1:]:
            raise ValueError(f"Label shape mismatch for {case.case_id}.")
        label = loaded_label.data_zyx.astype(np.uint8, copy=False)
    metadata = {
        "case_id": case.case_id,
        "split": case.split,
        "modalities": list(modalities),
        "source_shape_zyx": [int(item) for item in image.shape[1:]],
        "source_shape_xyz": [int(item) for item in shape_xyz],
        "source_spacing_zyx": [float(item) for item in spacing],
        "reference_modality": str(case.modality_paths[modalities[0]]),
        "modality_paths": {key: str(value) for key, value in case.modality_paths.items()},
        "seg_path": None if case.seg_path is None else str(case.seg_path),
    }
    return image, label, metadata


def label_properties(label: np.ndarray) -> dict[str, Any]:
    label = np.asarray(label)
    volumes = {str(class_id): int((label == class_id).sum()) for class_id in [1, 2, 3, 4]}
    wt_volume = int((label > 0).sum())
    tc_volume = int(np.isin(label, [1, 2, 3]).sum())
    return {
        "et_present": bool(volumes["1"] > 0),
        "net_present": bool(volumes["2"] > 0),
        "cc_present": bool(volumes["3"] > 0),
        "ed_present": bool(volumes["4"] > 0),
        "label_volumes": volumes,
        "wt_volume": wt_volume,
        "tc_volume": tc_volume,
    }


def preprocess_case(
    case: BraTSCase,
    output_path: str | Path,
    cfg: DataConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists() and not overwrite:
        with np.load(output, allow_pickle=False) as cached:
            return json.loads(str(cached["metadata"].item()))
    image, label, metadata = load_case_arrays(case, cfg.modalities)
    start, end = foreground_bbox(image, margin=cfg.crop_margin)
    image_crop = crop_zyx(image, start, end)
    foreground_mask = np.any(np.abs(image_crop) > 0, axis=0)
    image_crop, stats = _clip_and_normalize_modalities(
        image_crop,
        foreground_mask=foreground_mask,
        percentiles=cfg.intensity_clip_percentiles,
    )
    label_crop = crop_zyx(label, start, end) if label is not None else None
    metadata.update(
        {
            "crop_bbox_zyx": {"start": [int(item) for item in start], "end": [int(item) for item in end]},
            "crop_shape_zyx": [int(item) for item in image_crop.shape[1:]],
            "normalization": "foreground_zscore",
            "modality_stats": stats,
            "group_id": group_id_from_case_id(case.case_id),
        }
    )
    if label_crop is not None:
        metadata["label_properties"] = label_properties(label_crop)
    requested_dtype = np.dtype(cfg.image_dtype)
    if requested_dtype not in (np.float16, np.float32):
        raise ValueError("data.image_dtype must be float16 or float32.")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "image": image_crop.astype(requested_dtype, copy=False),
        "metadata": np.asarray(json.dumps(metadata)),
    }
    if label_crop is not None:
        payload["label"] = label_crop.astype(np.uint8, copy=False)
    np.savez_compressed(output, **payload)
    return metadata


def preprocess_dataset(
    cfg: DataConfig,
    *,
    include_training: bool = True,
    include_validation: bool = True,
    overwrite: bool | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    overwrite = cfg.overwrite_cache if overwrite is None else bool(overwrite)
    cache_root = ensure_dir(cfg.cache_dir)
    summary: dict[str, Any] = {"cache_dir": str(cache_root), "training": [], "validation": []}
    if include_training:
        cases = discover_training_cases(cfg)
        logger.info("Preprocessing %d training cases into %s", len(cases), cache_root / "train")
        for case in tqdm(cases, desc="training cache"):
            metadata = preprocess_case(case, cache_root / "train" / f"{case.case_id}.npz", cfg, overwrite=overwrite)
            summary["training"].append(metadata)
    if include_validation:
        cases = discover_validation_cases(cfg)
        logger.info("Preprocessing %d validation cases into %s", len(cases), cache_root / "validation")
        for case in tqdm(cases, desc="validation cache"):
            metadata = preprocess_case(case, cache_root / "validation" / f"{case.case_id}.npz", cfg, overwrite=overwrite)
            summary["validation"].append(metadata)
    save_json(summary, cache_root / "preprocess_summary.json")
    return summary


def cached_records(cfg: DataConfig, split: str) -> list[CachedCaseRecord]:
    split_key = "validation" if split == "validation" else "train"
    root = Path(cfg.cache_dir) / split_key
    records = []
    for path in sorted(root.glob("*.npz")):
        case_id = path.stem
        records.append(CachedCaseRecord(case_id=case_id, cache_path=path, split=split_key, group_id=group_id_from_case_id(case_id)))
    return records


def load_cached_case(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        payload: dict[str, Any] = {
            "image": data["image"],
            "metadata": json.loads(str(data["metadata"].item())),
        }
        if "label" in data.files:
            payload["label"] = data["label"]
        for optional_key in ["sampler_label", "hard_negative_mask", "brain_mask", "base_segmentation"]:
            if optional_key in data.files:
                payload[optional_key] = data[optional_key]
    return payload


def ensure_cache_available(cfg: DataConfig) -> None:
    missing = []
    for split in ["train", "validation"]:
        split_root = Path(cfg.cache_dir) / split
        if not split_root.exists() or not any(split_root.glob("*.npz")):
            missing.append(str(split_root))
    if missing:
        raise FileNotFoundError(
            "Missing NeuroTS cache directories. Run `python scripts/preprocess_brats26.py --config "
            f"configs/neurots_brats26_peds_5fold.yaml` first. Missing: {', '.join(missing)}"
        )
