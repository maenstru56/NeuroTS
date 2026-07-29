
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch

from neurots_net.losses import main_logits


LABEL_NAMES = {
    1: "ET",
    2: "NET",
    3: "CC",
    4: "ED",
}

LABEL_REGION_NAMES = {
    1: "ET",
    2: "NET",
    3: "CC",
    4: "ED",
}

REGIONS = {
    "ET": (1,),
    "NET": (2,),
    "CC": (3,),
    "ED": (4,),
    "TC": (1, 2, 3),
    "WT": (1, 2, 3, 4),
}

CHALLENGE_REGIONS = ("ED", "NET", "ET", "CC", "TC", "WT")
RICH_METRIC_NAMES = (
    "jaccard",
    "iou",
    "dice",
    "sdc",
    "nsd",
    "accuracy",
    "precision",
    "recall",
    "hd95",
)


def _dice_from_masks(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_sum = int(prediction.sum())
    target_sum = int(target.sum())
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    denominator = pred_sum + target_sum
    if denominator == 0:
        return 0.0
    intersection = int(np.logical_and(prediction, target).sum())
    return float(2.0 * intersection / denominator)


def _safe_divide(numerator: float, denominator: float, empty_value: float = 0.0) -> float:
    if denominator == 0:
        return float(empty_value)
    return float(numerator / denominator)


def _nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _nanstd(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size <= 1:
        return 0.0
    return float(finite.std(ddof=1))


def region_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    result: dict[str, float] = {}
    for region_name, labels in REGIONS.items():
        pred_mask = np.isin(prediction, labels)
        target_mask = np.isin(target, labels)
        intersection = int(np.logical_and(pred_mask, target_mask).sum())
        union = int(np.logical_or(pred_mask, target_mask).sum())
        result[f"dice_{region_name.lower()}"] = _dice_from_masks(pred_mask, target_mask)
        result[f"iou_{region_name.lower()}"] = _safe_divide(float(intersection), float(union), empty_value=1.0)
    result["mean_region_dice"] = float(np.mean([result[f"dice_{name.lower()}"] for name in REGIONS]))
    result["mean_region_iou"] = float(np.mean([result[f"iou_{name.lower()}"] for name in REGIONS]))
    return result


def _region_mask(label: np.ndarray, region_name: str) -> np.ndarray:
    return np.isin(label, REGIONS[region_name])


def _component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy import ndimage

    structure = np.ones((3, 3, 3), dtype=bool)
    return ndimage.label(mask.astype(bool, copy=False), structure=structure)


def _surface_coordinates(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    footprint = ndimage.generate_binary_structure(mask.ndim, int(connectivity))
    border = mask ^ ndimage.binary_erosion(mask, structure=footprint, border_value=0)
    return np.argwhere(border)


def _crop_to_union(test: np.ndarray, reference: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.where(np.logical_or(test, reference))
    if positions[0].size == 0:
        return test, reference
    lower = [max(0, int(axis.min()) - int(margin)) for axis in positions]
    upper = [min(test.shape[dim], int(axis.max()) + int(margin) + 1) for dim, axis in enumerate(positions)]
    slices = tuple(slice(lower[dim], upper[dim]) for dim in range(test.ndim))
    return test[slices], reference[slices]


def _sample_coordinates(coords: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or coords.shape[0] <= max_points:
        return coords
    if coords.shape[0] <= 1:
        return coords
    indices = np.linspace(0, coords.shape[0] - 1, num=int(max_points), dtype=np.int64)
    return coords[indices]


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    try:
        distances, _ = tree.query(source, k=1, workers=-1)
    except TypeError:
        distances, _ = tree.query(source, k=1)
    return np.asarray(distances, dtype=np.float64)


def approximate_surface_distances(
    test: np.ndarray,
    reference: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    connectivity: int = 1,
    max_points: int = 20_000,
    crop_margin: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    test = np.asarray(test, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    cropped_test, cropped_reference = _crop_to_union(test, reference, margin=crop_margin)
    test_coords = _surface_coordinates(cropped_test, connectivity=connectivity)
    reference_coords = _surface_coordinates(cropped_reference, connectivity=connectivity)
    if test_coords.size == 0 or reference_coords.size == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty
    test_coords = _sample_coordinates(test_coords, max_points=max_points)
    reference_coords = _sample_coordinates(reference_coords, max_points=max_points)
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    test_points = test_coords.astype(np.float64, copy=False) * spacing
    reference_points = reference_coords.astype(np.float64, copy=False) * spacing
    return _nearest_distances(test_points, reference_points), _nearest_distances(reference_points, test_points)


def approximate_hd95(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    max_points: int = 20_000,
    crop_margin: int = 2,
) -> float:
    prediction_mask = np.asarray(prediction_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    pred_empty = not bool(prediction_mask.any())
    target_empty = not bool(target_mask.any())
    if pred_empty and target_empty:
        return 0.0
    if pred_empty or target_empty:
        return float("nan")
    pred_to_target, target_to_pred = approximate_surface_distances(
        prediction_mask,
        target_mask,
        spacing_zyx=spacing_zyx,
        max_points=max_points,
        crop_margin=crop_margin,
    )
    distances = np.concatenate([pred_to_target, target_to_pred])
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def approximate_surface_dice(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    tolerance_mm: float = 1.0,
    max_points: int = 20_000,
    crop_margin: int = 2,
) -> float:
    prediction_mask = np.asarray(prediction_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    pred_empty = not bool(prediction_mask.any())
    target_empty = not bool(target_mask.any())
    if pred_empty and target_empty:
        return 1.0
    if pred_empty or target_empty:
        return 0.0
    pred_to_target, target_to_pred = approximate_surface_distances(
        prediction_mask,
        target_mask,
        spacing_zyx=spacing_zyx,
        max_points=max_points,
        crop_margin=crop_margin,
    )
    total = int(pred_to_target.size + target_to_pred.size)
    if total == 0:
        return 0.0
    close = int(np.count_nonzero(pred_to_target <= tolerance_mm) + np.count_nonzero(target_to_pred <= tolerance_mm))
    return float(close / total)


def _remove_small_components(mask: np.ndarray, spacing_zyx: tuple[float, float, float], threshold_mm3: float) -> np.ndarray:
    if threshold_mm3 <= 0:
        return mask.astype(bool, copy=False)
    labels, count = _component_labels(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    voxel_volume = float(np.prod(np.asarray(spacing_zyx, dtype=np.float64)))
    keep = np.zeros_like(mask, dtype=bool)
    for component_id in range(1, count + 1):
        component = labels == component_id
        if float(component.sum()) * voxel_volume >= float(threshold_mm3):
            keep |= component
    return keep


def lesionwise_dice(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    dilation_iterations: int = 5,
    lesion_volume_threshold_mm3: float = 10.0,
) -> float:
    from scipy import ndimage

    prediction_mask = _remove_small_components(prediction_mask, spacing_zyx, lesion_volume_threshold_mm3)
    target_mask = _remove_small_components(target_mask, spacing_zyx, lesion_volume_threshold_mm3)
    pred_empty = not bool(prediction_mask.any())
    target_empty = not bool(target_mask.any())
    if target_empty and pred_empty:
        return 1.0
    if target_empty or pred_empty:
        return 0.0

    target_labels, target_count = _component_labels(target_mask)
    pred_labels, pred_count = _component_labels(prediction_mask)
    structure = np.ones((3, 3, 3), dtype=bool)
    dilated_targets = [
        ndimage.binary_dilation(target_labels == component_id, structure=structure, iterations=int(dilation_iterations))
        for component_id in range(1, target_count + 1)
    ]
    dilated_predictions = [
        ndimage.binary_dilation(pred_labels == component_id, structure=structure, iterations=int(dilation_iterations))
        for component_id in range(1, pred_count + 1)
    ]
    matched_predictions: set[int] = set()
    scores: list[float] = []
    for target_idx, target_dilated in enumerate(dilated_targets, start=1):
        matched = [
            pred_idx
            for pred_idx, pred_dilated in enumerate(dilated_predictions, start=1)
            if bool(np.logical_and(target_dilated, pred_dilated).any())
        ]
        if not matched:
            scores.append(0.0)
            continue
        matched_predictions.update(matched)
        target_component = target_labels == target_idx
        prediction_union = np.zeros_like(prediction_mask, dtype=bool)
        for pred_idx in matched:
            prediction_union |= pred_labels == pred_idx
        scores.append(_dice_from_masks(prediction_union, target_component))
    unmatched_false_positives = pred_count - len(matched_predictions)
    scores.extend([0.0] * max(0, unmatched_false_positives))
    return float(np.mean(scores)) if scores else 0.0


def normalized_surface_dice(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    tolerance_mm: float = 1.0,
) -> float:
    from scipy import ndimage

    prediction_mask = prediction_mask.astype(bool, copy=False)
    target_mask = target_mask.astype(bool, copy=False)
    pred_empty = not bool(prediction_mask.any())
    target_empty = not bool(target_mask.any())
    if target_empty and pred_empty:
        return 1.0
    if target_empty or pred_empty:
        return 0.0
    structure = np.ones((3, 3, 3), dtype=bool)
    pred_surface = prediction_mask & ~ndimage.binary_erosion(prediction_mask, structure=structure, border_value=0)
    target_surface = target_mask & ~ndimage.binary_erosion(target_mask, structure=structure, border_value=0)
    if not bool(pred_surface.any()) or not bool(target_surface.any()):
        return _dice_from_masks(prediction_mask, target_mask)
    dist_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing_zyx)
    dist_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing_zyx)
    pred_close = int((dist_to_target[pred_surface] <= float(tolerance_mm)).sum())
    target_close = int((dist_to_pred[target_surface] <= float(tolerance_mm)).sum())
    denom = int(pred_surface.sum()) + int(target_surface.sum())
    return float((pred_close + target_close) / max(1, denom))


def rich_binary_metrics(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    surface_tolerance_mm: float = 1.0,
    surface_distance_max_points: int = 20_000,
    surface_distance_crop_margin: int = 2,
) -> dict[str, float]:
    prediction_mask = np.asarray(prediction_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    tp = int(np.logical_and(prediction_mask, target_mask).sum())
    fp = int(np.logical_and(prediction_mask, ~target_mask).sum())
    tn = int(np.logical_and(~prediction_mask, ~target_mask).sum())
    fn = int(np.logical_and(~prediction_mask, target_mask).sum())
    both_empty = (tp + fp + fn) == 0
    dice = 1.0 if both_empty else _safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    jaccard = 1.0 if both_empty else _safe_divide(tp, tp + fp + fn)
    surface_dice = approximate_surface_dice(
        prediction_mask,
        target_mask,
        spacing_zyx=spacing_zyx,
        tolerance_mm=surface_tolerance_mm,
        max_points=surface_distance_max_points,
        crop_margin=surface_distance_crop_margin,
    )
    return {
        "jaccard": jaccard,
        "iou": jaccard,
        "dice": dice,
        "sdc": surface_dice,
        "nsd": surface_dice,
        "accuracy": _safe_divide(tp + tn, tp + fp + tn + fn, empty_value=1.0),
        "precision": 1.0 if (tp + fp == 0 and tp + fn == 0) else _safe_divide(tp, tp + fp),
        "recall": 1.0 if (tp + fp == 0 and tp + fn == 0) else _safe_divide(tp, tp + fn),
        "hd95": approximate_hd95(
            prediction_mask,
            target_mask,
            spacing_zyx=spacing_zyx,
            max_points=surface_distance_max_points,
            crop_margin=surface_distance_crop_margin,
        ),
    }


def rich_metric_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    surface_tolerance_mm: float = 1.0,
    surface_distance_max_points: int = 20_000,
    surface_distance_crop_margin: int = 2,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id, class_name in LABEL_REGION_NAMES.items():
        metrics = rich_binary_metrics(
            prediction == class_id,
            target == class_id,
            spacing_zyx=spacing_zyx,
            surface_tolerance_mm=surface_tolerance_mm,
            surface_distance_max_points=surface_distance_max_points,
            surface_distance_crop_margin=surface_distance_crop_margin,
        )
        rows.append({"target_type": "class", "target": class_name, "class_id": class_id, **metrics})
    for region_name in ("TC", "WT"):
        metrics = rich_binary_metrics(
            _region_mask(prediction, region_name),
            _region_mask(target, region_name),
            spacing_zyx=spacing_zyx,
            surface_tolerance_mm=surface_tolerance_mm,
            surface_distance_max_points=surface_distance_max_points,
            surface_distance_crop_margin=surface_distance_crop_margin,
        )
        rows.append({"target_type": "region", "target": region_name, "class_id": None, **metrics})
    metrics = rich_binary_metrics(
        prediction > 0,
        target > 0,
        spacing_zyx=spacing_zyx,
        surface_tolerance_mm=surface_tolerance_mm,
        surface_distance_max_points=surface_distance_max_points,
        surface_distance_crop_margin=surface_distance_crop_margin,
    )
    rows.append({"target_type": "overall", "target": "foreground", "class_id": None, **metrics})
    return rows


def batch_rich_metric_rows(
    prediction: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]],
    target: torch.Tensor,
    *,
    spacing_zyx: tuple[float, float, float],
    surface_tolerance_mm: float = 1.0,
    surface_distance_max_points: int = 20_000,
    surface_distance_crop_margin: int = 2,
) -> list[dict[str, Any]]:
    logits = main_logits(prediction)
    pred = torch.argmax(logits.detach(), dim=1).cpu().numpy()
    tgt = target.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for sample_index in range(pred.shape[0]):
        for row in rich_metric_rows(
            pred[sample_index],
            tgt[sample_index],
            spacing_zyx=spacing_zyx,
            surface_tolerance_mm=surface_tolerance_mm,
            surface_distance_max_points=surface_distance_max_points,
            surface_distance_crop_margin=surface_distance_crop_margin,
        ):
            row["sample_index"] = sample_index
            rows.append(row)
    return rows


def summarize_rich_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_type"]), str(row["target"]))].append(row)
    by_target = []
    for (target_type, target), target_rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "target_type": target_type,
            "target": target,
            "num_rows": len(target_rows),
        }
        for metric_name in RICH_METRIC_NAMES:
            values = [float(row[metric_name]) for row in target_rows]
            item[f"{metric_name}_mean"] = _nanmean(values)
            item[f"{metric_name}_std"] = _nanstd(values)
        by_target.append(item)
    overall = {}
    for metric_name in RICH_METRIC_NAMES:
        values = [float(row[metric_name]) for row in rows]
        overall[f"{metric_name}_mean"] = _nanmean(values)
        overall[f"{metric_name}_std"] = _nanstd(values)
    return {"overall": overall, "by_target": by_target, "num_rows": len(rows)}


def batch_region_metrics(prediction: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]], target: torch.Tensor) -> dict[str, float]:
    logits = main_logits(prediction)
    pred = torch.argmax(logits.detach(), dim=1).cpu().numpy()
    tgt = target.detach().cpu().numpy()
    rows = [region_metrics(pred[idx], tgt[idx]) for idx in range(pred.shape[0])]
    return summarize_metric_dicts(rows)


def summarize_metric_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0])
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def confusion_counts(prediction: np.ndarray, target: np.ndarray, num_classes: int) -> dict[str, Any]:
    rows = {}
    for class_id in range(num_classes):
        pred_mask = prediction == class_id
        target_mask = target == class_id
        rows[str(class_id)] = {
            "tp": int(np.logical_and(pred_mask, target_mask).sum()),
            "fp": int(np.logical_and(pred_mask, ~target_mask).sum()),
            "fn": int(np.logical_and(~pred_mask, target_mask).sum()),
        }
    return rows
