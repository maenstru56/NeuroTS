from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


MAIN_CHANNEL_ORDER = ("background", "ET", "NET", "CC", "ED")


@dataclass
class PostprocessingParameters:
    cc_logit_bias: float = 0.0
    cc_prob_threshold: float | None = None
    cc_min_size: int = 0
    cc_max_components: int = 0
    et_logit_bias: float = 0.0
    et_prob_threshold: float | None = None
    et_min_size: int = 0
    et_max_components: int = 0
    wt_gate_threshold_for_core: float = 0.40
    protect_tc_wt: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_probability_npz(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        probabilities = payload["probabilities"].astype(np.float32, copy=False)
        metadata = json.loads(str(payload["metadata"].item())) if "metadata" in payload.files else {}
    return probabilities, metadata


def save_probability_npz(
    path: str | Path,
    probabilities: np.ndarray,
    *,
    case_id: str,
    channel_order: tuple[str, ...],
    source: str,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "case_id": str(case_id),
        "source": str(source),
        "channel_order": list(channel_order),
        "array": "probabilities",
        "dtype": "float16",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    np.savez_compressed(
        output,
        probabilities=np.asarray(probabilities, dtype=np.float16),
        metadata=np.asarray(json.dumps(metadata)),
        channel_order=np.asarray(channel_order),
    )


def hard_seg_from_main_probabilities(
    probabilities: np.ndarray,
    logit_biases: list[float] | tuple[float, ...] | None = None,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 4 or probs.shape[0] < 5:
        raise ValueError(f"Expected main probabilities with shape [5,D,H,W], got {probs.shape}.")
    biases = list(logit_biases or [0.0] * int(probs.shape[0]))
    if len(biases) < int(probs.shape[0]):
        biases.extend([0.0] * (int(probs.shape[0]) - len(biases)))
    bias = np.asarray(biases[: int(probs.shape[0])], dtype=np.float32).reshape((-1, 1, 1, 1))
    scores = np.log(np.maximum(probs, np.finfo(np.float32).tiny)) + bias
    return np.argmax(scores, axis=0).astype(np.uint8, copy=False)


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    if int(iterations) <= 0:
        return np.asarray(mask, dtype=bool)
    return ndimage.binary_dilation(
        np.asarray(mask, dtype=bool),
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=int(iterations),
    )


def _biased_binary_probability(probability: np.ndarray, logit_bias: float) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    if float(logit_bias) == 0.0:
        return prob
    logits = np.log(prob) - np.log1p(-prob) + float(logit_bias)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32, copy=False)


def _keep_best_components(
    mask: np.ndarray,
    probability: np.ndarray,
    *,
    min_size: int,
    max_components: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3, 3), dtype=bool))
    keep = np.zeros(labels.shape, dtype=bool)
    candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    for component_id in range(1, int(count) + 1):
        component = labels == component_id
        volume = int(component.sum())
        values = np.asarray(probability[component], dtype=np.float32)
        mean_probability = float(values.mean()) if values.size else 0.0
        row = {
            "component_id": int(component_id),
            "volume": volume,
            "mean_probability": mean_probability,
            "score": float(mean_probability * np.sqrt(max(volume, 1))),
            "drop_reason": "",
        }
        if volume < int(min_size):
            row["drop_reason"] = "min_size"
            rows.append(row)
        else:
            candidates.append((float(row["score"]), component, row))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for rank, (_, component, row) in enumerate(candidates, start=1):
        if int(max_components) > 0 and rank > int(max_components):
            row = dict(row)
            row["drop_reason"] = "max_components"
            rows.append(row)
            continue
        keep |= component
        rows.append(dict(row))
    return keep, rows


def postprocess_main_probabilities(
    main_probabilities: np.ndarray,
    params: PostprocessingParameters,
    *,
    base_segmentation: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    probs = np.asarray(main_probabilities, dtype=np.float32)
    if probs.ndim != 4 or probs.shape[0] < 5:
        raise ValueError(f"Expected main probabilities with shape [5,D,H,W], got {probs.shape}.")
    seg = (
        hard_seg_from_main_probabilities(probs)
        if base_segmentation is None
        else np.asarray(base_segmentation, dtype=np.uint8).copy()
    )
    summary: dict[str, Any] = {
        "parameters": params.to_dict(),
        "et_adjusted": False,
        "cc_adjusted": False,
        "et_component_rows": [],
        "cc_component_rows": [],
    }
    p_wt = probs[1:5].sum(axis=0)
    tc_touch = _dilate(np.isin(seg, [1, 2, 3]), 3)
    tumor_gate = (p_wt >= float(params.wt_gate_threshold_for_core)) | tc_touch | (seg > 0)
    if bool(params.protect_tc_wt):
        tumor_gate &= tc_touch | (seg > 0)

    if params.et_prob_threshold is not None or float(params.et_logit_bias) != 0.0 or int(params.et_min_size) > 0:
        p_et = _biased_binary_probability(probs[1], float(params.et_logit_bias))
        threshold = float(params.et_prob_threshold) if params.et_prob_threshold is not None else 0.5
        add_gate = tumor_gate & ((seg > 0) if bool(params.protect_tc_wt) else np.ones_like(seg, dtype=bool))
        et_mask = ((seg == 1) | ((p_et >= threshold) & add_gate)) & (seg != 3) & (seg != 4)
        et_mask, rows = _keep_best_components(
            et_mask,
            p_et,
            min_size=int(params.et_min_size),
            max_components=int(params.et_max_components),
        )
        seg[seg == 1] = 2
        seg[et_mask] = 1
        summary["et_adjusted"] = True
        summary["et_component_rows"] = rows

    if params.cc_prob_threshold is not None or float(params.cc_logit_bias) != 0.0 or int(params.cc_min_size) > 0:
        p_cc = _biased_binary_probability(probs[3], float(params.cc_logit_bias))
        threshold = float(params.cc_prob_threshold) if params.cc_prob_threshold is not None else 0.5
        add_gate = tumor_gate & ((seg > 0) if bool(params.protect_tc_wt) else np.ones_like(seg, dtype=bool))
        cc_mask = ((seg == 3) | ((p_cc >= threshold) & add_gate)) & (seg != 1) & (seg != 4)
        cc_mask, rows = _keep_best_components(
            cc_mask,
            p_cc,
            min_size=int(params.cc_min_size),
            max_components=int(params.cc_max_components),
        )
        old_cc = seg == 3
        seg[old_cc] = np.where(probs[1:4, old_cc].sum(axis=0) >= 0.35, 2, 0).astype(np.uint8, copy=False)
        seg[cc_mask] = 3
        summary["cc_adjusted"] = True
        summary["cc_component_rows"] = rows
    return seg.astype(np.uint8, copy=False), summary


def component_audit_rows(
    *,
    case_id: str,
    mask: np.ndarray,
    main_probabilities: np.ndarray | None = None,
    reference_segmentation: np.ndarray | None = None,
    final_keep_mask: np.ndarray | None = None,
    image: np.ndarray | None = None,
    label_prefix: str = "component",
) -> list[dict[str, Any]]:
    component_labels, count = ndimage.label(
        np.asarray(mask, dtype=bool),
        structure=np.ones((3, 3, 3), dtype=bool),
    )
    if count == 0:
        return []
    reference = (
        np.zeros(component_labels.shape, dtype=np.uint8)
        if reference_segmentation is None
        else np.asarray(reference_segmentation, dtype=np.uint8)
    )
    tc = np.isin(reference, [1, 2, 3])
    distance_to_tc = ndimage.distance_transform_edt(~tc) if bool(tc.any()) else np.full(tc.shape, np.inf)
    p_ed = main_probabilities[4] if main_probabilities is not None else None
    p_wt = main_probabilities[1:5].sum(axis=0) if main_probabilities is not None else None
    p_tc = main_probabilities[1:4].sum(axis=0) if main_probabilities is not None else None
    rows: list[dict[str, Any]] = []
    for component_id in range(1, int(count) + 1):
        component = component_labels == component_id
        distances = distance_to_tc[component]
        row: dict[str, Any] = {
            "case_id": str(case_id),
            "component_id": int(component_id),
            "component_source": str(label_prefix),
            "volume": int(component.sum()),
            "final_kept_voxels": int(np.logical_and(component, final_keep_mask).sum()) if final_keep_mask is not None else int(component.sum()),
            "overlap_TC": int(np.logical_and(component, tc).sum()),
            "overlap_dilateTC_3": int(np.logical_and(component, _dilate(tc, 3)).sum()),
            "overlap_dilateTC_5": int(np.logical_and(component, _dilate(tc, 5)).sum()),
            "min_distance_to_TC": float(np.min(distances)) if distances.size else float("inf"),
            "median_distance_to_TC": float(np.median(distances)) if distances.size else float("inf"),
        }
        if p_ed is not None:
            row["mean_main_pED"] = float(np.asarray(p_ed[component], dtype=np.float32).mean())
            row["mean_pWT"] = float(np.asarray(p_wt[component], dtype=np.float32).mean())
            row["mean_pTC"] = float(np.asarray(p_tc[component], dtype=np.float32).mean())
        if image is not None:
            for channel_index, channel_name in enumerate(("T1N", "T1C", "T2W", "T2F")[: int(image.shape[0])]):
                values = np.asarray(image[channel_index][component], dtype=np.float32)
                row[f"{channel_name}_mean"] = float(values.mean()) if values.size else 0.0
                row[f"{channel_name}_p95"] = float(np.percentile(values, 95)) if values.size else 0.0
        rows.append(row)
    return rows


def summarize_prediction_labels(segmentation: np.ndarray) -> dict[str, Any]:
    seg = np.asarray(segmentation)
    values, counts = np.unique(seg, return_counts=True)
    return {
        "label_counts": {str(int(label)): int(count) for label, count in zip(values, counts)},
        "ed_present": bool(np.any(seg == 4)),
        "ed_voxels": int(np.count_nonzero(seg == 4)),
        "cc_present": bool(np.any(seg == 3)),
        "cc_voxels": int(np.count_nonzero(seg == 3)),
        "et_present": bool(np.any(seg == 1)),
        "et_voxels": int(np.count_nonzero(seg == 1)),
        "wt_voxels": int(np.count_nonzero(seg > 0)),
        "tc_voxels": int(np.count_nonzero(np.isin(seg, [1, 2, 3]))),
    }
