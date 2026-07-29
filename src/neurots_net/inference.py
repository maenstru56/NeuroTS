
from __future__ import annotations

from contextlib import nullcontext
import itertools
import json
import logging
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

from neurots_net.config import ExperimentConfig
from neurots_net.data.brats26 import CachedCaseRecord, cached_records, load_cached_case
from neurots_net.data.dataset import extract_padded_patch
from neurots_net.data.nifti_io import save_segmentation_like_reference
from neurots_net.models.registry import build_model
from neurots_net.utils.io import ensure_dir, save_json


MAIN_CHANNEL_ORDER = ("background", "ET", "NET", "CC", "ED")


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_autocast_dtype(device: torch.device, requested: str | torch.dtype | None) -> torch.dtype:
    if isinstance(requested, torch.dtype):
        return requested
    key = str(requested or "float16").strip().lower()
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"bf16", "bfloat16"}:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if key == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    raise ValueError("amp_dtype must be one of float16, bfloat16, or auto.")


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def main_logits(prediction: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]) -> torch.Tensor:
    if isinstance(prediction, tuple):
        return prediction[0]
    return prediction


def _steps_for_dim(size: int, patch: int, overlap: float) -> list[int]:
    if size <= patch:
        return [0]
    step_fraction = max(1e-3, 1.0 - float(overlap))
    target_step = max(1, int(round(float(patch) * step_fraction)))
    num_steps = int(np.ceil((size - patch) / target_step)) + 1
    max_start = size - patch
    return [int(round(max_start * idx / max(1, num_steps - 1))) for idx in range(num_steps)]


def sliding_window_origins(shape: tuple[int, int, int], patch_size: tuple[int, int, int], overlap: float) -> list[tuple[int, int, int]]:
    steps = [_steps_for_dim(int(shape[axis]), int(patch_size[axis]), overlap) for axis in range(3)]
    return [(z, y, x) for z in steps[0] for y in steps[1] for x in steps[2]]


def _valid_slices(origin: tuple[int, int, int], patch_size: tuple[int, int, int], shape: tuple[int, int, int]) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    volume_slices = []
    patch_slices = []
    for axis, start in enumerate(origin):
        valid_start = max(0, int(start))
        valid_end = min(int(shape[axis]), int(start) + int(patch_size[axis]))
        patch_start = valid_start - int(start)
        patch_end = patch_start + (valid_end - valid_start)
        volume_slices.append(slice(valid_start, valid_end))
        patch_slices.append(slice(patch_start, patch_end))
    return tuple(volume_slices), tuple(patch_slices)


def gaussian_importance_map(patch_size: tuple[int, int, int], sigma_scale: float) -> np.ndarray:
    from scipy import ndimage

    importance = np.zeros(patch_size, dtype=np.float32)
    center = tuple(int(size // 2) for size in patch_size)
    importance[center] = 1.0
    sigma = [max(float(size) * float(sigma_scale), 1e-3) for size in patch_size]
    importance = ndimage.gaussian_filter(importance, sigma=sigma, mode="constant", cval=0.0)
    max_value = float(importance.max())
    if max_value > 0:
        importance /= max_value
    nonzero = importance[importance > 0]
    min_nonzero = float(nonzero.min()) if nonzero.size else 1.0
    importance[importance == 0] = min_nonzero
    return importance.astype(np.float32, copy=False)


def _tta_axis_sets(enabled: bool, axes: list[int] | tuple[int, ...]) -> list[tuple[int, ...]]:
    if not enabled:
        return [tuple()]
    clean_axes = tuple(int(axis) for axis in axes)
    result = []
    for length in range(len(clean_axes) + 1):
        result.extend(tuple(item) for item in itertools.combinations(clean_axes, length))
    return result


@torch.no_grad()
def predict_probability_sliding_window(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    num_classes: int,
    patch_size: list[int] | tuple[int, int, int],
    device: torch.device,
    mixed_precision: bool,
    amp_dtype: str | torch.dtype,
    window_batch_size: int,
    overlap: float,
    use_gaussian_weighting: bool,
    gaussian_sigma_scale: float,
    score_dtype: str,
    tta_mirroring: bool,
    tta_axes: list[int] | tuple[int, ...],
    activation: str = "auto",
) -> np.ndarray:
    model.eval()
    autocast_dtype = resolve_autocast_dtype(device, amp_dtype)
    volume = np.asarray(image, dtype=np.float32)
    shape = tuple(int(item) for item in volume.shape[1:])
    patch = tuple(int(item) for item in patch_size)
    origins = sliding_window_origins(shape, patch, overlap=float(overlap))
    scores = np.zeros((int(num_classes), *shape), dtype=np.dtype(score_dtype))
    normalizer = np.zeros(shape, dtype=np.float32)
    weight = gaussian_importance_map(patch, sigma_scale=gaussian_sigma_scale) if use_gaussian_weighting else np.ones(patch, dtype=np.float32)
    axis_sets = _tta_axis_sets(tta_mirroring, tta_axes)
    batch_size = max(1, int(window_batch_size))
    for start in range(0, len(origins), batch_size):
        batch_origins = origins[start : start + batch_size]
        patches = [
            extract_padded_patch(volume, origin=origin, patch_size=patch).astype(np.float32, copy=False)
            for origin in batch_origins
        ]
        batch = torch.from_numpy(np.stack(patches, axis=0)).to(device, non_blocking=True)
        probs_sum: torch.Tensor | None = None
        for axes in axis_sets:
            tensor = batch
            if axes:
                tensor = torch.flip(tensor, dims=[2 + axis for axis in axes])
            with autocast_context(device, mixed_precision, autocast_dtype):
                logits = main_logits(model(tensor))
            activation_key = str(activation).strip().lower()
            if activation_key == "sigmoid":
                probabilities = torch.sigmoid(logits.float())
            elif activation_key == "softmax":
                probabilities = torch.softmax(logits.float(), dim=1)
            elif logits.shape[1] == 1:
                probabilities = torch.sigmoid(logits.float())
            else:
                probabilities = torch.softmax(logits.float(), dim=1)
            if axes:
                probabilities = torch.flip(probabilities, dims=[2 + axis for axis in axes])
            probs_sum = probabilities if probs_sum is None else probs_sum + probabilities
        probabilities_np = (probs_sum / float(len(axis_sets))).detach().cpu().numpy()
        for item_idx, origin in enumerate(batch_origins):
            volume_slices, patch_slices = _valid_slices(origin, patch, shape)
            weighted = probabilities_np[(item_idx, slice(None), *patch_slices)] * weight[patch_slices][None]
            scores[(slice(None), *volume_slices)] += weighted.astype(scores.dtype, copy=False)
            normalizer[volume_slices] += weight[patch_slices]
    scores = scores.astype(np.float32, copy=False)
    scores /= np.maximum(normalizer[None], np.finfo(np.float32).tiny)
    return scores


def paste_crop_to_full(prediction_crop_zyx: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    full_shape = tuple(int(item) for item in metadata["source_shape_zyx"])
    full = np.zeros(full_shape, dtype=np.uint8)
    start = [int(item) for item in metadata["crop_bbox_zyx"]["start"]]
    end = [int(item) for item in metadata["crop_bbox_zyx"]["end"]]
    slices = tuple(slice(start[axis], end[axis]) for axis in range(3))
    crop_shape = tuple(end[axis] - start[axis] for axis in range(3))
    full[slices] = prediction_crop_zyx[tuple(slice(0, crop_shape[axis]) for axis in range(3))].astype(np.uint8, copy=False)
    return full


def _remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if int(min_voxels) <= 0:
        return np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3, 3), dtype=bool))
    keep = np.zeros(labels.shape, dtype=bool)
    for component_id in range(1, int(count) + 1):
        component = labels == component_id
        if int(component.sum()) > int(min_voxels):
            keep |= component
    return keep


def segment_probabilities(
    probabilities: np.ndarray,
    inference_cfg: Any,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 4:
        raise ValueError("segment_probabilities expects probabilities with shape [C,D,H,W].")
    if probs.shape[0] != 5:
        raise ValueError(
            "Public NeuroTS-Net inference expects channels [background, ET, NET, CC, ED]."
        )
    biases = list(getattr(inference_cfg, "logit_biases", [0.0] * 5))
    if len(biases) != 5:
        raise ValueError("inference.logit_biases must contain five values in [BG, ET, NET, CC, ED] order.")
    bias = np.asarray(biases, dtype=np.float32).reshape((-1, 1, 1, 1))
    biased_scores = np.log(np.maximum(probs, np.finfo(np.float32).tiny)) + bias
    seg = np.asarray(np.argmax(biased_scores, axis=0), dtype=np.uint8)
    if not bool(getattr(inference_cfg, "use_probability_fusion", False)):
        return seg

    p_et = probs[1]
    p_cc = probs[3]
    p_ed = probs[4]
    p_wt = probs[1] + probs[2] + probs[3] + probs[4]
    wt_gate = p_wt > float(getattr(inference_cfg, "wt_gate_threshold", 0.25))
    dilation = int(getattr(inference_cfg, "wt_gate_dilation", 4))
    if dilation > 0:
        wt_gate |= ndimage.binary_dilation(
            seg > 0,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=dilation,
        )
    et = (p_et >= float(getattr(inference_cfg, "threshold_et", 0.25))) & wt_gate
    cc = (p_cc >= float(getattr(inference_cfg, "threshold_cc", 0.20))) & wt_gate & ~et
    ed = (
        (p_ed >= float(getattr(inference_cfg, "threshold_ed", 0.15)))
        & wt_gate
        & ~et
        & ~cc
    )
    et = _remove_small_components(et, int(getattr(inference_cfg, "min_component_voxels_et", 10)))
    cc = _remove_small_components(cc, int(getattr(inference_cfg, "min_component_voxels_cc", 10)))
    ed = _remove_small_components(ed, int(getattr(inference_cfg, "min_component_voxels_ed", 10)))
    seg = np.where(wt_gate, seg, 0).astype(np.uint8, copy=False)
    seg[et] = 1
    seg[cc] = 3
    seg[ed] = 4
    return seg


def save_probability_npz(
    path: str | Path,
    probabilities: np.ndarray,
    *,
    case_id: str,
    channel_order: tuple[str, ...],
    source: str,
    checkpoint: str | Path | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "case_id": str(case_id),
        "source": str(source),
        "channel_order": list(channel_order),
        "array": "probabilities",
        "dtype": "float16",
        "checkpoint": None if checkpoint is None else str(checkpoint),
    }
    np.savez_compressed(
        output,
        probabilities=np.asarray(probabilities, dtype=np.float16),
        metadata=np.asarray(json.dumps(metadata)),
        channel_order=np.asarray(channel_order),
    )


def load_checkpoint_model(checkpoint_path: str | Path, cfg: ExperimentConfig, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg.model).to(device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    cleaned = {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def create_submission_zip(
    prediction_dir: str | Path,
    archive_path: str | Path,
    *,
    expected_suffix: str = ".nii.gz",
) -> dict[str, Any]:
    source = Path(prediction_dir)
    if not source.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {source}")
    files = sorted(path for path in source.glob(f"*{expected_suffix}") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No {expected_suffix} prediction files found in {source}")
    bad_files = [path.name for path in files if not path.name.endswith(expected_suffix)]
    if bad_files:
        raise ValueError(f"Invalid prediction filenames: {bad_files[:5]}")
    archive = Path(archive_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in files:
            handle.write(path, arcname=path.name)
    return {
        "archive_path": str(archive),
        "prediction_dir": str(source),
        "num_files": len(files),
        "files": [path.name for path in files],
    }


def create_submission_archives(
    prediction_root: str | Path,
    *,
    experiment_name: str,
    candidate: str,
) -> list[dict[str, Any]]:
    root = Path(prediction_root)
    archives: list[dict[str, Any]] = []
    for directory in sorted(item for item in root.iterdir() if item.is_dir()):
        if not any(directory.glob("*.nii.gz")):
            continue
        archive_name = f"{experiment_name}_{candidate}_{directory.name}.zip"
        archives.append(create_submission_zip(directory, root / archive_name))
    return archives


def find_fold_checkpoints(run_dir: str | Path, candidate: str = "voxel_dice") -> list[Path]:
    root = Path(run_dir)
    key = str(candidate).strip().lower()
    if key in {"voxel", "voxel_dice", "best_voxel_dice"}:
        patterns = ["best_voxel_dice.pt", "fold_*/best_voxel_dice.pt", "folds_all/checkpoints/fold_*_best_voxel_dice.pt"]
    elif key in {"rare", "rare_present", "best_rare_present_score"}:
        patterns = ["best_rare_present_score.pt", "fold_*/best_rare_present_score.pt", "folds_all/checkpoints/fold_*_best_rare_present_score.pt"]
    elif key in {"mixed", "all"}:
        patterns = [
            "best_voxel_dice.pt",
            "best_rare_present_score.pt",
            "fold_*/best_voxel_dice.pt",
            "fold_*/best_rare_present_score.pt",
            "folds_all/checkpoints/fold_*_best_voxel_dice.pt",
            "folds_all/checkpoints/fold_*_best_rare_present_score.pt",
        ]
    else:
        raise ValueError("candidate must be voxel_dice, rare_present, or mixed.")
    checkpoints: list[Path] = []
    for pattern in patterns:
        checkpoints.extend(sorted(root.glob(pattern)))
    seen = set()
    unique = []
    for path in checkpoints:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    checkpoints = unique
    if not checkpoints:
        raise FileNotFoundError(f"No {candidate} fold checkpoints found under {root}.")
    return checkpoints


@torch.no_grad()
def predict_validation_ensemble(
    cfg: ExperimentConfig,
    checkpoints: list[str | Path],
    output_dir: str | Path,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    device = choose_device()
    records = cached_records(cfg.data, "validation")
    if not records:
        raise FileNotFoundError(
            f"No cached validation cases found under {Path(cfg.data.cache_dir) / 'validation'}."
        )
    root = ensure_dir(output_dir)
    ensemble_dir = ensure_dir(root / "folds_all")
    fold_dirs = [
        ensure_dir(root / f"fold_{idx + 1:02d}")
        for idx in range(len(checkpoints))
    ]
    save_probability_outputs = bool(
        getattr(cfg.inference, "save_probabilities", False)
        or cfg.inference.save_fold_predictions
    )
    fold_probability_dirs = (
        [
            ensure_dir(root / f"fold_{idx + 1:02d}_probabilities")
            for idx in range(len(checkpoints))
        ]
        if save_probability_outputs
        else []
    )
    ensemble_probability_dir = (
        ensure_dir(root / "folds_all_probabilities")
        if save_probability_outputs
        else None
    )
    summary: dict[str, Any] = {
        "device": str(device),
        "checkpoints": [str(path) for path in checkpoints],
        "probability_fusion_enabled": bool(cfg.inference.use_probability_fusion),
        "probability_outputs_saved": save_probability_outputs,
        "probability_channel_order": list(
            MAIN_CHANNEL_ORDER[: int(cfg.model.num_classes)]
        ),
        "cases": [],
        "ensemble_dir": str(ensemble_dir),
    }

    for case_index, record in enumerate(records):
        payload = load_cached_case(record.cache_path)
        image = payload["image"].astype(np.float32, copy=False)
        metadata = payload["metadata"]
        logger.info(
            "Predicting validation case %d/%d | %s | crop_shape=%s",
            case_index + 1,
            len(records),
            record.case_id,
            tuple(image.shape[1:]),
        )
        probability_sum: np.ndarray | None = None
        fold_outputs: list[str] = []
        for fold_index, checkpoint_path in enumerate(checkpoints):
            model = load_checkpoint_model(checkpoint_path, cfg, device)
            probabilities = predict_probability_sliding_window(
                model,
                image,
                num_classes=cfg.model.num_classes,
                patch_size=cfg.inference.patch_size,
                device=device,
                mixed_precision=cfg.inference.mixed_precision,
                amp_dtype=cfg.inference.amp_dtype,
                window_batch_size=cfg.inference.window_batch_size,
                overlap=cfg.inference.sliding_window_overlap,
                use_gaussian_weighting=cfg.inference.use_gaussian_weighting,
                gaussian_sigma_scale=cfg.inference.gaussian_sigma_scale,
                score_dtype=cfg.inference.score_dtype,
                tta_mirroring=cfg.inference.tta_mirroring,
                tta_axes=cfg.inference.tta_axes,
            )
            if fold_probability_dirs:
                save_probability_npz(
                    fold_probability_dirs[fold_index] / f"{record.case_id}.npz",
                    probabilities,
                    case_id=record.case_id,
                    channel_order=MAIN_CHANNEL_ORDER[: int(cfg.model.num_classes)],
                    source=f"fold_{fold_index + 1:02d}",
                    checkpoint=checkpoint_path,
                )
            probability_sum = (
                probabilities
                if probability_sum is None
                else probability_sum + probabilities
            )
            if cfg.inference.save_fold_predictions:
                pred_crop = segment_probabilities(probabilities, cfg.inference)
                pred_full = paste_crop_to_full(pred_crop, metadata)
                fold_path = fold_dirs[fold_index] / f"{record.case_id}.nii.gz"
                save_segmentation_like_reference(
                    pred_full,
                    metadata["reference_modality"],
                    fold_path,
                )
                fold_outputs.append(str(fold_path))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if probability_sum is None:
            raise RuntimeError("No checkpoints were evaluated.")
        ensemble_probability = probability_sum / float(len(checkpoints))
        if ensemble_probability_dir is not None:
            save_probability_npz(
                ensemble_probability_dir / f"{record.case_id}.npz",
                ensemble_probability,
                case_id=record.case_id,
                channel_order=MAIN_CHANNEL_ORDER[: int(cfg.model.num_classes)],
                source="folds_all",
                checkpoint=None,
            )
        ensemble_crop = segment_probabilities(
            ensemble_probability,
            cfg.inference,
        )
        ensemble_full = paste_crop_to_full(ensemble_crop, metadata)
        ensemble_path = ensemble_dir / f"{record.case_id}.nii.gz"
        save_segmentation_like_reference(
            ensemble_full,
            metadata["reference_modality"],
            ensemble_path,
        )
        summary["cases"].append(
            {
                "case_id": record.case_id,
                "ensemble_prediction": str(ensemble_path),
                "fold_predictions": fold_outputs,
            }
        )

    save_json(summary, root / "prediction_summary.json")
    return summary
