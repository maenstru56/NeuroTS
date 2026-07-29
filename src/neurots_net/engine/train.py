from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neurots_net.config import ExperimentConfig
from neurots_net.data.brats26 import cached_records
from neurots_net.data.brats26 import load_cached_case
from neurots_net.data.dataset import CachedBraTSPatchDataset
from neurots_net.data.folds import load_or_create_folds, load_single_split, split_records_by_fold, split_records_by_single_split
from neurots_net.engine.optimizers import build_optimizer
from neurots_net.inference import autocast_context, choose_device, predict_probability_sliding_window, resolve_autocast_dtype, segment_probabilities
from neurots_net.losses import build_loss
from neurots_net.metrics import (
    CHALLENGE_REGIONS,
    LABEL_REGION_NAMES,
    batch_rich_metric_rows,
    batch_region_metrics,
    rich_metric_rows,
    summarize_rich_metric_rows,
)
from neurots_net.models.neurots_net import branch_usage_regularization_loss, configure_branch_selection
from neurots_net.models.registry import build_model
from neurots_net.utils.io import ensure_dir, save_json
from neurots_net.utils.plots import save_training_plots
from neurots_net.utils.resources import format_resource_snapshot, resource_snapshot
from neurots_net.utils.seed import seed_everything


def _make_loader(dataset: CachedBraTSPatchDataset, cfg: ExperimentConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=bool(cfg.data.pin_memory and torch.cuda.is_available()),
        drop_last=shuffle,
    )


def _format_patch_size(patch_size: list[int] | tuple[int, ...]) -> str:
    return "[" + ", ".join(str(int(item)) for item in patch_size) + "]"


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _validation_rich_metric_table_rows(epoch: int, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overall = summary.get("overall", {})
    if overall:
        rows.append(
            {
                "epoch": int(epoch),
                "target_type": "overall",
                "target": "all_targets",
                "num_rows": int(summary.get("num_rows", 0)),
                **overall,
            }
        )
    for row in summary.get("by_target", []):
        rows.append({"epoch": int(epoch), **row})
    return rows


def _save_validation_rich_metric_history(
    *,
    run_root: Path,
    history: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "description": "Compact per-epoch validation rich metrics for NeuroTS-Net patch validation.",
        "epochs": history,
    }
    save_json(payload, run_root / "validation_rich_metrics.json")
    pd.DataFrame(table_rows).to_csv(run_root / "validation_rich_metrics.csv", index=False)


def _cleanup_validation_rich_metric_outputs(run_root: Path) -> None:
    for pattern in [
        "validation_rich_metrics_epoch_*.csv",
        "validation_rich_metrics_epoch_*.json",
        "validation_rich_metrics_latest.csv",
        "validation_rich_metrics_latest.json",
    ]:
        for path in run_root.glob(pattern):
            if path.is_file():
                path.unlink()
    for filename in ["validation_rich_metrics.csv", "validation_rich_metrics.json"]:
        path = run_root / filename
        if path.is_file():
            path.unlink()


def _make_patch_dataset(
    records,
    cfg: ExperimentConfig,
    epoch_size: int,
    seed_offset: int,
    augment: bool,
) -> CachedBraTSPatchDataset:
    return CachedBraTSPatchDataset(
        records=records,
        patch_size=cfg.data.patch_size,
        epoch_size=epoch_size,
        foreground_patch_prob=cfg.data.foreground_oversample,
        min_foreground_voxels=cfg.data.min_foreground_voxels,
        label_mode=cfg.data.label_mode,
        patch_sampling_strategy=cfg.data.patch_sampling_strategy,
        patch_sampling_probabilities=cfg.data.patch_sampling_probabilities,
        target_class_jitter_voxels=cfg.data.target_class_jitter_voxels,
        targeted_patch_retries=cfg.data.targeted_patch_retries,
        cache_memory_cases=cfg.data.cache_memory_cases,
        augment=augment and cfg.augmentation.enabled,
        augmentation_config=cfg.augmentation,
        seed=int(cfg.train.seed) + int(seed_offset),
    )


def _learning_rate_for_epoch(cfg: ExperimentConfig, epoch: int) -> float:
    base_lr = float(cfg.train.learning_rate)
    warmup = int(cfg.train.warmup_epochs)
    if warmup > 0 and epoch <= warmup:
        return base_lr * float(epoch) / float(warmup)
    scheduler = str(cfg.train.scheduler).strip().lower()
    if scheduler == "cosine":
        eta_min = float(getattr(cfg.train, "cosine_eta_min", 1e-6))
        total = max(1, int(cfg.train.epochs) - warmup - 1)
        current = min(total, max(0, int(epoch) - warmup - 1))
        progress = float(current) / float(total)
        return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * progress))
    if scheduler != "poly":
        return base_lr
    total = max(1, int(cfg.train.epochs) - warmup - 1)
    current = min(total, max(0, int(epoch) - warmup - 1))
    return base_lr * (1.0 - float(current) / float(total)) ** 0.9


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _make_grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device.type, enabled=enabled and device.type == "cuda")


def _branch_regularization_weight(cfg: ExperimentConfig, epoch: int) -> float:
    total_epochs = max(1, int(cfg.train.epochs))
    progress = 0.0 if total_epochs == 1 else float(epoch - 1) / float(total_epochs - 1)
    if progress <= max(0.0, float(cfg.train.branch_usage_regularization_fraction)):
        return float(cfg.train.branch_usage_regularization_weight)
    return 0.0


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        images, labels, sample_info = batch
        return images, labels, sample_info
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        images, labels = batch
        return images, labels, None
    raise ValueError("Expected dataloader batch to contain images/labels or images/labels/sample_info.")


def _collated_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _new_sampling_stats() -> dict[str, Any]:
    return {
        "patches": 0,
        "requested": {},
        "successful": {},
        "fallback": {},
        "actual_contains": {"et": 0, "net": 0, "cc": 0, "ed": 0},
        "actual_voxels": {"et": 0, "net": 0, "cc": 0, "ed": 0},
        "target_voxels": {"et": 0, "net": 0, "cc": 0, "ed": 0},
    }


def _add_count(target: dict[str, int], key: Any, amount: int = 1) -> None:
    clean_key = str(key)
    if not clean_key:
        return
    target[clean_key] = int(target.get(clean_key, 0)) + int(amount)


def _update_sampling_stats(stats: dict[str, Any], sample_info: dict[str, Any] | None) -> None:
    if not sample_info:
        return
    requested = _collated_values(sample_info.get("requested_target"))
    successful = _collated_values(sample_info.get("successful_target"))
    fallback = _collated_values(sample_info.get("fallback_target"))
    target_class_ids = _collated_values(sample_info.get("target_class_id"))
    target_class_names = _collated_values(sample_info.get("target_class_name"))
    target_voxels = _collated_values(sample_info.get("target_voxels"))
    stats["patches"] += len(requested)
    for item in requested:
        _add_count(stats["requested"], item)
    for item in successful:
        _add_count(stats["successful"], item)
    for item in fallback:
        _add_count(stats["fallback"], item)
    class_name_by_id = {1: "et", 2: "net", 3: "cc", 4: "ed"}
    for idx, (class_id, voxels) in enumerate(zip(target_class_ids, target_voxels)):
        explicit_name = ""
        if idx < len(target_class_names):
            explicit_name = str(target_class_names[idx]).strip().lower()
        name = explicit_name if explicit_name in stats["target_voxels"] else class_name_by_id.get(int(class_id))
        if name is not None:
            stats["target_voxels"][name] += int(voxels)
    for name in ["et", "net", "cc", "ed"]:
        contains = _collated_values(sample_info.get(f"actual_contains_{name}"))
        voxels = _collated_values(sample_info.get(f"actual_voxels_{name}"))
        stats["actual_contains"][name] += int(sum(int(item) for item in contains))
        stats["actual_voxels"][name] += int(sum(int(item) for item in voxels))


def _format_sampling_stats(stats: dict[str, Any]) -> str:
    if not stats or int(stats.get("patches", 0)) <= 0:
        return "n/a"
    return (
        f"patches={int(stats['patches'])} | "
        f"requested={stats.get('requested', {})} | "
        f"successful={stats.get('successful', {})} | "
        f"fallback={stats.get('fallback', {})} | "
        f"contains={stats.get('actual_contains', {})} | "
        f"target_voxels={stats.get('target_voxels', {})} | "
        f"actual_voxels={stats.get('actual_voxels', {})}"
    )


def _run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mixed_precision: bool,
    amp_dtype: torch.dtype,
    grad_clip_norm: float,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {"loss": 0.0, "branch_loss": 0.0, "mean_region_dice": 0.0, "mean_region_iou": 0.0}
    sampling_stats = _new_sampling_stats()
    batches = 0
    for batch in loader:
        images, labels, sample_info = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, mixed_precision, amp_dtype):
            prediction = model(images)
            loss = criterion(prediction, labels)
            loss_components = dict(getattr(criterion, "last_components", {}) or {})
            branch_loss = branch_usage_regularization_loss(model)
            if branch_loss is not None:
                loss = loss + branch_loss
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            metrics = batch_region_metrics(prediction, labels)
        totals["loss"] += float(loss.detach().item())
        totals["branch_loss"] += 0.0 if branch_loss is None else float(branch_loss.detach().item())
        for component_name, component_value in loss_components.items():
            totals[component_name] = totals.get(component_name, 0.0) + float(component_value)
        totals["mean_region_dice"] += float(metrics.get("mean_region_dice", 0.0))
        totals["mean_region_iou"] += float(metrics.get("mean_region_iou", 0.0))
        _update_sampling_stats(sampling_stats, sample_info)
        batches += 1
    result: dict[str, Any] = {key: value / max(1, batches) for key, value in totals.items()}
    result["sampling_summary"] = sampling_stats
    return result


@torch.no_grad()
def _run_eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    mixed_precision: bool,
    amp_dtype: torch.dtype,
    *,
    detailed_metrics: bool = False,
    detailed_spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    surface_tolerance_mm: float = 1.0,
    surface_distance_max_points: int = 20_000,
    surface_distance_crop_margin: int = 2,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0}
    metric_totals: dict[str, float] = {}
    rich_rows: list[dict[str, Any]] = []
    sampling_stats = _new_sampling_stats()
    batches = 0
    for batch in loader:
        images, labels, sample_info = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, mixed_precision, amp_dtype):
            prediction = model(images)
            loss = criterion(prediction, labels)
            loss_components = dict(getattr(criterion, "last_components", {}) or {})
        metrics = batch_region_metrics(prediction, labels)
        if detailed_metrics:
            for row in batch_rich_metric_rows(
                prediction,
                labels,
                spacing_zyx=detailed_spacing_zyx,
                surface_tolerance_mm=surface_tolerance_mm,
                surface_distance_max_points=surface_distance_max_points,
                surface_distance_crop_margin=surface_distance_crop_margin,
            ):
                row["batch_index"] = batches
                rich_rows.append(row)
        totals["loss"] += float(loss.detach().item())
        for component_name, component_value in loss_components.items():
            totals[component_name] = totals.get(component_name, 0.0) + float(component_value)
        for key, value in metrics.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + float(value)
        _update_sampling_stats(sampling_stats, sample_info)
        batches += 1
    result = {key: value / max(1, batches) for key, value in totals.items()}
    result.update({key: value / max(1, batches) for key, value in metric_totals.items()})
    result["sampling_summary"] = sampling_stats
    if detailed_metrics:
        result["rich_metric_rows"] = rich_rows
        result["rich_metrics"] = summarize_rich_metric_rows(rich_rows) if rich_rows else {}
    return result


def _save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
    epoch: int,
    best_metric: float,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg.to_dict(),
        },
        output,
    )


def _spacing_from_metadata(metadata: dict[str, Any]) -> tuple[float, float, float]:
    spacing = metadata.get("source_spacing_zyx") or metadata.get("target_spacing_zyx") or [1.0, 1.0, 1.0]
    return tuple(float(item) for item in spacing)


def _region_mask(label: np.ndarray, region_name: str) -> np.ndarray:
    if region_name == "ET":
        return label == 1
    if region_name == "NET":
        return label == 2
    if region_name == "CC":
        return label == 3
    if region_name == "ED":
        return label == 4
    if region_name == "TC":
        return np.isin(label, [1, 2, 3])
    if region_name == "WT":
        return label > 0
    raise KeyError(region_name)


def _summarize_platform_like_case_rows(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_region: dict[str, Any] = {}
    for region_name in CHALLENGE_REGIONS:
        region_rows = []
        for row in case_rows:
            presence = row.get("presence_by_region", {}).get(region_name, {})
            region_rows.append(
                {
                    "gt_present": bool(presence.get("gt_present", False)),
                    "pred_present": bool(presence.get("pred_present", False)),
                    "lesionwise_dice": float(row["lesionwise_dice_by_region"][region_name]),
                    "nsd": float(row["surface_score_by_region"][region_name]),
                }
            )
        gt_present = int(sum(int(item["gt_present"]) for item in region_rows))
        pred_present = int(sum(int(item["pred_present"]) for item in region_rows))
        tp_cases = int(sum(int(item["gt_present"] and item["pred_present"] and item["lesionwise_dice"] > 0.0) for item in region_rows))
        fp_cases = int(sum(int((not item["gt_present"]) and item["pred_present"]) for item in region_rows))
        fn_cases = int(sum(int(item["gt_present"] and not (item["pred_present"] and item["lesionwise_dice"] > 0.0)) for item in region_rows))
        present_rows = [item for item in region_rows if item["gt_present"]]
        absent_rows = [item for item in region_rows if not item["gt_present"]]
        precision = float(tp_cases / max(1, tp_cases + fp_cases))
        recall = float(tp_cases / max(1, gt_present))
        presence_f1 = float(2.0 * precision * recall / max(1e-8, precision + recall))
        by_region[region_name] = {
            "gt_present_cases": gt_present,
            "predicted_present_cases": pred_present,
            "tp_cases": tp_cases,
            "fp_cases": fp_cases,
            "fn_cases": fn_cases,
            "present_only_lesionwise_dice": float(np.mean([item["lesionwise_dice"] for item in present_rows])) if present_rows else float("nan"),
            "present_only_nsd": float(np.mean([item["nsd"] for item in present_rows])) if present_rows else float("nan"),
            "absent_case_fp_rate": float(sum(int(item["pred_present"]) for item in absent_rows) / max(1, len(absent_rows))),
            "presence_precision": precision,
            "presence_recall": recall,
            "presence_f1": presence_f1,
        }
    ed = by_region.get("ED", {})
    ed_score = (
        0.45 * float(ed.get("present_only_lesionwise_dice", 0.0) if np.isfinite(ed.get("present_only_lesionwise_dice", 0.0)) else 0.0)
        + 0.25 * float(ed.get("present_only_nsd", 0.0) if np.isfinite(ed.get("present_only_nsd", 0.0)) else 0.0)
        + 0.20 * float(ed.get("presence_f1", 0.0))
        - 0.10 * float(ed.get("absent_case_fp_rate", 0.0))
    )
    return {
        "regions": by_region,
        "ed_present_selection_score": float(ed_score),
    }


RARE_EVAL_REGIONS: dict[str, tuple[int, ...]] = {
    "ET": (1,),
    "NETC": (2,),
    "CC": (3,),
    "ED": (4,),
    "TC": (1, 2, 3),
    "WT": (1, 2, 3, 4),
}


def _mask_for_labels(label: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    return np.isin(label, labels)


def _binary_dice_np(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_sum = int(prediction.sum())
    target_sum = int(target.sum())
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / max(1, pred_sum + target_sum))


def _remove_small_components_np(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if int(min_voxels) <= 0:
        return np.asarray(mask, dtype=bool)
    from scipy import ndimage

    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3, 3), dtype=bool))
    keep = np.zeros(mask.shape, dtype=bool)
    for component_id in range(1, int(count) + 1):
        component = labels == component_id
        if int(component.sum()) >= int(min_voxels):
            keep |= component
    return keep


def _safe_et_variant(
    prediction: np.ndarray,
    probabilities: np.ndarray,
    cfg: ExperimentConfig,
    *,
    disable_ed: bool,
) -> np.ndarray:
    output = np.asarray(prediction, dtype=np.uint8).copy()
    p_et = probabilities[1]
    p_tc = probabilities[1:4].sum(axis=0)
    old_et = output == 1
    output[old_et & (p_tc >= 0.45)] = 2
    output[old_et & (p_tc < 0.45)] = 0
    et = p_et >= float(cfg.inference.threshold_et)
    et &= output != 3
    et = _remove_small_components_np(et, int(cfg.inference.min_component_voxels_et))
    output[et] = 1
    if disable_ed:
        old_ed = output == 4
        output[old_ed & (p_tc >= 0.45)] = 2
        output[old_ed & (p_tc < 0.45)] = 0
    return output


def _rare_eval_rows_for_prediction(
    *,
    case_id: str,
    variant: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region_name, labels in RARE_EVAL_REGIONS.items():
        pred_mask = _mask_for_labels(prediction, labels)
        target_mask = _mask_for_labels(target, labels)
        gt_present = bool(target_mask.any())
        pred_present = bool(pred_mask.any())
        overlap = bool(np.logical_and(pred_mask, target_mask).any())
        tp_case = bool(gt_present and pred_present and overlap)
        fp_case = bool(pred_present and not overlap)
        fn_case = bool(gt_present and not overlap)
        rows.append(
            {
                "variant": variant,
                "case_id": case_id,
                "region": region_name,
                "gt_present": int(gt_present),
                "pred_present": int(pred_present),
                "overlap": int(overlap),
                "tp_case": int(tp_case),
                "fp_case": int(fp_case),
                "fn_case": int(fn_case),
                "dice": _binary_dice_np(pred_mask, target_mask),
                "gt_voxels": int(target_mask.sum()),
                "pred_voxels": int(pred_mask.sum()),
            }
        )
    return rows


def _summarize_rare_eval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_region: dict[str, dict[str, Any]] = {}
    for region_name in RARE_EVAL_REGIONS:
        region_rows = [row for row in rows if row["region"] == region_name]
        gt_present_rows = [row for row in region_rows if int(row["gt_present"]) > 0]
        absent_rows = [row for row in region_rows if int(row["gt_present"]) == 0]
        tp_cases = int(sum(int(row["tp_case"]) for row in region_rows))
        fp_cases = int(sum(int(row["fp_case"]) for row in region_rows))
        fn_cases = int(sum(int(row["fn_case"]) for row in region_rows))
        case_f1 = float(2.0 * tp_cases / max(1, 2 * tp_cases + fp_cases + fn_cases))
        by_region[region_name] = {
            "gt_present_cases": int(sum(int(row["gt_present"]) for row in region_rows)),
            "pred_present_cases": int(sum(int(row["pred_present"]) for row in region_rows)),
            "tp_cases": tp_cases,
            "fp_cases": fp_cases,
            "fn_cases": fn_cases,
            "present_only_dice": float(np.mean([float(row["dice"]) for row in gt_present_rows])) if gt_present_rows else 0.0,
            "case_f1": case_f1,
            "fp_rate": float(fp_cases / max(1, len(absent_rows))),
            "mean_dice_including_empty": float(np.mean([float(row["dice"]) for row in region_rows])) if region_rows else 0.0,
        }
    score = (
        0.20 * by_region["ET"]["present_only_dice"]
        + 0.20 * by_region["CC"]["present_only_dice"]
        + 0.15 * by_region["ED"]["present_only_dice"]
        + 0.15 * by_region["ET"]["case_f1"]
        + 0.15 * by_region["CC"]["case_f1"]
        + 0.10 * by_region["TC"]["present_only_dice"]
        + 0.05 * by_region["WT"]["present_only_dice"]
        - 0.10 * by_region["ED"]["fp_rate"]
        - 0.05 * by_region["CC"]["fp_rate"]
    )
    return {"rare_present_score": float(score), "regions": by_region}


@torch.no_grad()
def _run_rare_present_full_evaluation(
    model: torch.nn.Module,
    records,
    cfg: ExperimentConfig,
    device: torch.device,
    logger: logging.Logger,
    *,
    run_root: Path,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    variant_rows: dict[str, list[dict[str, Any]]] = {"raw": [], "safe_postproc": [], "safe_et_ed_disabled": []}
    for case_index, record in enumerate(records):
        payload = load_cached_case(record.cache_path)
        image = payload["image"].astype("float32", copy=False)
        target = payload["label"].astype("uint8", copy=False)
        logger.info(
            "Rare-present full evaluation %d/%d | case=%s | crop_shape=%s",
            case_index + 1,
            len(records),
            record.case_id,
            tuple(int(item) for item in image.shape[1:]),
        )
        probabilities = predict_probability_sliding_window(
            model,
            image,
            num_classes=cfg.model.num_classes,
            patch_size=cfg.inference.patch_size,
            device=device,
            mixed_precision=bool(cfg.inference.mixed_precision and device.type == "cuda"),
            amp_dtype=cfg.inference.amp_dtype,
            window_batch_size=cfg.inference.window_batch_size,
            overlap=cfg.inference.sliding_window_overlap,
            use_gaussian_weighting=cfg.inference.use_gaussian_weighting,
            gaussian_sigma_scale=cfg.inference.gaussian_sigma_scale,
            score_dtype=cfg.inference.score_dtype,
            tta_mirroring=bool(getattr(cfg.train, "rare_present_full_eval_use_tta", False)),
            tta_axes=cfg.inference.tta_axes,
        )
        raw = segment_probabilities(probabilities, cfg.inference)
        safe = _safe_et_variant(raw, probabilities, cfg, disable_ed=bool(getattr(cfg.train, "disable_ed_for_safe_eval", False)))
        safe_ed_disabled = _safe_et_variant(raw, probabilities, cfg, disable_ed=True)
        variant_rows["raw"].extend(_rare_eval_rows_for_prediction(case_id=record.case_id, variant="raw", prediction=raw, target=target))
        variant_rows["safe_postproc"].extend(_rare_eval_rows_for_prediction(case_id=record.case_id, variant="safe_postproc", prediction=safe, target=target))
        variant_rows["safe_et_ed_disabled"].extend(
            _rare_eval_rows_for_prediction(case_id=record.case_id, variant="safe_et_ed_disabled", prediction=safe_ed_disabled, target=target)
        )
    summaries = {variant: _summarize_rare_eval_rows(rows) for variant, rows in variant_rows.items()}
    if bool(getattr(cfg.train, "rare_present_full_eval_save_csv", True)):
        for variant, rows in variant_rows.items():
            filename = "full_eval_epoch_%04d_%s.csv" % (int(epoch), "raw" if variant == "raw" else variant)
            pd.DataFrame(rows).to_csv(run_root / filename, index=False)
    selection_variant = "safe_et_ed_disabled" if bool(getattr(cfg.train, "disable_ed_for_safe_eval", False)) else "safe_postproc"
    return {
        "epoch": int(epoch),
        "selection_variant": selection_variant,
        "rare_present_score": float(summaries[selection_variant]["rare_present_score"]),
        "variants": summaries,
    }


def run_training(
    cfg: ExperimentConfig,
    run_dir: str | Path,
    *,
    fold_index: int | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    seed_everything(cfg.train.seed + int(fold_index if fold_index is not None else cfg.data.fold_index))
    run_root = ensure_dir(run_dir)
    _cleanup_validation_rich_metric_outputs(run_root)
    all_records = cached_records(cfg.data, "train")
    if not all_records:
        raise FileNotFoundError(f"No cached training cases found under {Path(cfg.data.cache_dir) / 'train'}.")
    split_mode = str(getattr(cfg.data, "split_mode", "fold")).strip().lower()
    active_fold = int(cfg.data.fold_index if fold_index is None else fold_index)
    active_split: dict[str, Any] | None = None
    if split_mode == "single":
        active_fold = 0
        active_split = load_single_split(cfg.data.split_file, cfg.data.split_name)
        train_records, val_records = split_records_by_single_split(all_records, active_split)
        logger.info(
            "Using single split=%s from %s | train_cases=%d validation_cases=%d",
            active_split.get("split_name", cfg.data.split_name),
            cfg.data.split_file,
            len(train_records),
            len(val_records),
        )
    else:
        folds = load_or_create_folds(all_records, cfg.data.num_folds, cfg.train.seed, cfg.data.folds_file)
        train_records, val_records = split_records_by_fold(all_records, folds, active_fold)
    logger.info(
        "Starting NeuroTS-Net single training phase: native_%s | patch_size=%s | epochs=%d | lr=%g | init=scratch",
        "x".join(str(int(item)) for item in cfg.data.patch_size),
        _format_patch_size(cfg.data.patch_size),
        int(cfg.train.epochs),
        float(cfg.train.learning_rate),
    )
    train_dataset = _make_patch_dataset(
        train_records,
        cfg,
        epoch_size=int(cfg.train.iterations_per_epoch) * int(cfg.data.batch_size),
        seed_offset=10_000 * active_fold,
        augment=True,
    )
    val_dataset = _make_patch_dataset(
        val_records,
        cfg,
        epoch_size=int(cfg.train.val_iterations_per_epoch) * int(cfg.data.batch_size),
        seed_offset=20_000 + 10_000 * active_fold,
        augment=False,
    )
    train_loader = _make_loader(train_dataset, cfg, shuffle=False)
    val_loader = _make_loader(val_dataset, cfg, shuffle=False)
    device = choose_device()
    logger.info("Starting training on device=%s", device)
    model = build_model(cfg.model).to(device)
    logger.info("Model=%s parameters=%d", cfg.model.name, _parameter_count(model))
    if str(cfg.data.label_mode).strip().lower() != "multiclass":
        raise ValueError("The public NeuroTS-Net trainer supports data.label_mode='multiclass' only.")
    foreground_names = [LABEL_REGION_NAMES[index] for index in sorted(LABEL_REGION_NAMES)]
    logger.info(
        "Classes=%d foreground=%s challenge_regions=%s",
        int(cfg.model.num_classes),
        foreground_names,
        list(CHALLENGE_REGIONS),
    )
    logger.info(
        "Cases train=%d val=%d test=0 | epoch_eval_split=validation | patch_size=%s",
        len(train_records),
        len(val_records),
        _format_patch_size(cfg.data.patch_size),
    )
    criterion = build_loss(cfg.loss, num_classes=cfg.model.num_classes)
    if float(getattr(cfg.loss, "absent_rare_fp_weight", 0.0)) > 0.0 and not hasattr(criterion, "_absent_rare_fp_loss"):
        raise RuntimeError("loss.absent_rare_fp_weight > 0 but the configured loss does not compute loss_absent_rare_fp.")
    optimizer = build_optimizer(
        model.parameters(),
        cfg.train.optimizer,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        adam_eps=cfg.train.adam_eps,
    )
    amp_dtype = resolve_autocast_dtype(device, cfg.train.amp_dtype)
    mixed_precision = bool(cfg.train.mixed_precision and device.type == "cuda")
    scaler = _make_grad_scaler(device, mixed_precision)
    logger.info(
        "Optimizer=%s lr=%g adam_eps=%g weight_decay=%g scheduler=%s warmup_epochs=%d loss=%s amp_dtype=%s grad_scaler=%s",
        str(cfg.train.optimizer).lower(),
        float(cfg.train.learning_rate),
        float(cfg.train.adam_eps),
        float(cfg.train.weight_decay),
        str(cfg.train.scheduler).lower(),
        int(cfg.train.warmup_epochs),
        str(cfg.loss.name).lower(),
        str(cfg.train.amp_dtype),
        bool(scaler.is_enabled()),
    )
    logger.info("Resource training_start | %s", format_resource_snapshot(resource_snapshot("training_start")))
    best_voxel_metric = float("-inf")
    best_voxel_epoch = 0
    best_rare_present_score: float | None = None
    best_rare_present_epoch: int | None = None
    best_voxel_rich_metrics: dict[str, Any] = {}
    epochs_without_improvement = 0
    rows: list[dict[str, Any]] = []
    validation_rich_history: list[dict[str, Any]] = []
    validation_rich_table_rows: list[dict[str, Any]] = []
    metrics_csv = run_root / "metrics.csv"
    for epoch in range(1, int(cfg.train.epochs) + 1):
        start_time = time.time()
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        lr = _learning_rate_for_epoch(cfg, epoch)
        _set_optimizer_lr(optimizer, lr)
        progress = 0.0 if cfg.train.epochs <= 1 else float(epoch - 1) / float(cfg.train.epochs - 1)
        configure_branch_selection(model, progress=progress, regularization_weight=_branch_regularization_weight(cfg, epoch))
        train_metrics = _run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            mixed_precision,
            amp_dtype,
            cfg.train.grad_clip_norm,
        )
        train_sampling_summary = train_metrics.pop("sampling_summary", {})
        do_validation = epoch == 1 or epoch % int(cfg.train.validation_interval) == 0 or epoch == int(cfg.train.epochs)
        detailed_interval = max(1, int(cfg.train.validation_detailed_metrics_interval))
        compute_detailed_metrics = bool(cfg.train.compute_validation_detailed_metrics) and (
            epoch == 1 or epoch % detailed_interval == 0 or epoch == int(cfg.train.epochs)
        )
        val_metrics = (
            _run_eval_epoch(
                model,
                val_loader,
                criterion,
                device,
                mixed_precision,
                amp_dtype,
                detailed_metrics=compute_detailed_metrics,
                detailed_spacing_zyx=(1.0, 1.0, 1.0),
                surface_tolerance_mm=cfg.inference.surface_tolerance_mm,
                surface_distance_max_points=cfg.inference.surface_distance_max_points,
                surface_distance_crop_margin=cfg.inference.surface_distance_crop_margin,
            )
            if do_validation
            else {}
        )
        val_sampling_summary = val_metrics.pop("sampling_summary", {}) if val_metrics else {}
        val_rich_rows = val_metrics.pop("rich_metric_rows", []) if val_metrics else []
        val_rich_metrics = val_metrics.pop("rich_metrics", {}) if val_metrics else {}
        if val_rich_metrics:
            validation_rich_payload = {"epoch": epoch, **val_rich_metrics}
            validation_rich_history.append(validation_rich_payload)
            validation_rich_table_rows.extend(_validation_rich_metric_table_rows(epoch, val_rich_metrics))
            _save_validation_rich_metric_history(
                run_root=run_root,
                history=validation_rich_history,
                table_rows=validation_rich_table_rows,
            )
        val_metric = float(val_metrics.get("mean_region_dice", best_voxel_metric))
        voxel_improved = do_validation and val_metric > best_voxel_metric
        if voxel_improved:
            best_voxel_metric = val_metric
            best_voxel_epoch = epoch
            epochs_without_improvement = 0
            best_voxel_rich_metrics = val_rich_metrics
            voxel_checkpoint_path = run_root / "best_voxel_dice.pt"
            _save_checkpoint(voxel_checkpoint_path, model=model, optimizer=optimizer, cfg=cfg, epoch=epoch, best_metric=best_voxel_metric)
            logger.info("New best voxel Dice checkpoint saved: %s", voxel_checkpoint_path)
        elif do_validation:
            epochs_without_improvement += int(cfg.train.validation_interval)
        early_stop_now = bool(
            cfg.train.early_stopping_patience > 0 and epochs_without_improvement >= int(cfg.train.early_stopping_patience)
        )
        rare_full_metrics: dict[str, Any] = {}
        rare_present_improved = False
        rare_enabled = bool(getattr(cfg.train, "enable_rare_present_full_evaluation", False))
        rare_interval = max(1, int(getattr(cfg.train, "rare_present_full_eval_interval", 50)))
        rare_start = max(1, int(getattr(cfg.train, "rare_present_full_eval_start_epoch", rare_interval)))
        do_rare_full_eval = (
            rare_enabled
            and (
                epoch == int(cfg.train.epochs)
                or early_stop_now
                or (epoch >= rare_start and epoch % rare_interval == 0)
            )
        )
        if do_rare_full_eval:
            logger.info("Starting rare-present full evaluation at epoch=%d", epoch)
            rare_full_metrics = _run_rare_present_full_evaluation(model, val_records, cfg, device, logger, run_root=run_root, epoch=epoch)
            save_json(rare_full_metrics, run_root / f"rare_present_full_eval_epoch_{epoch:04d}.json")
            rare_metric = float(rare_full_metrics.get("rare_present_score", float("-inf")))
            if best_rare_present_score is None or rare_metric > best_rare_present_score:
                best_rare_present_score = rare_metric
                best_rare_present_epoch = epoch
                rare_present_improved = True
                rare_checkpoint_path = run_root / "best_rare_present_score.pt"
                _save_checkpoint(
                    rare_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    epoch=epoch,
                    best_metric=best_rare_present_score,
                )
                logger.info("New best rare-present checkpoint saved: %s", rare_checkpoint_path)
            selected_summary = rare_full_metrics.get("variants", {}).get(rare_full_metrics.get("selection_variant", ""), {})
            selected_regions = selected_summary.get("regions", {})
            logger.info(
                "Rare-present full eval epoch=%d | variant=%s score=%s | ET dice=%s f1=%s | CC dice=%s f1=%s | ED dice=%s f1=%s fp_rate=%s%s",
                epoch,
                rare_full_metrics.get("selection_variant"),
                _format_metric(rare_full_metrics.get("rare_present_score")),
                _format_metric(selected_regions.get("ET", {}).get("present_only_dice")),
                _format_metric(selected_regions.get("ET", {}).get("case_f1")),
                _format_metric(selected_regions.get("CC", {}).get("present_only_dice")),
                _format_metric(selected_regions.get("CC", {}).get("case_f1")),
                _format_metric(selected_regions.get("ED", {}).get("present_only_dice")),
                _format_metric(selected_regions.get("ED", {}).get("case_f1")),
                _format_metric(selected_regions.get("ED", {}).get("fp_rate")),
                " | rare improved" if rare_present_improved else "",
            )
        row = {
            "epoch": epoch,
            "learning_rate": lr,
            "train_loss": train_metrics.get("loss"),
            "train_branch_loss": train_metrics.get("branch_loss"),
            "train_mean_region_dice": train_metrics.get("mean_region_dice"),
            "train_mean_region_iou": train_metrics.get("mean_region_iou"),
            "val_loss": val_metrics.get("loss"),
            "val_mean_region_dice": val_metrics.get("mean_region_dice"),
            "val_mean_region_iou": val_metrics.get("mean_region_iou"),
            "best_val_mean_region_dice": best_voxel_metric,
            "rare_present_score": rare_full_metrics.get("rare_present_score"),
            "best_rare_present_score": best_rare_present_score,
            "epoch_seconds": time.time() - start_time,
        }
        for key, value in train_metrics.items():
            row[f"train_{key}"] = value
        for key, value in val_metrics.items():
            row[f"val_{key}"] = value
        rows.append(row)
        pd.DataFrame(rows).to_csv(metrics_csv, index=False)
        if cfg.output.save_plots:
            save_training_plots(metrics_csv, run_root)
        logger.info(
            "Epoch %d/%d | train loss %.4f dice %.4f iou %.4f absent_fp %s | val loss %s dice %s iou %s absent_fp %s | lr %.6g | seconds %.2f%s",
            epoch,
            cfg.train.epochs,
            float(train_metrics.get("loss", 0.0)),
            float(train_metrics.get("mean_region_dice", 0.0)),
            float(train_metrics.get("mean_region_iou", 0.0)),
            _format_metric(train_metrics.get("loss_absent_rare_fp")),
            _format_metric(val_metrics.get("loss")),
            _format_metric(val_metrics.get("mean_region_dice")),
            _format_metric(val_metrics.get("mean_region_iou")),
            _format_metric(val_metrics.get("loss_absent_rare_fp")),
            lr,
            float(row["epoch_seconds"]),
            " | voxel improved" if voxel_improved else "",
        )
        logger.info("Resource epoch_end epoch=%d | %s", epoch, format_resource_snapshot(resource_snapshot("epoch_end")))
        if train_sampling_summary:
            sampling_payload = {"epoch": epoch, "train": train_sampling_summary, "validation": val_sampling_summary}
            save_json(sampling_payload, run_root / "patch_sampling_latest.json")
            logger.info("Patch sampling epoch=%d train | %s", epoch, _format_sampling_stats(train_sampling_summary))
            if val_sampling_summary:
                logger.info("Patch sampling epoch=%d validation | %s", epoch, _format_sampling_stats(val_sampling_summary))
        if early_stop_now:
            logger.info("Early stopping at epoch %d after %d validation epochs without improvement.", epoch, epochs_without_improvement)
            break
        gc.collect()
    summary = {
        "fold_index": active_fold,
        "fold": active_fold + 1,
        "split_mode": split_mode,
        "split_name": None if active_split is None else active_split.get("split_name"),
        "split_file": cfg.data.split_file if split_mode == "single" else None,
        "best_epoch": best_voxel_epoch,
        "best_val_mean_region_dice": best_voxel_metric,
        "best_voxel_epoch": best_voxel_epoch,
        "best_voxel_dice": best_voxel_metric,
        "best_rare_present_epoch": best_rare_present_epoch,
        "best_rare_present_score": best_rare_present_score,
        "best_voxel_rich_metrics": best_voxel_rich_metrics,
        "best_checkpoint_path": str(run_root / "best_voxel_dice.pt"),
        "best_voxel_checkpoint_path": str(run_root / "best_voxel_dice.pt"),
        "best_rare_present_checkpoint_path": str(run_root / "best_rare_present_score.pt") if best_rare_present_score is not None else None,
        "metrics_csv": str(metrics_csv),
        "train_cases": [record.case_id for record in train_records],
        "validation_cases": [record.case_id for record in val_records],
    }
    save_json(summary, run_root / "summary.json")
    return summary
