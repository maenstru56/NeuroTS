from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from neurots_net.config import ExperimentConfig
from neurots_net.engine.train import run_training
from neurots_net.utils.io import ensure_dir, save_json


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _create_ensemble_directory(run_root: Path, fold_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    ensemble_root = ensure_dir(run_root / "folds_all")
    checkpoint_dir = ensure_dir(ensemble_root / "checkpoints")
    checkpoint_rows = []
    for summary in fold_summaries:
        voxel_source = Path(summary["best_voxel_checkpoint_path"])
        voxel_target = checkpoint_dir / f"fold_{int(summary['fold']):02d}_best_voxel_dice.pt"
        if voxel_source.exists():
            shutil.copy2(voxel_source, voxel_target)
        checkpoint_rows.append(
            {
                "fold": int(summary["fold"]),
                "fold_index": int(summary["fold_index"]),
                "best_voxel_source_checkpoint": str(voxel_source),
                "best_voxel_ensemble_checkpoint": str(voxel_target),
                "best_voxel_epoch": int(summary["best_voxel_epoch"]),
                "best_voxel_dice": float(summary["best_voxel_dice"]),
            }
        )
    manifest = {
        "description": "NeuroTS-Net v1 5-fold ensemble checkpoint set.",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoints": checkpoint_rows,
    }
    save_json(manifest, ensemble_root / "ensemble_manifest.json")
    return manifest


def _mean_std(values: list[float]) -> tuple[float, float]:
    numeric = pd.Series(values, dtype="float64").dropna()
    if numeric.empty:
        return float("nan"), 0.0
    return float(numeric.mean()), float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0


def _save_cross_fold_metric_summaries(run_root: Path, fold_summaries: list[dict[str, Any]]) -> dict[str, str]:
    scalar_rows = []
    scalar_keys = ["best_voxel_dice"]
    for key in scalar_keys:
        mean_value, std_value = _mean_std(
            [value for value in (_optional_float(item.get(key)) for item in fold_summaries) if value is not None]
        )
        scalar_rows.append({"metric": key, "mean": mean_value, "std": std_value, "mean_pm_std": f"{mean_value:.6f} +/- {std_value:.6f}"})
    scalar_path = run_root / "fold_scalar_metrics_mean_std.csv"
    pd.DataFrame(scalar_rows).to_csv(scalar_path, index=False)

    target_metric_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for summary in fold_summaries:
        rich_metrics = summary.get("best_voxel_rich_metrics") or {}
        for row in rich_metrics.get("by_target", []):
            grouped.setdefault((str(row["target_type"]), str(row["target"])), []).append(row)
    metric_names = ["jaccard", "iou", "dice", "sdc", "nsd", "accuracy", "precision", "recall", "hd95"]
    for (target_type, target), rows in sorted(grouped.items()):
        output_row: dict[str, Any] = {"target_type": target_type, "target": target, "folds": len(rows)}
        for metric_name in metric_names:
            values = [float(row.get(f"{metric_name}_mean", float("nan"))) for row in rows]
            mean_value, std_value = _mean_std(values)
            output_row[f"{metric_name}_mean"] = mean_value
            output_row[f"{metric_name}_std"] = std_value
            output_row[f"{metric_name}_mean_pm_std"] = f"{mean_value:.6f} +/- {std_value:.6f}"
        target_metric_rows.append(output_row)
    rich_path = run_root / "fold_rich_metrics_mean_std.csv"
    pd.DataFrame(target_metric_rows).to_csv(rich_path, index=False)
    return {"scalar_metrics": str(scalar_path), "rich_metrics": str(rich_path)}


def run_cross_validation(
    cfg: ExperimentConfig,
    run_dir: str | Path,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    run_root = ensure_dir(run_dir)
    cfg.save_yaml(run_root / "config.yaml")
    if str(getattr(cfg.data, "split_mode", "fold")).strip().lower() == "single":
        logger.info(
            "Starting single-split training for dataset=%s split=%s split_file=%s",
            cfg.output.experiment_name,
            cfg.data.split_name,
            cfg.data.split_file,
        )
        summary = run_training(cfg, run_root, fold_index=0, logger=logger)
        payload = {
            "experiment_name": cfg.output.experiment_name,
            "split_mode": "single",
            "split_name": cfg.data.split_name,
            "split_file": cfg.data.split_file,
            "run_summary": summary,
        }
        save_json(payload, run_root / "summary.json")
        return payload
    logger.info(
        "Starting %d-fold cross validation for dataset=%s. Each fold uses train/validation subsets only; "
        "the held-out validation fold is also the epoch optimization split.",
        int(cfg.data.num_folds),
        cfg.output.experiment_name,
    )
    fold_summaries = []
    for fold_index in range(int(cfg.data.num_folds)):
        fold_number = fold_index + 1
        fold_dir = ensure_dir(run_root / f"fold_{fold_number:02d}")
        fold_cfg = deepcopy(cfg)
        fold_cfg.data.fold_index = fold_index
        logger.info("Starting fold %d/%d in %s", fold_number, cfg.data.num_folds, fold_dir)
        summary = run_training(fold_cfg, fold_dir, fold_index=fold_index, logger=logger)
        fold_summaries.append(summary)
    rows = [
        {
            "fold": item["fold"],
            "best_voxel_epoch": item["best_voxel_epoch"],
            "best_voxel_dice": item["best_voxel_dice"],
            "best_voxel_checkpoint_path": item["best_voxel_checkpoint_path"],
        }
        for item in fold_summaries
    ]
    metrics_table = pd.DataFrame(rows)
    metrics_path = run_root / "fold_metrics.csv"
    metrics_table.to_csv(metrics_path, index=False)
    cross_fold_metric_paths = _save_cross_fold_metric_summaries(run_root, fold_summaries)
    ensemble_manifest = _create_ensemble_directory(run_root, fold_summaries) if cfg.output.create_ensemble_dir else None
    summary = {
        "experiment_name": cfg.output.experiment_name,
        "num_folds": int(cfg.data.num_folds),
        "fold_metrics_csv": str(metrics_path),
        "mean_best_voxel_dice": float(metrics_table["best_voxel_dice"].mean()) if not metrics_table.empty else None,
        "cross_fold_metric_summaries": cross_fold_metric_paths,
        "folds": fold_summaries,
        "ensemble_manifest": ensemble_manifest,
    }
    save_json(summary, run_root / "summary.json")
    return summary
