from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch

from neurots_net.config import ExperimentConfig
from neurots_net.data.brats26 import (
    CachedCaseRecord,
    cached_records,
    discover_training_cases,
    group_id_from_case_id,
    load_cached_case,
    preprocess_case,
)
from neurots_net.data.folds import load_single_split, split_records_by_single_split
from neurots_net.engine.train import _safe_et_variant, _spacing_from_metadata
from neurots_net.inference import (
    choose_device,
    load_checkpoint_model,
    predict_probability_sliding_window,
    segment_probabilities,
)
from neurots_net.metrics import rich_metric_rows, summarize_rich_metric_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a NeuroTS checkpoint on its internal full-volume validation split.")
    parser.add_argument("--config", required=True, help="Archived experiment YAML used for the run.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--output-prefix", required=True, help="Output path without .csv/.json suffix.")
    parser.add_argument("--variant", choices=("raw", "safe_postproc"), default="safe_postproc")
    parser.add_argument("--tta", action="store_true", help="Enable mirroring TTA. Disabled by default to match checkpoint selection.")
    parser.add_argument(
        "--analysis-cache",
        default="cache/internal_checkpoint_evaluation",
        help="Fallback cache used only when the original training cache is unavailable.",
    )
    return parser.parse_args()


def validation_records_for_split(cfg: ExperimentConfig, split: dict[str, object], analysis_cache: str | Path) -> list[CachedCaseRecord]:
    validation_ids = [str(item) for item in split.get("validation", split.get("val", []))]
    if not validation_ids:
        raise ValueError("The configured single split contains no validation case IDs.")

    records = cached_records(cfg.data, "train")
    record_by_id = {record.case_id: record for record in records}
    if all(case_id in record_by_id for case_id in validation_ids):
        _, validation_records = split_records_by_single_split(records, split)
        return validation_records

    raw_by_id = {case.case_id: case for case in discover_training_cases(cfg.data)}
    missing_raw = sorted(set(validation_ids) - set(raw_by_id))
    if missing_raw:
        raise KeyError(f"Validation IDs missing from raw training roots: {missing_raw[:10]}")
    cache_root = Path(analysis_cache)
    result: list[CachedCaseRecord] = []
    print(f"Original training cache is unavailable; preprocessing {len(validation_ids)} validation cases into {cache_root}")
    for case_index, case_id in enumerate(validation_ids, start=1):
        output = cache_root / f"{case_id}.npz"
        print(f"[cache {case_index:02d}/{len(validation_ids):02d}] {case_id}", flush=True)
        if output.exists():
            try:
                with zipfile.ZipFile(output, "r") as archive:
                    if archive.testzip() is not None or "metadata.npy" not in archive.namelist():
                        raise zipfile.BadZipFile(f"Incomplete analysis cache: {output}")
            except (OSError, ValueError, zipfile.BadZipFile):
                output.unlink()
        preprocess_case(raw_by_id[case_id], output, cfg.data, overwrite=False)
        result.append(CachedCaseRecord(case_id, output, "train", group_id_from_case_id(case_id)))
    return result


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if str(cfg.data.split_mode).strip().lower() != "single":
        raise ValueError("This evaluator currently requires data.split_mode=single.")

    split = load_single_split(cfg.data.split_file, cfg.data.split_name)
    validation_records = validation_records_for_split(cfg, split, args.analysis_cache)
    if not validation_records:
        raise ValueError("The configured single split contains no validation cases.")

    device = choose_device()
    model = load_checkpoint_model(args.checkpoint, cfg, device)
    rows: list[dict[str, object]] = []
    print(f"Evaluating {len(validation_records)} full cases on {device} | variant={args.variant} | tta={bool(args.tta)}")

    with torch.inference_mode():
        for case_index, record in enumerate(validation_records, start=1):
            payload = load_cached_case(record.cache_path)
            image = payload["image"].astype(np.float32, copy=False)
            target = payload["label"].astype(np.uint8, copy=False)
            metadata = payload["metadata"]
            print(f"[{case_index:02d}/{len(validation_records):02d}] {record.case_id} {tuple(image.shape[1:])}", flush=True)
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
                tta_mirroring=bool(args.tta),
                tta_axes=cfg.inference.tta_axes,
            )
            prediction = segment_probabilities(probabilities, cfg.inference)
            if args.variant == "safe_postproc":
                prediction = _safe_et_variant(prediction, probabilities, cfg, disable_ed=False)

            spacing = _spacing_from_metadata(metadata)
            for row in rich_metric_rows(
                prediction,
                target,
                spacing_zyx=spacing,
                surface_tolerance_mm=cfg.inference.surface_tolerance_mm,
                surface_distance_max_points=cfg.inference.surface_distance_max_points,
                surface_distance_crop_margin=cfg.inference.surface_distance_crop_margin,
            ):
                row["case_id"] = record.case_id
                target_name = str(row["target"])
                if target_name == "TC":
                    target_mask = np.isin(target, (1, 2, 3))
                elif target_name == "WT" or target_name == "foreground":
                    target_mask = target > 0
                else:
                    class_id = row.get("class_id")
                    target_mask = target == int(class_id) if class_id is not None else np.zeros_like(target, dtype=bool)
                row["gt_present"] = int(bool(target_mask.any()))
                rows.append(row)

    summary = summarize_rich_metric_rows(rows)
    present_rows = [row for row in rows if int(row["gt_present"]) > 0]
    summary["present_only"] = summarize_rich_metric_rows(present_rows)
    summary["checkpoint"] = str(args.checkpoint)
    summary["config"] = str(args.config)
    summary["variant"] = str(args.variant)
    summary["tta"] = bool(args.tta)
    summary["validation_cases"] = len(validation_records)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(prefix.with_suffix(".csv"), index=False)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"by_target": summary["by_target"], "present_only": summary["present_only"]["by_target"]}, indent=2))


if __name__ == "__main__":
    main()
