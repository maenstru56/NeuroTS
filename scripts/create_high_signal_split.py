from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurots_net.config import ExperimentConfig
from neurots_net.data.brats26 import cached_records, discover_training_cases, group_id_from_case_id, load_cached_case
from neurots_net.data.nifti_io import load_nifti_zyx
from neurots_net.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a high-signal single validation split for NeuroTS-Net.")
    parser.add_argument("--config", default="configs/neurots_brats26_peds_single_split.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--split-name", default="single_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _component_stats(mask: np.ndarray) -> dict[str, Any]:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3, 3), dtype=bool))
    sizes = [int((labels == component_id).sum()) for component_id in range(1, int(count) + 1)]
    return {
        "components": int(count),
        "small_components_10_200": int(sum(10 <= size <= 200 for size in sizes)),
        "min_component": int(min(sizes)) if sizes else 0,
        "max_component": int(max(sizes)) if sizes else 0,
    }


def _case_row_from_label(case_id: str, group_id: str, label: np.ndarray) -> dict[str, Any]:
    volumes = {str(class_id): int(np.count_nonzero(label == class_id)) for class_id in [1, 2, 3, 4]}
    wt_volume = int(np.count_nonzero(label > 0))
    tc_volume = int(np.count_nonzero(np.isin(label, [1, 2, 3])))
    row: dict[str, Any] = {
        "case_id": case_id,
        "group_id": group_id,
        "label_volumes": volumes,
        "wt_volume": wt_volume,
        "tc_volume": tc_volume,
        "et_present": volumes["1"] > 0,
        "net_present": volumes["2"] > 0,
        "cc_present": volumes["3"] > 0,
        "ed_present": volumes["4"] > 0,
    }
    for class_id, name in [(1, "et"), (3, "cc"), (4, "ed")]:
        stats = _component_stats(label == class_id)
        row[f"{name}_components"] = stats["components"]
        row[f"{name}_small_components_10_200"] = stats["small_components_10_200"]
        row[f"{name}_min_component"] = stats["min_component"]
        row[f"{name}_max_component"] = stats["max_component"]
    return row


def _load_case_rows(cfg: ExperimentConfig) -> list[dict[str, Any]]:
    records = cached_records(cfg.data, "train")
    rows: list[dict[str, Any]] = []
    if records:
        for record in records:
            payload = load_cached_case(record.cache_path)
            metadata = payload["metadata"]
            if "label" in payload:
                rows.append(_case_row_from_label(record.case_id, metadata.get("group_id", record.group_id), payload["label"]))
                continue
            properties = metadata.get("label_properties")
            if properties is None:
                raise ValueError(f"Missing label properties in {record.cache_path}.")
            volumes = {str(key): int(value) for key, value in properties.get("label_volumes", {}).items()}
            rows.append(
                {
                    "case_id": record.case_id,
                    "group_id": metadata.get("group_id", record.group_id),
                    "label_volumes": volumes,
                    "wt_volume": int(properties.get("wt_volume", 0)),
                    "tc_volume": int(properties.get("tc_volume", 0)),
                    "et_present": bool(properties.get("et_present", False)),
                    "net_present": bool(properties.get("net_present", False)),
                    "cc_present": bool(properties.get("cc_present", False)),
                    "ed_present": bool(properties.get("ed_present", False)),
                    "et_components": 0,
                    "cc_components": 0,
                    "ed_components": 0,
                    "et_small_components_10_200": 0,
                    "cc_small_components_10_200": 0,
                    "ed_small_components_10_200": 0,
                }
            )
        return sorted(rows, key=lambda item: item["case_id"])

    cases = discover_training_cases(cfg.data)
    if not cases:
        raise FileNotFoundError("No training cases found in data.train_roots and no training cache is available.")
    for case in cases:
        label = load_nifti_zyx(case.seg_path, dtype=np.uint8).data_zyx
        rows.append(_case_row_from_label(case.case_id, group_id_from_case_id(case.case_id), label))
    return sorted(rows, key=lambda item: item["case_id"])


def _quantile_threshold(values: list[int], q: float) -> float:
    positives = [int(value) for value in values if int(value) > 0]
    return float(np.quantile(np.asarray(positives, dtype=np.float64), q)) if positives else 0.0


def _augment_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wt_values = [int(row["wt_volume"]) for row in rows]
    tc_values = [int(row["tc_volume"]) for row in rows]
    wt_bins = [float(item) for item in np.quantile(np.asarray(wt_values, dtype=np.float64), [0.25, 0.5, 0.75])]
    tc_bins = [float(item) for item in np.quantile(np.asarray(tc_values, dtype=np.float64), [0.25, 0.5, 0.75])]
    small_thresholds = {
        "et": _quantile_threshold([int(row["label_volumes"].get("1", 0)) for row in rows], 0.25),
        "cc": _quantile_threshold([int(row["label_volumes"].get("3", 0)) for row in rows], 0.25),
        "ed": _quantile_threshold([int(row["label_volumes"].get("4", 0)) for row in rows], 0.25),
    }
    for row in rows:
        row["wt_quantile"] = int(np.searchsorted(wt_bins, int(row["wt_volume"]), side="right"))
        row["tc_quantile"] = int(np.searchsorted(tc_bins, int(row["tc_volume"]), side="right"))
        row["small_et"] = bool(0 < int(row["label_volumes"].get("1", 0)) <= small_thresholds["et"])
        row["small_cc"] = bool(0 < int(row["label_volumes"].get("3", 0)) <= small_thresholds["cc"])
        row["small_ed"] = bool(0 < int(row["label_volumes"].get("4", 0)) <= small_thresholds["ed"])
        row["hard_ed_negative"] = bool(
            not row["ed_present"]
            and (
                row["cc_present"]
                or row["et_present"]
                or int(row["wt_quantile"]) in {0, 3}
                or int(row.get("cc_small_components_10_200", 0)) > 0
                or int(row.get("et_small_components_10_200", 0)) > 0
            )
        )
    return {"wt_bins": wt_bins, "tc_bins": tc_bins, "small_thresholds": small_thresholds}


def _summary(rows: list[dict[str, Any]], case_ids: list[str]) -> dict[str, Any]:
    selected = [row for row in rows if row["case_id"] in set(case_ids)]
    label_volume_sums = {
        str(class_id): int(sum(int(row["label_volumes"].get(str(class_id), 0)) for row in selected))
        for class_id in [1, 2, 3, 4]
    }
    return {
        "cases": len(selected),
        "ed_present": int(sum(bool(row["ed_present"]) for row in selected)),
        "cc_present": int(sum(bool(row["cc_present"]) for row in selected)),
        "et_present": int(sum(bool(row["et_present"]) for row in selected)),
        "net_present": int(sum(bool(row["net_present"]) for row in selected)),
        "ed_negative": int(sum(not bool(row["ed_present"]) for row in selected)),
        "ed_negative_hard": int(sum(bool(row.get("hard_ed_negative", False)) for row in selected)),
        "small_et": int(sum(bool(row.get("small_et", False)) for row in selected)),
        "small_cc": int(sum(bool(row.get("small_cc", False)) for row in selected)),
        "small_ed": int(sum(bool(row.get("small_ed", False)) for row in selected)),
        "wt_quantiles": {str(idx): int(sum(int(row["wt_quantile"]) == idx for row in selected)) for idx in range(4)},
        "tc_quantiles": {str(idx): int(sum(int(row["tc_quantile"]) == idx for row in selected)) for idx in range(4)},
        "label_volume_sums": label_volume_sums,
        "wt_volume_mean": float(np.mean([int(row["wt_volume"]) for row in selected])) if selected else 0.0,
        "wt_volume_median": float(np.median([int(row["wt_volume"]) for row in selected])) if selected else 0.0,
        "tc_volume_mean": float(np.mean([int(row["tc_volume"]) for row in selected])) if selected else 0.0,
        "tc_volume_median": float(np.median([int(row["tc_volume"]) for row in selected])) if selected else 0.0,
    }


def _selection_score(rows: list[dict[str, Any]], selected_ids: set[str], candidate: dict[str, Any], targets: dict[str, Any]) -> float:
    proposed = sorted(selected_ids | {candidate["case_id"]})
    summary = _summary(rows, proposed)
    score = 0.0
    weighted_targets = {
        "cases": (targets["cases"], 0.15),
        "ed_present": (targets["ed_present"], 3.0),
        "cc_present": (targets["cc_present"], 2.4),
        "et_present": (targets["et_present"], 1.6),
        "ed_negative_hard": (targets["ed_negative_hard"], 2.0),
        "small_et": (targets["small_et"], 1.2),
        "small_cc": (targets["small_cc"], 1.8),
        "small_ed": (targets["small_ed"], 1.8),
    }
    for key, (target, weight) in weighted_targets.items():
        deficit = max(0.0, float(target) - float(summary[key]))
        overshoot = max(0.0, float(summary[key]) - float(target))
        score += float(weight) * (deficit * deficit + 0.35 * overshoot * overshoot)
    for quantile_key in ["wt_quantiles", "tc_quantiles"]:
        for idx in range(4):
            target = float(targets[quantile_key][str(idx)])
            observed = float(summary[quantile_key][str(idx)])
            score += 0.35 * (target - observed) ** 2
    rarity_bonus = (
        0.15 * int(candidate["ed_present"])
        + 0.12 * int(candidate["cc_present"])
        + 0.08 * int(candidate["et_present"])
        + 0.10 * int(candidate.get("hard_ed_negative", False))
        + 0.08 * int(candidate.get("small_cc", False))
        + 0.08 * int(candidate.get("small_ed", False))
    )
    return float(score - rarity_bonus)


def create_split(rows: list[dict[str, Any]], *, val_size: int, seed: int) -> dict[str, Any]:
    metadata = _augment_rows(rows)
    rng = np.random.default_rng(int(seed))
    val_size = int(np.clip(val_size, 1, max(1, len(rows) - 1)))
    targets = {
        "cases": val_size,
        "ed_present": min(16, max(12, int(round(val_size * 0.35)))),
        "cc_present": min(18, max(14, int(round(val_size * 0.40)))),
        "et_present": min(24, max(18, int(round(val_size * 0.55)))),
        "ed_negative_hard": min(16, max(12, int(round(val_size * 0.35)))),
        "small_et": max(5, int(round(val_size * 0.18))),
        "small_cc": max(5, int(round(val_size * 0.18))),
        "small_ed": max(4, int(round(val_size * 0.15))),
        "wt_quantiles": {str(idx): val_size // 4 for idx in range(4)},
        "tc_quantiles": {str(idx): val_size // 4 for idx in range(4)},
    }
    for key in ["wt_quantiles", "tc_quantiles"]:
        for idx in range(val_size % 4):
            targets[key][str(idx)] += 1

    selected_ids: set[str] = set()
    candidates = list(rows)
    rng.shuffle(candidates)
    while len(selected_ids) < val_size:
        remaining = [row for row in candidates if row["case_id"] not in selected_ids]
        if not remaining:
            break
        best = min(remaining, key=lambda row: _selection_score(rows, selected_ids, row, targets))
        selected_ids.add(best["case_id"])
    validation_ids = sorted(selected_ids)
    train_ids = sorted(row["case_id"] for row in rows if row["case_id"] not in selected_ids)
    split = {
        "train": train_ids,
        "validation": validation_ids,
        "targets": targets,
        "summary": {
            "train": _summary(rows, train_ids),
            "validation": _summary(rows, validation_ids),
            "all": _summary(rows, [row["case_id"] for row in rows]),
        },
        "metadata": metadata,
        "case_properties": rows,
    }
    return split


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    output = Path(args.output or cfg.data.split_file)
    if output.exists() and not args.force:
        raise FileExistsError(f"Split file already exists: {output}. Use --force to overwrite.")
    rows = _load_case_rows(cfg)
    split = create_split(rows, val_size=int(args.val_size), seed=int(args.seed))
    payload = {
        "default_split": args.split_name,
        "splits": {
            args.split_name: {
                "train": split["train"],
                "validation": split["validation"],
                "targets": split["targets"],
            }
        },
        "summary": {args.split_name: split["summary"]},
        "metadata": {
            "seed": int(args.seed),
            "val_size": int(args.val_size),
            "source_config": str(args.config),
            **split["metadata"],
        },
        "case_properties": split["case_properties"],
    }
    save_json(payload, output)
    summary = split["summary"]["validation"]
    print(f"Wrote split to {output}")
    print(
        "validation cases={cases} ED+={ed_present} CC+={cc_present} ET+={et_present} "
        "ED-={ed_negative} hard_ED-={ed_negative_hard} small_ET={small_et} small_CC={small_cc} small_ED={small_ed}".format(**summary)
    )


if __name__ == "__main__":
    main()
