from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from neurots_net.data.brats26 import CachedCaseRecord, load_cached_case
from neurots_net.utils.io import save_json


def _quantile_bins(values: list[int], num_bins: int = 4) -> list[float]:
    if not values:
        return [0.0] * (num_bins - 1)
    return [float(item) for item in np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, num_bins + 1)[1:-1])]


def _bin_index(value: int, bins: list[float]) -> int:
    result = 0
    for threshold in bins:
        if float(value) > float(threshold):
            result += 1
    return result


def _case_feature_vector(properties: dict[str, Any], wt_bins: list[float], tc_bins: list[float], et_small_threshold: float, cc_small_threshold: float) -> list[float]:
    volumes = properties["label_volumes"]
    et_volume = int(volumes.get("1", 0))
    cc_volume = int(volumes.get("3", 0))
    wt_bin = _bin_index(int(properties["wt_volume"]), wt_bins)
    tc_bin = _bin_index(int(properties["tc_volume"]), tc_bins)
    vector = [
        float(properties["et_present"]),
        float(properties["net_present"]),
        float(properties["cc_present"]),
        float(properties["ed_present"]),
        float(et_volume > 0 and et_volume <= et_small_threshold),
        float(cc_volume > 0 and cc_volume <= cc_small_threshold),
    ]
    vector.extend(float(wt_bin == idx) for idx in range(4))
    vector.extend(float(tc_bin == idx) for idx in range(4))
    vector.extend([np.log1p(float(properties["wt_volume"])) / 12.0, np.log1p(float(properties["tc_volume"])) / 12.0])
    return [float(item) for item in vector]


def build_folds(records: list[CachedCaseRecord], num_folds: int, seed: int, output_path: str | Path) -> dict[str, Any]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    case_rows: list[dict[str, Any]] = []
    for record in records:
        payload = load_cached_case(record.cache_path)
        metadata = payload["metadata"]
        properties = metadata.get("label_properties")
        if properties is None:
            raise ValueError(f"Missing label properties in cache file {record.cache_path}.")
        case_rows.append(
            {
                "case_id": record.case_id,
                "group_id": metadata.get("group_id", record.group_id),
                "properties": properties,
            }
        )
    wt_bins = _quantile_bins([int(row["properties"]["wt_volume"]) for row in case_rows], 4)
    tc_bins = _quantile_bins([int(row["properties"]["tc_volume"]) for row in case_rows], 4)
    et_positive = [int(row["properties"]["label_volumes"].get("1", 0)) for row in case_rows if int(row["properties"]["label_volumes"].get("1", 0)) > 0]
    cc_positive = [int(row["properties"]["label_volumes"].get("3", 0)) for row in case_rows if int(row["properties"]["label_volumes"].get("3", 0)) > 0]
    et_small_threshold = float(np.quantile(et_positive, 0.25)) if et_positive else 0.0
    cc_small_threshold = float(np.quantile(cc_positive, 0.25)) if cc_positive else 0.0
    for row in case_rows:
        row["wt_quantile"] = _bin_index(int(row["properties"]["wt_volume"]), wt_bins)
        row["tc_quantile"] = _bin_index(int(row["properties"]["tc_volume"]), tc_bins)
        row["feature_vector"] = _case_feature_vector(row["properties"], wt_bins, tc_bins, et_small_threshold, cc_small_threshold)

    groups: dict[str, dict[str, Any]] = {}
    for row in case_rows:
        group = groups.setdefault(row["group_id"], {"group_id": row["group_id"], "case_ids": [], "feature_vector": None})
        group["case_ids"].append(row["case_id"])
        vector = np.asarray(row["feature_vector"], dtype=np.float64)
        group["feature_vector"] = vector if group["feature_vector"] is None else group["feature_vector"] + vector

    rng = np.random.default_rng(int(seed))
    group_items = list(groups.values())
    rng.shuffle(group_items)
    rarity = np.asarray([sum(group["feature_vector"]) for group in group_items], dtype=np.float64)
    order = np.argsort(-rarity)
    group_items = [group_items[int(idx)] for idx in order]
    feature_dim = len(case_rows[0]["feature_vector"]) if case_rows else 1
    total_features = np.sum([group["feature_vector"] for group in group_items], axis=0) if group_items else np.zeros(feature_dim)
    target_features = total_features / float(num_folds)
    target_count = len(case_rows) / float(num_folds)
    fold_features = [np.zeros(feature_dim, dtype=np.float64) for _ in range(num_folds)]
    fold_case_ids = [[] for _ in range(num_folds)]
    fold_group_ids = [[] for _ in range(num_folds)]
    feature_scale = np.maximum(target_features, 1.0)
    max_fold_count = int(np.ceil(len(case_rows) / float(num_folds))) if case_rows else 0

    for group in group_items:
        best_fold = 0
        best_score = float("inf")
        group_vector = np.asarray(group["feature_vector"], dtype=np.float64)
        group_count = len(group["case_ids"])
        for fold_idx in range(num_folds):
            projected_count = len(fold_case_ids[fold_idx]) + group_count
            overflow_penalty = max(0, projected_count - max_fold_count) * 10.0
            projected = fold_features[fold_idx] + group_vector
            balance = np.square((projected - target_features) / feature_scale).sum()
            count_balance = ((projected_count - target_count) / max(target_count, 1.0)) ** 2
            score = float(balance + count_balance + overflow_penalty)
            if score < best_score:
                best_score = score
                best_fold = fold_idx
        fold_features[best_fold] += group_vector
        fold_case_ids[best_fold].extend(sorted(group["case_ids"]))
        fold_group_ids[best_fold].append(group["group_id"])

    folds = []
    all_case_ids = sorted(row["case_id"] for row in case_rows)
    for fold_idx in range(num_folds):
        validation_ids = sorted(fold_case_ids[fold_idx])
        validation_set = set(validation_ids)
        folds.append(
            {
                "fold_index": fold_idx,
                "fold": fold_idx + 1,
                "validation": validation_ids,
                "train": [case_id for case_id in all_case_ids if case_id not in validation_set],
                "groups": sorted(fold_group_ids[fold_idx]),
            }
        )
    payload = {
        "num_folds": num_folds,
        "seed": int(seed),
        "wt_volume_bins": wt_bins,
        "tc_volume_bins": tc_bins,
        "et_small_threshold": et_small_threshold,
        "cc_small_threshold": cc_small_threshold,
        "folds": folds,
        "case_properties": case_rows,
    }
    save_json(payload, output_path)
    return payload


def load_or_create_folds(records: list[CachedCaseRecord], num_folds: int, seed: int, path: str | Path, force: bool = False) -> dict[str, Any]:
    fold_path = Path(path)
    if fold_path.exists() and not force:
        return json.loads(fold_path.read_text(encoding="utf-8"))
    return build_folds(records, num_folds=num_folds, seed=seed, output_path=fold_path)


def split_records_by_fold(records: list[CachedCaseRecord], folds: dict[str, Any], fold_index: int) -> tuple[list[CachedCaseRecord], list[CachedCaseRecord]]:
    record_by_id = {record.case_id: record for record in records}
    fold_items = folds["folds"]
    if fold_index < 0 or fold_index >= len(fold_items):
        raise IndexError(f"fold_index {fold_index} outside available folds 0..{len(fold_items) - 1}.")
    fold = fold_items[fold_index]
    train = [record_by_id[case_id] for case_id in fold["train"] if case_id in record_by_id]
    validation = [record_by_id[case_id] for case_id in fold["validation"] if case_id in record_by_id]
    return train, validation


def load_single_split(path: str | Path, split_name: str | None = None) -> dict[str, Any]:
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Single split file does not exist: {split_path}. "
            "Run `python scripts/create_high_signal_split.py --config <config> --output <split_file>` first."
        )
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if "splits" in payload:
        name = str(split_name or payload.get("default_split") or "")
        if not name:
            if len(payload["splits"]) != 1:
                raise ValueError(f"Split file {split_path} contains multiple splits; set data.split_name.")
            name = next(iter(payload["splits"]))
        if name not in payload["splits"]:
            raise KeyError(f"Split {name!r} not found in {split_path}. Available: {sorted(payload['splits'])}")
        split = dict(payload["splits"][name])
        split["split_name"] = name
        split["metadata"] = payload.get("metadata", {})
        split["summary"] = payload.get("summary", {}).get(name, payload.get("summary", {}))
        return split
    if "train" not in payload or "validation" not in payload:
        raise ValueError(f"Single split file {split_path} must contain train and validation case ID lists.")
    split = dict(payload)
    split["split_name"] = str(split_name or payload.get("split_name") or "single")
    return split


def split_records_by_single_split(records: list[CachedCaseRecord], split: dict[str, Any]) -> tuple[list[CachedCaseRecord], list[CachedCaseRecord]]:
    record_by_id = {record.case_id: record for record in records}
    train_ids = [str(item) for item in split.get("train", [])]
    validation_ids = [str(item) for item in split.get("validation", split.get("val", []))]
    if not train_ids or not validation_ids:
        raise ValueError("Single split must contain non-empty train and validation case ID lists.")
    missing = sorted((set(train_ids) | set(validation_ids)) - set(record_by_id))
    if missing:
        raise KeyError(f"Split references case IDs not present in cache: {missing[:10]}")
    overlap = sorted(set(train_ids) & set(validation_ids))
    if overlap:
        raise ValueError(f"Single split train/validation overlap detected: {overlap[:10]}")
    train = [record_by_id[case_id] for case_id in train_ids]
    validation = [record_by_id[case_id] for case_id in validation_ids]
    return train, validation
