from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurots_net.config import ExperimentConfig, apply_overrides
from neurots_net.engine.crossval import run_cross_validation
from neurots_net.utils.io import ensure_dir
from neurots_net.utils.logging import setup_logger


def _parse_int3(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--patch-size must contain D,H,W.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuroTS-Net training with cross-validation or a configured single split.")
    parser.add_argument("--config", type=str, default="configs/neurots_brats26_peds_5fold.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--iterations-per-epoch", type=int, default=None)
    parser.add_argument("--val-iterations-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patch-size", type=str, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--disable-augment", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg = apply_overrides(
        cfg,
        {
            "train.epochs": args.epochs,
            "train.iterations_per_epoch": args.iterations_per_epoch,
            "train.val_iterations_per_epoch": args.val_iterations_per_epoch,
            "data.batch_size": args.batch_size,
            "data.patch_size": _parse_int3(args.patch_size),
            "inference.patch_size": _parse_int3(args.patch_size),
            "model.base_channels": args.base_channels,
            "data.num_workers": args.num_workers,
            "augmentation.enabled": False if args.disable_augment else None,
            "train.mixed_precision": False if args.disable_amp else None,
        },
    )
    run_root = ensure_dir(Path(cfg.output.output_root) / cfg.output.experiment_name)
    logger = setup_logger("neurots_net.crossval", run_root / "training.log")
    run_cross_validation(cfg, run_root, logger=logger)


if __name__ == "__main__":
    main()
