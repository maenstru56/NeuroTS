from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurots_net.config import ExperimentConfig
from neurots_net.data.brats26 import discover_training_cases, discover_validation_cases, preprocess_dataset
from neurots_net.data.folds import load_or_create_folds
from neurots_net.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the NeuroTS-Net BraTS-PED cache.")
    parser.add_argument("--config", type=str, default="configs/neurots_brats26_peds_5fold.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--rebuild-folds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    logger = setup_logger("neurots_net.preprocess", Path(cfg.data.cache_dir) / "preprocess.log")
    train_cases = discover_training_cases(cfg.data)
    validation_cases = discover_validation_cases(cfg.data)
    logger.info("Discovered training_cases=%d validation_cases=%d", len(train_cases), len(validation_cases))
    preprocess_dataset(
        cfg.data,
        include_training=not args.skip_training,
        include_validation=not args.skip_validation,
        overwrite=args.overwrite,
        logger=logger,
    )
    if not args.skip_training and str(cfg.data.split_mode).strip().lower() == "fold":
        from neurots_net.data.brats26 import cached_records

        records = cached_records(cfg.data, "train")
        folds = load_or_create_folds(
            records,
            num_folds=cfg.data.num_folds,
            seed=cfg.train.seed,
            path=cfg.data.folds_file,
            force=args.rebuild_folds,
        )
        logger.info("Prepared %d folds at %s", len(folds["folds"]), cfg.data.folds_file)


if __name__ == "__main__":
    main()
