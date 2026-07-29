from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurots_net.config import ExperimentConfig
from neurots_net.inference import create_submission_archives, find_fold_checkpoints, predict_validation_ensemble
from neurots_net.utils.io import ensure_dir
from neurots_net.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict BraTS validation cases with one checkpoint or a NeuroTS-Net ensemble.")
    parser.add_argument("--config", default=None, help="Config YAML. Defaults to <run-dir>/config.yaml when available.")
    parser.add_argument("--run-dir", default=None, help="Experiment directory used to discover checkpoints.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--candidate",
        default=None,
        choices=["voxel_dice", "rare_present", "mixed"],
        help="Checkpoint family to discover when --checkpoint is not supplied.",
    )
    parser.add_argument("--checkpoint", action="append", default=None, help="Explicit checkpoint. Repeat to average an ensemble.")
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--disable-fold-predictions", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def _has_candidate(run_dir: Path, candidate: str) -> bool:
    try:
        return bool(find_fold_checkpoints(run_dir, candidate=candidate))
    except FileNotFoundError:
        return False


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else None
    config_path = Path(args.config) if args.config else None
    if config_path is None and run_dir is not None and (run_dir / "config.yaml").exists():
        config_path = run_dir / "config.yaml"
    if config_path is None:
        config_path = Path("configs/neurots_brats26_peds_5fold.yaml")

    cfg = ExperimentConfig.from_yaml(config_path)
    run_dir = run_dir or Path(cfg.output.output_root) / cfg.output.experiment_name
    if args.disable_tta:
        cfg.inference.tta_mirroring = False
    if args.disable_fold_predictions:
        cfg.inference.save_fold_predictions = False
    if args.save_probabilities:
        cfg.inference.save_probabilities = True

    candidate = args.candidate
    if candidate is None:
        candidate = "rare_present" if _has_candidate(run_dir, "rare_present") else "voxel_dice"
    checkpoints = [Path(item) for item in args.checkpoint] if args.checkpoint else find_fold_checkpoints(run_dir, candidate)
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    output_dir = ensure_dir(args.output_dir or (run_dir / "validation_predictions" / candidate))
    logger = setup_logger("neurots_net.predict_validation", output_dir / "prediction.log")
    logger.info("Selected prediction candidate=%s | run_dir=%s", candidate, run_dir)
    logger.info("Using %d checkpoint(s): %s", len(checkpoints), [str(path) for path in checkpoints])
    logger.info("Main-model ED predictions are enabled.")

    predict_validation_ensemble(cfg, checkpoints, output_dir, logger=logger)
    if not args.no_zip:
        archives = create_submission_archives(
            output_dir,
            experiment_name=cfg.output.experiment_name,
            candidate=candidate,
        )
        for archive in archives:
            logger.info("Created archive: %s | files=%d", archive["archive_path"], archive["num_files"])


if __name__ == "__main__":
    main()
