from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurots_net.config import ExperimentConfig
from neurots_net.data.brats26 import cached_records, load_cached_case
from neurots_net.data.nifti_io import save_segmentation_like_reference
from neurots_net.postprocessing import (
    PostprocessingParameters,
    hard_seg_from_main_probabilities,
    postprocess_main_probabilities,
    summarize_prediction_labels,
)
from neurots_net.inference import (
    MAIN_CHANNEL_ORDER,
    choose_device,
    create_submission_zip,
    load_checkpoint_model,
    paste_crop_to_full,
    predict_probability_sliding_window,
    save_probability_npz,
)
from neurots_net.utils.io import ensure_dir, save_json
from neurots_net.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeuroTS-Net ensemble inference with configurable ET/CC component postprocessing."
    )
    parser.add_argument("--config", default="configs/neurots_brats26_peds_single_split.yaml")
    parser.add_argument("--checkpoint", action="append", required=True, help="Checkpoint to average. Repeat for ensembles.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-name", default="neurots_postprocessed")
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cc-logit-bias", type=float, default=0.75)
    parser.add_argument("--cc-prob-thr", type=float, default=0.15)
    parser.add_argument("--cc-min-size", type=int, default=20)
    parser.add_argument("--cc-max-components", type=int, default=3)
    parser.add_argument("--et-logit-bias", type=float, default=0.25)
    parser.add_argument("--et-prob-thr", type=float, default=0.30)
    parser.add_argument("--et-min-size", type=int, default=20)
    parser.add_argument("--et-max-components", type=int, default=4)
    parser.add_argument("--protect-tc-wt", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.cache_dir:
        cfg.data.cache_dir = str(args.cache_dir)
    if args.disable_tta:
        cfg.inference.tta_mirroring = False

    checkpoints = [Path(item) for item in args.checkpoint]
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    records = cached_records(cfg.data, "validation")
    if not records:
        raise FileNotFoundError(
            f"No cached validation cases under {Path(cfg.data.cache_dir) / 'validation'}. "
            "Run scripts/preprocess_brats26.py first."
        )

    output_root = ensure_dir(args.output)
    prediction_dir = ensure_dir(output_root / "predictions")
    probability_dir = ensure_dir(output_root / "probabilities") if args.save_probabilities else None
    logger = setup_logger("neurots_net.predict_postprocessed", output_root / "prediction.log")
    device = choose_device()
    params = PostprocessingParameters(
        cc_logit_bias=float(args.cc_logit_bias),
        cc_prob_threshold=float(args.cc_prob_thr),
        cc_min_size=int(args.cc_min_size),
        cc_max_components=int(args.cc_max_components),
        et_logit_bias=float(args.et_logit_bias),
        et_prob_threshold=float(args.et_prob_thr),
        et_min_size=int(args.et_min_size),
        et_max_components=int(args.et_max_components),
        protect_tc_wt=bool(args.protect_tc_wt),
    )
    logger.info("Starting postprocessed inference on device=%s | cases=%d", device, len(records))
    logger.info("Checkpoints (%d): %s", len(checkpoints), [str(path) for path in checkpoints])
    logger.info("Main-model ED predictions are enabled and are not globally suppressed.")
    logger.info("Postprocessing parameters: %s", params.to_dict())

    summary = {
        "candidate_name": str(args.candidate_name),
        "device": str(device),
        "checkpoints": [str(path) for path in checkpoints],
        "parameters": params.to_dict(),
        "main_ed_enabled": True,
        "probability_channel_order": list(MAIN_CHANNEL_ORDER),
        "cases": [],
    }
    for case_index, record in enumerate(records, start=1):
        payload = load_cached_case(record.cache_path)
        image = payload["image"].astype(np.float32, copy=False)
        metadata = payload["metadata"]
        logger.info("Predicting case %d/%d | %s | crop_shape=%s", case_index, len(records), record.case_id, tuple(image.shape[1:]))
        probability_sum: np.ndarray | None = None
        for checkpoint_path in checkpoints:
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
            probability_sum = probabilities if probability_sum is None else probability_sum + probabilities
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if probability_sum is None:
            raise RuntimeError("No model probabilities were produced.")
        average_probability = probability_sum / float(len(checkpoints))
        base_segmentation = hard_seg_from_main_probabilities(average_probability)
        prediction_crop, fusion_summary = postprocess_main_probabilities(
            average_probability,
            params,
            base_segmentation=base_segmentation,
        )
        prediction_full = paste_crop_to_full(prediction_crop, metadata)
        prediction_path = prediction_dir / f"{record.case_id}.nii.gz"
        save_segmentation_like_reference(prediction_full, metadata["reference_modality"], prediction_path)
        if probability_dir is not None:
            save_probability_npz(
                probability_dir / f"{record.case_id}.npz",
                average_probability,
                case_id=record.case_id,
                channel_order=MAIN_CHANNEL_ORDER,
                source="ensemble",
            )
        summary["cases"].append(
            {
                "case_id": record.case_id,
                "prediction": str(prediction_path),
                **summarize_prediction_labels(prediction_full),
                "core_adjustments": fusion_summary,
            }
        )

    save_json(summary, output_root / "prediction_summary.json")
    if args.zip:
        archive = create_submission_zip(prediction_dir, output_root / f"{args.candidate_name}.zip")
        save_json(archive, output_root / "zip_summary.json")
        logger.info("Created archive: %s | files=%d", archive["archive_path"], archive["num_files"])
    logger.info("Finished postprocessed inference | cases=%d", len(records))


if __name__ == "__main__":
    main()
