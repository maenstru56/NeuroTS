from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

from neurots_net.utils.io import load_yaml, save_yaml


T = TypeVar("T")


@dataclass
class DataConfig:
    train_roots: list[str] = field(default_factory=lambda: ["data/BraTS26_PED_training", "data/BraTS26_PED_training_Batch2_Release"])
    validation_root: str = "data/BraTS26_PED_validation"
    cache_dir: str = "cache/brats26_peds_task2_native_margin32"
    folds_file: str = "cache/brats26_peds_task2_native_margin32/folds_5.json"
    modalities: list[str] = field(default_factory=lambda: ["t1n", "t1c", "t2w", "t2f"])
    crop_margin: int = 32
    patch_size: list[int] = field(default_factory=lambda: [128, 192, 128])
    batch_size: int = 2
    num_folds: int = 5
    fold_index: int = 0
    split_mode: str = "fold"
    split_file: str = ""
    split_name: str = "single"
    foreground_oversample: float | None = 0.33
    min_foreground_voxels: int = 16
    label_mode: str = "multiclass"
    patch_sampling_strategy: str = "component_balanced"
    patch_sampling_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "random": 0.20,
            "foreground": 0.20,
            "et": 0.15,
            "net": 0.10,
            "cc": 0.175,
            "ed": 0.175,
        }
    )
    target_class_jitter_voxels: int = 24
    targeted_patch_retries: int = 8
    cache_memory_cases: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    image_dtype: str = "float16"
    overwrite_cache: bool = False
    intensity_clip_percentiles: list[float] = field(default_factory=lambda: [0.5, 99.5])


@dataclass
class ModelConfig:
    name: str = "NeuroTS_Net_v1"
    in_channels: int = 4
    num_classes: int = 5
    base_channels: int = 32
    model_id: str = "L"
    kernel_size: int = 3
    expansion_ratios: list[int] | None = None
    block_counts: list[int] | None = None
    normalization: str = "group"
    dropout: float = 0.0
    deep_supervision: bool = True
    residual_blocks: bool = True
    residual_resampling: bool = True
    checkpoint_style: str | None = "outside_block"
    use_global_response_norm: bool = True
    use_selective_branches: bool = True
    branch_dropout_prob: float = 0.05
    branch_selection_topk: int = 2
    branch_temperature_start: float = 2.0
    branch_temperature_end: float = 0.5
    branch_residual_gain_init: float = 0.005
    use_raw_detail_stream: bool = True
    raw_detail_channels: int = 8
    raw_detail_injection_init: float = 0.001


@dataclass
class LossConfig:
    name: str = "region_aware"
    include_background: bool = False
    batch_dice: bool = True
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    label_smoothing: float = 0.0
    main_ce_weight: float = 0.40
    main_label_dice_weight: float = 0.30
    main_region_weight: float = 0.30
    absent_rare_fp_weight: float = 0.0
    ce_class_weights: list[float] | None = field(default_factory=lambda: [0.05, 2.50, 1.00, 4.00, 2.75])
    label_dice_class_weights: list[float] = field(default_factory=lambda: [1.50, 0.50, 2.50, 2.00])
    region_loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "ED": 2.50,
            "CC": 2.00,
            "ET": 1.50,
            "NET": 0.75,
            "TC": 0.75,
            "WT": 0.50,
        }
    )
    rare_tversky_alpha_fp: float = 0.40
    rare_tversky_beta_fn: float = 0.60
    et_tversky_alpha_fp: float | None = None
    et_tversky_beta_fn: float | None = None
    cc_tversky_alpha_fp: float | None = None
    cc_tversky_beta_fn: float | None = None
    ed_tversky_alpha_fp_main: float | None = None
    ed_tversky_beta_fn_main: float | None = None
    absent_rare_fp_classes: list[int] = field(default_factory=lambda: [1, 3, 4])
    absent_rare_fp_power: float = 2.0
    deep_supervision_full_rare_outputs: int = 1


@dataclass
class TrainConfig:
    epochs: int = 1000
    iterations_per_epoch: int = 250
    val_iterations_per_epoch: int = 50
    optimizer: str = "adamw"
    learning_rate: float = 1.5e-3
    adam_eps: float = 1e-4
    weight_decay: float = 3e-5
    grad_clip_norm: float = 12.0
    mixed_precision: bool = True
    amp_dtype: str = "bfloat16"
    scheduler: str = "poly"
    cosine_eta_min: float = 1e-6
    warmup_epochs: int = 5
    early_stopping_patience: int = 300
    seed: int = 42
    validation_interval: int = 1
    compute_validation_detailed_metrics: bool = True
    validation_detailed_metrics_interval: int = 1
    enable_rare_present_full_evaluation: bool = False
    rare_present_full_eval_interval: int = 50
    rare_present_full_eval_start_epoch: int = 50
    rare_present_full_eval_use_tta: bool = False
    rare_present_full_eval_save_csv: bool = True
    rare_present_full_eval_postprocess_et: bool = True
    disable_ed_for_safe_eval: bool = False
    branch_usage_regularization_weight: float = 1e-4
    branch_usage_regularization_fraction: float = 0.3


@dataclass
class AugmentationConfig:
    enabled: bool = True
    rotation_p: float = 0.2
    rotation_degrees: float = 30.0
    scaling_p: float = 0.2
    scale_range: list[float] = field(default_factory=lambda: [0.7, 1.4])
    elastic_enabled: bool = False
    gaussian_noise_p: float = 0.1
    gaussian_noise_variance: list[float] = field(default_factory=lambda: [0.0, 0.1])
    gaussian_blur_p: float = 0.2
    gaussian_blur_sigma: list[float] = field(default_factory=lambda: [0.5, 1.0])
    gaussian_blur_channel_p: float = 0.5
    brightness_p: float = 0.15
    brightness_multiplier: list[float] = field(default_factory=lambda: [0.75, 1.25])
    contrast_p: float = 0.15
    contrast_range: list[float] = field(default_factory=lambda: [0.75, 1.25])
    lowres_p: float = 0.25
    lowres_scale: list[float] = field(default_factory=lambda: [0.5, 1.0])
    lowres_channel_p: float = 0.5
    gamma_inverted_p: float = 0.1
    gamma_normal_p: float = 0.3
    gamma_range: list[float] = field(default_factory=lambda: [0.7, 1.5])
    gamma_retain_stats: bool = True
    mirror_axes: list[int] = field(default_factory=lambda: [0, 1, 2])
    mirror_axis_p: float = 0.5


@dataclass
class InferenceConfig:
    patch_size: list[int] = field(default_factory=lambda: [128, 192, 128])
    sliding_window_overlap: float = 0.625
    window_batch_size: int = 1
    use_gaussian_weighting: bool = True
    gaussian_sigma_scale: float = 0.125
    mixed_precision: bool = True
    amp_dtype: str = "bfloat16"
    score_dtype: str = "float32"
    tta_mirroring: bool = True
    tta_axes: list[int] = field(default_factory=lambda: [0, 1, 2])
    save_fold_predictions: bool = True
    save_probabilities: bool = False
    lesion_dilation_iterations: int = 5
    lesion_volume_threshold_mm3: float = 10.0
    surface_tolerance_mm: float = 1.0
    surface_distance_max_points: int = 20_000
    surface_distance_crop_margin: int = 2
    use_probability_fusion: bool = False
    logit_biases: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    wt_gate_threshold: float = 0.25
    wt_gate_dilation: int = 4
    threshold_et: float = 0.25
    threshold_cc: float = 0.20
    threshold_ed: float = 0.15
    min_component_voxels_et: int = 10
    min_component_voxels_cc: int = 10
    min_component_voxels_ed: int = 10


@dataclass
class OutputConfig:
    output_root: str = "outputs"
    experiment_name: str = "neurots_brats26_peds"
    save_plots: bool = True
    create_ensemble_dir: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        cfg = cls()
        data = load_yaml(path)
        _update_dataclass(cfg, data)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        save_yaml(self, path)


def _is_dataclass_type(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__")


def _coerce_value(current: Any, value: Any, annotation: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin in {list, tuple}:
        return list(value)
    if origin is not None and type(None) in get_args(annotation):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if args:
            return _coerce_value(current, value, args[0])
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, str):
        return str(value)
    return value


def _update_dataclass(instance: Any, updates: dict[str, Any]) -> None:
    field_map = {item.name: item for item in fields(instance)}
    for key, value in updates.items():
        if key not in field_map:
            raise KeyError(f"Unknown configuration key: {key}")
        current = getattr(instance, key)
        if _is_dataclass_type(current):
            if not isinstance(value, dict):
                raise TypeError(f"Configuration section {key} must be a mapping.")
            _update_dataclass(current, value)
        else:
            setattr(instance, key, _coerce_value(current, value, field_map[key].type))


def apply_overrides(cfg: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        target: Any = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        current = getattr(target, parts[-1])
        setattr(target, parts[-1], _coerce_value(current, value, type(current)))
    return cfg
