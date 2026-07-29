# NeuroTS-Net Configuration Reference

NeuroTS-Net experiments are defined by YAML files in this directory. The two configurations use the same model and training recipe while selecting different data-splitting strategies:

| File | Purpose |
|---|---|
| `neurots_brats26_peds_5fold.yaml` | Grouped cross-validation using a fold-definition JSON file. |
| `neurots_brats26_peds_single_split.yaml` | One explicit training/validation split using a split-definition JSON file. |

Run commands from the repository root so relative paths such as `data/...`,
`cache/...`, and `outputs/...` resolve correctly. The YAML files are the
authoritative reproduction settings, meaning that fields omitted from a custom YAML file use
the defaults declared in `src/neurots_net/config.py`.

## Loading and Validation

The loader expects these seven top-level mappings:

```yaml
data: {}
model: {}
loss: {}
train: {}
augmentation: {}
inference: {}
output: {}
```

Configuration loading is intentionally strict:

- Duplicate YAML keys raise an error that includes the file, duplicated key,
  line, and column.
- Unknown fields raise an error instead of being ignored.
- Missing optional fields receive their dataclass defaults.
- YAML `null` is loaded as Python `None`.
- Class-weight arrays, model stage arrays, split settings, and sampling
  probabilities are validated before training.
- `patch_sampling_probabilities` must sum to `1.0`, within floating-point
  tolerance.

Validate either or both shipped files without starting training:

```bash
python scripts/check_config_yaml.py configs/neurots_brats26_peds_5fold.yaml
python scripts/check_config_yaml.py configs/neurots_brats26_peds_single_split.yaml
```

## Common Conventions

- Volumetric shapes use `[D, H, W]` order inside the configuration.
- Spatial augmentation axes `0`, `1`, and `2` refer to the internal depth,
  height, and width dimensions.
- Multiclass labels are ordered as `[background, ET, NET, CC, ED]`, or
  `[0, 1, 2, 3, 4]`.
- Evaluated composite regions are `TC = ET + NET + CC` and
  `WT = ET + NET + CC + ED`.
- Component-size postprocessing values are voxel counts. Metric volume
  thresholds with an `_mm3` suffix are physical volumes in cubic millimetres.
- Probabilities are softmax probabilities unless a script explicitly states
  otherwise.

## `data`

The `data` section controls data discovery, splitting, caching, patch creation,
and loading.

| Field | Type | Description |
|---|---|---|
| `train_roots` | list of paths | One or more labelled training roots. Cases are deduplicated by case ID across roots. |
| `validation_root` | path or `null` | Root containing unlabelled challenge/hold-out cases used by prediction scripts. It is not the epoch-validation split. |
| `cache_dir` | path | Reusable preprocessed cache root. Training data and external validation data are stored in separate subdirectories. |
| `split_mode` | string | `fold` for grouped cross-validation or `single` for one explicit train/validation split. |
| `folds_file` | path or `null` | JSON file containing fold assignments when `split_mode: fold`. It is generated from cached cases if missing. |
| `num_folds` | integer | Number of folds to create or use in fold mode. |
| `fold_index` | integer | Fold selected by single-fold commands. `train_crossval.py` iterates over all configured folds. |
| `split_file` | path or `null` | JSON file containing explicit `train` and `val` case-ID lists when `split_mode: single`. |
| `split_name` | string | Human-readable identifier recorded in logs and output metadata for a single split. |
| `modalities` | list of strings | Ordered input modalities. The public model expects `[t1n, t1c, t2w, t2f]`. This order defines the input-channel order. |
| `crop_margin` | integer | Margin, in voxels, added around the nonzero foreground bounding box during preprocessing. |
| `patch_size` | three integers | Training patch shape in `[D, H, W]` order. |
| `batch_size` | integer | Number of patches in one training or patch-validation batch. |
| `foreground_oversample` | float or `null` | Legacy probability of generic foreground-centred sampling. It is ignored when `patch_sampling_strategy: component_balanced`; `null` is valid in that mode. |
| `min_foreground_voxels` | integer | Minimum total foreground voxels requested for generic foreground and label-based hard-negative patches. It is not a target-class voxel minimum. |
| `label_mode` | string | Target representation. The public main model uses `multiclass`. |
| `patch_sampling_strategy` | string | `component_balanced` activates the class- and component-aware sampler. Other values use the retained legacy random/foreground path. |
| `patch_sampling_probabilities` | mapping | Probability assigned to every component-balanced patch request. See the table below. |
| `target_class_jitter_voxels` | integer | Maximum random centre displacement, in voxels, applied along each axis after selecting a target-class voxel. |
| `targeted_patch_retries` | integer | Maximum retries used to obtain a valid targeted or hard-negative patch before a logged fallback. |
| `cache_memory_cases` | integer | Maximum number of preprocessed cases retained in each worker's in-memory cache. |
| `num_workers` | integer | Number of data-loader worker processes. Set to `0` for in-process loading and easier debugging. |
| `pin_memory` | boolean | Pins CPU batches before GPU transfer when supported. |
| `image_dtype` | string | On-disk cached image dtype. `float32` is recommended; AMP casting occurs during training/inference. |
| `overwrite_cache` | boolean | Rebuilds existing preprocessed case entries when preprocessing is run. The preprocessing CLI `--overwrite` also enables this behavior. |
| `intensity_clip_percentiles` | two floats | Lower and upper foreground percentiles used to clip each modality before foreground-only z-score normalization. |

### Split Modes

Fold mode uses grouped assignments from `folds_file`:

```yaml
data:
  split_mode: fold
  folds_file: cache/brats26_peds_task2_native_margin32/folds_5.json
  num_folds: 5
  fold_index: 0
```

Single-split mode uses the exact case IDs in `split_file` and ignores
`num_folds`, `fold_index`, and `folds_file`:

```yaml
data:
  split_mode: single
  split_file: cache/brats26_peds_task2_native_margin32/splits_single_seed42.json
  split_name: single_seed42
```

If single-split mode is requested and the split file is missing or malformed,
training fails instead of silently falling back to fold mode. Generate the
shipped split format with:

```bash
python scripts/create_high_signal_split.py \
  --config configs/neurots_brats26_peds_single_split.yaml \
  --output cache/brats26_peds_task2_native_margin32/splits_single_seed42.json \
  --split-name single_seed42 \
  --seed 42 \
  --val-size 40
```

### Component-Balanced Sampling

The public recipe uses the following nested keys under
`patch_sampling_probabilities`:

| Key | Public value | Requested patch |
|---|---:|---|
| `random` | `0.15` | A random brain/case patch. |
| `foreground` | `0.20` | A generic patch centred on any non-background label. |
| `et` | `0.15` | Choose an ET-containing case, then an ET connected component uniformly, then a voxel in that component. |
| `net` | `0.10` | The equivalent component-balanced request for NET. |
| `cc` | `0.20` | The equivalent component-balanced request for CC. |
| `ed` | `0.12` | The equivalent component-balanced request for ED. |
| `hard_negative` | `0.08` | Prefer an ED-negative case and centre on non-ED tumour or a tumour-adjacent location; the accepted patch must contain no ED. |

Targeted class requests choose a case containing the class before selecting a
connected component. Components are sampled uniformly rather than in proportion
to their voxel volume. After centre jitter, the sampler verifies that the final
crop still contains the requested class. Requests, successes, fallbacks, actual
class presence, and sampled target-voxel counts are written to the training log
and sampling-statistics JSON.

## `model`

The `model` section defines the NeuroTS-Net backbone and its optional adaptive
mechanisms.

| Field | Type | Description |
|---|---|---|
| `name` | string | Architecture identifier. The public implementation uses `NeuroTS_Net_v1`. |
| `in_channels` | integer | Number of input MRI channels; `4` for T1N, T1C, T2W, and T2F. |
| `num_classes` | integer | Number of output logits; `5` for background plus four tumour labels. |
| `base_channels` | integer | Width of the first stage. Deeper encoder widths scale from this value. |
| `model_id` | string | Capacity preset: `S`, `B`, `M`, or `L`. It supplies expansion ratios and block counts when their explicit fields are `null`. |
| `kernel_size` | integer | Kernel size of the main depthwise spatial convolution, normally `3`. |
| `expansion_ratios` | nine integers or `null` | Optional explicit channel-expansion ratio for each encoder, bottleneck, and decoder stage. Exactly nine values are required when supplied. |
| `block_counts` | nine integers or `null` | Optional explicit number of blocks at each of the nine stages. Exactly nine values are required when supplied. |
| `normalization` | string | Normalization layer: `group`, `layer`, `instance`, `batch`, or `none`. |
| `dropout` | float | Dropout probability in the principal block path. |
| `deep_supervision` | boolean | Adds auxiliary segmentation heads at intermediate decoder/bottleneck resolutions during training. |
| `residual_blocks` | boolean | Enables residual shortcuts inside regular NeuroTS blocks. |
| `residual_resampling` | boolean | Enables projected residual shortcuts when spatial resolution or channel count changes. |
| `checkpoint_style` | string or `null` | Activation checkpointing policy. `outside_block` checkpoints complete blocks; `null` disables it. |
| `use_global_response_norm` | boolean | Enables Global Response Normalization after expanded pointwise features and in detail-fusion paths. |
| `use_selective_branches` | boolean | Enables adaptive high-level residual branches in deeper stages. |
| `branch_dropout_prob` | float | Training-time probability of dropping selected correction branches. |
| `branch_selection_topk` | integer | Maximum number of high-level branches retained by sparse branch selection. |
| `branch_temperature_start` | float | Initial soft branch-selection temperature. |
| `branch_temperature_end` | float | Final branch-selection temperature after scheduling through training. |
| `branch_residual_gain_init` | float | Initial gain of the weighted branch correction, keeping early optimization close to the main path. |
| `use_raw_detail_stream` | boolean | Enables the shallow raw-detail stream derived directly from the normalized MRI input. |
| `raw_detail_channels` | integer | Internal width of the raw-detail feature extractor. |
| `raw_detail_injection_init` | float | Initial learnable gain for gated detail injection into the first encoder stages. |

### Capacity Presets

The nine entries are ordered as four encoder stages, bottleneck, and four decoder
stages.

| `model_id` | Expansion ratios | Block counts |
|---|---|---|
| `S` | `[2, 2, 2, 2, 2, 2, 2, 2, 2]` | `[2, 2, 2, 2, 2, 2, 2, 2, 2]` |
| `B` | `[2, 3, 4, 4, 4, 4, 4, 3, 2]` | `[2, 2, 2, 2, 2, 2, 2, 2, 2]` |
| `M` | `[2, 3, 4, 4, 4, 4, 4, 3, 2]` | `[3, 4, 4, 4, 4, 4, 4, 4, 3]` |
| `L` | `[3, 4, 8, 8, 8, 8, 8, 4, 3]` | `[3, 4, 8, 8, 8, 8, 8, 4, 3]` |

Setting `expansion_ratios` or `block_counts` explicitly overrides that part of
the selected preset. `base_channels` remains independent of `model_id`; the
public `M` configuration uses `base_channels: 32`.

The anti-aliased multi-path downsampling mechanism is part of the architecture,
not a separate YAML mode. It combines the learned strided path with average- and
max-pooled detail paths, projects their concatenation, normalizes the correction,
and applies it through a learnable channel-wise gate. Selective 5x5x5 and global
context corrections are restricted to deeper stages; the 3x3x3 main path remains
present throughout the network.

## `loss`

The public configuration uses `name: region_aware`. For full-resolution logits,
the objective is

```text
L = w_ce * L_weighted_CE
  + w_label * L_weighted_label_Dice
  + w_region * L_region_Tversky_Dice
  + w_absent * L_absent_rare_FP
```

Region definitions are fixed as follows:

| Region | Probability/target union |
|---|---|
| `ET` | Label `1` |
| `NET` | Label `2` |
| `CC` | Label `3` |
| `ED` | Label `4` |
| `TC` | Labels `1 + 2 + 3` |
| `WT` | Labels `1 + 2 + 3 + 4` |

| Field | Type | Description |
|---|---|---|
| `name` | string | `region_aware` selects the public composite objective. `dice_ce` selects the simpler compatibility objective. |
| `include_background` | boolean | Includes background in the Dice part of `dice_ce`. Region-aware label Dice uses foreground classes explicitly. |
| `batch_dice` | boolean | Aggregates Dice statistics over the batch for `dice_ce`. The region-aware rare-label terms remain per-sample. |
| `dice_weight` | float | Dice multiplier for `dice_ce`; retained for simple-objective configurations. |
| `ce_weight` | float | cross-entropy multiplier for `dice_ce`; retained for simple-objective configurations. |
| `label_smoothing` | float | Cross-entropy label-smoothing factor. |
| `main_ce_weight` | float | Multiplier of weighted multiclass cross-entropy in `region_aware`. |
| `main_label_dice_weight` | float | Multiplier of weighted per-label foreground Dice in `region_aware`. |
| `main_region_weight` | float | Multiplier of the evaluated-region Tversky/Dice objective. |
| `absent_rare_fp_weight` | float | Multiplier of the absent rare-class false-positive penalty. Set to `0` to disable it. |
| `ce_class_weights` | five floats | Cross-entropy weights in `[background, ET, NET, CC, ED]` order. Exactly five values are required. |
| `label_dice_class_weights` | four floats | Label-Dice weights in `[ET, NET, CC, ED]` order. Exactly four values are required. |
| `region_loss_weights` | mapping | Relative weights for `ED`, `CC`, `ET`, `NET`, `TC`, and `WT` in the region objective. |
| `rare_tversky_alpha_fp` | float | Default false-positive coefficient for rare-region Tversky terms. |
| `rare_tversky_beta_fn` | float | Default false-negative coefficient for rare-region Tversky terms. |
| `et_tversky_alpha_fp` | float or `null` | Optional ET-specific false-positive coefficient; falls back to the shared value when `null`. |
| `et_tversky_beta_fn` | float or `null` | Optional ET-specific false-negative coefficient. |
| `cc_tversky_alpha_fp` | float or `null` | Optional CC-specific false-positive coefficient. |
| `cc_tversky_beta_fn` | float or `null` | Optional CC-specific false-negative coefficient. |
| `ed_tversky_alpha_fp_main` | float or `null` | Optional ED-specific false-positive coefficient for the main multiclass objective. |
| `ed_tversky_beta_fn_main` | float or `null` | Optional ED-specific false-negative coefficient for the main multiclass objective. |
| `absent_rare_fp_classes` | list of integers | Labels penalized when absent from an individual ground-truth patch. Public order/IDs are `[1, 3, 4]` for ET, CC, and ED. |
| `absent_rare_fp_power` | float | Exponent applied to absent-class probabilities before spatial averaging. |
| `deep_supervision_full_rare_outputs` | integer | Number of highest-resolution auxiliary outputs that receive the full ET/CC/ED rare-region terms. Coarser outputs avoid targets that may disappear after downsampling. |

For every configured class absent from a sample, the absent-class term computes
the spatial mean of `p_c ** absent_rare_fp_power`. It then averages all valid
sample/class terms. The term is differentiable, is applied to the main
full-resolution output, and is zero when no configured class is absent.

Deep-supervision losses use decreasing auxiliary weights and are normalized by
their total weight. Only the requested number of finest auxiliary outputs receive
all rare-region terms; this avoids forcing tiny ET, CC, or ED components into
resolutions where they no longer exist.

## `train`

| Field | Type | Description |
|---|---|---|
| `epochs` | integer | Maximum number of epochs. |
| `iterations_per_epoch` | integer | Number of optimizer steps (training batches) per epoch. |
| `val_iterations_per_epoch` | integer | Number of patch-validation batches evaluated at each validation epoch. |
| `optimizer` | string | Optimizer name. The public recipe uses `adamw`. |
| `learning_rate` | float | Peak/base learning rate reached after warmup. |
| `adam_eps` | float | AdamW numerical-stability epsilon. |
| `weight_decay` | float | AdamW decoupled weight-decay coefficient. |
| `grad_clip_norm` | float or `null` | Maximum global gradient norm. `null` disables clipping. |
| `mixed_precision` | boolean | Enables automatic mixed precision on supported accelerators. |
| `amp_dtype` | string | AMP compute dtype, normally `bfloat16` or `float16`. Cached inputs remain in `image_dtype`. |
| `scheduler` | string | Learning-rate schedule after warmup. Supported public choices are `cosine` and `poly`. |
| `cosine_eta_min` | float | Final learning rate of the cosine schedule. |
| `warmup_epochs` | integer | Number of epochs used for linear warmup from a small learning rate to `learning_rate`. |
| `early_stopping_patience` | integer | Number of validation opportunities without checkpoint improvement before stopping. A value equal to `epochs` effectively disables early stopping. |
| `seed` | integer | Seed used for split generation, sampling, augmentation, and model initialization where supported. |
| `validation_interval` | integer | Epoch interval for patch-level validation and ordinary checkpoint updates. |
| `compute_validation_detailed_metrics` | boolean | Computes class/region Dice, IoU/Jaccard, precision, recall, accuracy, NSD, and HD95 diagnostics during patch validation. |
| `validation_detailed_metrics_interval` | integer | Epoch interval for appending detailed validation metrics to the compact CSV and JSON histories. |
| `enable_rare_present_full_evaluation` | boolean | Enables full-volume sliding-window evaluation on the labelled validation split. |
| `rare_present_full_eval_interval` | integer | Epoch interval between rare-present full-volume evaluations. |
| `rare_present_full_eval_start_epoch` | integer | First epoch eligible for rare-present full-volume evaluation. The final epoch is also evaluated when enabled. |
| `rare_present_full_eval_use_tta` | boolean | Enables mirroring TTA for full-volume checkpoint evaluation. Disabling it reduces validation time. |
| `rare_present_full_eval_save_csv` | boolean | Saves per-case and aggregate CSV reports for each full evaluation. |
| `rare_present_full_eval_postprocess_et` | boolean | Compatibility switch retained in the schema. The current evaluator always reports its predefined safe ET variant; keep this `true` for the public recipe. |
| `disable_ed_for_safe_eval` | boolean | Chooses the ED-disabled diagnostic variant for safe-score checkpoint selection when `true`. It does not alter raw metrics. The public recipe keeps this `false`. |
| `branch_usage_regularization_weight` | float | Weight of the branch-load balancing regularizer. `0` disables it. |
| `branch_usage_regularization_fraction` | float | Fraction of total training during which branch-load regularization is active. |

The rare-present evaluator performs full-case inference and separates
ground-truth-present cases, predicted-present cases, overlapping TP cases, FP
cases, and FN cases. Empty/empty cases are diagnostic only and do not inflate the
checkpoint score. Each evaluation can save:

- `full_eval_epoch_XXXX_raw.csv`: unmodified model output;
- `full_eval_epoch_XXXX_safe_postproc.csv`: conservative ET postprocessing;
- `full_eval_epoch_XXXX_safe_et_ed_disabled.csv`: the same ET rule with ED removed,
  retained only as a diagnostic comparison.

Training saves `best_voxel_dice.pt` for patch-level mean foreground Dice and
`best_rare_present_score.pt` for the full-volume rare-present score. The latter
emphasizes present-only ET/CC/ED performance and case-level detection while
penalizing rare-class false-positive cases.

## `augmentation`

Augmentations are applied to training patches only.

| Field | Type | Description |
|---|---|---|
| `enabled` | boolean | Master switch for training augmentation. |
| `rotation_p` | float | Probability of random 3D rotation. |
| `rotation_degrees` | float | Maximum absolute rotation angle in degrees. |
| `scaling_p` | float | Probability of random isotropic/anisotropic scale augmentation. |
| `scale_range` | two floats | Minimum and maximum sampled scale. |
| `elastic_enabled` | boolean | Reserved compatibility field. Elastic deformation is not enabled in the public augmentation pipeline; keep this `false`. |
| `gaussian_noise_p` | float | Probability of additive Gaussian noise. |
| `gaussian_noise_variance` | two floats | Range from which noise variance is sampled. |
| `gaussian_blur_p` | float | Probability of Gaussian blur. |
| `gaussian_blur_sigma` | two floats | Gaussian sigma range. |
| `gaussian_blur_channel_p` | float | Conditional probability of blurring each modality after blur is selected. |
| `brightness_p` | float | Probability of multiplicative brightness augmentation. |
| `brightness_multiplier` | two floats | Range of brightness multipliers. |
| `contrast_p` | float | Probability of contrast augmentation. |
| `contrast_range` | two floats | Range of contrast factors. |
| `lowres_p` | float | Probability of low-resolution simulation through downsampling and upsampling. |
| `lowres_scale` | two floats | Range of simulated resolution scales. |
| `lowres_channel_p` | float | Conditional probability of low-resolution simulation per modality. |
| `gamma_inverted_p` | float | Probability of inverted gamma augmentation. |
| `gamma_normal_p` | float | Probability of standard gamma augmentation. |
| `gamma_range` | two floats | Range of sampled gamma values. |
| `gamma_retain_stats` | boolean | Restores the pre-augmentation mean and standard deviation after gamma transformation. |
| `mirror_axes` | list of integers | Spatial axes eligible for random mirroring. `[0, 1, 2]` enables all three. |
| `mirror_axis_p` | float | Independent probability of mirroring each eligible axis. |

Probabilities ending in `_p` are per-patch unless described as per-channel.
Geometric transforms are applied consistently to images and labels; intensity
transforms affect images only.

## `inference`

The `inference` section controls full-volume sliding-window prediction, TTA,
probability export, metric utilities, and optional probability-based fusion.

| Field | Type | Description |
|---|---|---|
| `patch_size` | three integers | Sliding-window patch shape in `[D, H, W]`. It should normally match the training patch size. |
| `sliding_window_overlap` | float | Fractional overlap between neighbouring windows. Must be in `[0, 1)`. |
| `window_batch_size` | integer | Number of sliding windows evaluated in one forward pass. Reduce this first if inference exceeds GPU memory. |
| `use_gaussian_weighting` | boolean | Blends overlapping windows with a centre-weighted Gaussian importance map. |
| `gaussian_sigma_scale` | float | Gaussian sigma relative to patch dimensions. |
| `mixed_precision` | boolean | Enables inference autocast. |
| `amp_dtype` | string | Inference autocast dtype, normally `bfloat16` or `float16`. |
| `score_dtype` | string | Dtype of accumulated full-volume score maps. `float32` is recommended for stable blending. |
| `tta_mirroring` | boolean | Averages predictions over mirror test-time augmentation. |
| `tta_axes` | list of integers | Axes used to construct mirror combinations. Three axes produce eight total variants, including the unflipped input. |
| `save_fold_predictions` | boolean | Saves each checkpoint's prediction in addition to the ensemble. In the current implementation this also retains per-fold probability NPZ files for reproducible fusion analysis. |
| `save_probabilities` | boolean | Saves compressed ensemble probability maps even when per-fold outputs are disabled. |
| `lesion_dilation_iterations` | integer | Connected-component matching dilation used by lesion-wise evaluation utilities; it does not change raw predictions. |
| `lesion_volume_threshold_mm3` | float | Small-lesion physical-volume threshold used by lesion-wise metric utilities. |
| `surface_tolerance_mm` | float | Surface-distance tolerance, in millimetres, used for NSD. |
| `surface_distance_max_points` | integer | Maximum sampled surface points used to bound surface-metric memory/time. |
| `surface_distance_crop_margin` | integer | Voxel margin around the region union before surface-distance calculation. |
| `use_probability_fusion` | boolean | Enables built-in rare-class thresholding, WT gating, priority fusion, and component filtering instead of plain biased argmax. |
| `logit_biases` | five floats | Additive biases in `[background, ET, NET, CC, ED]` order, applied before argmax/fusion. |
| `wt_gate_threshold` | float | Minimum `pWT` for the probability-fusion tumour gate. Inactive when `use_probability_fusion: false`. |
| `wt_gate_dilation` | integer | Dilation radius/iterations applied to the hard tumour gate during probability fusion. |
| `threshold_et` | float | ET probability threshold used by built-in probability fusion. |
| `threshold_cc` | float | CC probability threshold used by built-in probability fusion. |
| `threshold_ed` | float | ED probability threshold used by built-in probability fusion. |
| `min_component_voxels_et` | integer | Minimum retained ET connected-component size in voxels. |
| `min_component_voxels_cc` | integer | Minimum retained CC connected-component size in voxels. |
| `min_component_voxels_ed` | integer | Minimum retained ED connected-component size in voxels. |

With `use_probability_fusion: false`, inference applies `logit_biases` and uses
argmax; the WT gate, class thresholds, and minimum component sizes are inactive
in that raw path. The dedicated `scripts/predict_postprocessed.py` command uses
its explicit CLI arguments for tuned ET/CC component postprocessing, so its
settings should be recorded alongside the resulting experiment.

Predictions are generated on the foreground-cropped volume, pasted into the
original shape, and written with source NIfTI affine, spacing, orientation,
qform, and sform metadata. Ensemble probabilities are arithmetic means of the
selected checkpoints' probabilities.

## `output`

| Field | Type | Description |
|---|---|---|
| `output_root` | path | Parent directory for all experiment outputs. |
| `experiment_name` | string | Run-directory name under `output_root`; use a unique value for each experiment. |
| `save_plots` | boolean | Saves training curves and other compact run plots. |
| `create_ensemble_dir` | boolean | In fold mode, creates the consolidated `folds_all` checkpoint/manifest directory after cross-validation. |

The resulting run path is generally:

```text
<output_root>/<experiment_name>/
```

Single-split outputs are written directly inside that run directory. Fold-mode
outputs additionally use `fold_01`, `fold_02`, and corresponding fold
subdirectories.

## Command-Line Overrides

The training entry points support a small set of deliberate runtime overrides,
including epochs, iterations per epoch, validation iterations, batch size, patch
size, base channels, number of workers, and disabling augmentation or AMP.
`train_fold.py` also accepts a fold index. The YAML file remains the complete run
record, so prefer making persistent experiment changes in a copied YAML file and
giving it a unique `output.experiment_name`.

Inspect supported overrides directly:

```bash
python scripts/train_crossval.py --help
python scripts/train_fold.py --help
python scripts/predict_validation.py --help
python scripts/predict_postprocessed.py --help
```

## Cache and Reproducibility Notes

Rebuild the preprocessing cache after changing fields that alter cached content,
especially `train_roots`, `modalities`, `crop_margin`, `image_dtype`, or
`intensity_clip_percentiles`. Either set `overwrite_cache: true` temporarily or
run:

```bash
python scripts/preprocess_brats26.py \
  --config configs/neurots_brats26_peds_5fold.yaml \
  --overwrite
```

Changes limited to model, loss, optimizer, augmentation, inference, or output
settings do not require preprocessing again. For reproducibility, archive the
resolved YAML, split/fold JSON, checkpoint filename, prediction command, and any
postprocessing CLI arguments with each reported result.
