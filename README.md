<h1 align="center">
NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI
</h1>

<p align="center">
  <strong>English</strong> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md">Français</a> |
  <a href="README.pt.md">Português</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.ro.md">Română</a>
</p>

This repository provides the training, evaluation, inference, and postprocessing code for **NeuroTS-Net**, a three-dimensional, multi-class semantic segmentation architecture designed for pediatric brain tumor segmentation in multimodal MRI.

The NeuroTS-Net architecture was developed by Darius Peteleaza. For questions about the paper or code, please contact [darius.peteleaza@ulbsibiu.ro](mailto:darius.peteleaza@ulbsibiu.ro?subject=[GitHub]NeuroTS-Net)

The code accompanies the paper: **NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI**

The paper is currently under review at MICCAI 2026 as part of the [BraTS Challenge](https://challenges.synapse.org/Challenges/DetailsPage/Overview?id=syn74274097).

- MICCAI/BraTS publication link: to be added.
- arXiv preprint link: to be added.

> [**Citation.**](#how-to-cite) Please [cite](#how-to-cite) our paper if you use the NeuroTS-Net code, architecture or material from the accompanying paper. This repository is licensed under the [Creative Commons Attribution 4.0 International License](LICENSE).

<p align="center">
  <a href="figures/NeuroTS-Net.png"><img src="figures/NeuroTS-Net.png" alt="NeuroTS-Net architecture" width="32%"></a>
  <a href="figures/NeuroTS_Block.png"><img src="figures/NeuroTS_Block.png" alt="NeuroTS block" width="32%"></a>
  <a href="figures/NeuroTS_Downsampling.png"><img src="figures/NeuroTS_Downsampling.png" alt="NeuroTS downsampling mechanism" width="32%"></a>
</p>
<p align="center"><em>From left to right: overview of the NeuroTS-Net architecture, NeuroTS block, and NeuroTS downsampling mechanism.</em></p>

## Requirements

- Tested operating systems: Windows 11 and Ubuntu 22.04 LTS (the code is designed to run on any operating system supported by Python 3.11, PyTorch, and the listed dependencies).
- Python 3.11.
- A CUDA-compatible GPU is strongly recommended for training and inference. CPU execution is also supported but is considerably slower.

For a standard installation, install the runtime requirements:

```bash
python -m pip install -r requirements.txt
```

For an editable development installation, including the test dependencies, use:

```bash
python -m pip install -e ".[dev]"
```

## Configuration

This project uses YAML configuration files to define data paths and splitting, model architecture, loss, training, augmentation, inference, and output settings. 

For a field-by-field description of every setting, see the [configuration reference](configs/CONFIGURATION.md).

## Dataset

The default configurations target BraTS-PED data with the following modalities and labels:

| Channel/label | Meaning |
|---|---|
| t1n | Native T1-weighted MRI |
| t1c | Contrast-enhanced T1-weighted MRI |
| t2w | T2-weighted MRI |
| t2f | T2-FLAIR MRI |
| 0 | Background |
| 1 | Enhancing tumor (ET) |
| 2 | Non-enhancing tumor (NET) |
| 3 | Cystic component (CC) |
| 4 | Edema (ED) |

The expected dataset layout is:

~~~text
data/
|-- BraTS26_PED_training/
|   |-- BraTS-PED-00001-000/
|   |   |-- BraTS-PED-00001-000-t1n.nii.gz
|   |   |-- BraTS-PED-00001-000-t1c.nii.gz
|   |   |-- BraTS-PED-00001-000-t2w.nii.gz
|   |   |-- BraTS-PED-00001-000-t2f.nii.gz
|   |   |-- BraTS-PED-00001-000-seg.nii.gz
|   |-- ...
|-- BraTS26_PED_training_Batch2_Release/
|   |-- ...
|-- BraTS26_PED_validation/
    |-- BraTS-PED-XXXXX-XXX/
    |   |-- BraTS-PED-XXXXX-XXX-t1n.nii.gz
    |   |-- BraTS-PED-XXXXX-XXX-t1c.nii.gz
    |   |-- BraTS-PED-XXXXX-XXX-t2w.nii.gz
    |   |-- BraTS-PED-XXXXX-XXX-t2f.nii.gz
    |-- ...
~~~

Subjects are deduplicated across training roots by case ID before splitting.

## Data Preprocessing

The preprocessing pipeline:

1. Reads the four NIfTI modalities without resampling;
2. Computes a foreground bounding box over all modalities;
3. Expands the box by the configured margin (32 voxels by default);
4. Clips each modality to foreground intensity percentiles;
5. Applies foreground-only z-score normalization;
6. Stores images as float32 arrays together with labels and spatial metadata.

Build the reusable cache:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

To rebuild existing cache entries:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --overwrite
~~~

Predictions are pasted back into the original image shape and saved with the source affine, spacing, orientation, qform, and sform metadata.

## Model Training

Both grouped cross-validation and one explicit training/validation split are supported by the same training entry point.

### Five-Fold Cross-Validation

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

The fold file is generated from cached training cases when it does not exist. To train only one configured fold:

~~~bash
python scripts/train_fold.py --config configs/neurots_brats26_peds_5fold.yaml --fold-index 0
~~~

### Single Training/Validation Split

Create a label-stratified, high-signal split:

~~~bash
python scripts/create_high_signal_split.py --config configs/neurots_brats26_peds_single_split.yaml --output cache/brats26_peds_task2_native_margin32/splits_single_seed42.json --split-name single_seed42 --seed 42 --val-size 40
~~~

Train on that split:

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_single_split.yaml
~~~

When data.split_mode is single, the split file is mandatory. Training fails with a clear error instead of silently falling back to folds.

## Sampling and Objective

The public configurations reproduce the rare-region-aware training recipe used for NeuroTS-Net:

- Component-balanced patch sampling for ET, NET, CC, and ED;
- ED-negative hard-negative patches centered on non-ED tumor tissue;
- Weighted cross-entropy and per-sample label Dice;
- Evaluated-region Tversky/Dice terms for ET, NET, CC, ED, TC, and WT;
- An absent-class probability penalty for ET, CC, and ED;
- Deep supervision, cosine learning-rate decay, and linear warmup.

Sampling request, success, fallback, class-presence, and target-voxel statistics are written to the training log and patch_sampling_latest.json.

## Checkpoints and Metrics

Each training directory can contain:

- best_voxel_dice.pt: best patch-validation mean region Dice;
- best_rare_present_score.pt: best full-volume rare-present score;

The rare-present evaluator reports GT-present, prediction-present, TP, FP, and FN cases separately. Empty/empty rare-region cases are diagnostic only and do not improve rare-present checkpoint selection.

Training outputs include compact CSV/JSON metric histories, learning curves, full-volume validation reports, logs, and resource snapshots. Branch-selection telemetry is intentionally not printed or exported.

Validate YAML files, including duplicate-key rejection:

~~~bash
python scripts/check_config_yaml.py configs/neurots_brats26_peds_5fold.yaml configs/neurots_brats26_peds_single_split.yaml
~~~

## Inference

Preprocess validation data before inference:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --skip-training
~~~

### Raw Checkpoint or Fold Ensemble

Discover and average all rare-present checkpoints in a run:

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_5fold.yaml --run-dir outputs/neurots_brats26_peds_5fold --candidate rare_present
~~~

Explicit checkpoints can be repeated for any ensemble:

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_single_split.yaml --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --candidate rare_present
~~~

### ET/CC Component Postprocessing

The tuned public postprocessing adjusts ET and CC probabilities/components while preserving main-model ED predictions:

~~~bash
python scripts/predict_postprocessed.py --config configs/neurots_brats26_peds_single_split.yaml --output outputs/neurots_postprocessed --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --cc-logit-bias 0.75 --cc-prob-thr 0.15 --cc-min-size 20 --cc-max-components 3 --et-logit-bias 0.25 --et-prob-thr 0.30 --et-min-size 20 --et-max-components 4 --protect-tc-wt --zip
~~~

Repeat --checkpoint to average multiple checkpoints. NIfTI files are placed in the predictions directory, and --zip creates a flat archive suitable for challenge upload.

## Outputs

Generated data are written below the configured roots:

~~~text
cache/       preprocessed arrays and split/fold definitions
outputs/     checkpoints, logs, metrics, predictions, and archives
~~~

These directories, raw datasets, checkpoints, and medical images are excluded by .gitignore.

## Tests

Run the fast test suite:

~~~bash
python -m pytest -q
~~~

Run the model forward/backward smoke test:

~~~bash
python scripts/smoke_test.py
~~~

<a id="how-to-cite"></a>

## Citation

Please cite our paper when using this repository:

### Plaintext

```

```

### BibTeX

```

```

