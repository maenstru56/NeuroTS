from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from math import radians
from typing import Any, Iterator

import numpy as np


@contextmanager
def _temporary_numpy_seed(seed: int) -> Iterator[None]:
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        yield
    finally:
        np.random.set_state(state)


def _pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None or len(value) != 2:
        return default
    return float(value[0]), float(value[1])


@lru_cache(maxsize=16)
def _spatial_transform(
    patch_size: tuple[int, int, int],
    rotation_p: float,
    rotation_radians: float,
    scaling_p: float,
    scale_range: tuple[float, float],
):
    try:
        from batchgenerators.transforms.abstract_transforms import Compose
        from batchgenerators.transforms.spatial_transforms import SpatialTransform
        from batchgenerators.transforms.utility_transforms import RemoveLabelTransform
    except ImportError as error:
        raise ImportError("batchgenerators is required for NeuroTS spatial augmentation.") from error
    transform = SpatialTransform(
        patch_size,
        patch_center_dist_from_border=None,
        do_elastic_deform=False,
        alpha=(0.0, 0.0),
        sigma=(0.0, 0.0),
        do_rotation=True,
        angle_x=(-rotation_radians, rotation_radians),
        angle_y=(-rotation_radians, rotation_radians),
        angle_z=(-rotation_radians, rotation_radians),
        do_scale=True,
        scale=scale_range,
        border_mode_data="constant",
        border_cval_data=0,
        order_data=3,
        border_mode_seg="constant",
        border_cval_seg=-1,
        order_seg=0,
        random_crop=False,
        p_el_per_sample=0.0,
        p_scale_per_sample=float(scaling_p),
        p_rot_per_sample=float(rotation_p),
        independent_scale_for_each_axis=False,
        p_rot_per_axis=1.0,
        p_independent_scale_per_axis=1.0,
    )
    return Compose([transform, RemoveLabelTransform(-1, 0)])


def _apply_spatial(image: np.ndarray, label: np.ndarray, rng: np.random.Generator, cfg: dict[str, Any], patch_size: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    transform = _spatial_transform(
        patch_size,
        float(cfg.get("rotation_p", 0.2)),
        radians(float(cfg.get("rotation_degrees", 30.0))),
        float(cfg.get("scaling_p", 0.2)),
        _pair(cfg.get("scale_range"), (0.7, 1.4)),
    )
    seed = int(rng.integers(0, 2**31 - 1))
    data = image.astype(np.float32, copy=False)[None]
    seg = label.astype(np.float32, copy=False)[None, None]
    with _temporary_numpy_seed(seed):
        augmented = transform(data=data, seg=seg)
    return np.asarray(augmented["data"][0], dtype=np.float32), np.asarray(augmented["seg"][0, 0], dtype=np.int64)


def _gamma_channel(channel: np.ndarray, gamma: float, invert: bool, retain_stats: bool) -> np.ndarray:
    source = channel.astype(np.float32, copy=False)
    old_mean = float(source.mean())
    old_std = float(source.std())
    lo = float(source.min())
    hi = float(source.max())
    if hi <= lo:
        return source
    normalized = (source - lo) / (hi - lo)
    if invert:
        normalized = 1.0 - normalized
    adjusted = np.power(np.clip(normalized, 0.0, 1.0), float(gamma))
    if invert:
        adjusted = 1.0 - adjusted
    adjusted = adjusted * (hi - lo) + lo
    if retain_stats:
        new_std = float(adjusted.std())
        if new_std > 1e-6:
            adjusted = (adjusted - float(adjusted.mean())) / new_std * max(old_std, 1e-6) + old_mean
    return adjusted.astype(np.float32, copy=False)


def _resize_channel(channel: np.ndarray, scale: float) -> np.ndarray:
    from scipy import ndimage

    if scale >= 0.999:
        return channel
    shape = np.asarray(channel.shape, dtype=np.int64)
    low_shape = np.maximum(1, np.round(shape.astype(np.float64) * float(scale)).astype(np.int64))
    down = ndimage.zoom(channel, zoom=tuple(low_shape / shape), order=1)
    up = ndimage.zoom(down, zoom=tuple(shape / np.asarray(down.shape)), order=1)
    slices = tuple(slice(0, int(size)) for size in shape)
    if up.shape != tuple(shape):
        padded = np.zeros(tuple(shape), dtype=np.float32)
        copy_slices = tuple(slice(0, min(int(shape[idx]), int(up.shape[idx]))) for idx in range(3))
        padded[copy_slices] = up[copy_slices]
        up = padded
    return np.asarray(up[slices], dtype=np.float32)


def apply_neurots_augmentation(
    image: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    cfg: dict[str, Any],
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image, label = _apply_spatial(image, label, rng, cfg, patch_size)
    if rng.random() < float(cfg.get("gaussian_noise_p", 0.1)):
        low, high = _pair(cfg.get("gaussian_noise_variance"), (0.0, 0.1))
        variance = rng.uniform(low, high)
        image = image + rng.normal(0.0, np.sqrt(max(variance, 0.0)), size=image.shape).astype(np.float32)
    if rng.random() < float(cfg.get("gaussian_blur_p", 0.2)):
        from scipy import ndimage

        sigma_low, sigma_high = _pair(cfg.get("gaussian_blur_sigma"), (0.5, 1.0))
        channel_p = float(cfg.get("gaussian_blur_channel_p", 0.5))
        for channel in range(image.shape[0]):
            if rng.random() < channel_p:
                sigma = float(rng.uniform(sigma_low, sigma_high))
                image[channel] = ndimage.gaussian_filter(image[channel], sigma=sigma)
    if rng.random() < float(cfg.get("brightness_p", 0.15)):
        low, high = _pair(cfg.get("brightness_multiplier"), (0.75, 1.25))
        image = image * float(rng.uniform(low, high))
    if rng.random() < float(cfg.get("contrast_p", 0.15)):
        low, high = _pair(cfg.get("contrast_range"), (0.75, 1.25))
        factor = float(rng.uniform(low, high))
        means = image.mean(axis=(1, 2, 3), keepdims=True)
        image = (image - means) * factor + means
    if rng.random() < float(cfg.get("lowres_p", 0.25)):
        low, high = _pair(cfg.get("lowres_scale"), (0.5, 1.0))
        channel_p = float(cfg.get("lowres_channel_p", 0.5))
        for channel in range(image.shape[0]):
            if rng.random() < channel_p:
                image[channel] = _resize_channel(image[channel], float(rng.uniform(low, high)))
    if rng.random() < float(cfg.get("gamma_inverted_p", 0.1)):
        gamma_low, gamma_high = _pair(cfg.get("gamma_range"), (0.7, 1.5))
        for channel in range(image.shape[0]):
            image[channel] = _gamma_channel(image[channel], float(rng.uniform(gamma_low, gamma_high)), True, bool(cfg.get("gamma_retain_stats", True)))
    if rng.random() < float(cfg.get("gamma_normal_p", 0.3)):
        gamma_low, gamma_high = _pair(cfg.get("gamma_range"), (0.7, 1.5))
        for channel in range(image.shape[0]):
            image[channel] = _gamma_channel(image[channel], float(rng.uniform(gamma_low, gamma_high)), False, bool(cfg.get("gamma_retain_stats", True)))
    mirror_axes = [int(axis) for axis in cfg.get("mirror_axes", [0, 1, 2])]
    mirror_p = float(cfg.get("mirror_axis_p", 0.5))
    for axis in mirror_axes:
        if rng.random() < mirror_p:
            image = np.flip(image, axis=axis + 1)
            label = np.flip(label, axis=axis)
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(label, dtype=np.int64)
