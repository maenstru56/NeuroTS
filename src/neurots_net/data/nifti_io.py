from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NiftiArray:
    data_zyx: np.ndarray
    spacing_zyx: tuple[float, float, float]
    shape_xyz: tuple[int, int, int]


def require_nibabel():
    try:
        import nibabel as nib
    except ImportError as error:
        raise ImportError("nibabel is required for BraTS NIfTI I/O.") from error
    return nib


def load_nifti_zyx(path: str | Path, dtype: np.dtype | type = np.float32) -> NiftiArray:
    nib = require_nibabel()
    image = nib.load(str(path))
    data_xyz = np.asarray(image.dataobj)
    if dtype is not None:
        data_xyz = data_xyz.astype(dtype, copy=False)
    if data_xyz.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI image at {path}, got shape {data_xyz.shape}.")
    zooms_xyz = tuple(float(item) for item in image.header.get_zooms()[:3])
    data_zyx = np.moveaxis(data_xyz, -1, 0)
    return NiftiArray(
        data_zyx=np.ascontiguousarray(data_zyx),
        spacing_zyx=(zooms_xyz[2], zooms_xyz[1], zooms_xyz[0]),
        shape_xyz=tuple(int(item) for item in data_xyz.shape),
    )


def save_segmentation_like_reference(
    prediction_zyx: np.ndarray,
    reference_path: str | Path,
    output_path: str | Path,
) -> None:
    nib = require_nibabel()
    reference = nib.load(str(reference_path))
    prediction_xyz = np.moveaxis(np.asarray(prediction_zyx, dtype=np.uint8), 0, -1)
    if tuple(prediction_xyz.shape) != tuple(reference.shape[:3]):
        raise ValueError(
            f"Prediction shape {prediction_xyz.shape} does not match reference shape {tuple(reference.shape[:3])}."
        )
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(prediction_xyz, affine=reference.affine, header=header)
    nib.save(image, str(output))
