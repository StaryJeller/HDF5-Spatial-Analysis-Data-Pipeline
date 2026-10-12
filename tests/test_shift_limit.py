import numpy as np
import pytest

from transform_pipeline.core.models import ShiftMeasurement
from transform_pipeline.transforms.registration import (
    apply_subpixel_registration,
    shift_spectral_row,
)


def test_shift_just_over_one_pixel_is_allowed() -> None:
    row = np.arange(24, dtype=np.float32).reshape(12, 2)
    shifted, valid = shift_spectral_row(row, -1.0211924458103088)

    assert shifted.shape == row.shape
    assert valid.shape == (12,)
    assert np.count_nonzero(valid) < 12
    assert np.isfinite(shifted[valid]).all()


def test_shift_over_two_pixels_is_rejected() -> None:
    row = np.arange(24, dtype=np.float32).reshape(12, 2)
    with pytest.raises(ValueError, match="application limit"):
        shift_spectral_row(row, 2.01)


def test_registration_accepts_detector_profile_over_one_pixel() -> None:
    cube = np.arange(5 * 12 * 2, dtype=np.float32).reshape(5, 12, 2)
    shifts = [
        ShiftMeasurement(1, -1.0211924458103088, -1.0211924458103088, 0.9, 0.8, 0.1),
        ShiftMeasurement(3, 1.25, 1.25, 0.9, 0.8, 0.1),
    ]

    result = apply_subpixel_registration(cube, shifts)

    assert result.cube.shape == cube.shape
    assert np.array_equal(result.cube[0], cube[0])
    assert np.array_equal(result.cube[2], cube[2])
    assert np.array_equal(result.cube[4], cube[4])
    assert np.count_nonzero(~result.validity_mask[1]) > 0
    assert np.count_nonzero(~result.validity_mask[3]) > 0
