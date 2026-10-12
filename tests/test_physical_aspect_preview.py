import pytest
from transform_pipeline.stitching.preview import physical_preview_size


def test_equal_spacing_preserves_array_dimensions() -> None:
    assert physical_preview_size((546, 542), 1.0, 1.0) == (542, 546)


def test_anisotropic_spacing_corrects_display_aspect() -> None:
    width, height = physical_preview_size(
        (546, 542),
        x_spacing=0.025,
        y_spacing=0.0183333333333333,
    )
    assert (width, height) == (739, 546)
    assert width / height == pytest.approx(
        (542 * 0.025) / (546 * 0.0183333333333333),
        rel=0.002,
    )


def test_nonpositive_spacing_is_rejected() -> None:
    with pytest.raises(ValueError):
        physical_preview_size((100, 100), 0.0, 1.0)
