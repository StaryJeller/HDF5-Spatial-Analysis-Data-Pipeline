from pathlib import Path

import numpy as np
from PIL import Image

from transform_pipeline.io.writers.tiff import write_side_by_side_comparison


def test_side_by_side_comparison(tmp_path: Path) -> None:
    original = np.arange(48, dtype=np.float32).reshape(6, 8)
    corrected = original.copy()
    corrected[:, 1:] = original[:, :-1]
    corrected[:, 0] = np.nan
    validity_mask = np.isfinite(corrected)
    output_path = tmp_path / "comparison.png"

    low, high = write_side_by_side_comparison(
        path=output_path,
        original_image=original,
        corrected_image=corrected,
        corrected_validity_mask=validity_mask,
    )

    assert output_path.is_file()
    assert high > low
    with Image.open(output_path) as image:
        assert image.mode == "L"
        assert image.size == (18, 34)
