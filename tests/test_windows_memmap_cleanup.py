from pathlib import Path

import numpy as np

from transform_pipeline.stitching.mosaic import _close_memmap


def test_close_memmap_releases_underlying_mapping(tmp_path: Path) -> None:
    path = tmp_path / "temporary.dat"
    array = np.memmap(path, mode="w+", dtype=np.float32, shape=(4, 4))
    array[:] = 3.0
    mapping = array._mmap

    _close_memmap(array)

    assert mapping.closed
    path.unlink()
    assert not path.exists()
