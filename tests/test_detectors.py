from pathlib import Path

import numpy as np

from transform_pipeline.io.readers.cls_sgm import detector_sort_key
from transform_pipeline.transforms.detectors import combine_registered_detectors


def test_detector_sort_key_uses_detector_number() -> None:
    paths = [
        Path("sdd10_0.bin"),
        Path("sdd2_0.bin"),
        Path("sdd1_0.bin"),
    ]

    assert [path.name for path in sorted(paths, key=detector_sort_key)] == [
        "sdd1_0.bin",
        "sdd2_0.bin",
        "sdd10_0.bin",
    ]


def test_sum_and_mean_fuse_already_registered_detectors() -> None:
    cubes = np.array(
        [
            [[[1, 2], [3, 4]]],
            [[[10, 20], [30, 40]]],
        ],
        dtype=np.float32,
    )
    masks = np.ones((2, 1, 2), dtype=bool)

    summed, summed_total, summed_mask = combine_registered_detectors(
        cubes,
        masks,
        "sum",
    )
    meaned, meaned_total, meaned_mask = combine_registered_detectors(
        cubes,
        masks,
        "mean",
    )

    assert np.array_equal(
        summed,
        np.array([[[11, 22], [33, 44]]], dtype=np.float32),
    )
    assert np.array_equal(
        summed_total,
        np.array([[33, 77]], dtype=np.float32),
    )
    assert np.array_equal(
        meaned,
        np.array([[[5.5, 11], [16.5, 22]]], dtype=np.float32),
    )
    assert np.array_equal(
        meaned_total,
        np.array([[16.5, 38.5]], dtype=np.float32),
    )
    assert np.array_equal(summed_mask, meaned_mask)


def test_fusion_uses_intersection_of_detector_masks() -> None:
    cubes = np.ones((2, 2, 3, 4), dtype=np.float32)
    masks = np.ones((2, 2, 3), dtype=bool)
    masks[0, 0, 0] = False
    masks[1, 1, 2] = False

    combined, total, combined_mask = combine_registered_detectors(
        cubes,
        masks,
        "sum",
    )

    assert not combined_mask[0, 0]
    assert not combined_mask[1, 2]
    assert np.isnan(combined[0, 0]).all()
    assert np.isnan(combined[1, 2]).all()
    assert np.isnan(total[0, 0])
    assert np.isnan(total[1, 2])


def test_none_mode_preserves_detector_only_output() -> None:
    cubes = np.ones((2, 2, 3, 4), dtype=np.float32)
    masks = np.ones((2, 2, 3), dtype=bool)

    assert combine_registered_detectors(
        cubes,
        masks,
        "none",
    ) == (None, None, None)
