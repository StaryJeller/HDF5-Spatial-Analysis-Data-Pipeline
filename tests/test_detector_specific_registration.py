import numpy as np
from transform_pipeline.core.models import MultiDetectorReconstructionResult, RasterGeometry
import transform_pipeline.transforms.detectors as detector_module


def test_each_detector_gets_its_own_measured_profile(monkeypatch):
    geometry = RasterGeometry(3, 4, np.arange(4), np.arange(3))
    cubes = np.zeros((2, 3, 4, 2), dtype=np.float32)
    totals = np.stack([
        np.full((3, 4), 1.0, dtype=np.float32),
        np.full((3, 4), 2.0, dtype=np.float32),
    ])
    reconstruction = MultiDetectorReconstructionResult(
        detector_ids=("sdd1_0", "sdd2_0"),
        source_files=(), cubes=cubes, total_counts=totals, geometry=geometry,
    )
    measured_inputs = []
    applied_profiles = []

    def fake_measure(signal):
        measured_inputs.append(signal.copy())
        return [float(signal[0, 0])]

    class FakeResult:
        def __init__(self, cube, shifts):
            self.cube = cube + shifts[0]
            self.total_counts = cube.sum(axis=2) + shifts[0]
            self.validity_mask = np.ones(cube.shape[:2], dtype=bool)

    def fake_apply(cube, shifts):
        applied_profiles.append(list(shifts))
        return FakeResult(cube, shifts)

    monkeypatch.setattr(detector_module, "measure_subpixel_shifts", fake_measure)
    monkeypatch.setattr(detector_module, "apply_subpixel_registration", fake_apply)
    result = detector_module.register_detectors(reconstruction, "none")

    assert len(measured_inputs) == 2
    assert np.array_equal(measured_inputs[0], totals[0])
    assert np.array_equal(measured_inputs[1], totals[1])
    assert applied_profiles == [[1.0], [2.0]]
    assert result.shifts_by_detector == {"sdd1_0": [1.0], "sdd2_0": [2.0]}
