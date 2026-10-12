from pathlib import Path
import pytest
from transform_pipeline.quadrant_pipeline import discover_quadrant_directories
from transform_pipeline.quadrant_cli import default_four_quadrant_root

def create_quadrant(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / f"{name}.h5").write_bytes(b"h5")
    (directory / "sdd1_0.bin").write_bytes(b"sdd")
    return directory

def test_discover_quadrants_uses_immediate_processable_children(tmp_path: Path) -> None:
    first = create_quadrant(tmp_path, "150721_NCo_7_SW")
    second = create_quadrant(tmp_path, "152250_NCo_7_SE")
    (tmp_path / "stitched_output").mkdir()
    (tmp_path / "notes").mkdir()
    assert discover_quadrant_directories(tmp_path) == (first, second)

def test_discovery_requires_at_least_two_quadrants(tmp_path: Path) -> None:
    create_quadrant(tmp_path, "150721_NCo_7_SW")
    with pytest.raises(ValueError, match="at least two"):
        discover_quadrant_directories(tmp_path)

def test_default_four_quadrant_root_is_current() -> None:
    assert default_four_quadrant_root().name == "Testmodel_4_quadrant"
