from pathlib import Path
import subprocess

from transform_pipeline.maintenance.clean import build_cleanup_plan, execute_cleanup


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def setup_repo(tmp_path: Path) -> None:
    git(tmp_path, "init")
    source = tmp_path / "source.py"
    source.write_text("value=1\n", encoding="utf-8")
    git(tmp_path, "add", "source.py")


def test_workspace_plan_preserves_raw_data_and_generated_outputs(tmp_path: Path) -> None:
    setup_repo(tmp_path)
    (tmp_path / "example_patch").mkdir()
    (tmp_path / "md_output").mkdir()
    (tmp_path / "prototype_output").mkdir()
    (tmp_path / "scan.h5").write_bytes(b"raw")
    (tmp_path / "sdd1_0.bin").write_bytes(b"raw")
    paths = {item.path for item in build_cleanup_plan(tmp_path, "workspace")}
    assert "example_patch" in paths
    assert "md_output" in paths
    assert "prototype_output" not in paths
    assert "scan.h5" not in paths
    assert "sdd1_0.bin" not in paths


def test_generated_profile_includes_outputs_but_not_raw_inputs(tmp_path: Path) -> None:
    setup_repo(tmp_path)
    output = tmp_path / "Playtest_model" / "tile" / "prototype_output"
    output.mkdir(parents=True)
    raw = tmp_path / "Playtest_model" / "tile" / "sdd1_0.bin"
    raw.write_bytes(b"raw")
    paths = {item.path for item in build_cleanup_plan(tmp_path, "generated")}
    assert "Playtest_model/tile/prototype_output" in paths
    assert "Playtest_model/tile/sdd1_0.bin" not in paths


def test_tracked_candidate_is_never_deleted(tmp_path: Path) -> None:
    setup_repo(tmp_path)
    tracked_cache = tmp_path / "__pycache__"
    tracked_cache.mkdir()
    tracked_file = tracked_cache / "important.pyc"
    tracked_file.write_bytes(b"tracked")
    git(tmp_path, "add", "-f", "__pycache__/important.pyc")
    result = execute_cleanup(tmp_path, "workspace", apply=True)
    assert tracked_file.exists()
    assert "__pycache__" not in result["removed_paths"]


def test_dry_run_deletes_nothing(tmp_path: Path) -> None:
    setup_repo(tmp_path)
    delivery = tmp_path / "example_hotfix"
    delivery.mkdir()
    execute_cleanup(tmp_path, "workspace", apply=False)
    assert delivery.exists()
