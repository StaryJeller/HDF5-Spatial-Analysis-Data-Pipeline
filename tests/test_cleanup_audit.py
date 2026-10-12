from pathlib import Path
import subprocess

from transform_pipeline.maintenance import audit_repository


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_audit_classifies_generated_and_delivery_artifacts(tmp_path: Path) -> None:
    git(tmp_path, "init")
    (tmp_path / "transform_pipeline").mkdir()
    source = tmp_path / "transform_pipeline" / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    git(tmp_path, "add", "transform_pipeline/module.py")

    cache = tmp_path / "transform_pipeline" / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"cache")
    delivery = tmp_path / "example_hotfix.zip"
    delivery.write_bytes(b"zip")

    report = audit_repository(tmp_path)
    categories = {entry["path"]: entry["category"] for entry in report["candidates"]}
    assert categories["transform_pipeline/__pycache__"] == "generated"
    assert categories["example_hotfix.zip"] == "delivery_artifact"
    assert report["policy"]["automatic_deletion_performed"] is False


def test_audit_marks_tracked_candidate(tmp_path: Path) -> None:
    git(tmp_path, "init")
    output = tmp_path / "prototype_output"
    output.mkdir()
    file_path = output / "result.npy"
    file_path.write_bytes(b"data")
    git(tmp_path, "add", "-f", "prototype_output/result.npy")

    report = audit_repository(tmp_path)
    candidate = next(item for item in report["candidates"] if item["path"] == "prototype_output")
    assert candidate["tracked"] is True
