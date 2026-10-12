from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Iterable


GENERATED_DIRECTORY_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "prototype_output",
    "stitched_output",
}

BUILD_DIRECTORY_NAMES = {
    "build",
    "dist",
}

GENERATED_SUFFIXES = {
    ".pyc",
    ".pyo",
}

LARGE_DATA_SUFFIXES = {
    ".bin",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".tif",
    ".tiff",
    ".zarr",
}

DELIVERY_NAME_FRAGMENTS = (
    "_patch",
    "_hotfix",
)


@dataclass(frozen=True)
class AuditEntry:
    path: str
    category: str
    tracked: bool
    size_bytes: int
    reason: str


def _run_git(root: Path, *arguments: str) -> tuple[int, str]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout


def _tracked_paths(root: Path) -> set[str]:
    code, output = _run_git(root, "ls-files", "-z")
    if code != 0:
        return set()
    return {value for value in output.split("\0") if value}


def _git_status(root: Path) -> list[str]:
    code, output = _run_git(root, "status", "--short", "--untracked-files=all")
    return output.splitlines() if code == 0 else []


def _path_size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def _is_tracked(relative: Path, tracked: set[str]) -> bool:
    normalized = relative.as_posix()
    if normalized in tracked:
        return True
    prefix = normalized.rstrip("/") + "/"
    return any(item.startswith(prefix) for item in tracked)


def _classify(path: Path, relative: Path) -> tuple[str, str] | None:
    parts = set(relative.parts)
    name_lower = path.name.lower()

    if any(part in GENERATED_DIRECTORY_NAMES for part in relative.parts):
        return "generated", "generated output or cache directory"
    if any(part.endswith(".egg-info") for part in relative.parts):
        return "build_metadata", "generated Python package metadata"
    if any(part in BUILD_DIRECTORY_NAMES for part in relative.parts):
        return "build_artifact", "generated build directory"
    if path.suffix.lower() in GENERATED_SUFFIXES:
        return "generated", "compiled Python cache"
    if path.suffix.lower() in LARGE_DATA_SUFFIXES:
        return "scientific_data", "large scientific or analytical data"
    if any(fragment in name_lower for fragment in DELIVERY_NAME_FRAGMENTS):
        if path.suffix.lower() == ".zip" or path.is_dir():
            return "delivery_artifact", "extracted or archived patch delivery"
    if name_lower in {"thumbs.db", ".ds_store", "desktop.ini"}:
        return "operating_system", "operating-system metadata"
    if ".idea" in parts:
        return "ide_metadata", "PyCharm project metadata"
    return None


def audit_repository(root: Path) -> dict:
    root = Path(root).resolve()
    if not (root / ".git").exists():
        raise ValueError(f"Not a Git repository root: {root}")

    tracked = _tracked_paths(root)
    entries: list[AuditEntry] = []
    visited_directories: set[Path] = set()

    candidates = sorted(root.rglob("*"), key=lambda value: value.as_posix().lower())
    for path in candidates:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] == ".git":
            continue
        if any(parent in visited_directories for parent in path.parents):
            continue

        classification = _classify(path, relative)
        if classification is None:
            continue
        category, reason = classification
        entries.append(
            AuditEntry(
                path=relative.as_posix(),
                category=category,
                tracked=_is_tracked(relative, tracked),
                size_bytes=_path_size(path),
                reason=reason,
            )
        )
        if path.is_dir():
            visited_directories.add(path)

    tracked_candidates = [entry for entry in entries if entry.tracked]
    untracked_candidates = [entry for entry in entries if not entry.tracked]
    return {
        "repository_root": str(root),
        "git_status": _git_status(root),
        "summary": {
            "candidate_count": len(entries),
            "tracked_candidate_count": len(tracked_candidates),
            "untracked_candidate_count": len(untracked_candidates),
            "candidate_size_bytes": sum(entry.size_bytes for entry in entries),
        },
        "candidates": [asdict(entry) for entry in entries],
        "policy": {
            "automatic_deletion_performed": False,
            "tracked_files_require_review": True,
            "scientific_data_requires_review": True,
        },
    }


def write_audit_report(root: Path, output: Path) -> Path:
    report = audit_repository(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output
