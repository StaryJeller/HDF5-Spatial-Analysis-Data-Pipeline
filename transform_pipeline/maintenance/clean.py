from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess


PROTECTED_SUFFIXES = {".h5", ".hdf5", ".bin"}
PROTECTED_NAMES = {"mcc_flyer_0.csv"}
WORKSPACE_DIRECTORY_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
DELIVERY_FRAGMENTS = ("_patch", "_hotfix")
GENERATED_DIRECTORY_NAMES = {"prototype_output", "stitched_output"}
ROOT_WORKSPACE_DIRECTORIES = {"md_output"}
ROOT_WORKSPACE_FILES = {"repository_cleanup_audit.json"}


@dataclass(frozen=True)
class CleanupItem:
    path: str
    category: str
    size_bytes: int


def _git_tracked(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Unable to inspect Git tracked files under {root}")
    return {item for item in result.stdout.split("\0") if item}


def _contains_tracked(relative: Path, tracked: set[str]) -> bool:
    normalized = relative.as_posix()
    if normalized in tracked:
        return True
    prefix = normalized.rstrip("/") + "/"
    return any(item.startswith(prefix) for item in tracked)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file() and not child.is_symlink()
    )


def _is_delivery(path: Path) -> bool:
    lowered = path.name.lower()
    return any(fragment in lowered for fragment in DELIVERY_FRAGMENTS)


def _is_protected(path: Path) -> bool:
    return path.suffix.lower() in PROTECTED_SUFFIXES or path.name in PROTECTED_NAMES


def build_cleanup_plan(root: Path, profile: str = "workspace") -> list[CleanupItem]:
    if profile not in {"workspace", "generated"}:
        raise ValueError("profile must be 'workspace' or 'generated'")
    root = Path(root).resolve()
    if not (root / ".git").is_dir():
        raise ValueError(f"Not a Git repository root: {root}")
    tracked = _git_tracked(root)
    selected: dict[str, CleanupItem] = {}

    def add(path: Path, category: str) -> None:
        relative = path.relative_to(root)
        if _is_protected(path):
            return
        if _contains_tracked(relative, tracked):
            return
        key = relative.as_posix()
        selected[key] = CleanupItem(key, category, _size(path))

    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_dir() and path.name in WORKSPACE_DIRECTORY_NAMES:
            add(path, "cache")
        elif path.parent == root and path.is_dir() and path.name in ROOT_WORKSPACE_DIRECTORIES:
            add(path, "temporary_output")
        elif path.parent == root and _is_delivery(path) and (path.is_dir() or path.suffix.lower() == ".zip"):
            add(path, "delivery_artifact")
        elif path.parent == root and path.is_file() and path.name in ROOT_WORKSPACE_FILES:
            add(path, "audit_report")
        elif path.is_dir() and path.name.startswith("stitch_") and path.parent.name == "stitched_output":
            add(path, "abandoned_workspace")
        elif profile == "generated" and path.is_dir() and path.name in GENERATED_DIRECTORY_NAMES:
            add(path, "generated_pipeline_output")

    # Keep only top-level selected ancestors to avoid duplicate deletion entries.
    ordered = sorted(selected.values(), key=lambda item: (item.path.count("/"), item.path))
    final: list[CleanupItem] = []
    for item in ordered:
        prefix = item.path.rstrip("/") + "/"
        if any(item.path.startswith(existing.path.rstrip("/") + "/") for existing in final):
            continue
        final.append(item)
    return final


def execute_cleanup(
    root: Path,
    profile: str = "workspace",
    apply: bool = False,
    report_path: Path | None = None,
) -> dict:
    root = Path(root).resolve()
    plan = build_cleanup_plan(root, profile)
    removed = []
    failures = []
    if apply:
        for item in plan:
            path = root / item.path
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                removed.append(item.path)
            except OSError as error:
                failures.append({"path": item.path, "error": str(error)})

    payload = {
        "repository_root": str(root),
        "profile": profile,
        "apply_requested": apply,
        "protected_suffixes": sorted(PROTECTED_SUFFIXES),
        "protected_names": sorted(PROTECTED_NAMES),
        "planned_size_bytes": sum(item.size_bytes for item in plan),
        "planned_items": [asdict(item) for item in plan],
        "removed_paths": removed,
        "failures": failures,
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
