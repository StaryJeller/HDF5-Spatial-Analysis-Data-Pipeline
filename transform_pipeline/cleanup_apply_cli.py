from __future__ import annotations

import argparse
from pathlib import Path

from transform_pipeline.maintenance.clean import execute_cleanup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely clean untracked repository artifacts. Dry-run is the default."
    )
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--profile", choices=("workspace", "generated"), default="workspace")
    parser.add_argument("--apply", action="store_true", help="Perform the planned deletions.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("repository_cleanup_plan.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = execute_cleanup(
        args.root,
        profile=args.profile,
        apply=args.apply,
        report_path=args.report,
    )
    action = "Removed" if args.apply else "Would remove"
    print(f"{action} {len(result['planned_items'])} paths")
    print(f"Planned bytes: {result['planned_size_bytes']}")
    print(f"Report: {args.report.resolve()}")
    if not args.apply:
        print("Dry-run only. No files were deleted.")
    if result["failures"]:
        print(f"Failures: {len(result['failures'])}")


if __name__ == "__main__":
    main()
