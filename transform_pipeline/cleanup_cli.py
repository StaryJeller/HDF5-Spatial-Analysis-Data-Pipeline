from __future__ import annotations

import argparse
from pathlib import Path

from transform_pipeline.maintenance import write_audit_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit repository cleanup candidates without deleting files. "
            "The report distinguishes tracked and untracked content."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Git repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("repository_cleanup_audit.json"),
        help="JSON report path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report_path = write_audit_report(args.root, args.output)
    print(f"Cleanup audit: {report_path.resolve()}")
    print("No files were deleted.")


if __name__ == "__main__":
    main()
