from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _project_root(explicit_root: str | None = None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "openreview").is_dir() and (package_root / "autocode").is_dir():
        return package_root
    cwd = Path.cwd().resolve()
    if (cwd / "openreview").is_dir() and (cwd / "autocode").is_dir():
        return cwd
    return package_root


def _run_service(root: Path, service: str, script: str) -> int:
    service_dir = root / service
    script_path = service_dir / script
    if not script_path.exists():
        print(
            f"PatchLoop service script not found: {script_path}\n"
            "Run from a PatchLoop source checkout or pass --root /path/to/patchloop.",
            file=sys.stderr,
        )
        return 2
    os.chdir(service_dir)
    os.execv(sys.executable, [sys.executable, str(script_path)])
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patchloop",
        description="Run PatchLoop's local GitHub review and coding services.",
    )
    parser.add_argument(
        "--root",
        help="Path to a PatchLoop source checkout. Defaults to the current checkout or cwd.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("review", help="Run the read-only OpenReview service.")
    subparsers.add_parser("code", help="Run the AutoCode patch and repair service.")

    args = parser.parse_args(argv)
    root = _project_root(args.root)
    if args.command == "review":
        return _run_service(root, "openreview", "review.py")
    if args.command == "code":
        return _run_service(root, "autocode", "autocode.py")
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
