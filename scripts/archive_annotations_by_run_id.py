#!/usr/bin/env python3
"""
Archive (move or copy) annotation folders for staged retraining.

Why:
  When doing staged experiments (10 -> 20 -> ... -> 50 corrected images),
  it’s easy to accidentally retrain on *all* prior annotations. This helper
  lets you move the completed batch out of `data/annotations/` so the next
  training round uses only newly saved annotations.

How it works:
  - scans immediate subfolders of `data/annotations/`
  - reads `<annotation_id>/meta.json`
  - matches either:
      - `--run-id <id>`: `meta["run_id"] == <id>`
      - `--missing-run-id`: `meta["run_id"]` missing/empty
      - `--all`: match all annotation folders with meta.json
  - moves (or copies) matched folders to:
      `data/annotations_archive/<tag>/<annotation_id>/`

This is filesystem-only (does not rewrite DB paths).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple


def _read_run_id(meta_path: Path) -> str:
    try:
        meta = json.loads(meta_path.read_text("utf-8"))
    except Exception:
        return ""
    if isinstance(meta, dict) and isinstance(meta.get("run_id"), str):
        return (meta.get("run_id") or "").strip()
    nested = meta.get("meta") if isinstance(meta, dict) else None
    if isinstance(nested, dict) and isinstance(nested.get("run_id"), str):
        return (nested.get("run_id") or "").strip()
    return ""


def _resolve_dst(dst_dir: Path, name: str) -> Path:
    """
    Avoid collisions: if dst exists, append a numeric suffix.
    """
    base = dst_dir / name
    if not base.exists():
        return base
    for i in range(2, 10_000):
        cand = dst_dir / f"{name}__{i}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not resolve unique destination for {name}")


def _op(src: Path, dst: Path, *, mode: str) -> None:
    if mode == "move":
        shutil.move(str(src), str(dst))
        return
    if mode == "copy":
        shutil.copytree(src, dst)
        return
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None, help="Only archive annotations with this exact run_id.")
    ap.add_argument("--missing-run-id", action="store_true", help="Archive annotations whose run_id is missing/empty.")
    ap.add_argument("--all", action="store_true", help="Archive all annotations that have a meta.json.")
    ap.add_argument(
        "--tag",
        default=None,
        help="Archive folder name under dst-root (defaults to --run-id when provided).",
    )
    ap.add_argument("--src", default="data/annotations", help="Source annotations directory.")
    ap.add_argument("--dst-root", default="data/annotations_archive", help="Archive root directory.")
    ap.add_argument("--mode", choices=["move", "copy"], default="move", help="Move (default) or copy.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without moving/copying.")
    args = ap.parse_args()

    run_id = (str(args.run_id).strip() if args.run_id is not None else "")
    modes = int(bool(run_id)) + int(bool(args.missing_run_id)) + int(bool(args.all))
    if modes != 1:
        raise SystemExit("Choose exactly one: --run-id, --missing-run-id, or --all")

    src = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()
    tag = (str(args.tag).strip() if args.tag is not None else "") or (run_id if run_id else "")
    if not tag:
        raise SystemExit("--tag is required when using --all or --missing-run-id")
    dst_dir = dst_root / tag

    if not src.exists():
        raise SystemExit(f"Source not found: {src}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    matched = 0
    skipped = 0
    missing_meta = 0

    for child in sorted(src.iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / "meta.json"
        if not meta_path.exists():
            missing_meta += 1
            continue
        rid = _read_run_id(meta_path)
        if args.all:
            pass
        elif args.missing_run_id:
            if rid:
                skipped += 1
                continue
        else:
            if rid != run_id:
                skipped += 1
                continue

        matched += 1
        dst = _resolve_dst(dst_dir, child.name)
        if args.dry_run:
            print(f"[dry-run] {args.mode}: {child} -> {dst}")
        else:
            _op(child, dst, mode=args.mode)
            print(f"{args.mode}: {child.name} -> {dst}")

    print()
    if args.all:
        print("match=all")
    elif args.missing_run_id:
        print("match=missing_run_id")
    else:
        print(f"run_id={run_id!r}")
    print(f"matched={matched} skipped={skipped} missing_meta={missing_meta}")
    print(f"archive_dir={dst_dir}")
    if matched == 0:
        if run_id:
            print("No matching annotations found. Make sure you set the same run_id in /annotate.")
        elif args.missing_run_id:
            print("No annotations with missing run_id were found.")
        else:
            print("No matching annotations were found.")


if __name__ == "__main__":
    main()
