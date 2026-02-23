#!/usr/bin/env python3
"""
Backfill missing `run_id` in annotation `meta.json` files.

This is useful when early annotation rounds were saved before run_id was
recorded in the GUI/API, but later training/analysis expects run_id for
batch isolation and plotting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory to search recursively for meta.json")
    ap.add_argument(
        "--from-run-id",
        default=None,
        help="If set, also change existing meta.json where run_id equals this value.",
    )
    ap.add_argument("--run-id", required=True, help="run_id value to set when missing")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Not found: {root}")

    meta_paths = sorted(root.rglob("meta.json"))
    if not meta_paths:
        raise SystemExit(f"No meta.json found under: {root}")

    changed_missing = 0
    changed_rename = 0
    skipped = 0
    errors = 0
    for p in meta_paths:
        try:
            meta = json.loads(p.read_text("utf-8"))
        except Exception:
            errors += 1
            continue

        rid = str(meta.get("run_id") or "").strip()
        if rid:
            if args.from_run_id is not None and rid == args.from_run_id:
                meta["run_id"] = args.run_id
                changed_rename += 1
                if args.apply:
                    p.write_text(json.dumps(meta, indent=2, sort_keys=False) + "\n", encoding="utf-8")
            else:
                skipped += 1
            continue

        meta["run_id"] = args.run_id
        changed_missing += 1
        if args.apply:
            p.write_text(json.dumps(meta, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"{mode}: root={root}")
    print(
        f"changed_missing={changed_missing} changed_rename={changed_rename} "
        f"skipped={skipped} errors={errors} total_meta={len(meta_paths)}"
    )


if __name__ == "__main__":
    main()
