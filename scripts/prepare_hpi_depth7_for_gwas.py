#!/usr/bin/env python3
"""
Prepare HPI root-area data for GWAS in Subterra.

Steps:
1) Parse the "Root Phenotyper Root Area Data - (HPI) Data Round Scattered.csv" table.
2) Keep rows that contain all 7 depth values.
3) Normalize genotype IDs to SAP marker sample IDs (e.g., "PI 597950" -> "PI_597950").
4) Keep only genotypes present in the supplied marker panel.
5) Aggregate a single trait value per genotype (mean of row-wise depth sums).
6) Upsert rows into analyses + phenotype_sample_id_map with a dedicated run_id.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_sample_id(raw: str) -> str:
    v = (raw or "").strip()
    if not v:
        return ""
    m = re.match(r"^PI[\s_-]?(\d+)$", v, flags=re.IGNORECASE)
    if m:
        return f"PI_{m.group(1)}"
    return re.sub(r"\s+", "_", v)


def _parse_float(raw: str) -> Optional[float]:
    t = (raw or "").strip()
    if not t:
        return None
    try:
        return float(t.replace(",", ""))
    except Exception:
        return None


def _read_hpi_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rr = csv.reader(f)
        row1 = next(rr, [])
        row2 = next(rr, [])
        header = row2 if row2 and (row2[0] or "").strip().lower() == "type" else row1
        if not header:
            raise RuntimeError(f"No header found in {path}")
        idx = {c.strip(): i for i, c in enumerate(header)}
        depth_cols = [idx.get(f"Depth {i}") for i in range(1, 8)]
        out: list[dict] = []
        for r in rr:
            if not r:
                continue
            if len(r) < len(header):
                r = r + [""] * (len(header) - len(r))
            typ = r[idx.get("Type", 0)].strip() if "Type" in idx else ""
            genotype = r[idx.get("Genotype", 2)].strip() if "Genotype" in idx else ""
            tube_id = r[idx.get("Tube ID", 3)].strip() if "Tube ID" in idx else ""
            plot_id = r[idx.get("Plot ID", 1)].strip() if "Plot ID" in idx else ""
            n_depths_raw = r[idx.get("# of Depths", 4)].strip() if "# of Depths" in idx else ""
            try:
                n_depths = int(float(n_depths_raw)) if n_depths_raw else None
            except Exception:
                n_depths = None
            depths: list[Optional[float]] = []
            for di in depth_cols:
                if di is None or di >= len(r):
                    depths.append(None)
                else:
                    depths.append(_parse_float(r[di]))
            out.append(
                {
                    "type": typ,
                    "plot_id": plot_id,
                    "genotype_raw": genotype,
                    "sample_id": _normalize_sample_id(genotype),
                    "tube_id": tube_id,
                    "n_depths": n_depths,
                    "depths": depths,
                }
            )
    return header, out


def _load_hapmap_sample_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        head = f.readline().rstrip("\n").split("\t")
    if len(head) < 12:
        raise RuntimeError(f"Unexpected HapMap header in {path}")
    return {c.strip() for c in head[11:] if c.strip()}


@dataclass
class Prepared:
    rows_kept: list[dict]
    by_sample_id: dict[str, list[float]]
    summary: dict


def _prepare_rows(rows: Iterable[dict], marker_ids: set[str]) -> Prepared:
    hpi_rows = [r for r in rows if (r.get("type") or "HPI").strip().upper() == "HPI"]
    complete: list[dict] = []
    for r in hpi_rows:
        depths = r.get("depths", [])
        non_empty = [d for d in depths if d is not None]
        has7 = len(non_empty) == 7
        if r.get("n_depths") == 7 or has7:
            complete.append(r)

    overlap_rows = [r for r in complete if r.get("sample_id") and r.get("sample_id") in marker_ids]
    by_sample_id: dict[str, list[float]] = defaultdict(list)
    for r in overlap_rows:
        vals = [float(x) for x in r.get("depths", []) if x is not None]
        if len(vals) != 7:
            continue
        by_sample_id[str(r["sample_id"])].append(sum(vals))

    summary = {
        "rows_hpi_total": len(hpi_rows),
        "rows_with_7_depths": len(complete),
        "rows_with_7_depths_and_marker_overlap": len(overlap_rows),
        "unique_genotypes_with_7_depths": len({r.get("sample_id") for r in complete if r.get("sample_id")}),
        "unique_genotypes_with_7_depths_and_marker_overlap": len(by_sample_id),
    }
    return Prepared(rows_kept=overlap_rows, by_sample_id=dict(by_sample_id), summary=summary)


def _write_outputs(prep: Prepared, out_rows_csv: Path, out_trait_csv: Path, out_summary_json: Path) -> None:
    out_rows_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_rows_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sample_id",
                "genotype_raw",
                "tube_id",
                "plot_id",
                "depth_1",
                "depth_2",
                "depth_3",
                "depth_4",
                "depth_5",
                "depth_6",
                "depth_7",
                "depth_sum",
            ]
        )
        for r in prep.rows_kept:
            depths = [float(x) if x is not None else "" for x in r["depths"]]
            ds = sum(float(x) for x in r["depths"] if x is not None)
            w.writerow(
                [
                    r["sample_id"],
                    r["genotype_raw"],
                    r["tube_id"],
                    r["plot_id"],
                    *depths,
                    ds,
                ]
            )

    with out_trait_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "n_records", "total_root_area"])
        for sid in sorted(prep.by_sample_id):
            vals = prep.by_sample_id[sid]
            mean_sum = sum(vals) / len(vals)
            w.writerow([sid, len(vals), mean_sum])

    out_summary_json.write_text(json.dumps(prep.summary, indent=2), encoding="utf-8")


def _upsert_analysis_rows(
    db_path: Path,
    *,
    run_id: str,
    source_file: str,
    sample_to_trait: dict[str, list[float]],
) -> tuple[int, int]:
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    cur = con.cursor()

    cur.execute(
        """
        SELECT filename FROM analyses
        WHERE json_extract(extra, '$.meta.run_id') = ?
        """,
        (run_id,),
    )
    existing_fns = [r[0] for r in cur.fetchall()]
    if existing_fns:
        q = ",".join(["?"] * len(existing_fns))
        cur.execute(f"DELETE FROM phenotype_sample_id_map WHERE filename IN ({q})", tuple(existing_fns))
        cur.execute(f"DELETE FROM analyses WHERE filename IN ({q})", tuple(existing_fns))

    inserted = 0
    mapped = 0
    ts = _utc_now_iso()
    for sid in sorted(sample_to_trait):
        vals = sample_to_trait[sid]
        if not vals:
            continue
        trait = float(sum(vals) / len(vals))
        filename = f"HPI_DEPTH7_{sid}.csv"
        extra = {
            "meta": {
                "run_id": run_id,
                "source": "Root Phenotyper Root Area Data - (HPI) Data Round Scattered.csv",
                "minirhizotron": {
                    "genotype": sid,
                    "tube_id": "",
                    "depth_coverage": "1-7",
                },
                "aggregation": {
                    "method": "mean_of_depth_sums",
                    "n_records": len(vals),
                },
            }
        }
        cur.execute(
            """
            INSERT INTO analyses
            (created_at, filename, model_type, model_version, threshold_area, scaling_factor, confidence_threshold,
             root_count, average_root_diameter, total_root_length, total_root_area, total_root_volume, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                filename,
                "external_hpi",
                "hpi_depth7_v1",
                0,
                1.0,
                0.0,
                0,
                0.0,
                0.0,
                trait,
                0.0,
                json.dumps(extra, ensure_ascii=False),
            ),
        )
        inserted += 1
        cur.execute(
            """
            INSERT INTO phenotype_sample_id_map (created_at, filename, sample_id)
            VALUES (?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET sample_id=excluded.sample_id
            """,
            (ts, filename, sid),
        )
        mapped += 1

    con.commit()
    con.close()
    return inserted, mapped


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare HPI depth-7 filtered phenotype rows for Subterra GWAS")
    ap.add_argument(
        "--input-csv",
        default="genotype_data/Root Phenotyper Root Area Data - (HPI) Data Round Scattered.csv",
    )
    ap.add_argument("--marker-hapmap", default="genotype_data/SAP_imputed.hmp")
    ap.add_argument("--db-path", default="data/subterra.sqlite3")
    ap.add_argument("--run-id", default=f"HPI_depth7_complete_{datetime.now().strftime('%Y%m%d')}")
    ap.add_argument("--out-rows-csv", default="genotype_data/hpi_depth7_rows_filtered.csv")
    ap.add_argument("--out-trait-csv", default="genotype_data/hpi_depth7_trait_by_genotype.csv")
    ap.add_argument("--out-summary-json", default="docs/figures/_exports/paper_figures/hpi_depth7_summary.json")
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    marker_hapmap = Path(args.marker_hapmap)
    db_path = Path(args.db_path)
    out_rows_csv = Path(args.out_rows_csv)
    out_trait_csv = Path(args.out_trait_csv)
    out_summary_json = Path(args.out_summary_json)

    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")
    if not marker_hapmap.exists():
        raise SystemExit(f"Marker HapMap not found: {marker_hapmap}")
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    _header, rows = _read_hpi_rows(input_csv)
    marker_ids = _load_hapmap_sample_ids(marker_hapmap)
    prep = _prepare_rows(rows, marker_ids)
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    _write_outputs(prep, out_rows_csv, out_trait_csv, out_summary_json)

    inserted, mapped = _upsert_analysis_rows(
        db_path,
        run_id=args.run_id,
        source_file=input_csv.name,
        sample_to_trait=prep.by_sample_id,
    )

    print(f"run_id={args.run_id}")
    for k, v in prep.summary.items():
        print(f"{k}={v}")
    print(f"analysis_rows_inserted={inserted}")
    print(f"sample_id_map_rows_upserted={mapped}")
    print(f"wrote_rows_csv={out_rows_csv}")
    print(f"wrote_trait_csv={out_trait_csv}")
    print(f"wrote_summary_json={out_summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
