#!/usr/bin/env python3
"""
Build phenotype filename->sample_id map for Sorghum GWAS runs.

Inputs:
- Tube tracker CSV (e.g. genotype_data/MR imaging tracker - Plot ID to Genotypes.csv)
  Required columns: Tube ID + Genotype (Plot ID optional if using --sample-id-source=plot)
- Phenotype rows from either:
  - API export CSV (/observability/analyses.csv), or
  - a local phenotype CSV with at least filename and tube_id columns

Output:
- CSV: filename,sample_id
  Upload in Subterra GUI: /genotype -> "Upload phenotype ID map"

Optional:
- Compare derived sample IDs against a VCF header sample list.
- Choose sample_id derivation:
  - --sample-id-source genotype (default; good for SAP PI_* IDs)
  - --sample-id-source plot
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _norm_tube(raw: str) -> str:
    v = (raw or "").strip()
    if not v:
        return ""
    m = re.match(r"^t?(\d+)$", v, flags=re.IGNORECASE)
    if m:
        return f"T{m.group(1)}"
    return v.upper()


def _plot_to_sample_id(plot_id: str) -> str:
    p = (plot_id or "").strip()
    if not p:
        return ""
    m = re.match(r"^(\d+)[_-](\d+)$", p)
    if m:
        # Common SorGSD sample style from plot IDs: 10_1 -> 101
        return f"{m.group(1)}{m.group(2)}"
    return re.sub(r"\s+", "", p)


def _normalize_sample_id(raw: str) -> str:
    v = (raw or "").strip()
    if not v:
        return ""
    m = re.match(r"^PI[\s_-]?(\d+)$", v, flags=re.IGNORECASE)
    if m:
        return f"PI_{m.group(1)}"
    return re.sub(r"\s+", "_", v)


def _load_tracker(path: Path, *, sample_id_source: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in (r.fieldnames or [])}
        tube_key = cols.get("tube id") or cols.get("tube_id") or cols.get("tube")
        plot_key = cols.get("plot id") or cols.get("plot_id") or cols.get("plot")
        geno_key = cols.get("genotype") or cols.get("line") or cols.get("accession")
        if not tube_key:
            raise ValueError("Tracker CSV must have column: Tube ID")
        if sample_id_source == "plot" and not plot_key:
            raise ValueError("Tracker CSV must have column: Plot ID when --sample-id-source=plot")
        if sample_id_source == "genotype" and not geno_key:
            raise ValueError("Tracker CSV must have column: Genotype when --sample-id-source=genotype")
        for row in r:
            if not row:
                continue
            tube = _norm_tube(row.get(tube_key, ""))
            if not tube:
                continue
            plot = (row.get(plot_key, "") or "").strip() if plot_key else ""
            genotype = (row.get(geno_key, "") or "").strip() if geno_key else ""
            if sample_id_source == "genotype":
                sample_id = _normalize_sample_id(genotype)
            else:
                sample_id = _plot_to_sample_id(plot)
            if not sample_id:
                continue
            out[tube] = {
                "sample_id": sample_id,
                "plot_id": plot,
                "genotype": genotype,
            }
    return out


def _fetch_csv(url: str) -> List[dict]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return [row for row in reader if row]


def _load_rows_from_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [row for row in r if row]


def _extract_tube_id(row: dict) -> str:
    for k in ("tube_id", "Tube ID", "tube"):
        v = (row.get(k) or "").strip()
        if v:
            return _norm_tube(v)
    return ""


def _extract_genotype(row: dict) -> str:
    for k in ("genotype", "Genotype", "line", "accession"):
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


def _extract_filename(row: dict) -> str:
    for k in ("filename", "file", "image_filename"):
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


def _tube_from_filename(filename: str) -> str:
    base = Path((filename or "").replace("\\", "/")).name
    if not base:
        return ""
    m = re.search(r"(?:^|[_-])T(\d+)(?:[_\.-]|$)", base, flags=re.IGNORECASE)
    if m:
        return _norm_tube(f"T{m.group(1)}")
    m = re.search(r"tube[_-]?(\d+)", base, flags=re.IGNORECASE)
    if m:
        return _norm_tube(f"T{m.group(1)}")
    return ""


def _load_vcf_samples(path: Path) -> List[str]:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:  # type: ignore[arg-type]
        for line in f:
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                return [c.strip() for c in cols[9:] if c.strip()]
    return []


def _build_map(phen_rows: Iterable[dict], tracker: Dict[str, Dict[str, str]]) -> Tuple[List[dict], List[str]]:
    out: List[dict] = []
    missing_labels: List[str] = []
    seen = set()
    genotype_index: Dict[str, Dict[str, str]] = {}
    for meta in tracker.values():
        g = _normalize_sample_id((meta.get("genotype") or "").strip())
        sid = (meta.get("sample_id") or "").strip()
        if g and sid:
            genotype_index[g] = meta

    for row in phen_rows:
        filename = _extract_filename(row)
        if not filename:
            continue
        tube = _extract_tube_id(row)
        if not tube:
            tube = _tube_from_filename(filename)
        genotype = _extract_genotype(row)
        genotype_norm = _normalize_sample_id(genotype)

        meta = tracker.get(tube) if tube else None
        if not meta and genotype_norm:
            meta = genotype_index.get(genotype_norm)
        if not meta:
            missing_labels.append(tube or genotype or "")
            continue
        sample_id = meta["sample_id"]
        if not sample_id:
            missing_labels.append(tube or genotype or "")
            continue
        key = (filename, sample_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "filename": filename,
                "sample_id": sample_id,
                "tube_id": tube or "",
                "plot_id": meta.get("plot_id", ""),
                "genotype": genotype or meta.get("genotype", ""),
            }
        )

    return out, missing_labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare phenotype sample_id map from Sorghum tube tracker CSV")
    ap.add_argument("--tracker-csv", default="genotype_data/MR imaging tracker - Plot ID to Genotypes.csv")
    ap.add_argument("--phenotype-csv", default="", help="Local analyses/phenotype CSV (optional). If omitted, fetch from API.")
    ap.add_argument("--analyses-url", default="http://localhost/api/observability/analyses.csv")
    ap.add_argument(
        "--sample-id-source",
        choices=["genotype", "plot"],
        default="genotype",
        help="How to derive sample_id from tracker CSV (default: genotype; recommended for SAP files).",
    )
    ap.add_argument("--vcf", default="", help="Optional VCF/VCF.GZ to report sample-ID overlap")
    ap.add_argument("--out", default="genotype_data/phenotype_sample_id_map.csv")
    args = ap.parse_args()

    tracker_path = Path(args.tracker_csv)
    if not tracker_path.exists():
        print(f"Tracker CSV not found: {tracker_path}", file=sys.stderr)
        sys.exit(1)
    tracker = _load_tracker(tracker_path, sample_id_source=args.sample_id_source)
    if not tracker:
        print("Tracker CSV parsed but no usable tube/sample rows found", file=sys.stderr)
        sys.exit(1)

    if args.phenotype_csv:
        phen_rows = _load_rows_from_csv(Path(args.phenotype_csv))
    else:
        phen_rows = _fetch_csv(args.analyses_url)

    mapped_rows, missing_tubes = _build_map(phen_rows, tracker)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "sample_id", "tube_id", "plot_id", "genotype"])
        w.writeheader()
        for r in mapped_rows:
            w.writerow(r)

    print(f"tracker_tubes={len(tracker)}")
    print(f"phenotype_rows={len(phen_rows)}")
    print(f"mapped_rows={len(mapped_rows)}")
    miss = sorted({m for m in missing_tubes if m})
    print(f"missing_tubes={len(miss)}")
    if miss:
        print("missing_tube_examples=", ",".join(miss[:15]))
    print(f"wrote={out_path}")

    if args.vcf:
        vcf_samples = set(_load_vcf_samples(Path(args.vcf)))
        derived = {r["sample_id"] for r in mapped_rows if r.get("sample_id")}
        overlap = sorted(derived & vcf_samples)
        print(f"vcf_samples={len(vcf_samples)}")
        print(f"derived_sample_ids={len(derived)}")
        print(f"vcf_overlap={len(overlap)}")
        if overlap:
            print("vcf_overlap_examples=", ",".join(overlap[:20]))


if __name__ == "__main__":
    main()
