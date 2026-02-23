#!/usr/bin/env python3
"""
Convert Arabidopsis MAGIC hapmap SNP file to Subterra genotype service CSV format.

Input: gzipped hapmap TSV (e.g. data/markers/MAGIC_SNP_chr1.tsv.gz)
  - First 11 columns: rs, allele, chr, bp, strand, assembly, center, protLSID, assayLSID, panel, Qcode
  - Remaining columns: one per sample (e.g. MAGIC.10), genotype as single letter (A/C/G/T) or N

Output: CSV with sample_id, then one column per marker (rs_chr_bp), values 0 (major allele), 1 (alt).
  Missing/N → empty. Subterra genotype service expects first column = sample_id, rest = numeric.

Usage:
  python scripts/hapmap_to_genotype_csv.py [--every N] [--max-markers M] [--out FILE] <input.tsv.gz>
  Default: --every 100 --max-markers 5000 --out data/markers/arabidopsis_magic_chr1_subset.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path


def _open(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _encode_allele(letter: str, ref: str, alt: str | None) -> str:
    """Encode single-letter genotype to 0/1/2. Missing/N → empty."""
    letter = (letter or "").strip().upper()
    if not letter or letter == "N" or letter not in "ACGT":
        return ""
    if letter == ref:
        return "0"
    if alt and letter == alt:
        return "1"
    # Biallelic: if we only have ref defined, any other is alt
    if not alt:
        return "1"
    # Unknown third allele → treat as missing
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert hapmap to Subterra genotype CSV")
    ap.add_argument("input", nargs="?", default="data/markers/MAGIC_SNP_chr1.tsv.gz", help="Input hapmap .tsv or .tsv.gz")
    ap.add_argument("--every", type=int, default=100, help="Take every Nth SNP to reduce size (default 100)")
    ap.add_argument("--max-markers", type=int, default=5000, help="Max marker columns (default 5000)")
    ap.add_argument("--out", default="data/markers/arabidopsis_magic_chr1_subset.csv", help="Output CSV path")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _open(input_path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if len(header) < 12:
            print("Error: expected at least 12 columns (11 hapmap + samples)", file=sys.stderr)
            sys.exit(1)
        sample_ids = header[11:]
        # marker name will be rs_chr_bp (e.g. 1s18_1_18)
        marker_names: list[str] = []
        rows_by_sample: dict[str, list[str]] = {sid: [] for sid in sample_ids}
        n_markers = 0
        for i, row in enumerate(reader):
            if len(row) < 11 + len(sample_ids):
                continue
            if (i % args.every) != 0:
                continue
            if n_markers >= args.max_markers:
                break
            rs, allele, chr_, bp = row[0], row[1], row[2], row[3]
            name = f"{rs}_{chr_}_{bp}"
            marker_names.append(name)
            # Determine ref/alt from first few samples (major allele as ref)
            letters = [c.strip().upper() for c in row[11 : 11 + len(sample_ids)]]
            counts: dict[str, int] = {}
            for c in letters:
                if c and c in "ACGT":
                    counts[c] = counts.get(c, 0) + 1
            ref = max(counts, key=counts.get) if counts else "N"
            alts = [c for c in "ACGT" if c != ref and counts.get(c, 0) > 0]
            alt = alts[0] if alts else None
            for j, sid in enumerate(sample_ids):
                val = _encode_allele(row[11 + j], ref, alt)
                rows_by_sample[sid].append(val)
            n_markers += 1

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id"] + marker_names)
        for sid in sample_ids:
            writer.writerow([sid] + rows_by_sample[sid])

    print(f"Wrote {out_path}: {len(sample_ids)} samples, {len(marker_names)} markers (from {args.input})")


if __name__ == "__main__":
    main()
