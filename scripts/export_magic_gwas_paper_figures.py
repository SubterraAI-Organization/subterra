#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def _http_get(url: str, timeout: int = 300) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def main() -> int:
    base = os.environ.get("SUBTERRA_BASE_URL", "http://127.0.0.1")
    out_dir = Path("docs/figures/_exports/gwas")
    out_dir.mkdir(parents=True, exist_ok=True)

    phenotype_field = os.environ.get("PHENOTYPE_FIELD", "total_root_volume")
    methods = ["mlm", "farmcpu"]
    p_adjusts = ["none", "bh", "bonferroni"]

    common = {
        "phenotype_field": phenotype_field,
        "max_markers": "2000",
        "min_n": os.environ.get("MIN_N", "20"),
        "n_pcs": os.environ.get("N_PCS", "3"),
        "kinship_markers": os.environ.get("KINSHIP_MARKERS", "1000"),
        "farmcpu_max_iters": os.environ.get("FARMCPU_MAX_ITERS", "3"),
        "farmcpu_max_qtn": os.environ.get("FARMCPU_MAX_QTN", "10"),
        "farmcpu_seed": os.environ.get("FARMCPU_SEED", "7"),
        "width": os.environ.get("PLOT_WIDTH", "1600"),
        "height": os.environ.get("PLOT_HEIGHT", "700"),
    }

    # quick health check
    stats = _http_get(f"{base}/genotype-api/stats", timeout=30).decode("utf-8", errors="replace")
    print("genotype stats:", stats.strip())

    for method in methods:
        for p_adjust in p_adjusts:
            params = dict(common)
            params["method"] = method
            params["p_adjust"] = p_adjust
            q = urllib.parse.urlencode(params)

            svg_url = f"{base}/genotype-api/mapping/plot.svg?{q}"
            csv_url = f"{base}/genotype-api/mapping/results.csv?{q}"

            tag = f"{phenotype_field}_{method}_{p_adjust}"
            svg_path = out_dir / f"gwas_{tag}.svg"
            csv_path = out_dir / f"gwas_{tag}.csv"

            print(f"downloading {tag} …")
            svg_path.write_bytes(_http_get(svg_url, timeout=600))
            csv_path.write_bytes(_http_get(csv_url, timeout=600))

    print(f"wrote figures to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

