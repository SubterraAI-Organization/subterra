#!/usr/bin/env python3
from __future__ import annotations

import os
import sqlite3
import math
import base64
from dataclasses import dataclass
from pathlib import Path

from fastapi.responses import Response

import app


def _write_response(resp: Response, path: Path) -> None:
    body = resp.body
    if body is None:
        raise RuntimeError(f"Empty response body for {path.name}")
    path.write_bytes(body)


def _save_figure5_png(
    out_path: Path,
    *,
    phenotype_field: str,
    trait_vals: list[float],
    phen_total: int,
    mapped: int,
    overlap: int,
    mlm_rows: list[app.MappingRow],
    farm_rows: list[app.MappingRow],
    top_marker: str,
    groups: dict[int, list[float]],
    sig_counts: dict[str, int],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    def _qq(rows: list[app.MappingRow]):
        ps = [float(r.p_value) for r in rows if r.p_value is not None and float(r.p_value) > 0 and math.isfinite(float(r.p_value))]
        ps.sort()
        m = len(ps)
        obs = [-math.log10(max(1e-300, v)) for v in ps]
        exp = [-math.log10((i + 1) / (m + 1)) for i in range(m)]
        return exp, obs

    def _manhattan_xy(rows: list[app.MappingRow]):
        pts = []
        for r in rows:
            pv = float(r.p_value) if r.p_value is not None else 1.0
            if not math.isfinite(pv) or pv <= 0:
                pv = 1e-300
            locus = None
            try:
                locus = app._parse_marker_locus(r.marker_name)  # type: ignore[attr-defined]
            except Exception:
                locus = None
            if locus:
                chrom, pos = locus
            else:
                chrom, pos = None, None
            pts.append((pv, chrom, pos))

        parsed = sum(1 for (_p, c, pos) in pts if c is not None and pos is not None)
        use_genomic = parsed >= max(10, int(0.4 * len(pts)))
        if use_genomic:
            pts.sort(key=lambda t: (int(t[1] or 0), int(t[2] or 0)))
            chr_max: dict[int, int] = {}
            for (_p, c, pos) in pts:
                if c is None or pos is None:
                    continue
                chr_max[int(c)] = max(chr_max.get(int(c), 0), int(pos))
            chroms = sorted(chr_max.keys())
            gap = 2_000_000
            off = 0
            chr_off: dict[int, int] = {}
            for c in chroms:
                chr_off[c] = off
                off += chr_max[c] + gap
            xs, ys, cs = [], [], []
            for p, c, pos in pts:
                if c is None or pos is None:
                    continue
                xs.append(chr_off[int(c)] + int(pos))
                ys.append(-math.log10(p))
                cs.append(int(c))
            return xs, ys, cs, chroms, chr_off, chr_max
        xs = list(range(len(pts)))
        ys = [-math.log10(p) for (p, _c, _pos) in pts]
        return xs, ys, [0] * len(xs), [], {}, {}

    def _plot_assoc(ax_m, ax_q, rows: list[app.MappingRow], title: str):
        xs, ys, cs, chroms, chr_off, chr_max = _manhattan_xy(rows)
        if chroms:
            colors = ["#60a5fa" if (c % 2 == 1) else "#fb923c" for c in cs]
            ax_m.scatter(xs, ys, s=10, c=colors, alpha=0.78, edgecolors="none")
            mids = []
            labs = []
            for c in chroms:
                start = chr_off[c]
                end = chr_off[c] + chr_max[c]
                mids.append((start + end) / 2)
                labs.append(str(c))
            ax_m.set_xticks(mids)
            ax_m.set_xticklabels([f"Chr{c}" for c in labs])
            ax_m.set_xlabel("marker position")
        else:
            ax_m.scatter(xs, ys, s=10, color="#60a5fa", alpha=0.78, edgecolors="none")
            ax_m.set_xlabel("marker index")
        ax_m.set_ylabel("-log10(p)")
        ax_m.set_title(title)

        exp, obs = _qq(rows)
        ax_q.scatter(exp, obs, s=10, color="#334155", alpha=0.65, edgecolors="none")
        mx = max(exp[-1] if exp else 1, obs[-1] if obs else 1, 5)
        ax_q.plot([0, mx], [0, mx], linestyle="--", color="#94a3b8", linewidth=1.5)
        ax_q.set_xlabel("expected -log10(p)")
        ax_q.set_ylabel("observed -log10(p)")
        ax_q.set_title("QQ")

    plt.rcParams.update(
        {
            # Match the Figure 2 paper style a bit more closely (larger, cleaner text).
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(12.6, 12.9))
    gs = fig.add_gridspec(3, 2, wspace=0.16, hspace=0.26, left=0.06, right=0.985, top=0.97, bottom=0.06)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(f"A. Trait distribution ({phenotype_field})")
    if trait_vals:
        ax.hist(trait_vals, bins=18, edgecolor="#334155", color="#60a5fa", alpha=0.8)
    else:
        ax.text(0.5, 0.5, "No mapped trait values", ha="center", va="center")
    ax.set_xlabel(phenotype_field)
    ax.set_ylabel("count")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B. Sample ID join and overlap")
    labels = ["phenotypes", "mapped IDs", "overlap"]
    vals = [phen_total, mapped, overlap]
    ax.bar(range(3), vals, color=["#60a5fa", "#a78bfa", "#22c55e"], edgecolor="#334155")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("n")
    for i, v in enumerate(vals):
        ax.text(i, v, f" {v}", va="bottom")

    axC = fig.add_subplot(gs[1, 0])
    axC.axis("off")
    _plot_assoc(axC.inset_axes([0.0, 0.0, 0.72, 1.0]), axC.inset_axes([0.75, 0.0, 0.25, 1.0]), mlm_rows, "C. Association (MLM, FDR/BH)")

    axD = fig.add_subplot(gs[1, 1])
    axD.axis("off")
    _plot_assoc(axD.inset_axes([0.0, 0.0, 0.72, 1.0]), axD.inset_axes([0.75, 0.0, 0.25, 1.0]), farm_rows, "D. Association (FarmCPU, FDR/BH)")

    ax = fig.add_subplot(gs[2, 0])
    ax.set_title(f"E. Top marker effect (FarmCPU FDR): {top_marker or '—'}")
    keys = sorted(k for k, v in groups.items() if v)
    if keys:
        data = [groups[k] for k in keys]
        ax.boxplot(data, tick_labels=[str(k) for k in keys], showfliers=False)
        for i, k in enumerate(keys, start=1):
            ax.scatter([i] * len(groups[k]), groups[k], s=10, color="#0f172a", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No top-marker groups", ha="center", va="center")
        ax.set_xticks([])
    ax.set_xlabel("marker genotype (0/1)")
    ax.set_ylabel(phenotype_field)

    ax = fig.add_subplot(gs[2, 1])
    ax.set_title("F. Multiple-testing correction (FarmCPU)")
    labels = ["p≤0.05", "FDR q≤0.05", "Bonf q≤0.05"]
    vals = [int(sig_counts.get("none", 0)), int(sig_counts.get("bh", 0)), int(sig_counts.get("bonferroni", 0))]
    ax.bar(range(3), vals, color=["#94a3b8", "#ef4444", "#0ea5e9"], edgecolor="#334155")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("significant markers")
    for i, v in enumerate(vals):
        ax.text(i, v, f" {v}", va="bottom")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

def _svg_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _svg_trait_histogram(values: list[float], *, title: str, xlabel: str, width: int = 900, height: int = 560) -> bytes:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        raise RuntimeError("No values to plot")
    vmin = min(vals)
    vmax = max(vals)
    if vmin == vmax:
        vmax = vmin + 1.0

    bins = 18
    counts = [0] * bins
    for v in vals:
        t = (v - vmin) / (vmax - vmin)
        i = min(bins - 1, max(0, int(t * bins)))
        counts[i] += 1
    cmax = max(1, max(counts))

    mean = sum(vals) / len(vals)
    med = sorted(vals)[len(vals) // 2]

    pad = 48
    x0, y0 = pad, pad
    x1, y1 = width - pad, height - pad

    def sx(x: float) -> float:
        return x0 + (x - vmin) / (vmax - vmin) * (x1 - x0)

    def sy(c: float) -> float:
        return y1 - (c / cmax) * (y1 - y0)

    axis = "#334155"
    grid = "rgba(51,65,85,0.18)"
    bar = "#60a5fa"
    mean_c = "#ef4444"
    med_c = "#0ea5e9"

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{pad}" y="{pad - 18}" font-family="Arial, sans-serif" font-size="16" fill="#0f172a">{_svg_escape(title)}</text>')

    # frame
    lines.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="white" stroke="{axis}" stroke-width="1"/>')
    # y grid/ticks
    for t in range(0, cmax + 1, max(1, cmax // 5)):
        y = sy(t)
        lines.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="{grid}" stroke-width="1"/>')
        lines.append(f'<text x="{x0 - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="{axis}">{t}</text>')

    # bars
    for i, c in enumerate(counts):
        bx0 = vmin + (i / bins) * (vmax - vmin)
        bx1 = vmin + ((i + 1) / bins) * (vmax - vmin)
        rx = sx(bx0)
        rw = max(1.0, sx(bx1) - sx(bx0) - 1.0)
        ry = sy(c)
        rh = y1 - ry
        lines.append(f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{rw:.2f}" height="{rh:.2f}" fill="{bar}" fill-opacity="0.75" stroke="{axis}" stroke-opacity="0.18"/>')

    # mean/median
    mx = sx(mean)
    mdx = sx(med)
    lines.append(f'<line x1="{mx:.2f}" y1="{y0}" x2="{mx:.2f}" y2="{y1}" stroke="{mean_c}" stroke-width="2"/>')
    lines.append(f'<line x1="{mdx:.2f}" y1="{y0}" x2="{mdx:.2f}" y2="{y1}" stroke="{med_c}" stroke-width="2" stroke-dasharray="4,3"/>')

    # labels
    lines.append(f'<text x="{(x0 + x1) / 2:.2f}" y="{height - 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{axis}">{_svg_escape(xlabel)}</text>')
    lines.append(f'<text x="{x0 - 34}" y="{(y0 + y1) / 2:.2f}" text-anchor="middle" transform="rotate(-90 {x0 - 34} {(y0 + y1) / 2:.2f})" font-family="Arial, sans-serif" font-size="12" fill="{axis}">count</text>')

    # small legend/stat line
    stat = f"n={len(vals)}  mean={mean:.3g}  median={med:.3g}"
    lines.append(f'<text x="{x1}" y="{pad - 18}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#475569">{_svg_escape(stat)}</text>')
    lines.append("</svg>")
    return ("\n".join(lines)).encode("utf-8")


@dataclass
class _Box:
    q1: float
    q2: float
    q3: float
    lo: float
    hi: float


def _box_stats(xs: list[float]) -> _Box:
    ys = sorted(float(x) for x in xs)
    n = len(ys)
    if n == 0:
        raise RuntimeError("empty group")
    def q(p: float) -> float:
        if n == 1:
            return ys[0]
        idx = p * (n - 1)
        i0 = int(math.floor(idx))
        i1 = min(n - 1, i0 + 1)
        t = idx - i0
        return ys[i0] * (1 - t) + ys[i1] * t
    q1 = q(0.25)
    q2 = q(0.5)
    q3 = q(0.75)
    iqr = q3 - q1
    lo = min(ys)
    hi = max(ys)
    # Tukey whiskers
    wlo = q1 - 1.5 * iqr
    whi = q3 + 1.5 * iqr
    lo = min([v for v in ys if v >= wlo] or [ys[0]])
    hi = max([v for v in ys if v <= whi] or [ys[-1]])
    return _Box(q1=q1, q2=q2, q3=q3, lo=lo, hi=hi)


def _svg_genotype_effect(groups: dict[int, list[float]], *, title: str, xlabel: str, ylabel: str, width: int = 900, height: int = 560) -> bytes:
    # filter empty groups; sort by genotype code
    keys = sorted(k for k, v in groups.items() if v)
    if not keys:
        raise RuntimeError("No groups to plot")
    all_vals = [v for k in keys for v in groups[k]]
    ymin = min(all_vals)
    ymax = max(all_vals)
    if ymin == ymax:
        ymax = ymin + 1.0
    pad = 56
    x0, y0 = pad, pad
    x1, y1 = width - pad, height - pad
    axis = "#334155"
    grid = "rgba(51,65,85,0.18)"
    box_fill = "#a78bfa"
    pt = "#0f172a"

    def sy(y: float) -> float:
        return y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)

    # x positions
    step = (x1 - x0) / max(1, len(keys))
    xs = {k: x0 + (i + 0.5) * step for i, k in enumerate(keys)}
    box_w = min(80.0, step * 0.45)

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{pad}" y="{pad - 18}" font-family="Arial, sans-serif" font-size="16" fill="#0f172a">{_svg_escape(title)}</text>')
    lines.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="white" stroke="{axis}" stroke-width="1"/>')

    # y grid/ticks
    for i in range(6):
        yv = ymin + (i / 5) * (ymax - ymin)
        y = sy(yv)
        lines.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="{grid}" stroke-width="1"/>')
        lines.append(f'<text x="{x0 - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="{axis}">{yv:.3g}</text>')

    # plot each group
    rng = 1337  # deterministic jitter
    def rand() -> float:
        nonlocal rng
        rng = (1103515245 * rng + 12345) & 0x7fffffff
        return rng / 0x7fffffff

    for k in keys:
        vals = groups[k]
        b = _box_stats(vals)
        cx = xs[k]
        # whiskers
        lines.append(f'<line x1="{cx:.2f}" y1="{sy(b.lo):.2f}" x2="{cx:.2f}" y2="{sy(b.hi):.2f}" stroke="{axis}" stroke-width="1.5"/>')
        lines.append(f'<line x1="{(cx - box_w/2):.2f}" y1="{sy(b.lo):.2f}" x2="{(cx + box_w/2):.2f}" y2="{sy(b.lo):.2f}" stroke="{axis}" stroke-width="1.5"/>')
        lines.append(f'<line x1="{(cx - box_w/2):.2f}" y1="{sy(b.hi):.2f}" x2="{(cx + box_w/2):.2f}" y2="{sy(b.hi):.2f}" stroke="{axis}" stroke-width="1.5"/>')
        # box
        yq3 = sy(b.q3)
        yq1 = sy(b.q1)
        lines.append(f'<rect x="{(cx - box_w/2):.2f}" y="{yq3:.2f}" width="{box_w:.2f}" height="{(yq1 - yq3):.2f}" fill="{box_fill}" fill-opacity="0.55" stroke="{axis}" stroke-width="1.2"/>')
        # median
        lines.append(f'<line x1="{(cx - box_w/2):.2f}" y1="{sy(b.q2):.2f}" x2="{(cx + box_w/2):.2f}" y2="{sy(b.q2):.2f}" stroke="{axis}" stroke-width="2"/>')
        # points
        for v in vals:
            j = (rand() - 0.5) * (box_w * 0.65)
            lines.append(f'<circle cx="{(cx + j):.2f}" cy="{sy(v):.2f}" r="2.0" fill="{pt}" fill-opacity="0.55"/>')
        # x tick label + n
        lines.append(f'<text x="{cx:.2f}" y="{y1 + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{axis}">{k}</text>')
        lines.append(f'<text x="{cx:.2f}" y="{y0 - 8}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#475569">n={len(vals)}</text>')

    # axis labels
    lines.append(f'<text x="{(x0 + x1) / 2:.2f}" y="{height - 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{axis}">{_svg_escape(xlabel)}</text>')
    lines.append(f'<text x="{x0 - 38}" y="{(y0 + y1) / 2:.2f}" text-anchor="middle" transform="rotate(-90 {x0 - 38} {(y0 + y1) / 2:.2f})" font-family="Arial, sans-serif" font-size="12" fill="{axis}">{_svg_escape(ylabel)}</text>')
    lines.append("</svg>")
    return ("\n".join(lines)).encode("utf-8")


def _svg_join_summary(*, phen_total: int, mapped: int, overlap: int, title: str, width: int = 900, height: int = 560) -> bytes:
    pad = 56
    x0, y0 = pad, pad
    x1, y1 = width - pad, height - pad
    axis = "#334155"
    grid = "rgba(51,65,85,0.18)"
    cols = ["#60a5fa", "#a78bfa", "#22c55e"]
    labels = ["phenotypes in DB", "mapped to sample_id", "overlap with markers"]
    vals = [int(phen_total), int(mapped), int(overlap)]
    vmax = max(1, max(vals))
    bw = (x1 - x0) / max(1, len(vals)) * 0.62

    def sy(v: float) -> float:
        return y1 - (v / vmax) * (y1 - y0)

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{pad}" y="{pad - 18}" font-family="Arial, sans-serif" font-size="16" fill="#0f172a">{_svg_escape(title)}</text>')
    lines.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="white" stroke="{axis}" stroke-width="1"/>')
    for i in range(6):
        yv = (i / 5) * vmax
        y = sy(yv)
        lines.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="{grid}" stroke-width="1"/>')
        lines.append(f'<text x="{x0 - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="{axis}">{int(yv)}</text>')
    for i, (lab, v, c) in enumerate(zip(labels, vals, cols)):
        cx = x0 + (i + 0.5) * (x1 - x0) / len(vals)
        rx = cx - bw / 2
        ry = sy(v)
        rh = y1 - ry
        lines.append(f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{bw:.2f}" height="{rh:.2f}" fill="{c}" fill-opacity="0.72" stroke="{axis}" stroke-opacity="0.2"/>')
        lines.append(f'<text x="{cx:.2f}" y="{ry - 8:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#0f172a">{v}</text>')
        lines.append(f'<text x="{cx:.2f}" y="{y1 + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{axis}">{_svg_escape(lab)}</text>')
    lines.append(f'<text x="{(x0 + x1)/2:.2f}" y="{height - 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#475569">ID join is required for real marker panels</text>')
    lines.append("</svg>")
    return ("\n".join(lines)).encode("utf-8")


def _svg_significance_summary(counts: dict[str, int], *, title: str, width: int = 900, height: int = 560) -> bytes:
    pad = 56
    x0, y0 = pad, pad
    x1, y1 = width - pad, height - pad
    axis = "#334155"
    grid = "rgba(51,65,85,0.18)"
    keys = ["none", "bh", "bonferroni"]
    labels = ["p≤0.05", "FDR q≤0.05", "Bonf q≤0.05"]
    cols = ["#94a3b8", "#ef4444", "#0ea5e9"]
    vals = [int(counts.get(k, 0)) for k in keys]
    vmax = max(1, max(vals))
    bw = (x1 - x0) / max(1, len(vals)) * 0.62

    def sy(v: float) -> float:
        return y1 - (v / vmax) * (y1 - y0)

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{pad}" y="{pad - 18}" font-family="Arial, sans-serif" font-size="16" fill="#0f172a">{_svg_escape(title)}</text>')
    lines.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="white" stroke="{axis}" stroke-width="1"/>')
    for i in range(6):
        yv = (i / 5) * vmax
        y = sy(yv)
        lines.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="{grid}" stroke-width="1"/>')
        lines.append(f'<text x="{x0 - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="{axis}">{int(yv)}</text>')
    for i, (lab, v, c) in enumerate(zip(labels, vals, cols)):
        cx = x0 + (i + 0.5) * (x1 - x0) / len(vals)
        rx = cx - bw / 2
        ry = sy(v)
        rh = y1 - ry
        lines.append(f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{bw:.2f}" height="{rh:.2f}" fill="{c}" fill-opacity="0.72" stroke="{axis}" stroke-opacity="0.2"/>')
        lines.append(f'<text x="{cx:.2f}" y="{ry - 8:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#0f172a">{v}</text>')
        lines.append(f'<text x="{cx:.2f}" y="{y1 + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{axis}">{_svg_escape(lab)}</text>')
    lines.append(f'<text x="{(x0 + x1)/2:.2f}" y="{height - 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#475569">more stringent correction → fewer significant hits</text>')
    lines.append("</svg>")
    return ("\n".join(lines)).encode("utf-8")


def _inline_svg_as_group(svg_bytes: bytes, *, x: float, y: float, w: float, h: float) -> str:
    # Minimal, dependency-free SVG inliner for our own generated SVGs.
    s = svg_bytes.decode("utf-8", errors="replace").strip()
    if "<svg" not in s or "</svg>" not in s:
        raise RuntimeError("Not an SVG")
    head, rest = s.split(">", 1)
    inner = rest.rsplit("</svg>", 1)[0]
    # viewBox preferred
    vb = None
    for key in ["viewBox", "viewbox"]:
        k = f'{key}="'
        if k in head:
            vb = head.split(k, 1)[1].split('"', 1)[0]
            break
    if vb:
        parts = [p for p in vb.replace(",", " ").split() if p]
        if len(parts) == 4:
            minx, miny, sw, sh = map(float, parts)
        else:
            minx, miny, sw, sh = 0.0, 0.0, 1.0, 1.0
    else:
        # fall back to width/height
        def _attr(name: str) -> float:
            k = f'{name}="'
            if k not in head:
                return 1.0
            v = head.split(k, 1)[1].split('"', 1)[0].strip().lower().replace("px", "")
            try:
                return float(v)
            except Exception:
                return 1.0
        minx, miny = 0.0, 0.0
        sw, sh = _attr("width"), _attr("height")
    if sw <= 0 or sh <= 0:
        sw, sh = 1.0, 1.0
    scale = min(w / sw, h / sh)
    dx = x + (w - sw * scale) / 2.0
    dy = y + (h - sh * scale) / 2.0
    # Apply viewBox origin offset
    tx = -minx
    ty = -miny
    return f'<g transform="translate({dx:.2f},{dy:.2f}) scale({scale:.6f}) translate({tx:.2f},{ty:.2f})">{inner}</g>'


def _compose_svg_grid(panels: list[tuple[str, bytes]], *, out_w: int, out_h: int, cols: int, title: str) -> bytes:
    pad = 52
    gutter = 22
    rows = (len(panels) + cols - 1) // cols
    grid_w = out_w - 2 * pad
    grid_h = out_h - 2 * pad - 22
    cell_w = (grid_w - (cols - 1) * gutter) / cols
    cell_h = (grid_h - (rows - 1) * gutter) / rows

    axis = "#334155"
    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{out_w}" height="{out_h}" viewBox="0 0 {out_w} {out_h}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{pad}" y="{pad - 18}" font-family="Arial, sans-serif" font-size="18" fill="#0f172a">{_svg_escape(title)}</text>')
    for i, (label, svg) in enumerate(panels):
        r = i // cols
        c = i % cols
        x = pad + c * (cell_w + gutter)
        y = pad + 22 + r * (cell_h + gutter)
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="white" stroke="{axis}" stroke-width="1"/>')
        lines.append(_inline_svg_as_group(svg, x=x, y=y, w=cell_w, h=cell_h))
        # panel letter
        lines.append(
            f'<text x="{(x + 10):.2f}" y="{(y + 18):.2f}" font-family="Arial, sans-serif" font-size="14" fill="#0f172a">{_svg_escape(label)}</text>'
        )
    lines.append("</svg>")
    return ("\n".join(lines)).encode("utf-8")


def main() -> int:
    out_root = Path(os.environ.get("OUT_DIR", "/app/exports")).resolve()
    out_dir = out_root / "gwas"
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = out_root / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)

    phenotype_field = os.environ.get("PHENOTYPE_FIELD", "total_root_volume")
    max_markers = int(os.environ.get("MAX_MARKERS", "2000"))
    min_n = int(os.environ.get("MIN_N", "20"))
    analysis_run_id = (os.environ.get("ANALYSIS_RUN_ID", "") or "").strip() or None

    methods = ["mlm", "farmcpu"]
    p_adjusts = ["none", "bh", "bonferroni"]

    png_only = os.environ.get("PNG_ONLY", "").strip().lower() in {"1", "true", "yes"}
    if png_only:
        phen_path = "/app/data/subterra.sqlite3"
        geno_path = "/app/data/subterra_genotype.sqlite3"

        # Trait values on mapped biological IDs.
        con = sqlite3.connect(phen_path)
        cur = con.cursor()
        if analysis_run_id:
            cur.execute(
                f"""
                SELECT m.sample_id, a.{phenotype_field}
                FROM analyses a
                JOIN phenotype_sample_id_map m ON m.filename = a.filename
                WHERE m.sample_id IS NOT NULL AND m.sample_id != '' AND a.{phenotype_field} IS NOT NULL
                  AND json_extract(a.extra, '$.meta.run_id') = ?
                """,
                (analysis_run_id,),
            )
        else:
            cur.execute(
                f"""
                SELECT m.sample_id, a.{phenotype_field}
                FROM analyses a
                JOIN phenotype_sample_id_map m ON m.filename = a.filename
                WHERE m.sample_id IS NOT NULL AND m.sample_id != '' AND a.{phenotype_field} IS NOT NULL
                """
            )
        rows = cur.fetchall()
        trait_vals = [float(v) for (_sid, v) in rows if v is not None]

        # Join summary.
        if analysis_run_id:
            cur.execute(
                f"""
                SELECT COUNT(DISTINCT filename)
                FROM analyses
                WHERE {phenotype_field} IS NOT NULL
                  AND json_extract(extra, '$.meta.run_id') = ?
                """,
                (analysis_run_id,),
            )
            phen_total = int(cur.fetchone()[0] or 0)
            cur.execute(
                """
                SELECT COUNT(*)
                FROM phenotype_sample_id_map m
                JOIN analyses a ON a.filename = m.filename
                WHERE m.sample_id IS NOT NULL AND m.sample_id != ''
                  AND json_extract(a.extra, '$.meta.run_id') = ?
                """,
                (analysis_run_id,),
            )
            mapped = int(cur.fetchone()[0] or 0)
            cur.execute(
                """
                SELECT m.sample_id
                FROM phenotype_sample_id_map m
                JOIN analyses a ON a.filename = m.filename
                WHERE m.sample_id IS NOT NULL AND m.sample_id != ''
                  AND json_extract(a.extra, '$.meta.run_id') = ?
                """,
                (analysis_run_id,),
            )
        else:
            cur.execute(f"SELECT COUNT(DISTINCT filename) FROM analyses WHERE {phenotype_field} IS NOT NULL")
            phen_total = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT COUNT(*) FROM phenotype_sample_id_map WHERE sample_id IS NOT NULL AND sample_id != ''")
            mapped = int(cur.fetchone()[0] or 0)
            cur.execute("SELECT sample_id FROM phenotype_sample_id_map WHERE sample_id IS NOT NULL AND sample_id != ''")
        mapped_sids = [r[0] for r in cur.fetchall()]
        con.close()

        # Overlap with genotype values (real markers).
        gcon = sqlite3.connect(geno_path)
        gcur = gcon.cursor()
        gcur.execute("SELECT id FROM genotype_markers WHERE name NOT LIKE 'demo_%'")
        mids = [int(r[0]) for r in gcur.fetchall()]
        overlap = 0
        if mids and mapped_sids:
            mids_q = ",".join(str(m) for m in mids[: min(len(mids), 3000)])
            chunk = 450
            seen = set()
            for i in range(0, len(mapped_sids), chunk):
                part = mapped_sids[i : i + chunk]
                qmarks = ",".join(["?"] * len(part))
                gcur.execute(
                    f"SELECT DISTINCT sample_id FROM genotype_values WHERE marker_id IN ({mids_q}) AND sample_id IN ({qmarks})",
                    tuple(part),
                )
                for (sid,) in gcur.fetchall():
                    seen.add(sid)
            overlap = len(seen)

        # Association results (MLM/FarmCPU with FDR/BH).
        req = app.MappingRunRequest(
            phenotype_field=phenotype_field,
            method="mlm",
            p_adjust="bh",
            max_markers=max_markers,
            min_n=min_n,
            analysis_run_id=analysis_run_id,
        )
        with app.SessionLocal() as db:
            mlm_rows, _m_tested, _n = app._mapping_compute(db, req)  # type: ignore[attr-defined]
        req = app.MappingRunRequest(
            phenotype_field=phenotype_field,
            method="farmcpu",
            p_adjust="bh",
            max_markers=max_markers,
            min_n=min_n,
            analysis_run_id=analysis_run_id,
        )
        with app.SessionLocal() as db:
            farm_rows, _m_tested, _n = app._mapping_compute(db, req)  # type: ignore[attr-defined]

        # Top marker effect groups.
        top_marker = farm_rows[0].marker_name if farm_rows else ""
        groups: dict[int, list[float]] = {}
        if top_marker:
            gcur.execute("SELECT id FROM genotype_markers WHERE name = ?", (top_marker,))
            r = gcur.fetchone()
            if r is not None:
                marker_id = int(r[0])
                phen_map = {sid: float(v) for (sid, v) in rows if sid and v is not None}
                sample_ids = list(phen_map.keys())
                chunk = 450
                for i in range(0, len(sample_ids), chunk):
                    part = sample_ids[i : i + chunk]
                    qmarks = ",".join(["?"] * len(part))
                    gcur.execute(
                        f"SELECT sample_id, value FROM genotype_values WHERE marker_id = ? AND sample_id IN ({qmarks})",
                        (marker_id, *part),
                    )
                    for sid, gv in gcur.fetchall():
                        if sid not in phen_map:
                            continue
                        try:
                            code = int(round(float(gv)))
                        except Exception:
                            continue
                        groups.setdefault(code, []).append(float(phen_map[sid]))

        # Multiple-testing summary (FarmCPU).
        sig_counts: dict[str, int] = {}
        for adj in ["none", "bh", "bonferroni"]:
            req = app.MappingRunRequest(
                phenotype_field=phenotype_field,
                method="farmcpu",
                p_adjust=adj,
                max_markers=max_markers,
                min_n=min_n,
                analysis_run_id=analysis_run_id,
            )
            with app.SessionLocal() as db:
                mrows, _m_tested, _n = app._mapping_compute(db, req)  # type: ignore[attr-defined]
            if adj == "none":
                sig_counts[adj] = sum(1 for r in mrows if r.p_value is not None and float(r.p_value) <= 0.05)
            else:
                sig_counts[adj] = sum(1 for r in mrows if r.p_adjusted is not None and float(r.p_adjusted) <= 0.05)

        gcon.close()

        out_png = paper_dir / "figure5_case_study.png"
        _save_figure5_png(
            out_png,
            phenotype_field=phenotype_field,
            trait_vals=trait_vals,
            phen_total=phen_total,
            mapped=mapped,
            overlap=overlap,
            mlm_rows=mlm_rows,
            farm_rows=farm_rows,
            top_marker=top_marker,
            groups=groups,
            sig_counts=sig_counts,
        )
        print(f"[export] wrote: {out_png}")
        print(f"[export] wrote: {paper_dir}")
        return 0

    for method in methods:
        for p_adjust in p_adjusts:
            tag = f"{phenotype_field}_{method}_{p_adjust}"
            print(f"[export] {tag}")

            svg = app.mapping_plot_svg(
                phenotype_field=phenotype_field,
                method=method,
                p_adjust=p_adjust,
                max_markers=max_markers,
                min_n=min_n,
                analysis_run_id=analysis_run_id,
                width=int(os.environ.get("PLOT_WIDTH", "1600")),
                height=int(os.environ.get("PLOT_HEIGHT", "700")),
            )
            csvr = app.mapping_results_csv(
                phenotype_field=phenotype_field,
                method=method,
                p_adjust=p_adjust,
                max_markers=max_markers,
                min_n=min_n,
                analysis_run_id=analysis_run_id,
            )

            _write_response(svg, out_dir / f"gwas_{tag}.svg")
            _write_response(csvr, out_dir / f"gwas_{tag}.csv")

    # Case study (Figure 5-style) plots.
    # 1) Trait distribution on mapped biological IDs.
    phen_path = "/app/data/subterra.sqlite3"
    con = sqlite3.connect(phen_path)
    cur = con.cursor()
    if analysis_run_id:
        cur.execute(
            f"""
            SELECT m.sample_id, a.{phenotype_field}
            FROM analyses a
            JOIN phenotype_sample_id_map m ON m.filename = a.filename
            WHERE m.sample_id IS NOT NULL AND m.sample_id != '' AND a.{phenotype_field} IS NOT NULL
              AND json_extract(a.extra, '$.meta.run_id') = ?
            """,
            (analysis_run_id,),
        )
    else:
        cur.execute(
            f"""
            SELECT m.sample_id, a.{phenotype_field}
            FROM analyses a
            JOIN phenotype_sample_id_map m ON m.filename = a.filename
            WHERE m.sample_id IS NOT NULL AND m.sample_id != '' AND a.{phenotype_field} IS NOT NULL
            """
        )
    rows = cur.fetchall()
    con.close()
    trait_vals = [float(v) for (_sid, v) in rows if v is not None]
    (paper_dir / f"fig5a_{phenotype_field}_distribution.svg").write_bytes(
        _svg_trait_histogram(
            trait_vals,
            title="Trait distribution",
            xlabel=phenotype_field,
            width=950,
            height=560,
        )
    )

    # 2) Join/overlap summary.
    # phenotype total: distinct filenames with non-null trait
    con = sqlite3.connect("/app/data/subterra.sqlite3")
    cur = con.cursor()
    if analysis_run_id:
        cur.execute(
            f"""
            SELECT COUNT(DISTINCT filename)
            FROM analyses
            WHERE {phenotype_field} IS NOT NULL
              AND json_extract(extra, '$.meta.run_id') = ?
            """,
            (analysis_run_id,),
        )
        phen_total = int(cur.fetchone()[0] or 0)
        cur.execute(
            """
            SELECT COUNT(*)
            FROM phenotype_sample_id_map m
            JOIN analyses a ON a.filename = m.filename
            WHERE m.sample_id IS NOT NULL AND m.sample_id != ''
              AND json_extract(a.extra, '$.meta.run_id') = ?
            """,
            (analysis_run_id,),
        )
        mapped = int(cur.fetchone()[0] or 0)
        # overlap: mapped sample_ids that exist in genotype_values for real markers
        cur.execute(
            """
            SELECT m.sample_id
            FROM phenotype_sample_id_map m
            JOIN analyses a ON a.filename = m.filename
            WHERE m.sample_id IS NOT NULL AND m.sample_id != ''
              AND json_extract(a.extra, '$.meta.run_id') = ?
            """,
            (analysis_run_id,),
        )
    else:
        cur.execute(f"SELECT COUNT(DISTINCT filename) FROM analyses WHERE {phenotype_field} IS NOT NULL")
        phen_total = int(cur.fetchone()[0] or 0)
        cur.execute("SELECT COUNT(*) FROM phenotype_sample_id_map WHERE sample_id IS NOT NULL AND sample_id != ''")
        mapped = int(cur.fetchone()[0] or 0)
        # overlap: mapped sample_ids that exist in genotype_values for real markers
        cur.execute("SELECT sample_id FROM phenotype_sample_id_map WHERE sample_id IS NOT NULL AND sample_id != ''")
    mapped_sids = [r[0] for r in cur.fetchall()]
    con.close()
    gcon = sqlite3.connect("/app/data/subterra_genotype.sqlite3")
    gcur = gcon.cursor()
    # Real marker ids
    gcur.execute("SELECT id FROM genotype_markers WHERE name NOT LIKE 'demo_%'")
    mids = [int(r[0]) for r in gcur.fetchall()]
    overlap = 0
    if mids and mapped_sids:
        # chunked IN query on sample_ids
        mids_q = ",".join(str(m) for m in mids[: min(len(mids), 3000)])
        chunk = 450
        seen = set()
        for i in range(0, len(mapped_sids), chunk):
            part = mapped_sids[i : i + chunk]
            qmarks = ",".join(["?"] * len(part))
            gcur.execute(
                f"SELECT DISTINCT sample_id FROM genotype_values WHERE marker_id IN ({mids_q}) AND sample_id IN ({qmarks})",
                tuple(part),
            )
            for (sid,) in gcur.fetchall():
                if sid not in seen:
                    seen.add(sid)
        overlap = len(seen)
    gcon.close()
    (paper_dir / "fig5b_id_join_summary.svg").write_bytes(
        _svg_join_summary(
            phen_total=phen_total,
            mapped=mapped,
            overlap=overlap,
            title="Sample ID join and overlap",
            width=950,
            height=560,
        )
    )

    # 3) Main GWAS plots for FDR (BH) as primary (regenerated with consistent titles).
    (paper_dir / "fig5c_gwas_mlm_fdr.svg").write_bytes(
        app.mapping_plot_svg(
            phenotype_field=phenotype_field,
            method="mlm",
            p_adjust="bh",
            max_markers=max_markers,
            min_n=min_n,
            analysis_run_id=analysis_run_id,
            width=1200,
            height=560,
            title="Marker–trait association (MLM, FDR/BH)",
        ).body
    )
    (paper_dir / "fig5d_gwas_farmcpu_fdr.svg").write_bytes(
        app.mapping_plot_svg(
            phenotype_field=phenotype_field,
            method="farmcpu",
            p_adjust="bh",
            max_markers=max_markers,
            min_n=min_n,
            analysis_run_id=analysis_run_id,
            width=1200,
            height=560,
            title="Marker–trait association (FarmCPU, FDR/BH)",
        ).body
    )

    # 4) Top-hit genotype effect plot (FarmCPU + FDR by default; fall back to MLM).
    best_method = "farmcpu"
    req = app.MappingRunRequest(
        phenotype_field=phenotype_field,
        method=best_method,
        p_adjust="bh",
        max_markers=max_markers,
        min_n=min_n,
        analysis_run_id=analysis_run_id,
    )
    with app.SessionLocal() as db:
        mapping_rows, _m_tested, _n = app._mapping_compute(db, req)  # type: ignore[attr-defined]
    if not mapping_rows:
        # fallback
        best_method = "mlm"
        req = app.MappingRunRequest(
            phenotype_field=phenotype_field,
            method=best_method,
            p_adjust="bh",
            max_markers=max_markers,
            min_n=min_n,
            analysis_run_id=analysis_run_id,
        )
        with app.SessionLocal() as db:
            mapping_rows, _m_tested, _n = app._mapping_compute(db, req)  # type: ignore[attr-defined]
    if mapping_rows:
        top = mapping_rows[0]
        top_marker = top.marker_name
        # pull genotype values for mapped sample_ids
        geno_con = sqlite3.connect("/app/data/subterra_genotype.sqlite3")
        gcur = geno_con.cursor()
        gcur.execute("SELECT id FROM genotype_markers WHERE name = ?", (top_marker,))
        r = gcur.fetchone()
        if r is not None:
            marker_id = int(r[0])
            # map sample_id -> phenotype value
            phen_map = {sid: float(v) for (sid, v) in rows if sid and v is not None}
            sample_ids = list(phen_map.keys())
            groups: dict[int, list[float]] = {}
            # query in chunks to avoid sqlite parameter limits
            chunk = 450
            for i in range(0, len(sample_ids), chunk):
                part = sample_ids[i : i + chunk]
                qmarks = ",".join(["?"] * len(part))
                gcur.execute(
                    f"SELECT sample_id, value FROM genotype_values WHERE marker_id = ? AND sample_id IN ({qmarks})",
                    (marker_id, *part),
                )
                for sid, gv in gcur.fetchall():
                    if sid not in phen_map:
                        continue
                    try:
                        code = int(round(float(gv)))
                    except Exception:
                        continue
                    groups.setdefault(code, []).append(float(phen_map[sid]))
            geno_con.close()

            (paper_dir / "fig5d_top_marker_effect.svg").write_bytes(
                _svg_genotype_effect(
                    groups,
                    title=f"Top marker effect ({best_method.upper()} FDR)",
                    xlabel="marker genotype (0/1)",
                    ylabel=phenotype_field,
                    width=950,
                    height=560,
                )
            )

    # 5) Multiple-testing stringency summary (counts of significant markers).
    sig_counts: dict[str, int] = {}
    for adj in ["none", "bh", "bonferroni"]:
        req = app.MappingRunRequest(
            phenotype_field=phenotype_field,
            method="farmcpu",
            p_adjust=adj,
            max_markers=max_markers,
            min_n=min_n,
            analysis_run_id=analysis_run_id,
        )
        with app.SessionLocal() as db:
            mrows, m_tested, _n = app._mapping_compute(db, req)  # type: ignore[attr-defined]
        if adj == "none":
            sig_counts[adj] = sum(1 for r in mrows if r.p_value is not None and float(r.p_value) <= 0.05)
        else:
            sig_counts[adj] = sum(1 for r in mrows if r.p_adjusted is not None and float(r.p_adjusted) <= 0.05)
    (paper_dir / "fig5f_multiple_testing_summary.svg").write_bytes(
        _svg_significance_summary(sig_counts, title="Multiple-testing correction (FarmCPU)", width=950, height=560)
    )

    # 6) Assemble 2-column multi-panel grids (4-panel and 6-panel).
    def _read(p: Path) -> bytes:
        return p.read_bytes()
    grid6 = [
        ("A", _read(paper_dir / f"fig5a_{phenotype_field}_distribution.svg")),
        ("B", _read(paper_dir / "fig5b_id_join_summary.svg")),
        ("C", _read(paper_dir / "fig5c_gwas_mlm_fdr.svg")),
        ("D", _read(paper_dir / "fig5d_gwas_farmcpu_fdr.svg")),
        ("E", _read(paper_dir / "fig5d_top_marker_effect.svg")),
        ("F", _read(paper_dir / "fig5f_multiple_testing_summary.svg")),
    ]
    (paper_dir / "fig5_case_study_grid_6panel.svg").write_bytes(
        _compose_svg_grid(grid6, out_w=2000, out_h=2100, cols=2, title="Figure 5 — Genotype association case study")
    )
    grid4 = [
        ("A", _read(paper_dir / f"fig5a_{phenotype_field}_distribution.svg")),
        ("B", _read(paper_dir / "fig5c_gwas_mlm_fdr.svg")),
        ("C", _read(paper_dir / "fig5d_gwas_farmcpu_fdr.svg")),
        ("D", _read(paper_dir / "fig5d_top_marker_effect.svg")),
    ]
    (paper_dir / "fig5_case_study_grid_4panel.svg").write_bytes(
        _compose_svg_grid(grid4, out_w=2000, out_h=1450, cols=2, title="Figure 5 — Genotype association case study")
    )

    # Copy the full GWAS plot set for easy manuscript linking.
    for p in out_dir.glob(f"gwas_{phenotype_field}_*.svg"):
        (paper_dir / p.name).write_bytes(p.read_bytes())
    for p in out_dir.glob(f"gwas_{phenotype_field}_*.csv"):
        (paper_dir / p.name).write_bytes(p.read_bytes())

    print(f"[export] wrote: {out_dir}")
    print(f"[export] wrote: {paper_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
