from __future__ import annotations

import csv
import gzip
import io
import math
import os
import random
import json
import re
import zipfile
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np

try:
    import scipy.stats as scipy_stats  # type: ignore
except Exception as e:  # pragma: no cover
    scipy_stats = None  # type: ignore[assignment]
    _SCIPY_IMPORT_ERROR = e
from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/subterra_genotype.sqlite3")
if DATABASE_URL.startswith("sqlite:"):
    Path("data").mkdir(parents=True, exist_ok=True)

PHENOTYPE_DATABASE_URL = os.getenv("PHENOTYPE_DATABASE_URL", "sqlite:///data/subterra.sqlite3")
if PHENOTYPE_DATABASE_URL.startswith("sqlite:"):
    Path("data").mkdir(parents=True, exist_ok=True)

MARKER_SOURCE_DIRS = os.getenv("MARKER_SOURCE_DIRS", "data/markers,genotype_data,../data/markers,../genotype_data")


def _connect_args(url: str) -> dict:
    if url.startswith("sqlite:"):
        return {"check_same_thread": False}
    return {}


engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args(DATABASE_URL))
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

phenotype_engine = create_engine(PHENOTYPE_DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args(PHENOTYPE_DATABASE_URL))
PhenotypeSessionLocal = sessionmaker(bind=phenotype_engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class GenotypeMarker(Base):
    __tablename__ = "genotype_markers"
    __table_args__ = (UniqueConstraint("name", name="uq_genotype_markers_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


class GenotypeSample(Base):
    __tablename__ = "genotype_samples"
    __table_args__ = (UniqueConstraint("sample_id", name="uq_genotype_samples_sample_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


class GenotypeValue(Base):
    __tablename__ = "genotype_values"
    __table_args__ = (UniqueConstraint("sample_id", "marker_id", name="uq_genotype_values_sample_marker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(String(512), nullable=False)
    marker_id: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)


class MappingResult(Base):
    __tablename__ = "mapping_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    phenotype_field: Mapped[str] = mapped_column(String(128), nullable=False)
    marker_name: Mapped[str] = mapped_column(String(256), nullable=False)
    n: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pearson_r: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class MappingRun(Base):
    __tablename__ = "mapping_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    phenotype_field: Mapped[str] = mapped_column(String(128), nullable=False)
    analysis_run_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    p_adjust: Mapped[str] = mapped_column(String(32), nullable=False)
    max_markers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    min_n: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    n_markers_tested: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_results: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class MappingHit(Base):
    __tablename__ = "mapping_hits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    run_id: Mapped[int] = mapped_column(Integer, nullable=False)
    marker_name: Mapped[str] = mapped_column(String(256), nullable=False)
    phenotype_field: Mapped[str] = mapped_column(String(128), nullable=False)
    analysis_run_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    p_adjust: Mapped[str] = mapped_column(String(32), nullable=False)
    n: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    effect: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    p_value: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    p_adjusted: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    r2: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


app = FastAPI(
    title="Subterra Genotype Mapping Service",
    description="Stores genetic marker tables and maps phenotype metrics to markers",
    version="0.1.0",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    Base.metadata.create_all(bind=engine)
    # Best-effort migrations for existing DBs.
    try:
        from sqlalchemy import inspect

        insp = inspect(engine)
        if "mapping_runs" in insp.get_table_names():
            cols = {c["name"] for c in insp.get_columns("mapping_runs")}
            if "analysis_run_id" not in cols:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE mapping_runs ADD COLUMN analysis_run_id VARCHAR(128) NOT NULL DEFAULT ''"))
        if "mapping_hits" in insp.get_table_names():
            cols = {c["name"] for c in insp.get_columns("mapping_hits")}
            if "analysis_run_id" not in cols:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE mapping_hits ADD COLUMN analysis_run_id VARCHAR(128) NOT NULL DEFAULT ''"))
    except Exception:
        pass


def _csv_response(filename: str, headers: List[str], rows: List[dict]) -> Response:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in headers})
    return Response(
        content=buf.getvalue().encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class HealthResponse(BaseModel):
    status: str
    database_url: str


@app.get("/")
def root():
    # Friendly entrypoint when accessed behind nginx prefixes (e.g. /genotype-api/).
    return {
        "service": "subterra-genotype",
        "status": "ok",
        "endpoints": [
            "/health",
            "/stats",
            "/markers/upload",
            "/markers/ingest",
            "/markers/list",
            "/markers/example.csv",
            "/mapping/run",
            "/mapping/history",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", database_url=DATABASE_URL)


class UploadMarkersResponse(BaseModel):
    samples_upserted: int
    markers_upserted: int
    values_upserted: int


class IngestMarkersRequest(BaseModel):
    name: str
    source_dir: Optional[str] = None
    max_markers: int = 5000
    every: int = 100
    replace_existing: bool = False


class IngestMarkersResponse(UploadMarkersResponse):
    source_file: str
    source_dir: str
    source_format: str


def _marker_source_dirs() -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for raw in MARKER_SOURCE_DIRS.split(","):
        token = raw.strip()
        if not token:
            continue
        p = Path(token).resolve()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _classify_marker_file(name: str) -> Tuple[str, bool, bool, str]:
    lower = name.lower()
    if lower.endswith(".csv"):
        return "csv", True, True, ""
    if lower.endswith(".vcf") or lower.endswith(".vcf.gz"):
        return "vcf", False, True, ""
    if (
        lower.endswith(".tsv")
        or lower.endswith(".tsv.gz")
        or lower.endswith(".hmp")
        or lower.endswith(".hmp.gz")
        or lower.endswith(".hmp.txt")
        or lower.endswith(".hapmap")
    ):
        return "hapmap", False, True, ""
    if lower.endswith(".vep"):
        return (
            "vep",
            False,
            False,
            "VEP annotation files do not contain per-sample genotypes; use VCF/HapMap/CSV genotype matrices.",
        )
    return "unknown", False, False, "Unsupported marker file type. Use CSV, VCF(.gz), or HapMap TSV(.gz)."


def _classify_marker_path(path: Path) -> Tuple[str, bool, bool, str]:
    fmt, uploadable, ingestable, reason = _classify_marker_file(path.name)
    if fmt != "csv":
        return fmt, uploadable, ingestable, reason

    # Lightweight CSV sniffing to hide helper/metadata tables from marker ingest pickers.
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except Exception:
        return fmt, uploadable, ingestable, reason

    h = [(c or "").strip().lower() for c in header]
    if not h:
        return fmt, uploadable, False, "CSV has no header"

    if ("tube id" in h or "tube_id" in h) and "genotype" in h:
        return fmt, uploadable, False, "Tube/Genotype lookup CSV (use in Phenotyping, not marker ingest)."
    if "filename" in h and "sample_id" in h and len(h) <= 8:
        return fmt, uploadable, False, "Phenotype sample-id map CSV (not a marker matrix)."
    return fmt, uploadable, ingestable, reason


def _resolve_marker_file(name: str, source_dir: Optional[str] = None) -> Tuple[Path, str]:
    safe_name = (name or "").strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="name is required")
    if Path(safe_name).name != safe_name:
        raise HTTPException(status_code=400, detail="invalid file name")

    roots = _marker_source_dirs()
    if source_dir:
        source_norm = str(Path(source_dir).resolve())
        roots = [p for p in roots if str(p) == source_norm]
        if not roots:
            raise HTTPException(status_code=400, detail="invalid source_dir")

    for root in roots:
        path = (root / safe_name).resolve()
        if root not in path.parents:
            continue
        if path.exists() and path.is_file():
            return path, str(root)
    raise HTTPException(status_code=404, detail="file not found")


def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _parse_numeric(raw: str) -> Optional[float]:
    token = (raw or "").strip()
    if token == "" or token.lower() in {"na", "nan", "n", "."}:
        return None
    try:
        v = float(token)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def _make_unique_marker_names(marker_names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for name in marker_names:
        base = (name or "").strip() or "marker"
        n = seen.get(base, 0) + 1
        seen[base] = n
        out.append(base if n == 1 else f"{base}__{n}")
    return out


def _parse_marker_csv(contents: bytes) -> Tuple[List[str], List[Tuple[str, List[Optional[float]]]]]:
    text_data = contents.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text_data))
    if not reader.fieldnames:
        raise ValueError("CSV has no header")
    fieldnames = [f.strip() for f in reader.fieldnames if f is not None]
    fieldnames_l = [f.lower() for f in fieldnames]
    if ("tube id" in fieldnames_l or "tube_id" in fieldnames_l) and "genotype" in fieldnames_l and len(fieldnames) <= 4:
        raise ValueError(
            "Detected a Tube/Genotype lookup CSV. Use this in Phenotyping (Tube→Genotype map), not marker upload."
        )
    if len(fieldnames) < 2:
        raise ValueError("CSV must contain at least: sample_id, <marker1>, ...")
    if fieldnames[0].lower() not in {"sample_id", "sample", "id", "filename"}:
        raise ValueError("First column must be sample_id (or sample/id/filename)")
    marker_names = _make_unique_marker_names(fieldnames[1:])
    rows: List[Tuple[str, List[Optional[float]]]] = []
    for r in reader:
        if not r:
            continue
        clean = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        sid = clean.get(fieldnames[0], "")
        if not sid:
            continue
        vals = [_parse_numeric(clean.get(src_name, "")) for src_name in fieldnames[1:]]
        rows.append((sid, vals))
    return marker_names, rows


def _decode_hapmap_call(raw: str) -> Optional[Tuple[str, str]]:
    """
    Decode HapMap genotype call to two alleles.
    Handles common forms: A, AA, A/T, AT, and IUPAC heterozygous codes.
    """
    token = (raw or "").strip().upper()
    if not token or token in {"N", "NN", ".", "./.", ".|.", "-"}:
        return None
    if "/" in token or "|" in token:
        parts = [p for p in re.split(r"[\/|]", token) if p]
        if len(parts) == 2 and all(p in {"A", "C", "G", "T"} for p in parts):
            return parts[0], parts[1]
        return None
    if len(token) == 1:
        if token in {"A", "C", "G", "T"}:
            return token, token
        iupac = {
            "R": ("A", "G"),
            "Y": ("C", "T"),
            "S": ("G", "C"),
            "W": ("A", "T"),
            "K": ("G", "T"),
            "M": ("A", "C"),
        }
        return iupac.get(token)
    if len(token) == 2 and all(c in {"A", "C", "G", "T"} for c in token):
        return token[0], token[1]
    return None


def _hapmap_dosage(raw: str, alt: Optional[str]) -> Optional[float]:
    call = _decode_hapmap_call(raw)
    if call is None:
        return None
    if not alt:
        return 0.0
    a1, a2 = call
    return float((1 if a1 == alt else 0) + (1 if a2 == alt else 0))


def _parse_hapmap(path: Path, *, every: int, max_markers: int) -> Tuple[List[str], List[Tuple[str, List[Optional[float]]]]]:
    try:
        with _open_text(path) as f:
            reader = csv.reader(f, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("HapMap file is empty")
            if len(header) < 12:
                raise ValueError("Expected HapMap TSV with 11 metadata columns plus sample columns")

            sample_cols: List[Tuple[int, str]] = []
            for idx, sid in enumerate(header[11:], start=11):
                sid2 = (sid or "").strip()
                if sid2:
                    sample_cols.append((idx, sid2))
            if not sample_cols:
                raise ValueError("No sample columns found in HapMap file")

            rows_by_sample: Dict[str, List[Optional[float]]] = {sid: [] for _idx, sid in sample_cols}
            marker_names: List[str] = []
            stride = max(1, int(every))
            max_m = max(1, int(max_markers))

            for i, row in enumerate(reader):
                if len(marker_names) >= max_m:
                    break
                if (i % stride) != 0:
                    continue
                if len(row) <= sample_cols[-1][0]:
                    continue

                rs = (row[0] or "").strip() or f"marker{i+1}"
                chr_ = (row[2] or "").strip() or "NA"
                bp = (row[3] or "").strip() or str(i + 1)
                marker_names.append(f"{rs}_{chr_}_{bp}")

                counts: Dict[str, int] = {}
                geno_calls: Dict[str, str] = {}
                for col_idx, sid in sample_cols:
                    c = (row[col_idx] if col_idx < len(row) else "").strip().upper()
                    geno_calls[sid] = c
                    call = _decode_hapmap_call(c)
                    if call is None:
                        continue
                    a1, a2 = call
                    counts[a1] = counts.get(a1, 0) + 1
                    counts[a2] = counts.get(a2, 0) + 1

                ref = max(counts, key=counts.get) if counts else None
                alt = None
                if ref:
                    alts = sorted(
                        [c for c, n in counts.items() if c != ref and n > 0],
                        key=lambda c: counts[c],
                        reverse=True,
                    )
                    alt = alts[0] if alts else None

                for _col_idx, sid in sample_cols:
                    rows_by_sample[sid].append(_hapmap_dosage(geno_calls.get(sid, ""), alt))
    except EOFError:
        raise ValueError(f"File appears truncated/corrupted: {path.name}")

    if not marker_names:
        raise ValueError(f"HapMap has no marker rows after filtering: {path.name}")
    marker_names = _make_unique_marker_names(marker_names)
    rows = [(sid, rows_by_sample[sid]) for _idx, sid in sample_cols]
    return marker_names, rows


def _encode_vcf_gt(gt: str) -> Optional[float]:
    token = (gt or "").strip()
    if not token or token in {".", "./.", ".|."}:
        return None
    alleles = re.split(r"[\/|]", token)
    if not alleles or any(a in {"", "."} for a in alleles):
        return None
    total = 0
    for a in alleles:
        if not a.isdigit():
            return None
        total += int(a)
    return float(total)


def _parse_vcf(path: Path, *, every: int, max_markers: int) -> Tuple[List[str], List[Tuple[str, List[Optional[float]]]]]:
    sample_ids: List[str] = []
    rows_by_sample: Dict[str, List[Optional[float]]] = {}
    marker_names: List[str] = []
    stride = max(1, int(every))
    max_m = max(1, int(max_markers))
    n_seen = 0

    try:
        with _open_text(path) as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line:
                    continue
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    cols = line.split("\t")
                    if len(cols) < 10:
                        raise ValueError("VCF has no sample columns")
                    sample_ids = [s.strip() for s in cols[9:] if s.strip()]
                    if not sample_ids:
                        raise ValueError("VCF sample IDs are empty")
                    rows_by_sample = {sid: [] for sid in sample_ids}
                    continue
                if line.startswith("#"):
                    continue
                if not sample_ids:
                    raise ValueError("Invalid VCF: missing #CHROM header")

                if len(marker_names) >= max_m:
                    break
                if (n_seen % stride) != 0:
                    n_seen += 1
                    continue
                n_seen += 1

                cols = line.split("\t")
                if len(cols) < 10:
                    continue
                chrom = (cols[0] or "").strip()
                pos = (cols[1] or "").strip()
                ref = (cols[3] or "").strip()
                alt = ((cols[4] or "").strip().split(",")[0] if cols[4] else "")
                marker_names.append(f"{chrom}_{pos}_{ref}_{alt}")

                fmt = cols[8].split(":")
                gt_idx = fmt.index("GT") if "GT" in fmt else 0
                for j, sid in enumerate(sample_ids):
                    col_idx = 9 + j
                    sval = cols[col_idx] if col_idx < len(cols) else ""
                    parts = sval.split(":")
                    gt = parts[gt_idx] if gt_idx < len(parts) else (parts[0] if parts else "")
                    rows_by_sample[sid].append(_encode_vcf_gt(gt))
    except EOFError:
        raise ValueError(f"File appears truncated/corrupted: {path.name}")

    if not marker_names:
        raise ValueError(f"VCF has no variant rows after filtering: {path.name}")
    marker_names = _make_unique_marker_names(marker_names)
    rows = [(sid, rows_by_sample[sid]) for sid in sample_ids]
    return marker_names, rows


def _upsert_markers_matrix(
    marker_names: List[str],
    rows: List[Tuple[str, List[Optional[float]]]],
    *,
    replace_existing: bool = False,
) -> UploadMarkersResponse:
    if not marker_names:
        raise ValueError("No marker columns found")
    if not rows:
        raise ValueError("No marker rows found")

    rows2: List[Tuple[str, List[Optional[float]]]] = []
    for sid, vals in rows:
        sid2 = (sid or "").strip()
        if not sid2:
            continue
        if len(vals) < len(marker_names):
            vals = vals + [None] * (len(marker_names) - len(vals))
        elif len(vals) > len(marker_names):
            vals = vals[: len(marker_names)]
        rows2.append((sid2, vals))
    if not rows2:
        raise ValueError("No non-empty sample IDs found in marker rows")
    non_missing_values = sum(1 for _sid, vals in rows2 for v in vals if v is not None)
    if non_missing_values == 0:
        raise ValueError(
            "No numeric marker values found. If this file is a Tube/Genotype lookup table, use it in the Phenotyping page."
        )

    sample_ids = sorted({sid for sid, _vals in rows2})

    with SessionLocal() as db:
        if replace_existing:
            db.execute(text("DELETE FROM genotype_values"))
            db.execute(text("DELETE FROM genotype_markers"))
            db.execute(text("DELETE FROM genotype_samples"))
            db.flush()

        existing_markers = {
            m.name: m
            for m in db.execute(select(GenotypeMarker).where(GenotypeMarker.name.in_(marker_names))).scalars().all()
        }
        markers_upserted = 0
        for name in marker_names:
            if name not in existing_markers:
                m = GenotypeMarker(name=name)
                db.add(m)
                existing_markers[name] = m
                markers_upserted += 1

        existing_samples = {
            s.sample_id: s
            for s in db.execute(select(GenotypeSample).where(GenotypeSample.sample_id.in_(sample_ids))).scalars().all()
        }
        samples_upserted = 0
        for sid in sample_ids:
            if sid not in existing_samples:
                s = GenotypeSample(sample_id=sid)
                db.add(s)
                existing_samples[sid] = s
                samples_upserted += 1

        db.flush()
        marker_ids: List[int] = []
        for name in marker_names:
            mid = existing_markers[name].id
            if mid is None:
                raise ValueError(f"Failed to resolve marker id for {name}")
            marker_ids.append(int(mid))

        values_upserted = 0
        for sid, vals in rows2:
            for j, val in enumerate(vals):
                if val is None:
                    continue
                db.execute(
                    text(
                        "INSERT INTO genotype_values (sample_id, marker_id, value, created_at) "
                        "VALUES (:sample_id, :marker_id, :value, :created_at) "
                        "ON CONFLICT (sample_id, marker_id) DO UPDATE SET value = EXCLUDED.value"
                    ),
                    {"sample_id": sid, "marker_id": int(marker_ids[j]), "value": float(val), "created_at": _utc_now()},
                )
                values_upserted += 1

        db.commit()

    return UploadMarkersResponse(
        samples_upserted=samples_upserted,
        markers_upserted=markers_upserted,
        values_upserted=values_upserted,
    )


@app.post("/markers/upload", response_model=UploadMarkersResponse)
async def upload_markers(file: UploadFile = File(...), replace_existing: bool = False) -> UploadMarkersResponse:
    contents = await file.read()
    try:
        marker_names, rows = _parse_marker_csv(contents)
        return _upsert_markers_matrix(marker_names, rows, replace_existing=bool(replace_existing))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/markers/ingest", response_model=IngestMarkersResponse)
def ingest_markers(req: IngestMarkersRequest) -> IngestMarkersResponse:
    path, src = _resolve_marker_file(req.name, req.source_dir)
    fmt, _uploadable, ingestable, reason = _classify_marker_path(path)
    if not ingestable:
        detail = reason or f"File format '{fmt}' is not ingestable"
        raise HTTPException(status_code=400, detail=detail)

    try:
        if fmt == "csv":
            marker_names, rows = _parse_marker_csv(path.read_bytes())
        elif fmt == "hapmap":
            marker_names, rows = _parse_hapmap(path, every=req.every, max_markers=req.max_markers)
        elif fmt == "vcf":
            marker_names, rows = _parse_vcf(path, every=req.every, max_markers=req.max_markers)
        else:
            raise ValueError(f"Unsupported ingest format: {fmt}")
        out = _upsert_markers_matrix(marker_names, rows, replace_existing=bool(req.replace_existing))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return IngestMarkersResponse(
        source_file=path.name,
        source_dir=src,
        source_format=fmt,
        samples_upserted=out.samples_upserted,
        markers_upserted=out.markers_upserted,
        values_upserted=out.values_upserted,
    )


@app.get("/markers/example.csv")
def download_example_markers() -> Response:
    path = Path("data/markers/arabidopsis_magic_chr1_subset.csv")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Example marker CSV not found on server")
    data = path.read_bytes()
    return Response(
        content=data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="arabidopsis_magic_chr1_subset.csv"'},
    )


@app.get("/markers/list")
def list_marker_files() -> dict:
    """
    List marker-related files bundled on the server.
    Source dirs are configured by MARKER_SOURCE_DIRS
    (default: data/markers,genotype_data,../data/markers,../genotype_data).
    """
    files = []
    for root in _marker_source_dirs():
        if not root.exists():
            continue
        for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
            if not p.is_file():
                continue
            if p.name.startswith("."):
                continue
            try:
                st = p.stat()
            except Exception:
                continue
            fmt, uploadable, ingestable, reason = _classify_marker_path(p)
            files.append(
                {
                    "name": p.name,
                    "source_dir": str(root),
                    "format": fmt,
                    "bytes": int(st.st_size),
                    "modified_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                    "uploadable": uploadable,
                    "ingestable": ingestable,
                    "ingest_note": reason,
                }
            )
    return {"files": files}


@app.get("/markers/download")
def download_marker_file(name: str, source_dir: Optional[str] = None) -> Response:
    path, _src = _resolve_marker_file(name, source_dir)
    safe_name = path.name
    data = path.read_bytes()
    media = "text/csv; charset=utf-8" if safe_name.lower().endswith(".csv") else "application/octet-stream"
    return Response(content=data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{safe_name}"'})


class MappingRunRequest(BaseModel):
    phenotype_field: str = "total_root_length"
    method: str = "linear"  # pearson | linear | glm | mlm | farmcpu | anova | lod
    p_adjust: str = "bh"  # none | bonferroni | bh
    max_markers: int = 5000
    min_n: int = 6
    analysis_run_id: Optional[str] = None
    allow_filename_fallback: bool = False
    n_pcs: int = 3
    kinship_markers: int = 1000
    farmcpu_max_iters: int = 3
    farmcpu_max_qtn: int = 10
    farmcpu_seed: int = 7


class MappingRow(BaseModel):
    marker_name: str
    n: int
    effect: Optional[float] = None
    p_value: Optional[float] = None
    p_adjusted: Optional[float] = None
    r2: Optional[float] = None
    lod: Optional[float] = None


class MappingRunResponse(BaseModel):
    phenotype_field: str
    method: str
    p_adjust: str
    rows: List[MappingRow]


class MarkerEffectClass(BaseModel):
    genotype_class: str
    n: int
    mean: float
    median: float
    q1: float
    q3: float
    whisker_low: float
    whisker_high: float
    min: float
    max: float


class MarkerEffectResponse(BaseModel):
    marker_name: str
    phenotype_field: str
    n_samples: int
    classes: List[MarkerEffectClass]


def _parse_marker_locus(name: str) -> Optional[Tuple[int, int]]:
    """
    Best-effort parse of marker locus from common GWAS marker names.
    Returns (chrom, pos) if parseable.
    Examples handled:
      - 1s12222091_1_12222091  -> (1, 12222091)
      - Chr1_12222091          -> (1, 12222091)
      - 1:12222091             -> (1, 12222091)
      - 1_12222091             -> (1, 12222091)
    """
    s = (name or "").strip()
    if not s:
        return None
    # Prefer the first "chr,pos" pair if present in the string.
    m = re.search(r"(?:chr)?([0-9]{1,2})[^0-9]+([0-9]{3,})", s, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        chrom = int(m.group(1))
        pos = int(m.group(2))
    except Exception:
        return None
    if chrom <= 0 or pos <= 0:
        return None
    return chrom, pos


def _svg_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _mapping_compute(db: Session, req: MappingRunRequest) -> Tuple[List[MappingRow], int, int]:
    """
    Compute mapping results (full list), returning:
      (rows, n_markers_tested, n_samples_used)
    """
    try:
        phen = _load_phenotypes(
            req.phenotype_field,
            analysis_run_id=req.analysis_run_id,
            allow_filename_fallback=bool(req.allow_filename_fallback),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read phenotypes from DB: {e}")

    if not phen:
        raise HTTPException(status_code=400, detail="No phenotypes found in DB (run phenotyping first)")

    # Real markers only: exclude any historical demo/synthetic markers if present.
    marker_q = (
        select(GenotypeMarker)
        .where(~GenotypeMarker.name.like("demo_%"))
        .order_by(GenotypeMarker.id.asc())
    )
    marker_rows = db.execute(marker_q.limit(req.max_markers)).scalars().all()
    if not marker_rows:
        raise HTTPException(status_code=400, detail="No genotype markers found (upload/ingest marker matrix first)")

    if scipy_stats is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"scipy is required for mapping methods ({_SCIPY_IMPORT_ERROR})")

    method = req.method.lower().strip()
    if method not in {"pearson", "linear", "glm", "mlm", "farmcpu", "anova", "lod"}:
        raise HTTPException(
            status_code=400,
            detail="method must be one of: pearson, linear, glm, mlm, farmcpu, anova, lod",
        )
    p_adjust = req.p_adjust.lower().strip()
    if p_adjust not in {"none", "bonferroni", "bh"}:
        raise HTTPException(status_code=400, detail="p_adjust must be one of: none, bonferroni, bh")

    results: List[MappingRow] = []
    n_samples_used = 0

    if method in {"glm", "mlm", "farmcpu"}:
        sample_ids, G, marker_names = _build_genotype_matrix(db, marker_rows, phen, min_n=req.min_n)
        n_samples_used = int(len(sample_ids))
        y = np.array([float(phen[sid]) for sid in sample_ids], dtype=float)

        Gstd = _standardize_cols(G)
        pcs = _compute_pcs(Gstd, req.n_pcs, req.kinship_markers, seed=req.farmcpu_seed)
        cov = np.concatenate([np.ones((y.shape[0], 1), dtype=float), pcs], axis=1)

        use = min(Gstd.shape[1], max(10, int(req.kinship_markers)))
        # random subset for kinship to avoid ordering bias
        idx = list(range(Gstd.shape[1]))
        rng = random.Random(req.farmcpu_seed)
        rng.shuffle(idx)
        idx = idx[:use]
        K = (Gstd[:, idx] @ Gstd[:, idx].T) / float(use)

        if method == "glm":
            betas, pvals, r2s = _glm_scan(y, G, cov)
        elif method == "mlm":
            betas, pvals, r2s = _mlm_scan_emmax(y, G, cov, K)
        else:
            betas, pvals, r2s = _farmcpu_scan(
                y,
                G,
                cov,
                K,
                max_iters=int(req.farmcpu_max_iters),
                max_qtn=int(req.farmcpu_max_qtn),
            )

        for name, b, p, r2 in zip(marker_names, betas, pvals, r2s):
            if p is None or not math.isfinite(float(p)):
                continue
            results.append(MappingRow(marker_name=name, n=int(y.shape[0]), effect=b, p_value=float(p), r2=r2))

    else:
        # For each marker, gather sample overlap and compute association.
        for marker in marker_rows:
            values = db.execute(
                select(GenotypeValue.sample_id, GenotypeValue.value).where(GenotypeValue.marker_id == marker.id)
            ).all()
            xs: List[float] = []
            ys: List[float] = []
            for sid, x in values:
                if sid in phen:
                    xs.append(float(x))
                    ys.append(float(phen[sid]))
            if len(xs) < req.min_n:
                continue
            n_samples_used = max(n_samples_used, int(len(xs)))
            row: Optional[MappingRow]
            if method == "pearson":
                row = _assoc_pearson(xs, ys)
            elif method == "linear":
                row = _assoc_linear(xs, ys)
            elif method == "anova":
                row = _assoc_anova(xs, ys)
            else:  # lod
                row = _assoc_lod(xs, ys)
            if row is None:
                continue
            payload = row.model_dump() if hasattr(row, "model_dump") else row.dict()  # pydantic v2/v1 compat
            payload["marker_name"] = marker.name
            results.append(MappingRow(**payload))

    pvals = [r.p_value for r in results]
    try:
        padj = _adjust_pvalues(pvals, p_adjust)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    for r, q in zip(results, padj):
        r.p_adjusted = q

    if method == "pearson":
        results.sort(key=lambda r: abs(r.effect or 0.0), reverse=True)
    else:
        results.sort(key=lambda r: (r.p_adjusted if r.p_adjusted is not None else 1.0, r.p_value if r.p_value is not None else 1.0))

    return results, int(len(marker_rows)), int(n_samples_used)


def _adjust_pvalues(p_values: List[Optional[float]], method: str) -> List[Optional[float]]:
    p = [v for v in p_values]
    if method == "none":
        return p

    idx = [i for i, v in enumerate(p) if v is not None]
    if not idx:
        return p

    if method == "bonferroni":
        m = len(idx)
        for i in idx:
            p[i] = min(1.0, float(p[i]) * m)  # type: ignore[arg-type]
        return p

    if method != "bh":
        raise ValueError("Unsupported p_adjust method")

    # Benjamini–Hochberg FDR
    sorted_idx = sorted(idx, key=lambda i: float(p[i]))  # type: ignore[arg-type]
    m = len(sorted_idx)
    q: List[Optional[float]] = [None] * len(p)
    prev = 1.0
    for rank, i in enumerate(reversed(sorted_idx), start=1):
        # reverse traversal to enforce monotonicity
        # rank in reversed order means i has larger p; compute on original rank:
        orig_rank = m - rank + 1
        pv = float(p[i])  # type: ignore[arg-type]
        val = min(prev, pv * m / orig_rank)
        prev = val
        q[i] = min(1.0, val)
    for i in idx:
        if q[i] is not None:
            p[i] = float(q[i])
    return p


def _as_float_list(vals: List[float]) -> Tuple[List[float], bool]:
    if scipy_stats is None:  # pragma: no cover
        raise RuntimeError(f"scipy is required for mapping methods ({_SCIPY_IMPORT_ERROR})")
    xs = [float(v) for v in vals]
    var = float(scipy_stats.tvar(xs)) if len(xs) >= 2 else 0.0  # type: ignore[union-attr]
    return xs, var > 0


def _assoc_pearson(xs: List[float], ys: List[float]) -> Optional[MappingRow]:
    n = len(xs)
    if n < 3:
        return None
    xs_f, okx = _as_float_list(xs)
    ys_f, oky = _as_float_list(ys)
    if not okx or not oky:
        return None
    r, p = scipy_stats.pearsonr(xs_f, ys_f)  # type: ignore[union-attr]
    return MappingRow(marker_name="", n=n, effect=float(r), p_value=float(p), r2=float(r * r))


def _assoc_linear(xs: List[float], ys: List[float]) -> Optional[MappingRow]:
    # y = b0 + b1*x; p-value for b1
    n = len(xs)
    if n < 3:
        return None
    xs_f, okx = _as_float_list(xs)
    ys_f, oky = _as_float_list(ys)
    if not okx or not oky:
        return None

    x = xs_f
    y = ys_f
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0:
        return None
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b1 = sxy / sxx
    b0 = my - b1 * mx
    resid = [yi - (b0 + b1 * xi) for xi, yi in zip(x, y)]
    rss = sum(r * r for r in resid)
    tss = sum((yi - my) ** 2 for yi in y)
    df = n - 2
    if df <= 0:
        return None
    sigma2 = rss / df
    se_b1 = (sigma2 / sxx) ** 0.5
    if se_b1 == 0:
        return None
    t_stat = b1 / se_b1
    p = float(2 * scipy_stats.t.sf(abs(t_stat), df=df))  # type: ignore[union-attr]
    r2 = 0.0 if tss == 0 else float(1 - rss / tss)
    return MappingRow(marker_name="", n=n, effect=float(b1), p_value=p, r2=r2)


def _assoc_anova(xs: List[float], ys: List[float]) -> Optional[MappingRow]:
    # one-way ANOVA across genotype groups (requires at least 2 groups)
    n = len(xs)
    if n < 3:
        return None
    groups: Dict[int, List[float]] = {}
    for x, y in zip(xs, ys):
        try:
            g = int(round(float(x)))
        except Exception:
            continue
        groups.setdefault(g, []).append(float(y))
    if len(groups) < 2:
        return None
    arrays = [groups[k] for k in sorted(groups.keys()) if len(groups[k]) >= 2]
    if len(arrays) < 2:
        return None
    f_stat, p = scipy_stats.f_oneway(*arrays)  # type: ignore[union-attr]
    return MappingRow(marker_name="", n=n, effect=float(f_stat), p_value=float(p))


def _assoc_lod(xs: List[float], ys: List[float]) -> Optional[MappingRow]:
    # QTL-style LOD from regression y~x vs null mean-only; also provide F-test p-value
    n = len(xs)
    if n < 3:
        return None
    xs_f, okx = _as_float_list(xs)
    ys_f, oky = _as_float_list(ys)
    if not okx or not oky:
        return None
    x = xs_f
    y = ys_f
    my = sum(y) / n
    rss0 = sum((yi - my) ** 2 for yi in y)

    mx = sum(x) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0:
        return None
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b1 = sxy / sxx
    b0 = my - b1 * mx
    rss1 = sum((yi - (b0 + b1 * xi)) ** 2 for xi, yi in zip(x, y))
    if rss1 <= 0 or rss0 <= 0:
        return None
    lod = float((n / 2.0) * math.log10(rss0 / rss1))
    # F test
    df1 = 1
    df2 = n - 2
    if df2 <= 0:
        return MappingRow(marker_name="", n=n, lod=lod)
    f = ((rss0 - rss1) / df1) / (rss1 / df2) if rss1 > 0 else 0.0
    p = float(scipy_stats.f.sf(f, df1, df2)) if f >= 0 else None  # type: ignore[union-attr]
    return MappingRow(marker_name="", n=n, lod=lod, p_value=p, effect=float(f))


def _standardize_cols(G: np.ndarray) -> np.ndarray:
    mu = np.mean(G, axis=0)
    sd = np.std(G, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (G - mu) / sd


def _compute_pcs(Gstd: np.ndarray, k: int, max_markers: int, seed: int) -> np.ndarray:
    if k <= 0:
        return np.zeros((Gstd.shape[0], 0), dtype=float)
    m = Gstd.shape[1]
    use = min(m, max(10, max_markers))
    if use < m:
        rng = random.Random(seed)
        idx = list(range(m))
        rng.shuffle(idx)
        idx = idx[:use]
        X = Gstd[:, idx]
    else:
        X = Gstd
    U, _s, _vt = np.linalg.svd(X, full_matrices=False)
    return U[:, : min(k, U.shape[1])]


def _ols_ttest(y: np.ndarray, X: np.ndarray, j: int) -> Optional[Tuple[float, float, float]]:
    # return (beta_j, p_value, r2) for column j
    n, p = X.shape
    df = n - p
    if df <= 0:
        return None
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return None
    yhat = X @ beta
    resid = y - yhat
    rss = float(np.sum(resid * resid))
    tss = float(np.sum((y - float(np.mean(y))) ** 2))
    sigma2 = rss / df if df > 0 else 0.0
    if sigma2 <= 0:
        return None
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except Exception:
        xtx_inv = np.linalg.pinv(X.T @ X)
    se2 = float(sigma2 * xtx_inv[j, j])
    if se2 <= 0:
        return None
    t = float(beta[j]) / math.sqrt(se2)
    pval = float(2 * scipy_stats.t.sf(abs(t), df=df))  # type: ignore[union-attr]
    r2 = 0.0 if tss == 0 else float(1.0 - rss / tss)
    return float(beta[j]), pval, r2


def _build_genotype_matrix(
    db: Session,
    marker_rows: List[GenotypeMarker],
    phen: Dict[str, float],
    *,
    min_n: int,
) -> Tuple[List[str], np.ndarray, List[str]]:
    marker_ids = [int(m.id) for m in marker_rows if m.id is not None]
    id_to_name = {int(m.id): m.name for m in marker_rows if m.id is not None}

    rows = db.execute(
        select(GenotypeValue.sample_id, GenotypeValue.marker_id, GenotypeValue.value).where(
            GenotypeValue.marker_id.in_(marker_ids)
        )
    ).all()
    phen_ids = set(phen.keys())
    sample_ids = sorted({sid for (sid, _mid, _v) in rows if sid in phen_ids})
    if len(sample_ids) < min_n:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough overlapping samples (n={len(sample_ids)}; need >= {min_n})",
        )
    sample_index = {sid: i for i, sid in enumerate(sample_ids)}
    marker_index = {mid: j for j, mid in enumerate(marker_ids)}

    G = np.full((len(sample_ids), len(marker_ids)), np.nan, dtype=float)
    for sid, mid, v in rows:
        i = sample_index.get(sid)
        if i is None:
            continue
        j = marker_index.get(int(mid))
        if j is None:
            continue
        G[i, j] = float(v)

    keep_cols: List[int] = []
    cols: List[np.ndarray] = []
    for j in range(G.shape[1]):
        col = G[:, j]
        mask = ~np.isnan(col)
        if not np.any(mask):
            continue
        m = float(np.nanmean(col))
        col = np.where(mask, col, m)
        if float(np.var(col)) <= 0.0:
            continue
        keep_cols.append(j)
        cols.append(col)

    if not keep_cols:
        raise HTTPException(status_code=400, detail="No usable markers after filtering (all missing or zero variance)")

    G2 = np.stack(cols, axis=1)
    marker_names = [id_to_name[int(marker_ids[j])] for j in keep_cols]
    return sample_ids, G2, marker_names


def _glm_scan(y: np.ndarray, G: np.ndarray, cov: np.ndarray) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    betas: List[Optional[float]] = []
    pvals: List[Optional[float]] = []
    r2s: List[Optional[float]] = []
    for j in range(G.shape[1]):
        X = np.concatenate([cov, G[:, j][:, None]], axis=1)
        out = _ols_ttest(y, X, X.shape[1] - 1)
        if out is None:
            betas.append(None)
            pvals.append(None)
            r2s.append(None)
        else:
            b, p, r2 = out
            betas.append(b)
            pvals.append(p)
            r2s.append(r2)
    return betas, pvals, r2s


def _mlm_scan_emmax(
    y: np.ndarray,
    G: np.ndarray,
    cov: np.ndarray,
    K: np.ndarray,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    # EMMAX-like scan: estimate delta once on null model, then GLS via eigen transform.
    n = y.shape[0]
    s, U = np.linalg.eigh(K)
    s = np.maximum(s, 0.0)
    Uy = U.T @ y
    UX = U.T @ cov

    grid = np.logspace(-3, 3, 25)
    best_ll = -1e300
    best_delta = float(grid[0])
    for delta in grid:
        d = s + float(delta)
        if np.any(d <= 0):
            continue
        w = 1.0 / np.sqrt(d)
        yw = w * Uy
        Xw = w[:, None] * UX
        try:
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except Exception:
            continue
        resid = yw - Xw @ beta
        rss = float(np.sum(resid * resid))
        if rss <= 0:
            continue
        sigma2 = rss / n
        logdet = float(np.sum(np.log(d)))
        ll = -0.5 * (n * math.log(2 * math.pi * sigma2) + logdet + n)
        if ll > best_ll:
            best_ll = ll
            best_delta = float(delta)

    d = s + float(best_delta)
    w = 1.0 / np.sqrt(d)
    yw = w * Uy
    X0w = w[:, None] * UX

    betas: List[Optional[float]] = []
    pvals: List[Optional[float]] = []
    r2s: List[Optional[float]] = []
    for j in range(G.shape[1]):
        Ug = U.T @ G[:, j]
        Xw = np.concatenate([X0w, (w * Ug)[:, None]], axis=1)
        out = _ols_ttest(yw, Xw, Xw.shape[1] - 1)
        if out is None:
            betas.append(None)
            pvals.append(None)
            r2s.append(None)
        else:
            b, p, r2 = out
            betas.append(b)
            pvals.append(p)
            r2s.append(r2)
    return betas, pvals, r2s


def _select_pseudo_qtn(G: np.ndarray, pvals: List[Optional[float]], max_qtn: int, *, corr_thresh: float = 0.8) -> List[int]:
    idx = [(i, p) for i, p in enumerate(pvals) if p is not None and math.isfinite(float(p))]
    idx.sort(key=lambda t: float(t[1]))
    selected: List[int] = []
    for i, _p in idx:
        if len(selected) >= max_qtn:
            break
        ok = True
        for j in selected:
            r = float(np.corrcoef(G[:, i], G[:, j])[0, 1])
            if abs(r) >= corr_thresh:
                ok = False
                break
        if ok:
            selected.append(i)
    return selected


def _farmcpu_scan(
    y: np.ndarray,
    G: np.ndarray,
    cov: np.ndarray,
    K: np.ndarray,
    *,
    max_iters: int,
    max_qtn: int,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    # Simple FarmCPU-style loop:
    # 1) Seed pseudo-QTNs from an MLM scan.
    # 2) Iterate GLM scans with pseudo-QTNs as fixed effects until stable.
    _b0, p0, _r0 = _mlm_scan_emmax(y, G, cov, K)
    pseudo = _select_pseudo_qtn(G, p0, max_qtn=max_qtn)
    betas: List[Optional[float]] = []
    pvals: List[Optional[float]] = []
    r2s: List[Optional[float]] = []
    for _ in range(max(1, max_iters)):
        cov_iter = np.concatenate([cov, G[:, pseudo]], axis=1) if pseudo else cov
        betas, pvals, r2s = _glm_scan(y, G, cov_iter)
        new_pseudo = _select_pseudo_qtn(G, pvals, max_qtn=max_qtn)
        if set(new_pseudo) == set(pseudo):
            break
        pseudo = new_pseudo
    return betas, pvals, r2s


def _normalize_tube_id(raw: str) -> str:
    token = (raw or "").strip()
    if not token:
        return ""
    m = re.match(r"^t?(\d+)$", token, flags=re.IGNORECASE)
    if m:
        return f"T{m.group(1)}"
    return token.upper()


def _load_phenotypes(
    field: str,
    analysis_run_id: Optional[str] = None,
    limit: int = 200000,
    *,
    allow_filename_fallback: bool = False,
) -> Dict[str, float]:
    # Reads from the API's `analyses` table (created by api.py) using PHENOTYPE_DATABASE_URL.
    # Join priority:
    # 1) explicit phenotype_sample_id_map.sample_id
    # 2) minirhizotron genotype metadata (string label)
    # 3) minirhizotron tube_id metadata (T###)
    # 4) optional filename aliases (legacy behavior)
    if field not in {
        "root_count",
        "average_root_diameter",
        "total_root_length",
        "total_root_area",
        "total_root_volume",
    }:
        raise ValueError("Unsupported phenotype_field")

    rid = (analysis_run_id or "").strip()
    with PhenotypeSessionLocal() as db:
        try:
            rows = db.execute(
                text(
                    f"SELECT a.filename, a.{field}, a.extra, m.sample_id "
                    "FROM analyses a "
                    "LEFT JOIN phenotype_sample_id_map m ON m.filename = a.filename "
                    "ORDER BY a.created_at DESC LIMIT :lim"
                ),
                {"lim": int(limit)},
            ).all()
        except Exception:
            # Backwards compatible: mapping table may not exist in older DBs.
            rows = db.execute(
                text(f"SELECT filename, {field}, extra, '' as sample_id FROM analyses ORDER BY created_at DESC LIMIT :lim"),
                {"lim": int(limit)},
            ).all()
    out: Dict[str, float] = {}
    for filename, value, extra, mapped_sample_id in rows:
        if not filename or value is None:
            continue
        if rid:
            try:
                e = json.loads(extra) if isinstance(extra, str) else (extra if isinstance(extra, dict) else {})
            except Exception:
                e = {}
            meta = e.get("meta") if isinstance(e, dict) and isinstance(e.get("meta"), dict) else {}
            if str((meta or {}).get("run_id") or "").strip() != rid:
                continue

        try:
            y = float(value)
        except Exception:
            continue

        try:
            e = json.loads(extra) if isinstance(extra, str) else (extra if isinstance(extra, dict) else {})
        except Exception:
            e = {}
        meta = e.get("meta") if isinstance(e, dict) and isinstance(e.get("meta"), dict) else {}
        mini = meta.get("minirhizotron") if isinstance(meta, dict) and isinstance(meta.get("minirhizotron"), dict) else {}

        keys: List[str] = []
        sid = str(mapped_sample_id or "").strip()
        if sid:
            keys.append(sid)

        genotype_raw = str((mini or {}).get("genotype") or (meta or {}).get("genotype") or "").strip()
        if genotype_raw:
            keys.append(genotype_raw)

        tube_raw = str((mini or {}).get("tube_id") or (meta or {}).get("tube_id") or "").strip()
        tube = _normalize_tube_id(tube_raw)
        if tube:
            keys.append(tube)

        if allow_filename_fallback:
            filename_key = str(filename)
            keys.append(filename_key)
            base = PurePosixPath(filename_key.replace("\\", "/")).name
            if base:
                keys.append(base)
            if "." in base:
                stem = base.rsplit(".", 1)[0]
                if stem:
                    keys.append(stem)

        for k in keys:
            if k and k not in out:
                out[k] = y
    return out


def _dosage_class(v: float) -> str:
    # Bin dosage/imputed dosage into canonical diploid genotype classes.
    if v < 0.5:
        return "0"
    if v < 1.5:
        return "1"
    return "2"


def _boxplot_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        raise ValueError("no finite values")
    q1 = float(np.percentile(arr, 25))
    med = float(np.percentile(arr, 50))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    inliers = arr[(arr >= low_fence) & (arr <= high_fence)]
    whisk_lo = float(np.min(inliers)) if inliers.size > 0 else float(np.min(arr))
    whisk_hi = float(np.max(inliers)) if inliers.size > 0 else float(np.max(arr))
    return {
        "mean": float(np.mean(arr)),
        "median": med,
        "q1": q1,
        "q3": q3,
        "whisker_low": whisk_lo,
        "whisker_high": whisk_hi,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


@app.get("/mapping/marker-effect", response_model=MarkerEffectResponse)
def mapping_marker_effect(
    phenotype_field: str = "total_root_volume",
    marker_name: str = "",
    analysis_run_id: Optional[str] = None,
    allow_filename_fallback: bool = False,
) -> MarkerEffectResponse:
    marker_key = (marker_name or "").strip()
    if not marker_key:
        raise HTTPException(status_code=400, detail="marker_name is required")

    try:
        phen = _load_phenotypes(
            phenotype_field,
            analysis_run_id=analysis_run_id,
            allow_filename_fallback=bool(allow_filename_fallback),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not phen:
        raise HTTPException(status_code=400, detail="No phenotype rows available for requested field/run")

    with SessionLocal() as db:
        marker = db.execute(select(GenotypeMarker).where(GenotypeMarker.name == marker_key)).scalars().first()
        if marker is None or marker.id is None:
            raise HTTPException(status_code=404, detail="marker not found")
        rows = db.execute(
            select(GenotypeValue.sample_id, GenotypeValue.value).where(GenotypeValue.marker_id == int(marker.id))
        ).all()

    groups: Dict[str, List[float]] = {"0": [], "1": [], "2": []}
    for sid, dosage in rows:
        y = phen.get(str(sid))
        if y is None:
            continue
        try:
            d = float(dosage)
            yy = float(y)
        except Exception:
            continue
        if not (math.isfinite(d) and math.isfinite(yy)):
            continue
        groups[_dosage_class(d)].append(yy)

    out_classes: List[MarkerEffectClass] = []
    for cls in ("0", "1", "2"):
        vals = groups.get(cls, [])
        if not vals:
            continue
        stats = _boxplot_stats(vals)
        out_classes.append(
            MarkerEffectClass(
                genotype_class=cls,
                n=int(len(vals)),
                mean=float(stats["mean"]),
                median=float(stats["median"]),
                q1=float(stats["q1"]),
                q3=float(stats["q3"]),
                whisker_low=float(stats["whisker_low"]),
                whisker_high=float(stats["whisker_high"]),
                min=float(stats["min"]),
                max=float(stats["max"]),
            )
        )

    if not out_classes:
        raise HTTPException(status_code=400, detail="No overlapping sample data for marker effect plot")

    return MarkerEffectResponse(
        marker_name=marker_key,
        phenotype_field=phenotype_field,
        n_samples=int(sum(c.n for c in out_classes)),
        classes=out_classes,
    )


@app.post("/mapping/run", response_model=MappingRunResponse)
def run_mapping(req: MappingRunRequest) -> MappingRunResponse:
    with SessionLocal() as db:
        results, n_markers_tested, _n_samples_used = _mapping_compute(db, req)
        method = req.method.lower().strip()
        p_adjust = req.p_adjust.lower().strip()

        # persist top results
        try:
            run = MappingRun(
                phenotype_field=req.phenotype_field,
                analysis_run_id=str(req.analysis_run_id or ""),
                method=method,
                p_adjust=p_adjust,
                max_markers=int(req.max_markers),
                min_n=int(req.min_n),
                n_markers_tested=int(n_markers_tested),
                n_results=int(len(results)),
            )
            db.add(run)
            db.flush()

            for row in results[:200]:
                db.add(
                    MappingHit(
                        run_id=int(run.id),
                        marker_name=row.marker_name,
                        phenotype_field=req.phenotype_field,
                        analysis_run_id=str(req.analysis_run_id or ""),
                        method=method,
                        p_adjust=p_adjust,
                        n=int(row.n),
                        effect=float(row.effect or 0.0),
                        p_value=float(row.p_value or 1.0),
                        p_adjusted=float(row.p_adjusted or 1.0),
                        r2=float(row.r2 or 0.0),
                    )
                )

            for row in results[:200]:
                db.add(
                    MappingResult(
                        phenotype_field=req.phenotype_field,
                        marker_name=row.marker_name,
                        n=row.n,
                        pearson_r=float(row.effect or 0.0),
                    )
                )
            db.commit()
        except Exception:
            db.rollback()

        return MappingRunResponse(phenotype_field=req.phenotype_field, method=method, p_adjust=p_adjust, rows=results[:200])


class MappingHistoryItem(BaseModel):
    created_at: str
    phenotype_field: str
    marker_name: str
    n: int
    effect: float


@app.get("/mapping/results.csv")
def mapping_results_csv(
    phenotype_field: str = "total_root_volume",
    method: str = "mlm",
    p_adjust: str = "bh",
    max_markers: int = 2000,
    min_n: int = 6,
    analysis_run_id: Optional[str] = None,
    allow_filename_fallback: bool = False,
    n_pcs: int = 3,
    kinship_markers: int = 1000,
    farmcpu_max_iters: int = 3,
    farmcpu_max_qtn: int = 10,
    farmcpu_seed: int = 7,
) -> Response:
    """
    Export full mapping results as CSV (paper-friendly).
    """
    req = MappingRunRequest(
        phenotype_field=phenotype_field,
        method=method,
        p_adjust=p_adjust,
        max_markers=int(max_markers),
        min_n=int(min_n),
        analysis_run_id=analysis_run_id,
        allow_filename_fallback=bool(allow_filename_fallback),
        n_pcs=int(n_pcs),
        kinship_markers=int(kinship_markers),
        farmcpu_max_iters=int(farmcpu_max_iters),
        farmcpu_max_qtn=int(farmcpu_max_qtn),
        farmcpu_seed=int(farmcpu_seed),
    )
    with SessionLocal() as db:
        rows, n_markers_tested, n_samples_used = _mapping_compute(db, req)
    out = []
    for r in rows:
        out.append(
            {
                "phenotype_field": phenotype_field,
                "method": method,
                "p_adjust": p_adjust,
                "n_samples": n_samples_used,
                "n_markers_tested": n_markers_tested,
                "marker_name": r.marker_name,
                "n": r.n,
                "effect": r.effect,
                "p_value": r.p_value,
                "p_adjusted": r.p_adjusted,
                "r2": r.r2,
                "lod": r.lod,
            }
        )
    headers = [
        "phenotype_field",
        "method",
        "p_adjust",
        "n_samples",
        "n_markers_tested",
        "marker_name",
        "n",
        "effect",
        "p_value",
        "p_adjusted",
        "r2",
        "lod",
    ]
    fn = f"mapping_{phenotype_field}_{method}_{p_adjust}.csv"
    return _csv_response(fn, headers, out)


@app.get("/mapping/plot.svg")
def mapping_plot_svg(
    phenotype_field: str = "total_root_volume",
    method: str = "mlm",
    p_adjust: str = "bh",
    max_markers: int = 2000,
    min_n: int = 6,
    analysis_run_id: Optional[str] = None,
    allow_filename_fallback: bool = False,
    n_pcs: int = 3,
    kinship_markers: int = 1000,
    farmcpu_max_iters: int = 3,
    farmcpu_max_qtn: int = 10,
    farmcpu_seed: int = 7,
    title: Optional[str] = None,
    width: int = 1200,
    height: int = 560,
) -> Response:
    """
    Paper-friendly GWAS plot (Manhattan + QQ) as a standalone SVG.
    """
    width = max(720, min(2400, int(width)))
    height = max(420, min(1600, int(height)))
    req = MappingRunRequest(
        phenotype_field=phenotype_field,
        method=method,
        p_adjust=p_adjust,
        max_markers=int(max_markers),
        min_n=int(min_n),
        analysis_run_id=analysis_run_id,
        allow_filename_fallback=bool(allow_filename_fallback),
        n_pcs=int(n_pcs),
        kinship_markers=int(kinship_markers),
        farmcpu_max_iters=int(farmcpu_max_iters),
        farmcpu_max_qtn=int(farmcpu_max_qtn),
        farmcpu_seed=int(farmcpu_seed),
    )
    with SessionLocal() as db:
        rows, n_markers_tested, n_samples_used = _mapping_compute(db, req)

    if not rows:
        raise HTTPException(status_code=400, detail="No mapping rows to plot")

    pts = []
    for r in rows:
        pv = float(r.p_value) if r.p_value is not None else 1.0
        if not math.isfinite(pv) or pv <= 0:
            pv = 1e-300
        locus = _parse_marker_locus(r.marker_name)
        pts.append(
            {
                "marker": r.marker_name,
                "p": pv,
                "q": float(r.p_adjusted) if r.p_adjusted is not None and math.isfinite(float(r.p_adjusted)) else None,
                "chrom": locus[0] if locus else None,
                "pos": locus[1] if locus else None,
            }
        )

    parsed = sum(1 for p in pts if p["chrom"] is not None and p["pos"] is not None)
    use_genomic = parsed >= max(10, int(0.4 * len(pts)))

    if use_genomic:
        # Order by chrom/pos and compute cumulative x coordinate.
        pts.sort(key=lambda p: (int(p["chrom"] or 0), int(p["pos"] or 0)))
        chr_max = {}
        for p in pts:
            c = int(p["chrom"] or 0)
            pos = int(p["pos"] or 0)
            if c <= 0 or pos <= 0:
                continue
            chr_max[c] = max(chr_max.get(c, 0), pos)
        chroms = sorted(chr_max.keys())
        gap = 2_000_000  # fixed visual spacer
        offset = 0
        chr_offset = {}
        for c in chroms:
            chr_offset[c] = offset
            offset += int(chr_max[c]) + gap
        total_span = max(1, offset - gap)
        for p in pts:
            c = p["chrom"]
            if c is None or p["pos"] is None:
                p["x"] = None
            else:
                p["x"] = int(chr_offset[int(c)]) + int(p["pos"])
        x_min = 0
        x_max = total_span
    else:
        # Fallback: x by index.
        for i, p in enumerate(pts):
            p["x"] = i
        x_min = 0
        x_max = max(1, len(pts) - 1)

    y_vals = [-math.log10(p["p"]) for p in pts]
    y_max = max(5.0, float(max(y_vals)) + 0.5)

    # significance helpers
    bonf = -math.log10(max(1e-300, 0.05 / max(1, int(n_markers_tested))))
    sig = [p for p in pts if p["q"] is not None and float(p["q"]) <= 0.05]

    # Layout
    pad = 44
    gutter = 26
    left_w = int(width * 0.68)
    right_w = width - left_w - gutter
    top_h = height - pad * 2
    man_x0, man_y0 = pad, pad
    man_x1, man_y1 = pad + left_w, pad + top_h
    qq_x0, qq_y0 = man_x1 + gutter, pad
    qq_x1, qq_y1 = qq_x0 + right_w, pad + top_h

    def sx(x: float) -> float:
        return man_x0 + (float(x) - x_min) / (x_max - x_min) * (man_x1 - man_x0) if x_max != x_min else man_x0

    def sy(y: float) -> float:
        return man_y1 - float(y) / y_max * (man_y1 - man_y0)

    def qx(x: float, x_max2: float) -> float:
        return qq_x0 + float(x) / max(1e-9, x_max2) * (qq_x1 - qq_x0)

    def qy(y: float, y_max2: float) -> float:
        return qq_y1 - float(y) / max(1e-9, y_max2) * (qq_y1 - qq_y0)

    # QQ data
    ps = sorted(p["p"] for p in pts)
    m = len(ps)
    obs = [-math.log10(max(1e-300, v)) for v in ps]
    exp = [-math.log10((i + 1) / (m + 1)) for i in range(m)]
    qq_max = max(5.0, max(max(obs), max(exp)) + 0.25)

    main_title = title or f"Marker–trait association (MAGIC chr1 panel): {phenotype_field} · {method.upper()} · {p_adjust.upper()} (n={n_samples_used}, m={n_markers_tested})"

    # Colors
    c1 = "#60a5fa"
    c2 = "#fb923c"
    sigc = "#ef4444"
    axis = "#334155"
    grid = "rgba(51,65,85,0.18)"

    # Build SVG
    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{pad}" y="{pad - 16}" font-family="Arial, sans-serif" font-size="16" fill="#0f172a">{_svg_escape(main_title)}</text>')

    # Manhattan axes
    lines.append(f'<rect x="{man_x0}" y="{man_y0}" width="{man_x1 - man_x0}" height="{man_y1 - man_y0}" fill="white" stroke="{axis}" stroke-width="1"/>')
    # grid + y ticks
    for t in range(0, int(math.ceil(y_max)) + 1, 2):
        y = sy(t)
        lines.append(f'<line x1="{man_x0}" y1="{y:.2f}" x2="{man_x1}" y2="{y:.2f}" stroke="{grid}" stroke-width="1"/>')
        lines.append(f'<text x="{man_x0 - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="{axis}">{t}</text>')
    lines.append(f'<text x="{(man_x0 + man_x1)/2:.2f}" y="{man_y1 + 32}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{axis}">marker position</text>')
    lines.append(f'<text x="{man_x0 - 34}" y="{(man_y0 + man_y1)/2:.2f}" text-anchor="middle" transform="rotate(-90 {man_x0 - 34} {(man_y0 + man_y1)/2:.2f})" font-family="Arial, sans-serif" font-size="12" fill="{axis}">-log10(p)</text>')

    # Bonferroni line
    yb = sy(bonf)
    lines.append(f'<line x1="{man_x0}" y1="{yb:.2f}" x2="{man_x1}" y2="{yb:.2f}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,4"/>')
    lines.append(f'<text x="{man_x1 - 6}" y="{yb - 6:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#64748b">Bonferroni 0.05</text>')

    # Points
    for p in pts:
        x = p["x"]
        if x is None:
            continue
        yy = -math.log10(p["p"])
        cx = sx(float(x))
        cy = sy(yy)
        chrom = int(p["chrom"] or 0)
        base = c1 if chrom % 2 == 1 else c2
        col = sigc if (p["q"] is not None and float(p["q"]) <= 0.05) else base
        lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.0" fill="{col}" fill-opacity="0.85"/>')

    # Label a few top hits (by p-value). Keep this small to avoid clutter.
    top_n = 3
    top_hits = sorted([p for p in pts if p.get("x") is not None], key=lambda p: float(p["p"]))[:top_n]
    for j, p in enumerate(top_hits):
        x = p["x"]
        if x is None:
            continue
        yy = -math.log10(p["p"])
        cx = sx(float(x))
        cy = sy(yy)
        chrom = p.get("chrom")
        pos = p.get("pos")
        if chrom is not None and pos is not None:
            label = f"Chr{int(chrom)}:{int(pos) / 1e6:.2f}Mb"
        else:
            label = str(p.get("marker") or "")[:18]
        # highlight point and add a small label above it
        lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3.2" fill="none" stroke="#0f172a" stroke-width="1"/>')
        lines.append(
            f'<text x="{(cx + 4):.2f}" y="{(cy - 10 - 12*j):.2f}" font-family="Arial, sans-serif" font-size="11" fill="#0f172a">{_svg_escape(label)}</text>'
        )

    # QQ panel
    lines.append(f'<rect x="{qq_x0}" y="{qq_y0}" width="{qq_x1 - qq_x0}" height="{qq_y1 - qq_y0}" fill="white" stroke="{axis}" stroke-width="1"/>')
    # grid + ticks
    for t in range(0, int(math.ceil(qq_max)) + 1, 2):
        y = qy(t, qq_max)
        lines.append(f'<line x1="{qq_x0}" y1="{y:.2f}" x2="{qq_x1}" y2="{y:.2f}" stroke="{grid}" stroke-width="1"/>')
        lines.append(f'<text x="{qq_x0 - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="{axis}">{t}</text>')
        x = qx(t, qq_max)
        lines.append(f'<line x1="{x:.2f}" y1="{qq_y0}" x2="{x:.2f}" y2="{qq_y1}" stroke="{grid}" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{qq_y1 + 16}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{axis}">{t}</text>')
    lines.append(f'<text x="{(qq_x0 + qq_x1)/2:.2f}" y="{qq_y1 + 32}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{axis}">expected -log10(p)</text>')
    lines.append(f'<text x="{qq_x0 - 34}" y="{(qq_y0 + qq_y1)/2:.2f}" text-anchor="middle" transform="rotate(-90 {qq_x0 - 34} {(qq_y0 + qq_y1)/2:.2f})" font-family="Arial, sans-serif" font-size="12" fill="{axis}">observed -log10(p)</text>')
    # diagonal
    d0x, d0y = qx(0, qq_max), qy(0, qq_max)
    d1x, d1y = qx(qq_max, qq_max), qy(qq_max, qq_max)
    lines.append(f'<line x1="{d0x:.2f}" y1="{d0y:.2f}" x2="{d1x:.2f}" y2="{d1y:.2f}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,4"/>')

    # QQ points (subsample if huge)
    step = max(1, m // 1500)
    for i in range(0, m, step):
        cx = qx(exp[i], qq_max)
        cy = qy(obs[i], qq_max)
        lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.0" fill="{axis}" fill-opacity="0.65"/>')

    # Legend-ish
    lx = qq_x0
    ly = pad - 14
    lines.append(f'<text x="{lx}" y="{ly}" font-family="Arial, sans-serif" font-size="12" fill="{axis}">QQ</text>')
    lines.append(f'<text x="{man_x0}" y="{pad - 14}" font-family="Arial, sans-serif" font-size="12" fill="{axis}">Manhattan</text>')
    if sig:
        lines.append(f'<text x="{man_x1 - 6}" y="{pad - 14}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{sigc}">q≤0.05: {len(sig)} markers</text>')

    lines.append("</svg>")
    svg = "\n".join(lines).encode("utf-8")
    fn = f"gwas_{phenotype_field}_{method}_{p_adjust}.svg"
    return Response(content=svg, media_type="image/svg+xml; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{fn}"'})


@app.get("/mapping/history", response_model=List[MappingHistoryItem])
def mapping_history(limit: int = 50) -> List[MappingHistoryItem]:
    with SessionLocal() as db:
        rows = db.execute(select(MappingResult).order_by(MappingResult.created_at.desc()).limit(limit)).scalars().all()
        return [
            MappingHistoryItem(
                created_at=r.created_at.isoformat(),
                phenotype_field=r.phenotype_field,
                marker_name=r.marker_name,
                n=r.n,
                effect=r.pearson_r,
            )
            for r in rows
        ]


class GenotypeStats(BaseModel):
    samples: int
    markers: int
    values: int


@app.get("/stats", response_model=GenotypeStats)
def stats() -> GenotypeStats:
    with SessionLocal() as db:
        # Real markers only (non-demo). Demo markers are not used by the GUI.
        real_marker_ids = (
            db.execute(select(GenotypeMarker.id).where(~GenotypeMarker.name.like("demo_%")))
            .scalars()
            .all()
        )
        markers = int(len(real_marker_ids))
        if real_marker_ids:
            samples = int(
                db.execute(
                    select(func.count(func.distinct(GenotypeValue.sample_id))).where(
                        GenotypeValue.marker_id.in_(real_marker_ids)
                    )
                ).scalar_one()
            )
            values = int(
                db.execute(select(func.count(GenotypeValue.id)).where(GenotypeValue.marker_id.in_(real_marker_ids))).scalar_one()
            )
        else:
            samples = 0
            values = 0
        return GenotypeStats(samples=samples, markers=markers, values=values)


@app.get("/observability/markers.csv")
def export_markers_csv(limit: int = 200000) -> Response:
    with SessionLocal() as db:
        rows = (
            db.execute(
                select(GenotypeMarker)
                .where(~GenotypeMarker.name.like("demo_%"))
                .order_by(GenotypeMarker.id.asc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        out = [{"id": r.id, "name": r.name, "created_at": r.created_at.isoformat()} for r in rows]
        headers = ["id", "name", "created_at"]
        return _csv_response("markers.csv", headers, out)


@app.get("/observability/samples.csv")
def export_samples_csv(limit: int = 200000) -> Response:
    with SessionLocal() as db:
        # Export samples that appear in real (non-demo) genotype values.
        real_marker_ids = (
            db.execute(select(GenotypeMarker.id).where(~GenotypeMarker.name.like("demo_%")))
            .scalars()
            .all()
        )
        if not real_marker_ids:
            rows = []
        else:
            sample_ids = (
                db.execute(
                    select(func.distinct(GenotypeValue.sample_id)).where(GenotypeValue.marker_id.in_(real_marker_ids)).limit(limit)
                )
                .scalars()
                .all()
            )
            rows = (
                db.execute(select(GenotypeSample).where(GenotypeSample.sample_id.in_(sample_ids)).order_by(GenotypeSample.id.asc()))
                .scalars()
                .all()
            )
        out = [{"id": r.id, "sample_id": r.sample_id, "created_at": r.created_at.isoformat()} for r in rows]
        headers = ["id", "sample_id", "created_at"]
        return _csv_response("samples.csv", headers, out)


@app.get("/observability/values.csv")
def export_values_csv(limit: int = 300000) -> Response:
    with SessionLocal() as db:
        # Export real marker values only.
        real_marker_ids = (
            db.execute(select(GenotypeMarker.id).where(~GenotypeMarker.name.like("demo_%")))
            .scalars()
            .all()
        )
        if not real_marker_ids:
            rows = []
        else:
            rows = (
                db.execute(
                    select(GenotypeValue)
                    .where(GenotypeValue.marker_id.in_(real_marker_ids))
                    .order_by(GenotypeValue.id.asc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
        out = [
            {
                "id": r.id,
                "sample_id": r.sample_id,
                "marker_id": r.marker_id,
                "value": r.value,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
        headers = ["id", "sample_id", "marker_id", "value", "created_at"]
        return _csv_response("values.csv", headers, out)


@app.get("/observability/mapping_runs.csv")
def export_mapping_runs_csv(limit: int = 5000) -> Response:
    with SessionLocal() as db:
        rows = db.execute(select(MappingRun).order_by(MappingRun.created_at.desc()).limit(limit)).scalars().all()
        out = [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "phenotype_field": r.phenotype_field,
                "analysis_run_id": r.analysis_run_id,
                "method": r.method,
                "p_adjust": r.p_adjust,
                "max_markers": r.max_markers,
                "min_n": r.min_n,
                "n_markers_tested": r.n_markers_tested,
                "n_results": r.n_results,
            }
            for r in rows
        ]
        headers = [
            "id",
            "created_at",
            "phenotype_field",
            "analysis_run_id",
            "method",
            "p_adjust",
            "max_markers",
            "min_n",
            "n_markers_tested",
            "n_results",
        ]
        return _csv_response("mapping_runs.csv", headers, out)


@app.get("/observability/mapping_hits.csv")
def export_mapping_hits_csv(run_id: Optional[int] = None, limit: int = 200000) -> Response:
    with SessionLocal() as db:
        rid = run_id
        if rid is None:
            r = db.execute(select(MappingRun).order_by(MappingRun.created_at.desc()).limit(1)).scalars().first()
            rid = int(r.id) if r else None
        if rid is None:
            return _csv_response(
                "mapping_hits.csv",
                ["run_id", "marker_name", "phenotype_field", "method", "p_adjust", "n", "effect", "p_value", "p_adjusted", "r2", "created_at"],
                [],
            )
        rows = (
            db.execute(select(MappingHit).where(MappingHit.run_id == rid).order_by(MappingHit.p_adjusted.asc()).limit(limit))
            .scalars()
            .all()
        )
        out = [
            {
                "run_id": r.run_id,
                "marker_name": r.marker_name,
                "phenotype_field": r.phenotype_field,
                "analysis_run_id": r.analysis_run_id,
                "method": r.method,
                "p_adjust": r.p_adjust,
                "n": r.n,
                "effect": r.effect,
                "p_value": r.p_value,
                "p_adjusted": r.p_adjusted,
                "r2": r.r2,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
        headers = list(out[0].keys()) if out else [
            "run_id","marker_name","phenotype_field","analysis_run_id","method","p_adjust","n","effect","p_value","p_adjusted","r2","created_at"
        ]
        return _csv_response("mapping_hits.csv", headers, out)


@app.get("/observability/export.zip")
def export_observability_zip() -> Response:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # JSON summary
        with SessionLocal() as db:
            summary = {
                "generated_at": _utc_now().isoformat(),
                "stats": {
                    "samples": int(db.execute(select(func.count(GenotypeSample.id))).scalar_one()),
                    "markers": int(db.execute(select(func.count(GenotypeMarker.id))).scalar_one()),
                    "values": int(db.execute(select(func.count(GenotypeValue.id))).scalar_one()),
                },
            }
        z.writestr("genotype_summary.json", json.dumps(summary, indent=2))

        def _add_csv(name: str, resp: Response):
            z.writestr(name, resp.body.decode("utf-8"))

        _add_csv("markers.csv", export_markers_csv())
        _add_csv("samples.csv", export_samples_csv())
        _add_csv("values.csv", export_values_csv(limit=200000))
        _add_csv("mapping_runs.csv", export_mapping_runs_csv())
        _add_csv("mapping_hits.csv", export_mapping_hits_csv(limit=200000))

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="genotype_observability_export.zip"'},
    )
