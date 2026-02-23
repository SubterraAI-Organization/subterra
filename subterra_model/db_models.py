from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnnotationRow(Base):
    __tablename__ = "annotations"
    __table_args__ = (UniqueConstraint("annotation_id", name="uq_annotations_annotation_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    annotation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    original_filename: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    mask_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


class AnalysisRow(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    filename: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), default="", nullable=False)

    threshold_area: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    scaling_factor: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    root_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    average_root_diameter: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_root_length: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_root_area: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_root_volume: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    extra: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


class PhenotypeSampleIdMapRow(Base):
    """
    Optional mapping from analysis `filename` (often an image path) to a biological/sample identifier
    (e.g. an accession or line ID like MAGIC.10) for genotype association.
    """

    __tablename__ = "phenotype_sample_id_map"
    __table_args__ = (UniqueConstraint("filename", name="uq_phenotype_sample_id_map_filename"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    sample_id: Mapped[str] = mapped_column(String(256), nullable=False, default="")


class ModelVersionRow(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("version_id", name="uq_model_versions_version_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)

    model_type: Mapped[str] = mapped_column(String(32), nullable=False, default="unet")
    checkpoint_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    base_checkpoint_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    annotations_dir: Mapped[str] = mapped_column(String(1024), default="", nullable=False)

    train_config: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    is_current: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class TrainJobRow(Base):
    __tablename__ = "train_jobs"
    __table_args__ = (UniqueConstraint("job_id", name="uq_train_jobs_job_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str] = mapped_column(Text, default="", nullable=False)

    planned_version_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    produced_version_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)

    log: Mapped[list] = mapped_column(JSON, default=list, nullable=False)


class ApiKeyRow(Base):
    __tablename__ = "api_keys"
    __table_args__ = (UniqueConstraint("key_id", name="uq_api_keys_key_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key_id: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(256), default="", nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    hashed_key: Mapped[str] = mapped_column(String(128), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
