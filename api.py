from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import secrets
import time
import csv
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple
from uuid import uuid4

# Avoid ~/.matplotlib write issues in locked-down environments when rendering plots.
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "mplconfig"))

import cv2
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from subterra_model.db import SessionLocal, init_db, is_postgres
from subterra_model.db_models import AnalysisRow, AnnotationRow, ApiKeyRow, ModelVersionRow, PhenotypeSampleIdMapRow, TrainJobRow
from subterra_model.loading import load_model
from subterra_model.utils.masks import threshold as threshold_mask
from subterra_model.utils.root_analysis import calculate_metrics


class AnalysisRequest(BaseModel):
    model_type: str = "unet"  # "unet" or "yolo"
    threshold_area: int = 50
    scaling_factor: float = 1.0
    confidence_threshold: float = 0.3


class AnalysisResult(BaseModel):
    root_count: int
    average_root_diameter: float
    total_root_length: float
    total_root_area: float
    total_root_volume: float
    mask_image_base64: str
    original_image_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    device: str


app = FastAPI(
    title="Subterra Root Analysis API",
    description="API for analyzing root images using U-Net and YOLO segmentation models",
    version="1.0.0"
)


class ForwardedPrefixMiddleware:
    """
    Respect reverse-proxy prefixes (nginx sets X-Forwarded-Prefix=/api).

    This keeps Swagger/Redoc working when the API is mounted under a path prefix,
    while still allowing direct access (no prefix header) in dev/testing.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in {"http", "websocket"}:
            headers = dict(scope.get("headers") or [])
            raw = headers.get(b"x-forwarded-prefix")
            if raw:
                prefix = raw.decode("utf-8", errors="ignore").strip()
                if prefix:
                    if not prefix.startswith("/"):
                        prefix = "/" + prefix
                    scope = dict(scope)
                    scope["root_path"] = prefix.rstrip("/")
        await self.app(scope, receive, send)


app.add_middleware(ForwardedPrefixMiddleware)

# CORS (for local GUI dev at localhost:3000)
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

# Global model storage
models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANNOTATIONS_DIR = Path(os.getenv("SUBTERRA_ANNOTATIONS_DIR", "data/annotations"))
MODELS_DIR = Path(os.getenv("SUBTERRA_MODELS_DIR", "data/models"))
REGISTRY_PATH = MODELS_DIR / "registry.json"
AUDIT_DIR = Path(os.getenv("SUBTERRA_AUDIT_DIR", "data/audit"))
ANNOTATION_QC_REJECTIONS_PATH = AUDIT_DIR / "annotation_qc_rejections.jsonl"

_registry_lock = Lock()
_train_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=1)
_train_jobs: dict[str, dict] = {}


def _db_session() -> Session:
    return SessionLocal()


def _api_key_secret() -> str:
    return os.getenv("SUBTERRA_API_KEY_SECRET", "dev-secret-change-me")


def _hash_api_key(token: str) -> str:
    data = f"{token}:{_api_key_secret()}".encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _extract_api_key_header(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        value = authorization.strip()
        if value.lower().startswith("bearer "):
            return value.split(" ", 1)[1].strip()
    return None


def _require_api_key() -> bool:
    return os.getenv("SUBTERRA_REQUIRE_API_KEY", "0").lower() in {"1", "true", "yes"}


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def api_landing() -> str:
    loaded = ", ".join(sorted(models.keys())) if models else "none"
    require_key = "enabled" if _require_api_key() else "disabled"
    admin_token_hint = os.getenv("SUBTERRA_ADMIN_TOKEN", "")
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Subterra API</title>
    <style>
      :root {{
        --bg: #0b1220;
        --panel: rgba(255,255,255,.06);
        --panel2: rgba(255,255,255,.08);
        --border: rgba(255,255,255,.12);
        --text: #eef2ff;
        --muted: rgba(238,242,255,.72);
        --accent: #60a5fa;
        --accent2: #a78bfa;
        --code: rgba(255,255,255,.08);
      }}
      body {{
        margin: 0;
        background: radial-gradient(1200px 700px at 20% 10%, rgba(96,165,250,.25), transparent 55%),
                    radial-gradient(1000px 600px at 80% 20%, rgba(167,139,250,.22), transparent 60%),
                    var(--bg);
        color: var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      }}
      .wrap {{ max-width: 980px; margin: 0 auto; padding: 28px 20px 44px; }}
      .hero {{
        display: flex; justify-content: space-between; align-items: flex-end; gap: 12px; flex-wrap: wrap;
        padding: 18px 18px 14px;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.03));
      }}
      h1 {{ margin: 0; font-size: 18px; letter-spacing: .2px; }}
      .sub {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
      .pill {{
        display: inline-flex; align-items: center; gap: 8px;
        font-size: 12px; color: var(--muted);
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 999px; padding: 6px 10px;
      }}
      .grid {{ margin-top: 14px; display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
      .card {{
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px;
        background: var(--panel);
      }}
      .cardTitle {{ font-weight: 700; font-size: 13px; margin-bottom: 8px; }}
      a {{ color: var(--accent); text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      code {{ background: var(--code); padding: 2px 6px; border-radius: 7px; }}
      pre {{
        margin: 10px 0 0;
        padding: 12px;
        border-radius: 12px;
        background: rgba(0,0,0,.28);
        border: 1px solid var(--border);
        overflow: auto;
        color: #e5e7eb;
        font-size: 12px;
        line-height: 1.35;
      }}
      ul {{ margin: 6px 0 0 18px; color: var(--muted); font-size: 13px; }}
      li {{ margin: 4px 0; }}
      .k {{ color: #e5e7eb; }}
      .hint {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="hero">
        <div>
          <h1>Subterra Root Analysis API</h1>
          <div class="sub">Backend API served behind nginx at <code>/api</code>. Use Swagger/ReDoc for full schemas.</div>
        </div>
        <div class="pill">
          <span class="k">device</span> <code>{device}</code>
          <span class="k">models</span> <code>{loaded}</code>
          <span class="k">auth</span> <code>{require_key}</code>
        </div>
      </div>

      <div class="grid">
        <div class="card">
          <div class="cardTitle">Docs &amp; Status</div>
          <ul>
            <li><a href="docs">Swagger UI</a> (<code>/docs</code>)</li>
            <li><a href="redoc">ReDoc</a> (<code>/redoc</code>)</li>
            <li><a href="openapi.json">OpenAPI JSON</a> (<code>/openapi.json</code>)</li>
            <li><a href="health">Health</a> (<code>/health</code>)</li>
            <li><a href="dashboard">Dashboard</a> (<code>/dashboard</code>)</li>
            <li><a href="models">Models</a> (<code>/models</code>)</li>
          </ul>
          <div class="hint">If you're accessing via the browser, prefer <code>/api/docs</code>.</div>
        </div>

        <div class="card">
          <div class="cardTitle">Runtime</div>
          <ul>
            <li>annotations_dir: <code>{ANNOTATIONS_DIR}</code></li>
            <li>models_dir: <code>{MODELS_DIR}</code></li>
            <li>api_key_auth: <code>{require_key}</code> (<code>SUBTERRA_REQUIRE_API_KEY</code>)</li>
          </ul>
          <div class="hint">When auth is enabled, send <code>X-API-Key</code> or <code>Authorization: Bearer ...</code>.</div>
        </div>

        <div class="card">
          <div class="cardTitle">API Key Management (Admin)</div>
          <ul>
            <li>Create: <code>POST /api-keys</code> (header <code>X-Admin-Token</code>)</li>
            <li>List: <code>GET /api-keys</code> (header <code>X-Admin-Token</code>)</li>
            <li>Revoke: <code>POST /api-keys/&lt;key_id&gt;/revoke</code> (header <code>X-Admin-Token</code>)</li>
          </ul>
          <pre>curl -sS -X POST "http://localhost/api/api-keys" \\
  -H "X-Admin-Token: {admin_token_hint or "YOUR_ADMIN_TOKEN"}" \\
  -H "Content-Type: application/json" \\
  -d '{{"name":"my-script"}}'</pre>
          <div class="hint">Set <code>SUBTERRA_ADMIN_TOKEN</code> and rotate <code>SUBTERRA_API_KEY_SECRET</code> in production.</div>
        </div>

        <div class="card">
          <div class="cardTitle">Inference (Data Creation)</div>
          <ul>
            <li>Single image: <code>POST /analyze</code></li>
            <li>Batch: <code>POST /batch-analyze</code></li>
            <li>Save annotation pair: <code>POST /annotations</code> (image+mask)</li>
          </ul>
          <pre>curl -sS -X POST "http://localhost/api/analyze?model_type=unet&amp;threshold_area=50&amp;scaling_factor=1" \\
  -H "X-API-Key: $SUBTERRA_API_KEY" \\
  -F "file=@/path/to/image.png"</pre>
          <div class="hint">If auth is disabled, the <code>X-API-Key</code> header is optional.</div>
        </div>

        <div class="card">
          <div class="cardTitle">Ingestion (Store External Metrics)</div>
          <ul>
            <li><code>POST /ingest/analysis</code> stores metrics directly (no model inference).</li>
            <li>Useful when you compute phenotypes elsewhere and want them in Postgres for genotype mapping.</li>
          </ul>
          <pre>curl -sS -X POST "http://localhost/api/ingest/analysis" \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: $SUBTERRA_API_KEY" \\
  -d '{{"filename":"CS_T474_L007.PNG","model_type":"unet","model_version":"","threshold_area":50,"scaling_factor":1,
       "confidence_threshold":0.3,"root_count":50,"average_root_diameter":3.0,"total_root_length":6600,
       "total_root_area":33048,"total_root_volume":66341.0,"extra":{{"source":"external"}}}}'</pre>
        </div>
      </div>
    </div>
  </body>
</html>
"""


def _verify_api_key(token: str) -> Optional[ApiKeyRow]:
    if not token:
        return None
    hashed = _hash_api_key(token)
    db = _db_session()
    try:
        row = (
            db.execute(select(ApiKeyRow).where(ApiKeyRow.hashed_key == hashed, ApiKeyRow.revoked.is_(False)))
            .scalars()
            .first()
        )
        if row:
            row.last_used_at = datetime.now(timezone.utc)
            db.commit()
        return row
    except Exception:
        db.rollback()
        return None
    finally:
        db.close()


def require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> ApiKeyRow:
    token = _extract_api_key_header(authorization, x_api_key)
    row = _verify_api_key(token or "")
    if not row:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    return row


def maybe_require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Optional[ApiKeyRow]:
    if not _require_api_key():
        return None
    return require_api_key(authorization=authorization, x_api_key=x_api_key)


def _require_admin(admin_token: Optional[str]) -> None:
    expected = os.getenv("SUBTERRA_ADMIN_TOKEN", "")
    if not expected:
        # If no admin token is configured, do not allow key creation/revoke by default.
        raise HTTPException(status_code=403, detail="Admin token not configured (set SUBTERRA_ADMIN_TOKEN)")
    if not admin_token or admin_token != expected:
        raise HTTPException(status_code=403, detail="Invalid admin token")


def _current_unet_version_id(db: Session) -> str:
    row = (
        db.execute(
            select(ModelVersionRow)
            .where(ModelVersionRow.model_type == "unet", ModelVersionRow.is_current.is_(True))
            .order_by(ModelVersionRow.created_at.desc())
        )
        .scalars()
        .first()
    )
    return row.version_id if row else ""


def _next_unet_version_id_db(db: Session) -> str:
    version_ids = db.execute(select(ModelVersionRow.version_id).where(ModelVersionRow.model_type == "unet")).scalars().all()
    max_n = 0
    for vid in version_ids:
        m = re.match(r"unet_v(\d+)$", str(vid))
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"unet_v{max_n + 1:04d}"


def _bootstrap_db(db: Session) -> None:
    """
    Initialize DB with a base U-Net version and import existing artifacts if present.
    """
    # Ensure a "base" U-Net version exists so versioning can start from current checkpoint.
    has_any = db.execute(select(func.count(ModelVersionRow.id))).scalar_one()
    if int(has_any) == 0:
        base_ckpt = os.getenv("SUBTERRA_UNET_CHECKPOINT") or "subterra_model/models/saved_models/unet_saved.pth"
        if os.path.exists(base_ckpt):
            db.add(
                ModelVersionRow(
                    version_id="unet_v0000",
                    model_type="unet",
                    checkpoint_path=str(base_ckpt),
                    base_checkpoint_path="",
                    annotations_dir="",
                    train_config={},
                    metrics={},
                    is_current=True,
                )
            )
            db.commit()

    # Import existing registry.json (if any) into DB (best-effort).
    if REGISTRY_PATH.exists():
        try:
            registry = _load_registry()
            versions = registry.get("unet", {}).get("versions", []) or []
            current = registry.get("unet", {}).get("current")
            existing = set(
                db.execute(select(ModelVersionRow.version_id).where(ModelVersionRow.model_type == "unet")).scalars().all()
            )
            for v in versions:
                vid = str(v.get("id", ""))
                if not vid or vid in existing:
                    continue
                ckpt = str(v.get("checkpoint_path", ""))
                if not ckpt:
                    continue
                db.add(
                    ModelVersionRow(
                        version_id=vid,
                        model_type="unet",
                        checkpoint_path=ckpt,
                        base_checkpoint_path=str(v.get("base_checkpoint_path") or ""),
                        annotations_dir=str(v.get("annotations_dir") or ""),
                        train_config=dict(v.get("train_config") or {}),
                        metrics=dict(v.get("metrics") or {}),
                        is_current=(vid == current),
                    )
                )
            db.commit()
        except Exception:
            db.rollback()

    # Import existing saved annotations into DB (best-effort, only when table empty).
    anno_count = db.execute(select(func.count(AnnotationRow.id))).scalar_one()
    if int(anno_count) == 0 and ANNOTATIONS_DIR.exists():
        for child in ANNOTATIONS_DIR.iterdir():
            if not child.is_dir():
                continue
            meta = child / "meta.json"
            if not meta.exists():
                continue
            try:
                payload = json.loads(meta.read_text("utf-8"))
            except Exception:
                continue

            annotation_id = str(payload.get("annotation_id") or child.name)
            image_name = str(payload.get("image_filename") or "")
            mask_name = str(payload.get("mask_filename") or "")
            if not image_name or not mask_name:
                continue
            image_path = child / image_name
            mask_path = child / mask_name
            if not image_path.exists() or not mask_path.exists():
                continue

            db.add(
                AnnotationRow(
                    annotation_id=annotation_id,
                    original_filename=str(payload.get("original_filename") or ""),
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    meta=payload if isinstance(payload, dict) else {},
                )
            )
        try:
            db.commit()
        except Exception:
            db.rollback()

def _safe_filename(filename: str) -> str:
    basename = os.path.basename(filename or "")
    basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return basename or "file"


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _csv_bytes(headers: List[str], rows: List[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in headers})
    return buf.getvalue().encode("utf-8")


def _csv_response(filename: str, headers: List[str], rows: List[dict]) -> Response:
    content = _csv_bytes(headers, rows)
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class AnnotationSaveResponse(BaseModel):
    annotation_id: str
    image_path: str
    mask_path: str
    metadata_path: str


class SystemMetricsResponse(BaseModel):
    counts: dict
    annotation_times_utc: List[str]
    model_version_times_utc: List[str]
    model_final_loss: List[Optional[float]]
    model_num_samples: List[Optional[float]]
    qc_rejected_deltas_px: List[int]
    mask_nonzero_fraction: List[float]
    trait_total_root_length: List[float]


class SampleIdMapUploadResponse(BaseModel):
    rows_seen: int
    rows_upserted: int
    rows_skipped: int


class ModelVersion(BaseModel):
    id: str
    created_at: str
    checkpoint_path: str
    base_checkpoint_path: Optional[str] = None
    annotations_dir: str
    train_config: dict
    metrics: dict


class ModelsResponse(BaseModel):
    unet_current: Optional[str] = None
    unet_versions: List[ModelVersion] = []


class DashboardCounts(BaseModel):
    annotations: int
    analyses: int
    model_versions: int
    train_jobs: int
    api_keys: int


class DashboardAnnotation(BaseModel):
    annotation_id: str
    created_at: str
    original_filename: str


class DashboardAnalysis(BaseModel):
    id: int
    filename: str
    created_at: str
    model_type: str
    model_version: str
    root_count: int
    total_root_length: float


class DashboardJob(BaseModel):
    job_id: str
    status: str
    created_at: str
    planned_version_id: str
    produced_version_id: str


class DashboardResponse(BaseModel):
    counts: DashboardCounts
    current_unet_version: Optional[str] = None
    recent_annotations: List[DashboardAnnotation] = []
    recent_analyses: List[DashboardAnalysis] = []
    recent_train_jobs: List[DashboardJob] = []


class ApiKeyCreateRequest(BaseModel):
    name: str = ""


class ApiKeyCreateResponse(BaseModel):
    key_id: str
    name: str
    created_at: str
    api_key: str


class ApiKeyListItem(BaseModel):
    key_id: str
    name: str
    created_at: str
    last_used_at: Optional[str] = None
    revoked: bool


class ApiKeyListResponse(BaseModel):
    keys: List[ApiKeyListItem]


class ApiKeyRevokeResponse(BaseModel):
    key_id: str
    revoked: bool


class IngestAnalysisRequest(BaseModel):
    filename: str
    model_type: str = "unet"
    model_version: str = ""
    run_id: str = ""
    threshold_area: int = 0
    scaling_factor: float = 1.0
    confidence_threshold: float = 0.0
    root_count: int
    average_root_diameter: float
    total_root_length: float
    total_root_area: float
    total_root_volume: float
    extra: dict = {}


class TrainUNetRequest(BaseModel):
    epochs: int = 3
    batch_size: int = 2
    lr: float = 1e-4
    image_size: int = 512
    base_checkpoint: Optional[str] = None
    camera_model: Optional[str] = None
    only_corrected: bool = False
    preserve_aspect_ratio: bool = True
    run_id: Optional[str] = None


class TrainJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    planned_version_id: Optional[str] = None
    produced_version_id: Optional[str] = None
    log: List[str] = []


class AnnotationStatsResponse(BaseModel):
    total_annotations: int
    by_camera_model: dict[str, int]
    missing_camera_model: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_camera_model_from_meta(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ""
    if isinstance(meta.get("camera_model"), str):
        return meta.get("camera_model") or ""
    camera = meta.get("camera") or {}
    if isinstance(camera, dict) and isinstance(camera.get("camera_model"), str):
        return camera.get("camera_model") or ""
    nested = meta.get("meta") or {}
    if isinstance(nested, dict) and isinstance(nested.get("camera_model"), str):
        return nested.get("camera_model") or ""
    mini = meta.get("minirhizotron") or {}
    if isinstance(mini, dict) and isinstance(mini.get("camera_model"), str):
        return mini.get("camera_model") or ""
    return ""


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"unet": {"current": None, "versions": []}}
    try:
        return json.loads(REGISTRY_PATH.read_text("utf-8"))
    except Exception:
        return {"unet": {"current": None, "versions": []}}


def _save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _next_unet_version_id(registry: dict) -> str:
    versions = registry.get("unet", {}).get("versions", [])
    max_n = 0
    for v in versions:
        vid = str(v.get("id", ""))
        m = re.match(r"unet_v(\d+)$", vid)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"unet_v{max_n + 1:04d}"


def _resolve_unet_checkpoint_path() -> Optional[str]:
    env_path = os.getenv("SUBTERRA_UNET_CHECKPOINT")
    if env_path and os.path.exists(env_path):
        return env_path

    try:
        db = _db_session()
        try:
            row = (
                db.execute(
                    select(ModelVersionRow)
                    .where(ModelVersionRow.model_type == "unet", ModelVersionRow.is_current.is_(True))
                    .order_by(ModelVersionRow.created_at.desc())
                )
                .scalars()
                .first()
            )
            if row and row.checkpoint_path and os.path.exists(row.checkpoint_path):
                return row.checkpoint_path
        finally:
            db.close()
    except Exception:
        pass

    with _registry_lock:
        registry = _load_registry()
        current = registry.get("unet", {}).get("current")
        if current:
            for v in registry.get("unet", {}).get("versions", []):
                if v.get("id") == current and v.get("checkpoint_path") and os.path.exists(v["checkpoint_path"]):
                    return v["checkpoint_path"]

    default_path = "subterra_model/models/saved_models/unet_saved.pth"
    return default_path if os.path.exists(default_path) else None


def load_saved_models():
    """Load the saved models on startup"""
    global models

    model_paths = {
        "unet": _resolve_unet_checkpoint_path(),
        "yolo": "subterra_model/models/saved_models/yolo_saved.pt",
    }

    for model_name, model_path in model_paths.items():
        if model_path and os.path.exists(model_path):
            try:
                models[model_name] = load_model(model_name, model_path, device=device)
                print(f"Loaded {model_name} model successfully")
            except Exception as e:
                print(f"Failed to load {model_name} model: {e}")
        else:
            print(f"Model file not found: {model_path}")


def preprocess_image(image: Image.Image, max_size: int = 1024) -> torch.Tensor:
    """Preprocess PIL image for model inference with automatic resizing for large images"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize large images to prevent memory issues
    width, height = image.size
    if max(width, height) > max_size:
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)

        print(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0

    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

    return tensor


def postprocess_mask(mask_tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Convert model output to binary mask"""
    mask = mask_tensor.squeeze().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]  # Take first channel if multi-channel
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    return binary_mask


def yolo_predict(model, image: torch.Tensor, confidence_threshold: float) -> np.ndarray:
    """Run YOLO inference and combine masks"""
    # Convert tensor back to PIL for YOLO
    image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

    # Run prediction
    results = model.predict(image_pil, conf=confidence_threshold)

    if len(results) == 0 or results[0].masks is None:
        return np.zeros((image.shape[2], image.shape[3]), dtype=np.uint8)

    # Get masks and combine them
    masks = results[0].masks.data.cpu().numpy()
    if masks.ndim == 3:
        # Combine multiple instance masks
        combined_mask = np.max(masks, axis=0)
    else:
        combined_mask = masks

    # Convert to binary mask
    binary_mask = (combined_mask > 0).astype(np.uint8) * 255
    return binary_mask


def encode_image_to_base64(image_array: np.ndarray) -> str:
    """Encode numpy array image to base64 string"""
    if image_array.ndim == 2:
        # Grayscale mask
        image_pil = Image.fromarray(image_array.astype(np.uint8), mode='L')
    else:
        # RGB image
        image_pil = Image.fromarray(image_array.astype(np.uint8))

    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


@app.on_event("startup")
async def startup_event():
    """Load models when the application starts"""
    try:
        init_db()
    except Exception as e:
        # Don't prevent the API from starting if Postgres isn't ready yet.
        print(f"DB init failed (continuing without DB): {e}")
    _cleanup_stale_train_jobs_on_startup()
    load_saved_models()
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with _registry_lock:
        if not REGISTRY_PATH.exists():
            _save_registry({"unet": {"current": None, "versions": []}})
    try:
        db = _db_session()
        try:
            _bootstrap_db(db)
        finally:
            db.close()
    except Exception as e:
        print(f"DB bootstrap skipped: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the API and loaded models"""
    loaded_models = list(models.keys())
    return HealthResponse(
        status="healthy" if loaded_models else "unhealthy",
        models_loaded=loaded_models,
        device=str(device)
    )


@app.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard():
    try:
        db = _db_session()
        try:
            annotations_count = int(db.execute(select(func.count(AnnotationRow.id))).scalar_one())
            analyses_count = int(db.execute(select(func.count(AnalysisRow.id))).scalar_one())
            model_versions_count = int(db.execute(select(func.count(ModelVersionRow.id))).scalar_one())
            train_jobs_count = int(db.execute(select(func.count(TrainJobRow.id))).scalar_one())
            api_keys_count = int(db.execute(select(func.count(ApiKeyRow.id))).scalar_one())

            current_unet = _current_unet_version_id(db) or None

            recent_annotations = (
                db.execute(select(AnnotationRow).order_by(AnnotationRow.created_at.desc()).limit(10)).scalars().all()
            )
            recent_analyses = (
                db.execute(select(AnalysisRow).order_by(AnalysisRow.created_at.desc()).limit(10)).scalars().all()
            )
            recent_jobs = db.execute(select(TrainJobRow).order_by(TrainJobRow.created_at.desc()).limit(10)).scalars().all()

            return DashboardResponse(
                counts=DashboardCounts(
                    annotations=annotations_count,
                    analyses=analyses_count,
                    model_versions=model_versions_count,
                    train_jobs=train_jobs_count,
                    api_keys=api_keys_count,
                ),
                current_unet_version=current_unet,
                recent_annotations=[
                    DashboardAnnotation(
                        annotation_id=a.annotation_id,
                        created_at=a.created_at.isoformat(),
                        original_filename=a.original_filename,
                    )
                    for a in recent_annotations
                ],
                recent_analyses=[
                    DashboardAnalysis(
                        id=a.id,
                        filename=a.filename,
                        created_at=a.created_at.isoformat(),
                        model_type=a.model_type,
                        model_version=a.model_version,
                        root_count=a.root_count,
                        total_root_length=a.total_root_length,
                    )
                    for a in recent_analyses
                ],
                recent_train_jobs=[
                    DashboardJob(
                        job_id=j.job_id,
                        status=j.status,
                        created_at=j.created_at.isoformat(),
                        planned_version_id=j.planned_version_id,
                        produced_version_id=j.produced_version_id,
                    )
                    for j in recent_jobs
                ],
            )
        finally:
            db.close()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")


@app.post("/api-keys", response_model=ApiKeyCreateResponse)
async def create_api_key(req: ApiKeyCreateRequest, x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")):
    _require_admin(x_admin_token)

    raw = f"stk_{secrets.token_urlsafe(32)}"
    key_id = uuid4().hex
    hashed = _hash_api_key(raw)

    db = _db_session()
    try:
        db.add(ApiKeyRow(key_id=key_id, name=req.name.strip(), hashed_key=hashed, revoked=False))
        db.commit()
        return ApiKeyCreateResponse(key_id=key_id, name=req.name.strip(), created_at=_utc_now_iso(), api_key=raw)
    finally:
        db.close()


@app.get("/api-keys", response_model=ApiKeyListResponse)
async def list_api_keys(x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")):
    _require_admin(x_admin_token)
    db = _db_session()
    try:
        rows = db.execute(select(ApiKeyRow).order_by(ApiKeyRow.created_at.desc())).scalars().all()
        return ApiKeyListResponse(
            keys=[
                ApiKeyListItem(
                    key_id=r.key_id,
                    name=r.name,
                    created_at=r.created_at.isoformat(),
                    last_used_at=r.last_used_at.isoformat() if r.last_used_at else None,
                    revoked=bool(r.revoked),
                )
                for r in rows
            ]
        )
    finally:
        db.close()


@app.post("/api-keys/{key_id}/revoke", response_model=ApiKeyRevokeResponse)
async def revoke_api_key(key_id: str, x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token")):
    _require_admin(x_admin_token)
    db = _db_session()
    try:
        row = db.execute(select(ApiKeyRow).where(ApiKeyRow.key_id == key_id)).scalars().first()
        if not row:
            raise HTTPException(status_code=404, detail="Key not found")
        row.revoked = True
        db.commit()
        return ApiKeyRevokeResponse(key_id=key_id, revoked=True)
    finally:
        db.close()


@app.post("/ingest/analysis")
async def ingest_analysis(req: IngestAnalysisRequest, api_key: ApiKeyRow = Depends(require_api_key)):
    # Store externally computed phenotypes/metrics (no model inference).
    db = _db_session()
    try:
        extra = dict(req.extra or {})
        meta = extra.get("meta") if isinstance(extra.get("meta"), dict) else {}
        if req.run_id:
            meta = dict(meta)
            meta.setdefault("run_id", req.run_id)
            extra["meta"] = meta
        db.add(
            AnalysisRow(
                filename=req.filename,
                model_type=req.model_type,
                model_version=req.model_version,
                threshold_area=req.threshold_area,
                scaling_factor=req.scaling_factor,
                confidence_threshold=req.confidence_threshold,
                root_count=req.root_count,
                average_root_diameter=req.average_root_diameter,
                total_root_length=req.total_root_length,
                total_root_area=req.total_root_area,
                total_root_volume=req.total_root_volume,
                extra={**extra, "ingested_via": "api_key", "api_key_id": api_key.key_id},
            )
        )
        db.commit()
        return {"success": True}
    finally:
        db.close()


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    try:
        db = _db_session()
        try:
            current = _current_unet_version_id(db) or None
            rows = (
                db.execute(
                    select(ModelVersionRow)
                    .where(ModelVersionRow.model_type == "unet")
                    .order_by(ModelVersionRow.created_at.asc())
                )
                .scalars()
                .all()
            )
            versions = [
                ModelVersion(
                    id=r.version_id,
                    created_at=r.created_at.isoformat(),
                    checkpoint_path=r.checkpoint_path,
                    base_checkpoint_path=r.base_checkpoint_path or None,
                    annotations_dir=r.annotations_dir,
                    train_config=r.train_config or {},
                    metrics=r.metrics or {},
                )
                for r in rows
            ]
            return ModelsResponse(unet_current=current, unet_versions=versions)
        finally:
            db.close()
    except Exception:
        with _registry_lock:
            registry = _load_registry()
        versions = registry.get("unet", {}).get("versions", []) or []
        current = registry.get("unet", {}).get("current")
        parsed_versions: List[ModelVersion] = []
        for v in versions:
            try:
                parsed_versions.append(ModelVersion(**v))
            except Exception:
                continue
        return ModelsResponse(unet_current=current, unet_versions=parsed_versions)


def _set_job(job_id: str, **updates) -> None:
    with _train_lock:
        job = _train_jobs.get(job_id)
        if not job:
            return
        job.update(updates)
        persisted = dict(job)

    try:
        db = _db_session()
        try:
            row = db.execute(select(TrainJobRow).where(TrainJobRow.job_id == job_id)).scalars().first()
            if not row:
                return
            if "status" in persisted and persisted["status"] is not None:
                row.status = str(persisted["status"])
            if "error" in persisted and persisted["error"] is not None:
                row.error = str(persisted["error"] or "")
            if "planned_version_id" in persisted and persisted["planned_version_id"] is not None:
                row.planned_version_id = str(persisted["planned_version_id"] or "")
            if "produced_version_id" in persisted and persisted["produced_version_id"] is not None:
                row.produced_version_id = str(persisted["produced_version_id"] or "")
            if persisted.get("started_at"):
                row.started_at = datetime.fromisoformat(persisted["started_at"])
            if persisted.get("finished_at"):
                row.finished_at = datetime.fromisoformat(persisted["finished_at"])
            db.commit()
        finally:
            db.close()
    except Exception:
        pass


def _append_job_log(job_id: str, line: str) -> None:
    with _train_lock:
        job = _train_jobs.get(job_id)
        if not job:
            return
        job.setdefault("log", []).append(line)
        job["log"] = job["log"][-200:]
        log = list(job["log"])

    try:
        db = _db_session()
        try:
            row = db.execute(select(TrainJobRow).where(TrainJobRow.job_id == job_id)).scalars().first()
            if not row:
                return
            row.log = log
            db.commit()
        finally:
            db.close()
    except Exception:
        pass


def _cleanup_stale_train_jobs_on_startup() -> None:
    """
    Training runs execute in-process (ThreadPoolExecutor). If the API restarts,
    any previously "queued"/"running" jobs stored in the DB cannot complete.

    Mark them failed so the GUI doesn't show a zombie run and new training isn't blocked.
    """
    try:
        db = _db_session()
    except Exception:
        return
    try:
        rows = (
            db.execute(select(TrainJobRow).where(TrainJobRow.status.in_(["queued", "running"])))
            .scalars()
            .all()
        )
        if not rows:
            return
        now = datetime.now(timezone.utc)
        for r in rows:
            r.status = "failed"
            if not r.finished_at:
                r.finished_at = now
            if not (r.error or "").strip():
                r.error = "stale_job_after_restart"
            log = list(r.log or [])
            log.append(f"Marked failed on startup at {now.isoformat()} (stale job after restart).")
            r.log = log[-200:]
        db.commit()
    finally:
        db.close()


def _run_unet_finetune_job(job_id: str, request: TrainUNetRequest) -> None:
    _set_job(job_id, status="running", started_at=_utc_now_iso())
    _append_job_log(job_id, f"Starting U-Net fine-tune on device={device}")

    try:
        from subterra_model.training.unet_finetune import finetune_unet_from_annotations

        base_ckpt = request.base_checkpoint or _resolve_unet_checkpoint_path()
        if not base_ckpt or not os.path.exists(base_ckpt):
            raise RuntimeError("Base checkpoint not found; set SUBTERRA_UNET_CHECKPOINT or add a trained version.")

        try:
            db = _db_session()
            try:
                version_id = _next_unet_version_id_db(db)
            finally:
                db.close()
        except Exception:
            with _registry_lock:
                registry = _load_registry()
            version_id = _next_unet_version_id(registry)

        version_dir = MODELS_DIR / "unet" / version_id
        checkpoint_path = version_dir / "unet.pth"

        _append_job_log(job_id, f"Base checkpoint: {base_ckpt}")
        _append_job_log(job_id, f"Output checkpoint: {checkpoint_path}")
        _append_job_log(job_id, f"Annotations dir: {ANNOTATIONS_DIR}")

        metrics = finetune_unet_from_annotations(
            annotations_dir=ANNOTATIONS_DIR,
            base_checkpoint_path=base_ckpt,
            output_checkpoint_path=checkpoint_path,
            device=device,
            epochs=request.epochs,
            batch_size=request.batch_size,
            lr=request.lr,
            image_size=request.image_size,
            camera_model=request.camera_model,
            only_corrected=bool(request.only_corrected),
            preserve_aspect_ratio=bool(request.preserve_aspect_ratio),
            run_id=(request.run_id or "").strip() or None,
            log_fn=lambda s: _append_job_log(job_id, s),
        )

        version_record = {
            "id": version_id,
            "created_at": _utc_now_iso(),
            "checkpoint_path": str(checkpoint_path),
            "base_checkpoint_path": str(base_ckpt),
            "annotations_dir": str(ANNOTATIONS_DIR),
            "train_config": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
            "metrics": metrics,
        }

        try:
            db = _db_session()
            try:
                # mark all previous as not current
                prev = (
                    db.execute(select(ModelVersionRow).where(ModelVersionRow.model_type == "unet", ModelVersionRow.is_current.is_(True)))
                    .scalars()
                    .all()
                )
                for r in prev:
                    r.is_current = False
                db.add(
                    ModelVersionRow(
                        version_id=version_id,
                        model_type="unet",
                        checkpoint_path=str(checkpoint_path),
                        base_checkpoint_path=str(base_ckpt),
                        annotations_dir=str(ANNOTATIONS_DIR),
                        train_config=version_record.get("train_config") or {},
                        metrics=metrics,
                        is_current=True,
                    )
                )
                db.commit()
            finally:
                db.close()
        except Exception:
            pass

        # Keep registry.json updated for backward compatibility
        with _registry_lock:
            registry = _load_registry()
            registry.setdefault("unet", {}).setdefault("versions", []).append(version_record)
            registry.setdefault("unet", {})["current"] = version_id
            _save_registry(registry)

        # Hot-reload UNet in memory for immediate inference
        models["unet"] = load_model("unet", str(checkpoint_path), device=device)

        _append_job_log(job_id, f"Training complete. Current version set to {version_id}")
        _set_job(job_id, status="succeeded", finished_at=_utc_now_iso(), produced_version_id=version_id)
    except Exception as e:
        _append_job_log(job_id, f"FAILED: {e}")
        _set_job(job_id, status="failed", finished_at=_utc_now_iso(), error=str(e))


@app.post("/train/unet", response_model=TrainJobResponse)
async def train_unet(req: TrainUNetRequest, _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key)):
    # prevent concurrent training
    try:
        db = _db_session()
        try:
            running = db.execute(select(func.count(TrainJobRow.id)).where(TrainJobRow.status == "running")).scalar_one()
            if int(running) > 0:
                raise HTTPException(status_code=409, detail="A training job is already running")
            planned_version = _next_unet_version_id_db(db)
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception:
        with _train_lock:
            if any(j.get("status") == "running" for j in _train_jobs.values()):
                raise HTTPException(status_code=409, detail="A training job is already running")
        with _registry_lock:
            registry = _load_registry()
            planned_version = _next_unet_version_id(registry)

    job_id = uuid4().hex
    run_id = (req.run_id or "").strip()
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "planned_version_id": planned_version,
        "produced_version_id": None,
        "log": [f"Queued job {job_id} (planned {planned_version})", *( [f"run_id={run_id}"] if run_id else [] )],
    }
    with _train_lock:
        _train_jobs[job_id] = job
    try:
        db = _db_session()
        try:
            db.add(
                TrainJobRow(
                    job_id=job_id,
                    status="queued",
                    error="",
                    planned_version_id=str(planned_version or ""),
                    produced_version_id="",
                    log=job["log"],
                )
            )
            db.commit()
        finally:
            db.close()
    except Exception:
        pass

    _executor.submit(_run_unet_finetune_job, job_id, req)
    return TrainJobResponse(**job)


@app.get("/train/jobs", response_model=List[TrainJobResponse])
async def list_train_jobs():
    try:
        db = _db_session()
        try:
            rows = db.execute(select(TrainJobRow).order_by(TrainJobRow.created_at.desc())).scalars().all()
            return [
                TrainJobResponse(
                    job_id=r.job_id,
                    status=r.status,
                    created_at=r.created_at.isoformat(),
                    started_at=r.started_at.isoformat() if r.started_at else None,
                    finished_at=r.finished_at.isoformat() if r.finished_at else None,
                    error=r.error or None,
                    planned_version_id=r.planned_version_id or None,
                    produced_version_id=r.produced_version_id or None,
                    log=[str(x) for x in (r.log or [])],
                )
                for r in rows
            ]
        finally:
            db.close()
    except Exception:
        with _train_lock:
            jobs = list(_train_jobs.values())
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return [TrainJobResponse(**j) for j in jobs]


@app.get("/train/jobs/{job_id}", response_model=TrainJobResponse)
async def get_train_job(job_id: str):
    try:
        db = _db_session()
        try:
            r = db.execute(select(TrainJobRow).where(TrainJobRow.job_id == job_id)).scalars().first()
            if not r:
                raise HTTPException(status_code=404, detail="Job not found")
            return TrainJobResponse(
                job_id=r.job_id,
                status=r.status,
                created_at=r.created_at.isoformat(),
                started_at=r.started_at.isoformat() if r.started_at else None,
                finished_at=r.finished_at.isoformat() if r.finished_at else None,
                error=r.error or None,
                planned_version_id=r.planned_version_id or None,
                produced_version_id=r.produced_version_id or None,
                log=[str(x) for x in (r.log or [])],
            )
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception:
        with _train_lock:
            job = _train_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return TrainJobResponse(**job)


@app.post("/analyze", response_model=AnalysisResult)
def analyze_root_image(
    file: UploadFile = File(...),
    model_type: str = "unet",
    threshold_area: int = 50,
    scaling_factor: float = 1.0,
    confidence_threshold: float = 0.3,
    metadata_json: Optional[str] = Form(None),
    run_id: Optional[str] = Form(None),
    _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key),
):
    """
    Analyze a root image using the specified model.

    - **file**: Image file to analyze (PNG, JPG, JPEG)
    - **model_type**: Model to use ("unet" or "yolo")
    - **threshold_area**: Minimum area threshold for root detection
    - **scaling_factor**: Scaling factor for metric calculations
    - **confidence_threshold**: Confidence threshold for YOLO model (ignored for U-Net)
    """

    # Validate model type
    if model_type not in ["unet", "yolo"]:
        raise HTTPException(status_code=400, detail="model_type must be 'unet' or 'yolo'")

    # Check if model is loaded
    if model_type not in models:
        raise HTTPException(status_code=503, detail=f"{model_type} model not loaded")

    try:
        t0 = time.perf_counter()
        print(f"Processing image: {file.filename}, model: {model_type}")

        # Read and validate image
        t_read0 = time.perf_counter()
        contents = file.file.read()
        t_read1 = time.perf_counter()
        image = Image.open(io.BytesIO(contents))
        print(f"Image loaded: {image.size}, mode: {image.mode}")

        # Preprocess image
        t_pre0 = time.perf_counter()
        tensor_image = preprocess_image(image)
        t_pre1 = time.perf_counter()
        print(f"Tensor shape: {tensor_image.shape}")

        # Run inference
        model = models[model_type]
        print(f"Running {model_type} inference...")

        if model_type == "unet":
            with torch.no_grad():
                print("Starting U-Net forward pass...")
                t_inf0 = time.perf_counter()
                output = model.model(tensor_image.to(device))
                t_inf1 = time.perf_counter()
                print(f"U-Net output shape: {output.shape}")
                t_post0 = time.perf_counter()
                mask = postprocess_mask(output)
                t_post1 = time.perf_counter()
                print(f"Mask shape: {mask.shape}")
        else:  # yolo
            print("Starting YOLO inference...")
            t_inf0 = time.perf_counter()
            mask = yolo_predict(model.model, tensor_image, confidence_threshold)
            t_inf1 = time.perf_counter()
            t_post0 = t_post1 = t_inf1
            print(f"YOLO mask shape: {mask.shape}")

        # Apply area thresholding
        if threshold_area > 0:
            print(f"Applying area thresholding with threshold: {threshold_area}")
            t_thr0 = time.perf_counter()
            mask = threshold_mask(mask, threshold_area)
            t_thr1 = time.perf_counter()
        else:
            t_thr0 = t_thr1 = time.perf_counter()

        # Calculate metrics
        print("Calculating metrics...")
        t_met0 = time.perf_counter()
        metrics = calculate_metrics(mask, scaling_factor)
        t_met1 = time.perf_counter()
        print(f"Metrics calculated: {metrics}")

        # Convert images to base64
        t_enc0 = time.perf_counter()
        mask_base64 = encode_image_to_base64(mask)
        original_image_array = np.array(image)
        original_base64 = encode_image_to_base64(original_image_array)
        t_enc1 = time.perf_counter()
        t1 = time.perf_counter()

        result = AnalysisResult(
            **metrics,
            mask_image_base64=mask_base64,
            original_image_base64=original_base64
        )

        metadata: dict = {}
        if metadata_json:
            try:
                parsed = json.loads(metadata_json)
                if isinstance(parsed, dict):
                    metadata = parsed
            except json.JSONDecodeError:
                metadata = {"metadata_json_error": "Invalid JSON", "metadata_json_raw": metadata_json}
        if run_id:
            metadata = dict(metadata)
            metadata.setdefault("run_id", run_id)
        # Persist analysis metadata + metrics (best-effort)
        try:
            db = _db_session()
            try:
                model_version = _current_unet_version_id(db) if model_type == "unet" else ""
                db.add(
                    AnalysisRow(
                        filename=str(file.filename or ""),
                        model_type=str(model_type),
                        model_version=str(model_version or ""),
                        threshold_area=int(threshold_area),
                        scaling_factor=float(scaling_factor),
                        confidence_threshold=float(confidence_threshold),
                        root_count=int(metrics.get("root_count", 0)),
                        average_root_diameter=float(metrics.get("average_root_diameter", 0.0)),
                        total_root_length=float(metrics.get("total_root_length", 0.0)),
                        total_root_area=float(metrics.get("total_root_area", 0.0)),
                        total_root_volume=float(metrics.get("total_root_volume", 0.0)),
                        extra={
                            "image_size": list(getattr(image, "size", ())),
                            "image_mode": str(getattr(image, "mode", "")),
                            "file_bytes": int(len(contents)),
                            "timing_ms": {
                                "total": round((t1 - t0) * 1000, 3),
                                "read": round((t_read1 - t_read0) * 1000, 3),
                                "preprocess": round((t_pre1 - t_pre0) * 1000, 3),
                                "inference": round((t_inf1 - t_inf0) * 1000, 3),
                                "postprocess": round((t_post1 - t_post0) * 1000, 3),
                                "threshold": round((t_thr1 - t_thr0) * 1000, 3),
                                "metrics": round((t_met1 - t_met0) * 1000, 3),
                                "encode": round((t_enc1 - t_enc0) * 1000, 3),
                            },
                            "meta": metadata,
                        },
                    )
                )
                db.commit()
            finally:
                db.close()
        except Exception:
            pass

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/batch-analyze")
def batch_analyze_root_images(
    files: List[UploadFile] = File(...),
    model_type: str = "unet",
    threshold_area: int = 50,
    scaling_factor: float = 1.0,
    confidence_threshold: float = 0.3,
    metadata_json: Optional[List[str]] = Form(None),
    run_id: Optional[str] = Form(None),
    _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key),
):
    """
    Analyze multiple root images in batch.

    Returns a list of analysis results for each image.
    """
    results = []
    for idx, file in enumerate(files):
        try:
            meta = None
            if metadata_json and idx < len(metadata_json):
                meta = metadata_json[idx]
            result = analyze_root_image(
                file=file,
                model_type=model_type,
                threshold_area=threshold_area,
                scaling_factor=scaling_factor,
                confidence_threshold=confidence_threshold,
                metadata_json=meta,
                run_id=run_id,
            )
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result
            })
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {"results": results}


def _collect_system_metrics(db: Optional[Session], run_id: Optional[str] = None) -> SystemMetricsResponse:
    rid = (run_id or "").strip() or None
    # Annotation activity + mask coverage from meta.json
    ann_times: List[str] = []
    mask_fracs: List[float] = []
    trait_lengths_from_annotations: List[float] = []
    try:
        if ANNOTATIONS_DIR.exists():
            for child in sorted(ANNOTATIONS_DIR.iterdir()):
                if not child.is_dir():
                    continue
                meta_path = child / "meta.json"
                if not meta_path.exists():
                    continue
                try:
                    meta = json.loads(meta_path.read_text("utf-8"))
                except Exception:
                    meta = {}
                if rid and str(meta.get("run_id") or "").strip() != rid:
                    continue
                ts = meta.get("saved_at")
                if isinstance(ts, str) and ts:
                    ann_times.append(ts)
                frac = meta.get("mask_nonzero_fraction")
                if isinstance(frac, (int, float)):
                    mask_fracs.append(float(frac))
                metrics = meta.get("metrics")
                if isinstance(metrics, dict):
                    v = metrics.get("total_root_length")
                    if isinstance(v, (int, float)):
                        trait_lengths_from_annotations.append(float(v))
    except Exception:
        pass

    # Model versions from registry.json (has created_at + training loss)
    v_times: List[str] = []
    v_loss: List[Optional[float]] = []
    v_nsamp: List[Optional[float]] = []
    try:
        if REGISTRY_PATH.exists():
            reg = json.loads(REGISTRY_PATH.read_text("utf-8"))
            unet = reg.get("unet") if isinstance(reg, dict) else None
            versions = (unet or {}).get("versions") if isinstance(unet, dict) else None
            if isinstance(versions, list):
                for v in versions:
                    if not isinstance(v, dict):
                        continue
                    if rid:
                        tc = v.get("train_config") if isinstance(v.get("train_config"), dict) else {}
                        if str((tc or {}).get("run_id") or "").strip() != rid:
                            continue
                    ct = v.get("created_at")
                    if isinstance(ct, str) and ct:
                        v_times.append(ct)
                    met = v.get("metrics") if isinstance(v.get("metrics"), dict) else {}
                    loss = met.get("final_loss")
                    ns = met.get("num_samples")
                    v_loss.append(float(loss) if isinstance(loss, (int, float)) else None)
                    v_nsamp.append(float(ns) if isinstance(ns, (int, float)) else None)
    except Exception:
        pass

    # QC rejection deltas (dimension mismatches rejected by /annotations)
    rejected_deltas: List[int] = []
    try:
        if ANNOTATION_QC_REJECTIONS_PATH.exists():
            for line in ANNOTATION_QC_REJECTIONS_PATH.read_text("utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("reason") != "dimension_mismatch":
                    continue
                if rid and str(obj.get("run_id") or "").strip() != rid:
                    continue
                iw = obj.get("image_w")
                ih = obj.get("image_h")
                mw = obj.get("mask_w")
                mh = obj.get("mask_h")
                if all(isinstance(v, int) for v in (iw, ih, mw, mh)):
                    rejected_deltas.append(abs(iw - mw) + abs(ih - mh))
    except Exception:
        pass

    # Trait distribution from analyses table (preferred; has more samples than corrected annotations)
    trait_lengths: List[float] = []
    images_phenotyped = 0
    if db is not None:
        try:
            q = select(func.count(func.distinct(AnalysisRow.filename))).where(AnalysisRow.filename != "")
            if rid:
                if is_postgres():
                    q = q.where(text("(extra->'meta'->>'run_id') = :rid")).params(rid=rid)
                else:
                    q = q.where(text("json_extract(extra, '$.meta.run_id') = :rid")).params(rid=rid)
            images_phenotyped = int(db.execute(q).scalar_one())
        except Exception:
            images_phenotyped = 0
        try:
            q = select(AnalysisRow.total_root_length).order_by(AnalysisRow.created_at.desc()).limit(5000)
            if rid:
                if is_postgres():
                    q = q.where(text("(extra->'meta'->>'run_id') = :rid")).params(rid=rid)
                else:
                    q = q.where(text("json_extract(extra, '$.meta.run_id') = :rid")).params(rid=rid)
            rows = db.execute(q).all()
            for (v,) in rows:
                if isinstance(v, (int, float)):
                    trait_lengths.append(float(v))
        except Exception:
            trait_lengths = []

    if not trait_lengths and trait_lengths_from_annotations:
        trait_lengths = trait_lengths_from_annotations

    counts = {
        "images_phenotyped": images_phenotyped,
        "corrected_pairs": len(ann_times) if ann_times else 0,
        "model_versions": len(v_times) if v_times else 0,
        "qc_rejections": len(rejected_deltas),
    }

    return SystemMetricsResponse(
        counts=counts,
        annotation_times_utc=sorted(ann_times),
        model_version_times_utc=v_times,
        model_final_loss=v_loss,
        model_num_samples=v_nsamp,
        qc_rejected_deltas_px=rejected_deltas,
        mask_nonzero_fraction=mask_fracs,
        trait_total_root_length=trait_lengths,
    )


@app.get("/system/metrics", response_model=SystemMetricsResponse)
def system_metrics(run_id: Optional[str] = None, _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key)) -> SystemMetricsResponse:
    """
    End-to-end system metrics for Figure 1-style inset plots (collected from stored artifacts + DB).
    """
    try:
        db = _db_session()
    except Exception:
        db = None
    try:
        # Filter by run_id (best-effort). When set, attempts to only include items tagged with that run_id.
        return _collect_system_metrics(db, run_id=run_id)
    finally:
        try:
            if db is not None:
                db.close()
        except Exception:
            pass


@app.get("/system/figure1-inset.png")
def system_figure1_inset_png(run_id: Optional[str] = None, _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key)) -> Response:
    """
    Render a Figure 1 inset plot (2x3 panels) from real, stored system data.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"matplotlib is required to render plots: {e}")

    try:
        db = _db_session()
    except Exception:
        db = None
    try:
        data = _collect_system_metrics(db, run_id=run_id).model_dump()
    finally:
        try:
            if db is not None:
                db.close()
        except Exception:
            pass

    # Parse times
    def _dt_list(xs: List[str]) -> List[datetime]:
        out: List[datetime] = []
        for s in xs:
            try:
                out.append(datetime.fromisoformat(s.replace("Z", "+00:00")))
            except Exception:
                continue
        return sorted(out)

    ann_times = _dt_list(list(data.get("annotation_times_utc") or []))
    v_times = _dt_list(list(data.get("model_version_times_utc") or []))
    v_loss = list(data.get("model_final_loss") or [])
    v_nsamp = list(data.get("model_num_samples") or [])
    rejected = [int(x) for x in (data.get("qc_rejected_deltas_px") or []) if isinstance(x, int)]
    cover = [float(x) for x in (data.get("mask_nonzero_fraction") or []) if isinstance(x, (int, float))]
    trait = [float(x) for x in (data.get("trait_total_root_length") or []) if isinstance(x, (int, float))]
    counts = data.get("counts") or {}

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(12.6, 6.6))
    # Leave extra whitespace on the right so panel labels/ticks don't get clipped (notably C and F).
    gs = fig.add_gridspec(2, 3, wspace=0.18, hspace=0.32, left=0.06, right=0.972, top=0.98, bottom=0.12)

    if ann_times:
        start = min(ann_times)
        end = max(ann_times)
    elif v_times:
        start = min(v_times)
        end = max(v_times)
    else:
        start = datetime.now(timezone.utc) - timedelta(days=7)
        end = datetime.now(timezone.utc)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
    formatter = mdates.ConciseDateFormatter(locator)

    def _format_date_axis(ax):
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", rotation=0, pad=3)

    # A
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A. System counts")
    labels = ["images phenotyped", "corrected pairs", "model versions", "QC rejections"]
    vals = [int(counts.get("images_phenotyped") or 0), int(counts.get("corrected_pairs") or 0), int(counts.get("model_versions") or 0), int(counts.get("qc_rejections") or 0)]
    ax.bar(range(len(vals)), vals, edgecolor="#334155")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("n")
    for i, v in enumerate(vals):
        ax.text(i, v, f" {v}", va="bottom")

    # B
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B. Annotation activity")
    if ann_times:
        ys = np.arange(1, len(ann_times) + 1)
        ax.plot(ann_times, ys, marker="o", markersize=3.5, linewidth=1.6, color="#f59e0b")
        ax.set_ylabel("cumulative saved")
        _format_date_axis(ax)
    else:
        ax.text(0.5, 0.5, "No annotations yet", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    # C
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("C. Versioned fine-tuning")
    if v_times and len(v_times) == len(v_loss):
        losses = np.array([float(x) if x is not None else np.nan for x in v_loss], dtype=float)
        ns = np.array([float(x) if x is not None else np.nan for x in v_nsamp], dtype=float)
        sizes = 28 + 0.22 * np.nan_to_num(ns, nan=20.0)
        ax.plot(v_times, losses, linewidth=1.6, color="#a78bfa")
        ax.scatter(v_times, losses, s=sizes, edgecolor="#0f172a", linewidth=0.45, zorder=3, color="#a78bfa")
        ax.set_ylabel("final training loss")
        _format_date_axis(ax)
    else:
        ax.text(0.5, 0.5, "No model versions yet", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    # D
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("D. QC: dimension deltas")
    saved = np.zeros(max(1, int(counts.get("corrected_pairs") or 0)))
    if rejected:
        mx = max(rejected)
        bins = np.arange(0, max(10, int(mx) + 10), 10)
        ax.hist(rejected, bins=bins, alpha=0.7, label="rejected attempts", color="#fb923c", edgecolor="#334155")
        ax.hist(saved, bins=bins, alpha=0.9, label="saved annotations", color="#38bdf8", edgecolor="#334155")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    else:
        ax.hist(saved, bins=np.arange(0, 20, 2), alpha=0.9, label="saved annotations", color="#38bdf8", edgecolor="#334155")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xlabel("|Δw|+|Δh| (px)")
    ax.set_ylabel("count")

    # E
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("E. Mask coverage")
    if cover:
        ax.scatter(np.arange(1, len(cover) + 1), cover, s=14, edgecolor="#0f172a", linewidth=0.25, color="#0ea5e9")
        ax.set_xlabel("annotation #")
        ax.set_ylabel("non-zero / total")
        ax.set_ylim(0, min(1.0, float(max(cover)) * 1.2))
        ax.tick_params(pad=3)
    else:
        ax.text(0.5, 0.5, "No mask coverage yet", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    # F
    ax = fig.add_subplot(gs[1, 2])
    ax.set_title("F. Trait outputs")
    if trait:
        ax.hist(trait, bins=14, edgecolor="#334155", color="#6ee7b7")
        ax.set_xlabel("total_root_length")
        ax.set_ylabel("count")
        ax.tick_params(pad=3)
    else:
        ax.text(0.5, 0.5, "No phenotypes yet", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.24)
    plt.close(fig)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/observability/analyses.csv")
def export_analyses_csv(run_id: Optional[str] = None, limit: int = 200000, _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key)) -> Response:
    try:
        db = _db_session()
    except Exception:
        raise HTTPException(status_code=500, detail="Database not available for analyses export")
    try:
        q = select(AnalysisRow).order_by(AnalysisRow.created_at.asc()).limit(limit)
        rid = (run_id or "").strip() or None
        if rid:
            if is_postgres():
                q = q.where(text("(extra->'meta'->>'run_id') = :rid")).params(rid=rid)
            else:
                q = q.where(text("json_extract(extra, '$.meta.run_id') = :rid")).params(rid=rid)
        rows = db.execute(q).scalars().all()
        out: List[dict] = []
        for r in rows:
            extra = r.extra or {}
            timing = extra.get("timing_ms") if isinstance(extra, dict) else None
            meta = extra.get("meta") if isinstance(extra, dict) and isinstance(extra.get("meta"), dict) else {}
            mini = meta.get("minirhizotron") if isinstance(meta, dict) and isinstance(meta.get("minirhizotron"), dict) else {}
            out.append(
                {
                    "created_at": r.created_at.isoformat(),
                    "filename": r.filename,
                    "model_type": r.model_type,
                    "model_version": r.model_version,
                    "run_id": (meta or {}).get("run_id") if isinstance(meta, dict) else "",
                    "field": (mini or {}).get("field") if isinstance(mini, dict) else "",
                    "tube_id": (mini or {}).get("tube_id") if isinstance(mini, dict) else "",
                    "genotype": (mini or {}).get("genotype") if isinstance(mini, dict) else "",
                    "depth": (mini or {}).get("depth") if isinstance(mini, dict) else "",
                    "depth_length_cm": (mini or {}).get("depth_length_cm") if isinstance(mini, dict) else "",
                    "timepoint": (mini or {}).get("timepoint") if isinstance(mini, dict) else "",
                    "session_label": (mini or {}).get("session_label") if isinstance(mini, dict) else "",
                    "session_time": (mini or {}).get("session_time") if isinstance(mini, dict) else "",
                    "camera_model": (meta.get("camera") or {}).get("camera_model")
                    if isinstance(meta, dict) and isinstance(meta.get("camera"), dict)
                    else ((mini or {}).get("camera_model") if isinstance(mini, dict) else ""),
                    "camera_dpi": (meta.get("camera") or {}).get("camera_dpi")
                    if isinstance(meta, dict) and isinstance(meta.get("camera"), dict)
                    else ((mini or {}).get("camera_dpi") if isinstance(mini, dict) else ""),
                    "pixel_to_cm": (meta.get("camera") or {}).get("pixel_to_cm")
                    if isinstance(meta, dict) and isinstance(meta.get("camera"), dict)
                    else ((mini or {}).get("pixel_to_cm") if isinstance(mini, dict) else ""),
                    "image_size_cm": (meta.get("camera") or {}).get("image_size_cm")
                    if isinstance(meta, dict) and isinstance(meta.get("camera"), dict)
                    else ((mini or {}).get("image_size_cm") if isinstance(mini, dict) else ""),
                    "threshold_area": r.threshold_area,
                    "scaling_factor": r.scaling_factor,
                    "confidence_threshold": r.confidence_threshold,
                    "root_count": r.root_count,
                    "average_root_diameter": r.average_root_diameter,
                    "total_root_length": r.total_root_length,
                    "total_root_area": r.total_root_area,
                    "total_root_volume": r.total_root_volume,
                    "image_width_px": (extra.get("image_size") or [None, None])[0] if isinstance(extra, dict) else None,
                    "image_height_px": (extra.get("image_size") or [None, None])[1] if isinstance(extra, dict) else None,
                    "file_bytes": extra.get("file_bytes") if isinstance(extra, dict) else None,
                    "timing_total_ms": (timing or {}).get("total") if isinstance(timing, dict) else None,
                    "timing_read_ms": (timing or {}).get("read") if isinstance(timing, dict) else None,
                    "timing_preprocess_ms": (timing or {}).get("preprocess") if isinstance(timing, dict) else None,
                    "timing_inference_ms": (timing or {}).get("inference") if isinstance(timing, dict) else None,
                    "timing_postprocess_ms": (timing or {}).get("postprocess") if isinstance(timing, dict) else None,
                    "timing_threshold_ms": (timing or {}).get("threshold") if isinstance(timing, dict) else None,
                    "timing_metrics_ms": (timing or {}).get("metrics") if isinstance(timing, dict) else None,
                    "timing_encode_ms": (timing or {}).get("encode") if isinstance(timing, dict) else None,
                    "extra_json": json.dumps(extra, ensure_ascii=False) if isinstance(extra, dict) else "",
                }
            )
        headers = list(out[0].keys()) if out else [
            "created_at","filename","model_type","model_version","run_id",
            "field","tube_id","genotype","depth","depth_length_cm","timepoint","session_label","session_time",
            "camera_model","camera_dpi","pixel_to_cm","image_size_cm",
            "threshold_area","scaling_factor","confidence_threshold",
            "root_count","average_root_diameter","total_root_length","total_root_area","total_root_volume",
            "image_width_px","image_height_px","file_bytes",
            "timing_total_ms","timing_read_ms","timing_preprocess_ms","timing_inference_ms","timing_postprocess_ms",
            "timing_threshold_ms","timing_metrics_ms","timing_encode_ms","extra_json"
        ]
        return _csv_response("analyses.csv", headers, out)
    finally:
        db.close()


def _parse_sample_id_map_csv(contents: bytes) -> List[Tuple[str, str]]:
    """
    Parse a CSV mapping table: filename -> sample_id.
    Accepts headers: filename,sample_id (recommended), with a few aliases for filename.
    """
    text_data = contents.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text_data))
    if not reader.fieldnames:
        raise ValueError("CSV has no header")
    fields = {f.strip().lower(): f for f in reader.fieldnames if f}
    fn_key = fields.get("filename") or fields.get("image_filename") or fields.get("analysis_filename") or fields.get("file")
    sid_key = fields.get("sample_id") or fields.get("sample") or fields.get("accession") or fields.get("line_id")
    if not fn_key or not sid_key:
        raise ValueError("CSV must include columns: filename, sample_id")
    out: List[Tuple[str, str]] = []
    for r in reader:
        if not r:
            continue
        filename = (r.get(fn_key) or "").strip()
        sample_id = (r.get(sid_key) or "").strip()
        if not filename:
            continue
        out.append((filename, sample_id))
    return out


@app.get("/observability/sample_id_map.csv")
def export_sample_id_map_csv(
    run_id: Optional[str] = None,
    limit: int = 200000,
    _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key),
) -> Response:
    """
    Download a CSV mapping table (filename -> sample_id) to align phenotypes with external marker panels.
    """
    try:
        db = _db_session()
    except Exception:
        raise HTTPException(status_code=500, detail="Database not available for sample_id_map export")
    try:
        q = select(AnalysisRow.filename, AnalysisRow.extra).order_by(AnalysisRow.created_at.desc()).limit(limit)
        rid = (run_id or "").strip() or None
        if rid:
            if is_postgres():
                q = q.where(text("(extra->'meta'->>'run_id') = :rid")).params(rid=rid)
            else:
                q = q.where(text("json_extract(extra, '$.meta.run_id') = :rid")).params(rid=rid)
        rows = db.execute(q).all()
        filenames: List[str] = []
        seen = set()
        for (fn, _extra) in rows:
            if not fn:
                continue
            s = str(fn)
            if s in seen:
                continue
            seen.add(s)
            filenames.append(s)

        mapping_rows = db.execute(select(PhenotypeSampleIdMapRow)).scalars().all()
        mapping = {r.filename: (r.sample_id or "") for r in mapping_rows}

        out = [{"filename": fn, "sample_id": mapping.get(fn, "")} for fn in filenames]
        return _csv_response("sample_id_map.csv", ["filename", "sample_id"], out)
    finally:
        db.close()


@app.post("/observability/sample_id_map/upload", response_model=SampleIdMapUploadResponse)
async def upload_sample_id_map_csv(
    file: UploadFile = File(...),
    _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key),
) -> SampleIdMapUploadResponse:
    """
    Upload a CSV mapping table (filename -> sample_id).

    This does NOT rename any files; it only stores an ID mapping used for genotype association.
    """
    contents = await file.read()
    try:
        pairs = _parse_sample_id_map_csv(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rows_seen = len(pairs)
    if not pairs:
        return SampleIdMapUploadResponse(rows_seen=0, rows_upserted=0, rows_skipped=0)

    try:
        db = _db_session()
    except Exception:
        raise HTTPException(status_code=500, detail="Database not available for sample_id_map upload")
    try:
        filenames = [fn for (fn, _sid) in pairs]
        existing = {
            r.filename: r
            for r in db.execute(select(PhenotypeSampleIdMapRow).where(PhenotypeSampleIdMapRow.filename.in_(filenames))).scalars().all()
        }
        upserted = 0
        skipped = 0
        for filename, sample_id in pairs:
            if filename in existing:
                existing[filename].sample_id = sample_id
                upserted += 1
            else:
                db.add(PhenotypeSampleIdMapRow(filename=filename, sample_id=sample_id))
                upserted += 1
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
        return SampleIdMapUploadResponse(rows_seen=rows_seen, rows_upserted=upserted, rows_skipped=skipped)
    finally:
        db.close()


@app.get("/observability/annotations.csv")
def export_annotations_csv(run_id: Optional[str] = None, limit: int = 200000, _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key)) -> Response:
    try:
        db = _db_session()
    except Exception:
        raise HTTPException(status_code=500, detail="Database not available for annotations export")
    try:
        q = select(AnnotationRow).order_by(AnnotationRow.created_at.asc()).limit(limit)
        rid = (run_id or "").strip() or None
        if rid:
            if is_postgres():
                q = q.where(text("(meta->>'run_id') = :rid")).params(rid=rid)
            else:
                q = q.where(text("json_extract(meta, '$.run_id') = :rid")).params(rid=rid)
        rows = db.execute(q).scalars().all()
        out: List[dict] = []
        for r in rows:
            meta = r.meta or {}
            out.append(
                {
                    "created_at": r.created_at.isoformat(),
                    "annotation_id": r.annotation_id,
                    "original_filename": r.original_filename,
                    "image_path": r.image_path,
                    "mask_path": r.mask_path,
                    "run_id": meta.get("run_id") if isinstance(meta, dict) else "",
                    "saved_at": meta.get("saved_at"),
                    "image_filename": meta.get("image_filename"),
                    "mask_filename": meta.get("mask_filename"),
                    "image_width_px": meta.get("image_width_px"),
                    "image_height_px": meta.get("image_height_px"),
                    "mask_width_px": meta.get("mask_width_px"),
                    "mask_height_px": meta.get("mask_height_px"),
                    "mask_nonzero_fraction": meta.get("mask_nonzero_fraction"),
                    "meta_json": json.dumps(meta, ensure_ascii=False) if isinstance(meta, dict) else "",
                }
            )
        headers = list(out[0].keys()) if out else [
            "created_at","annotation_id","original_filename","image_path","mask_path","run_id","saved_at","image_filename","mask_filename",
            "image_width_px","image_height_px","mask_width_px","mask_height_px","mask_nonzero_fraction","meta_json"
        ]
        return _csv_response("annotations.csv", headers, out)
    finally:
        db.close()


@app.get("/observability/model_versions.csv")
def export_model_versions_csv(run_id: Optional[str] = None, limit: int = 50000, _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key)) -> Response:
    try:
        db = _db_session()
    except Exception:
        raise HTTPException(status_code=500, detail="Database not available for model_versions export")
    try:
        q = select(ModelVersionRow).order_by(ModelVersionRow.created_at.asc()).limit(limit)
        rid = (run_id or "").strip() or None
        if rid:
            if is_postgres():
                q = q.where(text("(train_config->>'run_id') = :rid")).params(rid=rid)
            else:
                q = q.where(text("json_extract(train_config, '$.run_id') = :rid")).params(rid=rid)
        rows = db.execute(q).scalars().all()
        out: List[dict] = []
        for r in rows:
            out.append(
                {
                    "created_at": r.created_at.isoformat(),
                    "version_id": r.version_id,
                    "model_type": r.model_type,
                    "checkpoint_path": r.checkpoint_path,
                    "base_checkpoint_path": r.base_checkpoint_path,
                    "annotations_dir": r.annotations_dir,
                    "is_current": bool(r.is_current),
                    "run_id": (r.train_config or {}).get("run_id") if isinstance(r.train_config, dict) else "",
                    "train_config_json": json.dumps(r.train_config or {}, ensure_ascii=False),
                    "metrics_json": json.dumps(r.metrics or {}, ensure_ascii=False),
                }
            )
        headers = list(out[0].keys()) if out else [
            "created_at","version_id","model_type","checkpoint_path","base_checkpoint_path","annotations_dir","is_current","run_id","train_config_json","metrics_json"
        ]
        return _csv_response("model_versions.csv", headers, out)
    finally:
        db.close()


@app.get("/observability/train_jobs.csv")
def export_train_jobs_csv(run_id: Optional[str] = None, limit: int = 50000, _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key)) -> Response:
    try:
        db = _db_session()
    except Exception:
        raise HTTPException(status_code=500, detail="Database not available for train_jobs export")
    try:
        rows = db.execute(select(TrainJobRow).order_by(TrainJobRow.created_at.asc()).limit(limit)).scalars().all()
        out: List[dict] = []
        rid = (run_id or "").strip() or None
        for r in rows:
            log = [str(x) for x in (r.log or [])]
            run_tag = ""
            for line in log:
                if line.startswith("run_id="):
                    run_tag = line.split("=", 1)[1].strip()
                    break
            if rid and run_tag != rid:
                continue
            out.append(
                {
                    "created_at": r.created_at.isoformat(),
                    "job_id": r.job_id,
                    "status": r.status,
                    "started_at": r.started_at.isoformat() if r.started_at else "",
                    "finished_at": r.finished_at.isoformat() if r.finished_at else "",
                    "planned_version_id": r.planned_version_id,
                    "produced_version_id": r.produced_version_id,
                    "error": r.error,
                    "run_id": run_tag,
                    "log_json": json.dumps(log, ensure_ascii=False),
                }
            )
        headers = list(out[0].keys()) if out else [
            "created_at","job_id","status","started_at","finished_at","planned_version_id","produced_version_id","error","run_id","log_json"
        ]
        return _csv_response("train_jobs.csv", headers, out)
    finally:
        db.close()


@app.get("/observability/qc_rejections.csv")
def export_qc_rejections_csv(run_id: Optional[str] = None, _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key)) -> Response:
    out: List[dict] = []
    rid = (run_id or "").strip() or None
    if ANNOTATION_QC_REJECTIONS_PATH.exists():
        for line in ANNOTATION_QC_REJECTIONS_PATH.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if rid and str(obj.get("run_id") or "").strip() != rid:
                continue
            iw = obj.get("image_w")
            ih = obj.get("image_h")
            mw = obj.get("mask_w")
            mh = obj.get("mask_h")
            delta = ""
            if all(isinstance(v, int) for v in (iw, ih, mw, mh)):
                delta = str(abs(iw - mw) + abs(ih - mh))
            out.append(
                {
                    "ts": obj.get("ts") or "",
                    "reason": obj.get("reason") or "",
                    "annotation_id": obj.get("annotation_id") or "",
                    "original_filename": obj.get("original_filename") or "",
                    "run_id": obj.get("run_id") or "",
                    "image_w": iw if isinstance(iw, int) else "",
                    "image_h": ih if isinstance(ih, int) else "",
                    "mask_w": mw if isinstance(mw, int) else "",
                    "mask_h": mh if isinstance(mh, int) else "",
                    "delta_px": delta,
                }
            )
    headers = list(out[0].keys()) if out else ["ts","reason","annotation_id","original_filename","run_id","image_w","image_h","mask_w","mask_h","delta_px"]
    return _csv_response("qc_rejections.csv", headers, out)


@app.get("/observability/export.zip")
def export_observability_zip(run_id: Optional[str] = None, _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key)) -> Response:
    # Bundle CSV exports + system metrics JSON into a single downloadable zip.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # system metrics
        try:
            db = _db_session()
        except Exception:
            db = None
        try:
            metrics = (
                _collect_system_metrics(db, run_id=run_id).model_dump()
                if hasattr(SystemMetricsResponse, "model_dump")
                else _collect_system_metrics(db, run_id=run_id).dict()
            )
        finally:
            try:
                if db is not None:
                    db.close()
            except Exception:
                pass
        z.writestr("system_metrics.json", json.dumps(metrics, indent=2))

        # csv exports
        def _add_csv(name: str, resp: Response):
            z.writestr(name, resp.body.decode("utf-8"))

        _add_csv("analyses.csv", export_analyses_csv(run_id=run_id))
        _add_csv("annotations.csv", export_annotations_csv(run_id=run_id))
        _add_csv("model_versions.csv", export_model_versions_csv(run_id=run_id))
        _add_csv("train_jobs.csv", export_train_jobs_csv(run_id=run_id))
        _add_csv("qc_rejections.csv", export_qc_rejections_csv(run_id=run_id))

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="subterra_observability_export.zip"'},
    )


@app.get("/annotations/stats", response_model=AnnotationStatsResponse)
async def annotation_stats(_auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key)):
    """
    Lightweight summary of saved annotations (camera metadata coverage).
    """
    total = 0
    by_camera: dict[str, int] = {}
    missing = 0

    try:
        if ANNOTATIONS_DIR.exists():
            for child in sorted(ANNOTATIONS_DIR.iterdir()):
                if not child.is_dir():
                    continue
                meta_path = child / "meta.json"
                if not meta_path.exists():
                    continue
                try:
                    meta = json.loads(meta_path.read_text("utf-8"))
                except Exception:
                    meta = {}
                total += 1
                cam = _extract_camera_model_from_meta(meta).strip()
                if not cam:
                    missing += 1
                else:
                    by_camera[cam] = by_camera.get(cam, 0) + 1
    except Exception:
        pass

    return AnnotationStatsResponse(total_annotations=total, by_camera_model=by_camera, missing_camera_model=missing)


@app.post("/annotations", response_model=AnnotationSaveResponse)
async def save_annotation_pair(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    original_filename: str = Form(""),
    metadata_json: Optional[str] = Form(None),
    run_id: Optional[str] = Form(None),
    _auth: Optional[ApiKeyRow] = Depends(maybe_require_api_key),
):
    """
    Save an (image, mask) pair to disk for human-in-the-loop fine-tuning.

    Writes under `SUBTERRA_ANNOTATIONS_DIR` (default: `data/annotations/`) and stores `meta.json`.
    """
    annotation_id = uuid4().hex
    image_name = _safe_filename(original_filename or image.filename or "image")
    mask_name = _safe_filename(mask.filename or "mask.png")

    annotation_dir = ANNOTATIONS_DIR / annotation_id
    image_bytes = await image.read()
    mask_bytes = await mask.read()

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        msk = Image.open(io.BytesIO(mask_bytes))
        msk.load()
        img_w, img_h = img.size
        msk_w, msk_h = msk.size
        if (img_w, img_h) != (msk_w, msk_h):
            try:
                _append_jsonl(
                    ANNOTATION_QC_REJECTIONS_PATH,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "reason": "dimension_mismatch",
                        "annotation_id": annotation_id,
                        "original_filename": original_filename or image.filename or "",
                        "run_id": run_id or "",
                        "image_w": img_w,
                        "image_h": img_h,
                        "mask_w": msk_w,
                        "mask_h": msk_h,
                    },
                )
            except Exception:
                pass
            raise HTTPException(
                status_code=400,
                detail=f"Mask size {msk_w}x{msk_h} must match image size {img_w}x{img_h}",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or mask: {e}")

    annotation_dir.mkdir(parents=True, exist_ok=True)

    image_path = annotation_dir / image_name
    mask_path = annotation_dir / mask_name
    metadata_path = annotation_dir / "meta.json"

    _write_bytes(image_path, image_bytes)
    _write_bytes(mask_path, mask_bytes)

    metadata: dict = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "annotation_id": annotation_id,
        "original_filename": original_filename or image.filename or "",
        "image_filename": image_name,
        "mask_filename": mask_name,
        "image_width_px": img_w,
        "image_height_px": img_h,
        "mask_width_px": msk_w,
        "mask_height_px": msk_h,
    }
    if run_id:
        metadata["run_id"] = run_id
    try:
        # Fast QC metric: how sparse is the mask (non-zero fraction)
        hist = msk.convert("L").histogram()
        total_px = int(msk_w) * int(msk_h)
        zero_px = int(hist[0]) if hist else 0
        nonzero_frac = 0.0 if total_px <= 0 else float((total_px - zero_px) / total_px)
        metadata["mask_nonzero_fraction"] = nonzero_frac
    except Exception:
        pass
    if metadata_json:
        try:
            metadata.update(json.loads(metadata_json))
        except json.JSONDecodeError:
            metadata["metadata_json_error"] = "Invalid JSON; stored raw string"
            metadata["metadata_json_raw"] = metadata_json

    _write_bytes(metadata_path, json.dumps(metadata, indent=2).encode("utf-8"))

    # Persist annotation record (best-effort)
    try:
        db = _db_session()
        try:
            db.add(
                AnnotationRow(
                    annotation_id=annotation_id,
                    original_filename=str(metadata.get("original_filename") or ""),
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    meta=metadata,
                )
            )
            db.commit()
        finally:
            db.close()
    except Exception:
        pass

    return AnnotationSaveResponse(
        annotation_id=annotation_id,
        image_path=str(image_path),
        mask_path=str(mask_path),
        metadata_path=str(metadata_path),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
