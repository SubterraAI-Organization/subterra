#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Resets Subterra annotation + retraining artifacts (safe default).

What it deletes:
  - data/annotations/*              (saved corrected pairs + meta.json)
  - data/models/unet/unet_v*        (fine-tuned checkpoints)
  - data/models/registry.json       (version registry)
  - data/audit/* (if present)       (QC audit logs)

What it keeps by default:
  - docs/figures/_exports/*         (paper exports)
  - data/subterra.sqlite3 analyses  (phenotype runs), but it will remove rows from:
      annotations, model_versions, train_jobs (if those tables exist)

Usage:
  scripts/reset_annotation_and_training_data.sh [--yes]
EOF
}

YES=0
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--yes" ]]; then
  YES=1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "This will DELETE annotation + retraining data under:"
echo "  - $repo_root/data/annotations/*"
echo "  - $repo_root/data/models/unet/unet_v*"
echo "  - $repo_root/data/models/registry.json"
echo "  - $repo_root/data/audit/* (if present)"
echo "And will CLEAN DB tables (if present) in:"
echo "  - $repo_root/data/subterra.sqlite3: annotations, model_versions, train_jobs"
echo

if [[ "$YES" -ne 1 ]]; then
  read -r -p "Proceed? Type 'delete' to confirm: " CONFIRM
  if [[ "$CONFIRM" != "delete" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

echo "Stopping services (best-effort)…"
docker compose stop api nginx gui genotype >/dev/null 2>&1 || true

echo "Deleting on-disk artifacts…"
rm -rf data/annotations/* 2>/dev/null || true
rm -rf data/models/unet/unet_v* 2>/dev/null || true
rm -f data/models/registry.json 2>/dev/null || true
rm -rf data/audit/* 2>/dev/null || true

echo "Cleaning sqlite DB tables (best-effort)…"
if [[ -f data/subterra.sqlite3 ]]; then
  # Only delete from tables that exist.
  tables="$(sqlite3 data/subterra.sqlite3 "SELECT name FROM sqlite_master WHERE type='table';" 2>/dev/null || true)"
  if echo "$tables" | grep -qx "annotations"; then
    sqlite3 data/subterra.sqlite3 "DELETE FROM annotations;" || true
  fi
  if echo "$tables" | grep -qx "model_versions"; then
    sqlite3 data/subterra.sqlite3 "DELETE FROM model_versions;" || true
  fi
  if echo "$tables" | grep -qx "train_jobs"; then
    sqlite3 data/subterra.sqlite3 "DELETE FROM train_jobs;" || true
  fi
fi

echo "Starting services…"
docker compose up -d api nginx gui genotype >/dev/null 2>&1 || true

echo "Done."

