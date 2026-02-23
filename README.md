# Subterra Submission Module

This folder is a clean, runnable module for code/data release and  distribution.
It keeps the application code, model assets, a compact sample dataset, and the final demo video.
Paper-writing drafts and figure-build docs are intentionally excluded.

## Video (YouTube)

- https://youtu.be/4C1wFmr4M04

## Included

- Full runtime code:
  - `api.py`
  - `gui/`
  - `genotype_service/`
  - `subterra_model/`
  - `scripts/`
- Container runtime:
  - `docker-compose.yml`
  - `Dockerfile`
  - `nginx.conf`
  - `requirements.txt`
- Model assets:
  - `subterra_model/models/saved_models/unet_saved.pth`
  - `subterra_model/models/saved_models/yolo_saved.pt`
- Sample data:
  - `data/Round5_CS2024/` (sample input images)
  - `data/annotations/` (sample corrected pairs)
  - `data/markers/` (small marker panel examples)
  - `data/subterra.sqlite3` (sample phenotype DB)
- Genotype helper CSVs:
  - `genotype_data/*.csv`
- Demo media:
  - `media/subterra_demo_instructional_toptext_v3.mov`

## Excluded on purpose

- Paper manuscript drafts and placement notes
- Figure-export trees used for manuscript assembly
- Large raw genotype dumps (e.g., full SAP raw chromosome VCF bundle)
- Large image corpora used during full internal experiments

## Quick Start

1. Build and run:

```bash
docker compose up --build -d
```

2. Open GUI:

- `http://localhost:8080`

3. Stop:

```bash
docker compose down
```

## Notes for GitHub Upload

This module contains large binary artifacts (model/video). Use Git LFS before first push:

```bash
git lfs install
git lfs track "*.pth" "*.pt" "*.mov" "*.hmp" "*.vcf.gz" "*.sqlite3"
```

Then add and commit normally.

## Optional: Add full sorghum marker panels

Place additional marker files in `genotype_data/` (e.g., `SAP_imputed.hmp`, `.vcf.gz`, HapMap, marker CSV).
The genotype service will auto-detect supported files from `data/markers` and `genotype_data`.
