# Subterra

Subterra is an end-to-end platform for human-in-the-loop root segmentation, phenotyping, and genotype association mapping (GWAS workflows).

## Demo Video (YouTube)

- https://youtu.be/4C1wFmr4M04

## What Subterra Does

- Segment root structures from tube images using deep learning models.
- Support manual correction (Add/Erase brush tools) in the annotation UI.
- Store corrected image-mask pairs for iterative model improvement.
- Re-train and version U-Net models from saved annotations.
- Run phenotyping to compute traits such as:
  - `total_root_length`
  - `total_root_area`
  - `total_root_volume`
  - `average_root_diameter`
  - `root_count`
- Ingest genotype markers from CSV, HapMap, or VCF.
- Run mapping methods (linear/GWAS-style methods) and inspect signal plots.
- Export observability and mapping artifacts for downstream reporting.

## System Components

- `gui/`: Next.js interface (Annotation, Re-training, Phenotyping, Genotyping, API, Observability pages)
- `api.py`: FastAPI service for segmentation, phenotyping, annotation I/O, and training orchestration
- `genotype_service/`: FastAPI service for marker ingestion and association mapping
- `subterra_model/`: model and training code, including packaged saved model weights
- `data/`: persisted runtime data (SQLite DBs, annotations, marker examples, sample images)
- `genotype_data/`: marker/ID-map datasets used by genotype mapping
- `media/`: demo media

## Repository Layout

```text
.
├── api.py
├── docker-compose.yml
├── Dockerfile
├── nginx.conf
├── requirements.txt
├── gui/
├── genotype_service/
├── subterra_model/
│   └── models/saved_models/
├── data/
│   ├── annotations/
│   ├── markers/
│   ├── Round5_CS2024/
│   ├── subterra.sqlite3
│   └── subterra_genotype.sqlite3
├── genotype_data/
├── sample_images/
└── media/
```

## Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)
- Recommended RAM: >= 8 GB for smooth local use

## Quick Start

1. Start all services:

```bash
docker compose up --build -d
```

2. Open the UI:

- `http://localhost:8080`

3. Stop services:

```bash
docker compose down
```

## Service Endpoints

- GUI: `http://localhost:8080`
- API via nginx prefix: `http://localhost:8080/api`
- Genotype service via nginx prefix: `http://localhost:8080/genotype-api`

Health checks:

- `GET /api/health`
- `GET /genotype-api/health`

## Typical Workflow

1. Annotation:
   - Upload images and run batch segmentation.
   - Correct masks with Add/Erase.
   - Save edited pairs into `data/annotations/`.
2. Re-training:
   - Start U-Net fine-tuning from corrected pairs.
   - New model versions are tracked in `data/models/`.
3. Phenotyping:
   - Run inference batches and produce trait rows.
   - Trait outputs are persisted in the API database.
4. Genotyping/Mapping:
   - Ingest marker matrix (CSV/HapMap/VCF).
   - Select phenotype field + method.
   - Run mapping and review Manhattan/QQ/effect summaries.

## Marker and Genotype Inputs

Supported marker formats:

- Marker matrix CSV
- HapMap (`.hmp`)
- VCF (`.vcf`, `.vcf.gz`)

Marker file locations scanned by genotype service:

- `data/markers/`
- `genotype_data/`

You can place larger external marker files in `genotype_data/` and ingest them through the Genotyping page.

## Data and Model Notes

- Primary phenotype database: `data/subterra.sqlite3`
- Genotype mapping database: `data/subterra_genotype.sqlite3`
- Bundled base weights:
  - `subterra_model/models/saved_models/unet_saved.pth`
  - `subterra_model/models/saved_models/yolo_saved.pt`

## Large Files

For model/video/large genetics assets, Git LFS can be used:

```bash
git lfs install
git lfs track "*.pth" "*.pt" "*.mov" "*.hmp" "*.vcf.gz" "*.sqlite3"
```

## Additional Docs

- `API_README.md`
- `MODEL_DETAILS.md`
