# Subterra Root Analysis API

A FastAPI-based service for analyzing root images using deep learning models (U-Net and YOLOv8).

## Features

- **Model Support**: U-Net and YOLOv8 segmentation models
- **Image Analysis**: Automatic root segmentation and morphological analysis
- **Metrics Calculation**: Root count, diameter, length, area, and volume
- **Post-processing**: Area-based thresholding for noise reduction
- **Batch Processing**: Analyze multiple images simultaneously
- **RESTful API**: Clean endpoints with JSON responses

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure model files are present in `subterra_model/models/saved_models/`:
   - `unet_saved.pth` - U-Net model checkpoint
   - `yolo_saved.pt` - YOLOv8 model checkpoint

## Running the API

Start the server:
```bash
python api.py
```

The API will be available at `http://localhost:8001`

## API Documentation

### Health Check
```http
GET /health
```

Returns the status of loaded models and device information.

### Single Image Analysis
```http
POST /analyze
```

Upload a single root image for analysis.

**Parameters:**
- `file` (required): Image file (PNG, JPG, JPEG)
- `model_type` (optional): "unet" or "yolo" (default: "unet")
- `threshold_area` (optional): Minimum area for root detection (default: 50)
- `scaling_factor` (optional): Scaling factor for metrics (default: 1.0)
- `confidence_threshold` (optional): YOLO confidence threshold (default: 0.3)

**Response:**
```json
{
  "root_count": 5,
  "average_root_diameter": 2.34,
  "total_root_length": 156.78,
  "total_root_area": 89.45,
  "total_root_volume": 23.67,
  "mask_image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "original_image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### Batch Image Analysis
```http
POST /batch-analyze
```

Upload multiple images for batch analysis.

**Parameters:**
- `files` (required): Multiple image files
- Same optional parameters as single analysis
 - `metadata_json` (optional): Repeatable form field aligned with `files[]`; each value should be a JSON object stored under `analyses.extra.meta` (e.g. minirhizotron + camera calibration)

**Response:**
```json
{
  "results": [
    {
      "filename": "root1.jpg",
      "success": true,
      "result": { ... }
    },
    {
      "filename": "root2.jpg",
      "success": false,
      "error": "Invalid image format"
    }
  ]
}
```

## Usage Examples

### Python Client
```python
import requests

# Single image analysis
files = {'file': open('root_image.jpg', 'rb')}
params = {
    'model_type': 'unet',
    'threshold_area': 50,
    'scaling_factor': 1.0
}

response = requests.post('http://localhost:8001/analyze', files=files, data=params)
result = response.json()

print(f"Root count: {result['root_count']}")
print(f"Total length: {result['total_root_length']}")

# Save mask image
import base64
mask_data = result['mask_image_base64'].split(',')[1]
with open('mask.png', 'wb') as f:
    f.write(base64.b64decode(mask_data))
```

### cURL Examples
```bash
# Health check
curl http://localhost:8001/health

# Single image analysis
curl -X POST "http://localhost:8001/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@root_image.jpg" \
  -F "model_type=unet" \
  -F "threshold_area=50"
```

## Human-in-the-loop annotation saving

### Save corrected (image, mask) pairs
```http
POST /annotations
```

**Form fields:**
- `image` (required): original image file
- `mask` (required): binary mask PNG file (white roots on black)
- `original_filename` (optional): original filename for record-keeping
- `metadata_json` (optional): JSON string stored in `meta.json`

**Response:**
```json
{
  "annotation_id": "b1c2d3...",
  "image_path": "data/annotations/b1c2d3.../root1.jpg",
  "mask_path": "data/annotations/b1c2d3.../root1_mask.png",
  "metadata_path": "data/annotations/b1c2d3.../meta.json"
}
```

### CORS for the Next.js GUI

By default the API allows requests from:
- `http://localhost:3000`
- `http://127.0.0.1:3000`

Override via `CORS_ALLOW_ORIGINS`, e.g.
```bash
export CORS_ALLOW_ORIGINS="http://localhost:3000"
```

## Model versioning + retraining (U-Net)

### List model versions
```http
GET /models
```

Tracks the current U-Net version and previous versions. Registry stored at `data/models/registry.json`.

### Start a fine-tune job from current model
```http
POST /train/unet
Content-Type: application/json
```

Body (example):
```json
{
  "epochs": 3,
  "batch_size": 2,
  "lr": 0.0001,
  "image_size": 512,
  "camera_model": "CID CI-600 In-Situ Root Imager",
  "only_corrected": true,
  "preserve_aspect_ratio": true
}
```

Uses saved `(image, mask)` pairs in `data/annotations/`, writes a new checkpoint under `data/models/unet/<version>/unet.pth`,
updates the registry, and hot-reloads the in-memory U-Net for inference.

Tips:
- Set `camera_model` to avoid mixing cameras/scales (recommended for minirhizotron workflows)
- Set `only_corrected=true` to train only on masks you actually edited
- Keep `preserve_aspect_ratio=true` to avoid distortion when resizing

### Check training job status/logs
```http
GET /train/jobs
GET /train/jobs/{job_id}
```

## Annotation stats (camera coverage)

```http
GET /annotations/stats
```

Returns how many annotations exist and whether they include `camera_model` metadata (useful before training with a camera filter).

## Postgres persistence + dashboard

When `DATABASE_URL` is set (docker compose does this automatically), the API stores:
- Saved annotations
- Analysis runs (metrics + parameters)
- Model versions + “current” pointer
- Training jobs + logs

Dashboard summary:
```http
GET /dashboard
```

## API keys + automatic ingestion

### Create/list/revoke API keys (admin)

These endpoints require `SUBTERRA_ADMIN_TOKEN` to be set on the API service and passed via `X-Admin-Token`.

```http
POST /api-keys
GET /api-keys
POST /api-keys/{key_id}/revoke
```

### Ingest externally computed phenotypes/metrics

```http
POST /ingest/analysis
```

Authenticate with `X-API-Key: <token>` or `Authorization: Bearer <token>`.

Example:
```bash
curl -X POST "http://localhost:8001/ingest/analysis" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"filename":"sample1.png","model_type":"unet","model_version":"unet_v0003","threshold_area":50,"scaling_factor":1.0,"confidence_threshold":0.0,"root_count":5,"average_root_diameter":2.34,"total_root_length":156.78,"total_root_area":89.45,"total_root_volume":23.67,"extra":{"source":"external_pipeline"}}'
```

### Enforce API keys for inference/annotation/training

Set:
```bash
export SUBTERRA_REQUIRE_API_KEY=1
```

Then `/analyze`, `/batch-analyze`, `/annotations`, and `/train/unet` require an API key as well.

## Genotyping / mapping (separate service)

The docker compose stack includes a separate service on `http://localhost:8002` that:
- stores genetic marker CSV uploads
- maps phenotypes from this API’s `analyses` table to markers (GWAS/QTL-style single-marker tests)

## Model Details

### U-Net Model
- **Architecture**: Custom U-Net with encoder-decoder structure
- **Input**: RGB images (automatically resized to multiples of 16)
- **Output**: Binary segmentation mask
- **Best for**: Precise root segmentation with smooth boundaries

### YOLOv8 Model
- **Architecture**: Ultralytics YOLOv8 segmentation model
- **Input**: RGB images
- **Output**: Instance segmentation masks (combined via max operation)
- **Best for**: Fast inference with instance-aware segmentation

## Metrics Explanation

- **root_count**: Number of distinct root structures detected
- **average_root_diameter**: Mean diameter of detected roots (in pixels, scaled by scaling_factor)
- **total_root_length**: Total length of all root structures (scaled)
- **total_root_area**: Total area covered by roots (scaled²)
- **total_root_volume**: Estimated volume assuming cylindrical roots (scaled³)

## Configuration

Models are loaded automatically on startup from:
- `subterra_model/models/saved_models/unet_saved.pth`
- `subterra_model/models/saved_models/yolo_saved.pt`

The API automatically detects CUDA availability and uses GPU if available.

## Error Handling

The API provides detailed error messages for:
- Invalid model types
- Missing model files
- Corrupted image files
- Processing failures

All errors return appropriate HTTP status codes with descriptive messages.
