# SemLayoutDiff Pipeline Server

Drop-in replacement for `backend/` that exposes the same three API endpoints
but uses the SLDN + APM ML models instead of the LLM-based pipeline.

## Endpoints (identical to original backend)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/pipeline/normalize-run` | Start an async layout job |
| GET | `/pipeline/normalize-run/{job_id}/status` | Poll job status |
| GET | `/pipeline/normalize-run/{job_id}/result` | Fetch final result |

## Setup

### 1. Install dependencies (into the semlayoutdiff conda env)

```bash
conda activate semlayoutdiff
pip install fastapi uvicorn[standard] httpx pydantic>=2.7 omegaconf
```

### 2. Set environment variables

```bash
# Required — paths to trained model checkpoints
export SLDN_MODEL_DIR=/path/to/sldn_checkpoints          # contains args.pickle + check/checkpoint.pt
export APM_CHECKPOINT=/path/to/apm_checkpoint.ckpt

# Optional — defaults shown
export METADATA_DIR=/path/to/backend_new/preprocess/metadata
export JOB_STORAGE_DIR=/path/to/backend_new/server_jobs
export NUM_OPTIONS=3               # layout options per request
export SLDN_CONDITION_TYPE=floor   # "floor" | "arch" | "uncon"

# Optional — catalog API (same defaults as original backend)
export TKNT_CATALOG_API_BASE_URL=https://auto-furniture-api2.a-star.group
export TKNT_CATALOG_ASSET_BASE_URL=https://storage.mazig.io
```

### 3. Run the server

```bash
cd /path/to/backend_new

# Option A: run directly
python -m server.app

# Option B: via uvicorn
uvicorn server.app:app --host 0.0.0.0 --port 8001

# Option C: development with reload
uvicorn server.app:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Point the frontend at this server

Update the frontend's `AUTO_FILL_ROOM_FURNITURE_BASE_URL` (in
`frontend/src/services/api/baseUrl.ts` or the relevant env variable) to point
to this server instead of the original backend.

---

## Architecture

```
Frontend  ──POST /pipeline/normalize-run──►  app.py
                                              │  create_job → background task
                                              │
                                              └──► pipeline_adapter.py
                                                     │
                                          ┌──────────┼──────────────┐
                                          ▼          ▼              ▼
                               floor_plan_utils  ml_runner.py  catalog_client.py
                               (rasterize        (SLDN → sem.  (category →
                                polygon)          map, APM →    modelUrl)
                                                  attributes)

app.py  ──GET /status──►  job_manager.py → job_repository.py (JSON on disk)
app.py  ──GET /result──►  job_repository.py (reads result.json)
```

## Input / Output contract

### Input (same as original backend)

```json
{
  "room": {
    "name": "Living Room",
    "polygons": [[-3, -2], [-3, 3], [3, 3], [3, -2]],
    "description": "Modern living room"
  },
  "walls": [],
  "openings": [],
  "source_unit": "m",
  "tenant_id": "demo",
  "user_id": "demo_user",
  "style": "modern",
  "split_largest_room": true,
  "allow_generated_accessories": false
}
```

### Output (same as original backend)

```json
{
  "objects": [{"name": "...", "modelUrl": "...", "position": {...}, "rotation": {...}, ...}],
  "openings": [],
  "selectedOptionId": "option_1",
  "options": [{"optionId": "option_1", "objects": [...], ...}, ...]
}
```

## Key conversions performed by the adapter

| Step | From | To |
|------|------|----|
| Polygon rasterisation | `room.polygons` (metres) | 120×120 binary PNG tensor for SLDN |
| Room type detection | `room.name` / `room.description` | 0=bedroom / 1=livingroom / 2=diningroom |
| SLDN inference | floor plan tensor | N semantic layout maps (one per option) |
| APM inference | semantic map (upscaled 1200×1200) | furniture list with size/position/rotation |
| Coordinate inverse transform | APM pixel space | World metres (polygon coordinate frame) |
| Rotation | APM class 0–3 (×90°) | Quaternion `{x, y, z, w}` |
| Catalog lookup | APM category name (e.g. `double_bed`) | `catalogItemId` + `modelUrl` |
