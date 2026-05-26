"""
FastAPI server exposing the same three pipeline endpoints as the original backend:

  POST  /pipeline/normalize-run
  GET   /pipeline/normalize-run/{job_id}/status
  GET   /pipeline/normalize-run/{job_id}/result

The core logic is provided by PipelineAdapter, which calls the SLDN and APM
ML models instead of the LLM-based pipeline in the original backend.

Environment variables
---------------------
SLDN_MODEL_DIR      Path to SLDN model directory (args.pickle + check/checkpoint.pt)
APM_CHECKPOINT      Path to APM checkpoint file (.ckpt)
METADATA_DIR        Path to preprocess/metadata directory
                    (default: <backend_new>/preprocess/metadata)
JOB_STORAGE_DIR     Directory for job state files
                    (default: <backend_new>/server_jobs)
NUM_OPTIONS         Number of layout options to generate per request (default: 3)
SLDN_CONDITION_TYPE "floor", "arch", or "uncon" (default: "arch")
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend_new root and src are importable
_SERVER_DIR = Path(__file__).parent.resolve()
_BACKEND_NEW_ROOT = _SERVER_DIR.parent
for _p in [str(_BACKEND_NEW_ROOT / "src"), str(_BACKEND_NEW_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from server.catalog_client import CatalogIndex, load_catalog_index
from server.domain import (
    ApiErrorDetail,
    ApiErrorReason,
    PipelineNormalizeRunJobResponse,
    PipelineNormalizeRunRequest,
    PipelineNormalizeRunResponse,
    PipelineNormalizeRunStatusResponse,
)
from server.job_manager import NormalizeRunJobManager
from server.job_repository import NormalizeRunJobRepository
from server.ml_runner import APMRunner, SLDNRunner
from server.pipeline_adapter import PipelineAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _metadata_dir_default() -> str:
    return str(_BACKEND_NEW_ROOT / "preprocess" / "metadata")


def _job_storage_default() -> str:
    return str(_BACKEND_NEW_ROOT / "server_jobs")


# ── Globals (set in lifespan) ─────────────────────────────────────────────────

_sldn_runner: SLDNRunner | None = None
_apm_runner: APMRunner | None = None
_catalog_index: CatalogIndex = CatalogIndex.empty()
_job_manager: NormalizeRunJobManager | None = None
_pipeline_adapter: PipelineAdapter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global _sldn_runner, _apm_runner, _catalog_index, _job_manager, _pipeline_adapter

    metadata_dir = _env("METADATA_DIR", _metadata_dir_default())
    job_storage_dir = _env("JOB_STORAGE_DIR", _job_storage_default())
    sldn_model_dir = _env("SLDN_MODEL_DIR", "")
    apm_checkpoint = _env("APM_CHECKPOINT", "")
    num_options = int(_env("NUM_OPTIONS", "3"))
    condition_type = _env("SLDN_CONDITION_TYPE", "arch")

    # Job manager
    repo = NormalizeRunJobRepository(Path(job_storage_dir))
    _job_manager = NormalizeRunJobManager(repo)

    # Catalog
    logger.info("Loading catalog from API …")
    try:
        _catalog_index = load_catalog_index()
        logger.info("Catalog loaded.")
    except Exception as exc:
        logger.warning("Catalog load failed (%s) — proceeding without catalog.", exc)
        _catalog_index = CatalogIndex.empty()

    # ML models (only if paths are provided)
    if sldn_model_dir and apm_checkpoint:
        _sldn_runner = SLDNRunner(
            model_dir=sldn_model_dir,
            metadata_dir=metadata_dir,
            condition_type=condition_type,
        )
        _apm_runner = APMRunner(
            checkpoint_path=apm_checkpoint,
            metadata_dir=metadata_dir,
        )
        try:
            logger.info("Loading SLDN model …")
            _sldn_runner.load()
            logger.info("Loading APM model …")
            _apm_runner.load()
        except Exception as exc:
            logger.error("Model loading failed: %s", exc)
            _sldn_runner = None
            _apm_runner = None

        if _sldn_runner and _apm_runner:
            _pipeline_adapter = PipelineAdapter(
                sldn_runner=_sldn_runner,
                apm_runner=_apm_runner,
                catalog_index=_catalog_index,
                num_options=num_options,
            )
    else:
        logger.warning(
            "SLDN_MODEL_DIR or APM_CHECKPOINT not set — ML pipeline unavailable."
        )

    yield

    # Cleanup (nothing to close for now)


app = FastAPI(
    title="SemLayoutDiff Pipeline API",
    description="Drop-in replacement for the original backend using SLDN+APM models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/")
def health_check() -> dict:
    return {
        "status": "ok",
        "models_loaded": _pipeline_adapter is not None,
        "catalog_types": (
            len(_catalog_index._by_type) if _catalog_index else 0
        ),
    }


# ── Helper ────────────────────────────────────────────────────────────────────

def _raise(status_code: int, reason: ApiErrorReason, message: str, **ctx: object) -> None:
    detail = ApiErrorDetail(
        reason=reason,
        message=message,
        context={k: str(v) for k, v in ctx.items()},
    )
    raise HTTPException(status_code=status_code, detail=detail.model_dump(mode="json"))


def _get_manager() -> NormalizeRunJobManager:
    if _job_manager is None:
        raise RuntimeError("Job manager not initialised.")
    return _job_manager


# ── Pipeline endpoints ────────────────────────────────────────────────────────

@app.post(
    "/pipeline/normalize-run",
    response_model=PipelineNormalizeRunJobResponse,
    summary="Start a normalize-run job",
)
def normalize_run_pipeline(
    req: PipelineNormalizeRunRequest,
    background_tasks: BackgroundTasks,
) -> PipelineNormalizeRunJobResponse:
    if _pipeline_adapter is None:
        _raise(
            503,
            ApiErrorReason.NORMALIZE_RUN_PIPELINE_FAILED,
            "ML pipeline is not available. Set SLDN_MODEL_DIR and APM_CHECKPOINT.",
        )

    manager = _get_manager()
    job = manager.create_job(req.user_id)
    background_tasks.add_task(
        manager.run_job, job.id, req, _run_pipeline
    )
    return job


@app.get(
    "/pipeline/normalize-run/{job_id}/status",
    response_model=PipelineNormalizeRunStatusResponse,
    summary="Get normalize-run job status",
)
def normalize_run_status(job_id: str) -> PipelineNormalizeRunStatusResponse:
    manager = _get_manager()
    if not manager.is_valid_job_id(job_id):
        _raise(
            400,
            ApiErrorReason.NORMALIZE_RUN_INVALID_JOB_ID,
            "Job id contains unsupported characters.",
            id=job_id,
        )
    status = manager.status_response(job_id)
    if status is None:
        _raise(
            404,
            ApiErrorReason.NORMALIZE_RUN_JOB_NOT_FOUND,
            "Normalize-run job was not found.",
            id=job_id,
        )
    return status  # type: ignore[return-value]


@app.get(
    "/pipeline/normalize-run/{job_id}/result",
    response_model=PipelineNormalizeRunResponse,
    summary="Get a completed normalize-run result",
)
def normalize_run_result(job_id: str) -> PipelineNormalizeRunResponse:
    manager = _get_manager()
    if not manager.is_valid_job_id(job_id):
        _raise(
            400,
            ApiErrorReason.NORMALIZE_RUN_INVALID_JOB_ID,
            "Job id contains unsupported characters.",
            id=job_id,
        )
    status = manager.status_response(job_id)
    if status is None:
        _raise(
            404,
            ApiErrorReason.NORMALIZE_RUN_JOB_NOT_FOUND,
            "Normalize-run job was not found.",
            id=job_id,
        )
    if status.status == "error":
        err = status.error or ApiErrorDetail(
            reason=ApiErrorReason.NORMALIZE_RUN_JOB_FAILED,
            message="Normalize-run job failed.",
            context={"id": job_id},
        )
        raise HTTPException(status_code=502, detail=err.model_dump(mode="json"))
    if status.status != "ready":
        _raise(
            409,
            ApiErrorReason.NORMALIZE_RUN_JOB_NOT_READY,
            "Normalize-run job is not ready yet.",
            id=job_id,
            status=status.status,
        )

    result = manager.read_result(job_id)
    if result is None:
        _raise(
            500,
            ApiErrorReason.NORMALIZE_RUN_RESULT_MISSING,
            "Job is ready but result payload is missing.",
            id=job_id,
        )
    return result  # type: ignore[return-value]


# ── Pipeline executor (called in background thread) ───────────────────────────

def _run_pipeline(
    req: PipelineNormalizeRunRequest,
    job_id: str | None,
) -> PipelineNormalizeRunResponse:
    if _pipeline_adapter is None:
        raise RuntimeError("Pipeline adapter not initialised.")
    return _pipeline_adapter.run(req)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
        reload=False,
    )
