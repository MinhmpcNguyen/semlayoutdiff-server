"""Job lifecycle management for normalize-run async jobs."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

from server.domain import (
    ApiErrorDetail,
    ApiErrorReason,
    NormalizeRunJobRecord,
    NormalizeRunJobStatus,
    PipelineNormalizeRunJobResponse,
    PipelineNormalizeRunRequest,
    PipelineNormalizeRunResponse,
    PipelineNormalizeRunStatusResponse,
)
from server.job_repository import NormalizeRunJobRepository

PipelineRunCallable = Callable[
    [PipelineNormalizeRunRequest, str | None],
    PipelineNormalizeRunResponse,
]

logger = logging.getLogger(__name__)


class NormalizeRunJobManager:
    _VALID_JOB_ID_RE: ClassVar[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_-]+")
    _MAX_JOB_ID_LENGTH: ClassVar[int] = 160

    def __init__(self, repository: NormalizeRunJobRepository) -> None:
        self._repo = repository

    def create_job(self, user_id: str | None) -> PipelineNormalizeRunJobResponse:
        job_id = self._make_job_id(user_id)
        now = self._now()
        self._repo.create(
            NormalizeRunJobRecord(
                id=job_id,
                status="queued",
                stage="queued",
                message="Job queued.",
                created_at_utc=now,
                updated_at_utc=now,
            )
        )
        urls = self._urls(job_id)
        return PipelineNormalizeRunJobResponse(
            id=job_id, status="queued", **urls
        )

    def update(
        self,
        job_id: str,
        *,
        status: NormalizeRunJobStatus | None = None,
        stage: str | None = None,
        message: str | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
        result_path: str | None = None,
        error: ApiErrorDetail | None = None,
    ) -> None:
        record = self._repo.get(job_id)
        if record is None:
            now = self._now()
            record = NormalizeRunJobRecord(
                id=job_id, status="queued", created_at_utc=now, updated_at_utc=now
            )
        data = record.model_dump()
        if status is not None:
            data["status"] = status
        if stage is not None:
            data["stage"] = stage
        if message is not None:
            data["message"] = message
        if progress_current is not None:
            data["progress_current"] = progress_current
        if progress_total is not None:
            data["progress_total"] = progress_total
        if result_path is not None:
            data["result_path"] = result_path
        if error is not None:
            data["error"] = error
        elif status != "error":
            data["error"] = None
        data["updated_at_utc"] = self._now()
        self._repo.save(NormalizeRunJobRecord.model_validate(data))

    def status_response(self, job_id: str) -> PipelineNormalizeRunStatusResponse | None:
        record = self._repo.get(job_id)
        if record is None:
            return None
        urls = self._urls(job_id)
        return PipelineNormalizeRunStatusResponse(
            id=record.id,
            status=record.status,
            stage=record.stage,
            message=record.message,
            progressCurrent=record.progress_current,
            progressTotal=record.progress_total,
            createdAtUtc=record.created_at_utc,
            updatedAtUtc=record.updated_at_utc,
            caseIds=record.case_ids,
            currentCaseId=record.current_case_id,
            error=record.error,
            **urls,
        )

    def read_result(self, job_id: str) -> PipelineNormalizeRunResponse | None:
        return self._repo.read_result(job_id)

    def is_valid_job_id(self, job_id: str) -> bool:
        return (
            bool(job_id)
            and len(job_id) <= self._MAX_JOB_ID_LENGTH
            and self._VALID_JOB_ID_RE.fullmatch(job_id) is not None
        )

    def run_job(
        self,
        job_id: str,
        request: PipelineNormalizeRunRequest,
        run_pipeline: PipelineRunCallable,
    ) -> None:
        try:
            self.update(
                job_id,
                status="running",
                stage="running_pipeline",
                message="Running ML layout pipeline.",
            )
            response = run_pipeline(request, job_id)
            result_path = self._repo.write_result(job_id, response)
            self.update(
                job_id,
                status="ready",
                stage="ready",
                message="Result ready.",
                result_path=str(result_path),
            )
        except Exception as exc:
            logger.exception("normalize-run job crashed: job_id=%s", job_id)
            self.update(
                job_id,
                status="error",
                stage="error",
                message="Job failed.",
                error=ApiErrorDetail(
                    reason=ApiErrorReason.NORMALIZE_RUN_JOB_FAILED,
                    message=str(exc),
                    context={"error_type": type(exc).__name__},
                ),
            )

    @staticmethod
    def _urls(job_id: str) -> dict[str, str]:
        return {
            "statusUrl": f"/pipeline/normalize-run/{job_id}/status",
            "resultUrl": f"/pipeline/normalize-run/{job_id}/result",
        }

    @staticmethod
    def _make_job_id(user_id: str | None) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", user_id or "user")
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{safe}_{ts}_{uuid4().hex[:12]}"

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
