"""
Pydantic domain models for the pipeline API.
Mirrors backend/domain/normalize_run.py but self-contained.
"""
from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

NormalizeRunJobStatus = Literal["queued", "running", "ready", "error"]


class ApiErrorReason(StrEnum):
    NORMALIZE_RUN_INVALID_JOB_ID = "normalize_run_invalid_job_id"
    NORMALIZE_RUN_JOB_NOT_FOUND = "normalize_run_job_not_found"
    NORMALIZE_RUN_JOB_NOT_READY = "normalize_run_job_not_ready"
    NORMALIZE_RUN_JOB_FAILED = "normalize_run_job_failed"
    NORMALIZE_RUN_RESULT_MISSING = "normalize_run_result_missing"
    NORMALIZE_RUN_INVALID_PAYLOAD = "normalize_run_invalid_payload"
    NORMALIZE_RUN_NO_PIPELINE_INPUTS = "normalize_run_no_pipeline_inputs"
    NORMALIZE_RUN_PIPELINE_FAILED = "normalize_run_pipeline_failed"


class ApiErrorDetail(BaseModel):
    reason: ApiErrorReason
    message: str
    context: dict = Field(default_factory=dict)


class FrontendRoomPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    key: str | None = None
    name: str | None = None
    polygons: list[list[float]] | None = None
    description: str | None = None


class FrontendOpeningPayload(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    id: str | None = None
    objectRole: str | None = None


class PipelineNormalizeRunRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    room: FrontendRoomPayload
    walls: list[dict] = Field(default_factory=list)
    openings: list[FrontendOpeningPayload] = Field(default_factory=list)
    source_unit: Literal[
        "auto", "m", "meter", "meters", "mm", "millimeter", "millimeters"
    ] = Field(default="auto")
    tenant_id: str | None = Field(default="demo_tenant")
    user_id: str | None = Field(default="demo_user")
    description: str | None = None
    special_notes: str | None = None
    style: str | None = Field(default="modern")
    split_largest_room: bool = Field(default=True)
    allow_generated_accessories: bool = Field(default=False)


class PipelineNormalizeRunPosition(BaseModel):
    x: float
    y: float
    z: float


class PipelineNormalizeRunRotation(BaseModel):
    x: float
    y: float
    z: float
    w: float


class PipelineNormalizeRunObject(BaseModel):
    name: str | None = None
    size: list[float] | None = None
    type: str | None = None
    color: str | None = None
    modelUrl: str
    position: PipelineNormalizeRunPosition
    rotation: PipelineNormalizeRunRotation
    objectRole: str | None = None
    catalogItemId: str | None = None
    collisionLayer: str | None = None
    placeOn: dict | None = None


class PipelineNormalizeRunOption(BaseModel):
    optionId: str
    label: str | None = None
    layoutScore: int | None = None
    hardValid: bool | None = None
    complete: bool | None = None
    coverageRatio: float | None = None
    disabledReason: str | None = None
    objects: list[PipelineNormalizeRunObject] = Field(default_factory=list)
    openings: list[FrontendOpeningPayload] = Field(default_factory=list)


class PipelineNormalizeRunDebugZone(BaseModel):
    roomId: str
    roomType: str
    areaM2: float | None = None
    polygon: list[tuple[float, float]] = Field(default_factory=list)


class PipelineNormalizeRunResponse(BaseModel):
    objects: list[PipelineNormalizeRunObject] = Field(default_factory=list)
    openings: list[FrontendOpeningPayload] = Field(default_factory=list)
    selectedOptionId: str | None = None
    options: list[PipelineNormalizeRunOption] = Field(default_factory=list)
    selectionSummary: dict | None = None
    debugSplitWall: dict | None = None
    debugZones: list[PipelineNormalizeRunDebugZone] = Field(default_factory=list)


class PipelineNormalizeRunJobResponse(BaseModel):
    id: str
    status: NormalizeRunJobStatus
    statusUrl: str
    resultUrl: str


class PipelineNormalizeRunStatusResponse(BaseModel):
    id: str
    status: NormalizeRunJobStatus
    stage: str | None = None
    message: str | None = None
    progressCurrent: int | None = None
    progressTotal: int | None = None
    createdAtUtc: str | None = None
    updatedAtUtc: str | None = None
    caseIds: list[str] = Field(default_factory=list)
    currentCaseId: str | None = None
    error: ApiErrorDetail | None = None
    statusUrl: str
    resultUrl: str


class NormalizeRunJobRecord(BaseModel):
    id: str
    status: NormalizeRunJobStatus
    stage: str | None = None
    message: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    created_at_utc: str
    updated_at_utc: str
    case_ids: list[str] = Field(default_factory=list)
    current_case_id: str | None = None
    result_path: str | None = None
    error: ApiErrorDetail | None = None
