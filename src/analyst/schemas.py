"""Pydantic data contracts for inter-agent handoffs and tracing."""

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field


class Finding(BaseModel):
    claim: str = Field(..., description="Verified technical fact or observation.")
    source: str = Field(..., description="URL or local file path with line range.")
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence based on direct tool evidence.",
    )


class ResearchBrief(BaseModel):
    topic: str = Field(..., description="Normalized topic or question.")
    findings: list[Finding] = Field(default_factory=list)
    open_gaps: list[str] = Field(
        default_factory=list,
        description="Questions or details that could not be verified.",
    )


class CritiqueResult(BaseModel):
    approved: bool = Field(..., description="True only if grounding, completeness, and structure pass.")
    score: int = Field(..., ge=1, le=10, description="Rubric score from 1 to 10.")
    required_fixes: list[str] = Field(
        default_factory=list,
        description="Concrete line-item fixes required if approved is False.",
    )


class TraceEvent(BaseModel):
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    phase: str
    status: str
    detail: dict[str, Any] = Field(default_factory=dict)
