"""Lightweight JSONL trace logger for agentic runs."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyst.schemas import TraceEvent


class TraceLogger:
    def __init__(self, trace_dir: Path | str = "traces") -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.trace_path = self.trace_dir / f"run_{stamp}.jsonl"

    def log(self, phase: str, status: str, **detail: Any) -> None:
        event = TraceEvent(phase=phase, status=status, detail=detail)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
