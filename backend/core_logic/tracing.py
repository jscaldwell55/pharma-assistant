# backend/core_logic/tracing.py
import os, json, logging
from typing import Dict, Any
from dataclasses import dataclass
from .config import settings

logger = logging.getLogger("trace")

@dataclass
class TraceExporter:
    out_dir: str = settings.trace.DIR
    export_jsonl: bool = settings.trace.EXPORT_JSONL

    def __post_init__(self):
        if self.export_jsonl:
            os.makedirs(self.out_dir, exist_ok=True)

    def export(self, trace: Dict[str, Any]) -> None:
        if not settings.trace.ENABLED:
            return
        if self.export_jsonl:
            fname = os.path.join(self.out_dir, f"{trace['trace_id']}.jsonl")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        # Breadcrumb to Cloud logs
        logger.info(
            "trace_id=%s latency=%sms allow=%s retrieved=%d",
            trace["trace_id"],
            trace["timestamps"]["latency_ms"],
            trace["guard"]["allow"],
            len(trace.get("retrieved", [])),
        )