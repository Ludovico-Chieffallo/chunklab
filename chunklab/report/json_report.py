"""JSON report: the EvalReport model serialized verbatim (stable schema)."""

from pathlib import Path

from chunklab.models import EvalReport


def write_json_report(report: EvalReport, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path
