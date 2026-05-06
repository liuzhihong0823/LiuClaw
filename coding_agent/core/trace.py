from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_core.trace import AgentTraceCollector, TraceRecord, TraceReplayLoader, TraceSerializer, build_trace_listener


@dataclass(slots=True)
class SessionTraceArtifacts:
    """描述一次 session run 生成的 trace 工件路径。"""

    run_id: str  # 本次运行的 trace ID。
    json_path: str  # `trace.json` 文件路径。
    markdown_path: str  # `replay.md` 文件路径。


def trace_dir_for_session_file(session_file: str | Path) -> Path:
    """根据 session 文件路径计算对应的 trace 目录。"""

    path = Path(session_file)
    return path.parent / f"{path.stem}.trace"


def write_trace_record(record: TraceRecord, session_file: str | Path) -> SessionTraceArtifacts:
    """将 trace 记录落盘为 JSON 和 Markdown 两个文件。"""

    trace_dir = trace_dir_for_session_file(session_file)
    trace_dir.mkdir(parents=True, exist_ok=True)
    json_path = trace_dir / f"{record.run_id}.json"
    markdown_path = trace_dir / f"{record.run_id}.md"
    TraceSerializer.dump_json(record, json_path)
    TraceSerializer.dump_markdown(record, markdown_path)
    return SessionTraceArtifacts(run_id=record.run_id, json_path=str(json_path), markdown_path=str(markdown_path))


def resolve_trace_file(session_file: str | Path, trace_ref: str) -> Path | None:
    """根据路径或 run id 解析具体的 trace 文件。"""

    ref = Path(trace_ref)
    if ref.exists() and ref.is_file():
        return ref.resolve()
    trace_dir = trace_dir_for_session_file(session_file)
    if not trace_dir.exists():
        return None
    json_candidate = trace_dir / f"{trace_ref}.json"
    md_candidate = trace_dir / f"{trace_ref}.md"
    if json_candidate.exists():
        return json_candidate
    if md_candidate.exists():
        return md_candidate
    return None


def summarize_trace(record: TraceRecord) -> str:
    """将一份 trace 记录压缩成结构化摘要 JSON 文本。"""

    tool_sequence: list[str] = []
    for turn in record.turns:
        for event in turn.events:
            if event.tool_name and event.type == "tool_execution_start":
                tool_sequence.append(event.tool_name)
    payload = {
        "run_id": record.run_id,
        "model_id": record.model_id,
        "thinking": record.thinking,
        "tool_execution_mode": record.tool_execution_mode,
        "turn_count": len(record.turns),
        "tool_sequence": tool_sequence,
        "outcome": {
            "status": record.outcome.status,
            "error_kind": record.outcome.error_kind,
            "error_message": record.outcome.error_message,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "AgentTraceCollector",
    "SessionTraceArtifacts",
    "TraceReplayLoader",
    "TraceSerializer",
    "build_trace_listener",
    "resolve_trace_file",
    "summarize_trace",
    "trace_dir_for_session_file",
    "write_trace_record",
]
