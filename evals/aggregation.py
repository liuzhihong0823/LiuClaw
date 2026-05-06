from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent_core.trace import TraceRecord, TraceReplayLoader

from .types import AggregateReport, EvalArtifact, EvalResult, EvalRunOutput


def aggregate_run_artifacts(
    run_output: EvalRunOutput,
    result: EvalResult,
    *,
    profile_label: str = "",
    role: str = "",
) -> AggregateReport:
    trace = TraceReplayLoader.load(run_output.trace_json_path)
    task_report = _task_report_from_trace(trace, run_output, result, profile_label=profile_label, role=role)
    failure_index = [f"{result.task_id}: {'; '.join(result.reasons)}"] if not result.passed else []
    metrics = {
        "pass_rate": 1.0 if result.passed else 0.0,
        "avg_score": result.score,
        "tool_success_rate": task_report["tool_success_rate"],
        "abort_rate": 1.0 if trace.outcome.status == "aborted" else 0.0,
        "retry_count": float(task_report["retry_count"]),
        "compaction_count": float(task_report["compaction_count"]),
    }
    summary = render_aggregate_markdown(
        AggregateReport(
            id=result.task_id,
            metrics=metrics,
            summary="single task aggregate",
            task_reports=[task_report],
            failure_index=failure_index,
            artifacts=_aggregate_artifacts(run_output, result),
        )
    )
    return AggregateReport(
        id=result.task_id,
        metrics=metrics,
        summary=summary,
        task_reports=[task_report],
        failure_index=failure_index,
        artifacts=_aggregate_artifacts(run_output, result),
    )


def aggregate_batch_reports(
    report_id: str,
    entries: list[tuple[EvalRunOutput, EvalResult, str, str]],
    *,
    extra_metrics: dict[str, float] | None = None,
) -> AggregateReport:
    task_reports: list[dict[str, Any]] = []
    failures: list[str] = []
    pass_count = 0
    score_total = 0.0
    tool_success_values: list[float] = []
    retry_total = 0
    compaction_total = 0
    artifacts: list[EvalArtifact] = []
    for run_output, result, profile_label, role in entries:
        trace = TraceReplayLoader.load(run_output.trace_json_path)
        report = _task_report_from_trace(trace, run_output, result, profile_label=profile_label, role=role)
        task_reports.append(report)
        pass_count += 1 if result.passed else 0
        score_total += result.score
        tool_success_values.append(report["tool_success_rate"])
        retry_total += report["retry_count"]
        compaction_total += report["compaction_count"]
        if not result.passed:
            failures.append(f"{result.task_id}: {'; '.join(result.reasons)}")
        artifacts.extend(_aggregate_artifacts(run_output, result))
    size = max(len(entries), 1)
    metrics = {
        "pass_rate": pass_count / size,
        "avg_score": score_total / size,
        "tool_success_rate": sum(tool_success_values) / max(len(tool_success_values), 1),
        "abort_rate": sum(1 for report in task_reports if report["outcome_status"] == "aborted") / size,
        "retry_count": float(retry_total),
        "compaction_count": float(compaction_total),
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    aggregate = AggregateReport(
        id=report_id,
        metrics=metrics,
        summary="",
        task_reports=task_reports,
        failure_index=failures,
        artifacts=artifacts,
    )
    aggregate.summary = render_aggregate_markdown(aggregate)
    return aggregate


def render_aggregate_markdown(report: AggregateReport) -> str:
    lines = [
        f"# Aggregate Report: {report.id}",
        "",
        "## Batch Summary",
        f"- pass_rate: `{report.metrics.get('pass_rate', 0.0):.2f}`",
        f"- avg_score: `{report.metrics.get('avg_score', 0.0):.2f}`",
        f"- tool_success_rate: `{report.metrics.get('tool_success_rate', 0.0):.2f}`",
        f"- abort_rate: `{report.metrics.get('abort_rate', 0.0):.2f}`",
        "",
        "## Domain / Batch Metrics",
    ]
    for key, value in sorted(report.metrics.items()):
        if key in {"pass_rate", "avg_score", "tool_success_rate", "abort_rate"}:
            continue
        lines.append(f"- {key}: `{value:.2f}`")
    lines.extend(["", "## Failure Index"])
    if report.failure_index:
        for item in report.failure_index:
            lines.append(f"- {item}")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Task Reports"])
    for task_report in report.task_reports:
        lines.append(
            f"- `{task_report['task_id']}` [{task_report['outcome_status']}] "
            f"profile=`{task_report['profile_label'] or 'default'}` "
            f"tools={task_report['tool_sequence']}"
        )
    lines.extend(["", "## Useful Artifacts"])
    seen: set[str] = set()
    for artifact in report.artifacts:
        key = f"{artifact.name}:{artifact.path}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {artifact.name}: {artifact.path}")
    return "\n".join(lines) + "\n"


def write_aggregate_report(report: AggregateReport, output_dir: str | Path) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "aggregate.json"
    markdown_path = out_dir / "aggregate.md"
    json_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(report.summary or render_aggregate_markdown(report), encoding="utf-8")
    return {"aggregate_json": str(json_path), "aggregate_markdown": str(markdown_path)}


def rebuild_aggregate_from_file(path: str | Path) -> AggregateReport:
    ref = Path(path)
    data = json.loads(ref.read_text(encoding="utf-8"))
    if "turns" in data and "outcome" in data:
        record = TraceReplayLoader.load(ref)
        fake_result = EvalResult(task_id=record.run_id, category="multi_step_completion", passed=record.outcome.status == "completed", score=1.0 if record.outcome.status == "completed" else 0.0)
        fake_output = EvalRunOutput(
            task_id=record.run_id,
            category="multi_step_completion",
            workspace=Path(record.session_metadata.get("workspace_root") or "."),
            session_file=str(record.session_metadata.get("session_file") or ""),
            step_results=[],
            final_message="",
            tool_sequence=[],
            artifacts=[],
            trace_json_path=str(ref),
            replay_markdown_path=str(ref.with_suffix(".md")),
        )
        return aggregate_run_artifacts(fake_output, fake_result)
    if "suite_id" in data and "task_reports" in data:
        return AggregateReport(
            id=str(data["suite_id"]),
            metrics={key: float(value) for key, value in data.get("metrics", {}).items()},
            summary=str(data.get("summary", "")),
            task_reports=list(data.get("task_reports", [])),
            failure_index=list(data.get("regression_failures", [])),
            artifacts=[EvalArtifact(**artifact) for artifact in data.get("artifacts", [])],
        )
    if "control" in data and "candidate" in data and "metric_deltas" in data:
        task_reports = []
        for arm_name in ("control", "candidate"):
            arm = data[arm_name]
            for result in arm.get("results", []):
                task_reports.append(
                    {
                        "task_id": result["task_id"],
                        "category": result["category"],
                        "profile_label": arm.get("profile", {}).get("name", arm_name),
                        "role": arm_name,
                        "passed": result["passed"],
                        "score": result["score"],
                        "reasons": result.get("reasons", []),
                        "metrics": result.get("metrics", {}),
                        "tool_sequence": [],
                        "retry_count": 0,
                        "abort_count": 0,
                        "compaction_count": 0,
                        "outcome_status": "completed" if result["passed"] else "error",
                    }
                )
        return AggregateReport(
            id=str(data["id"]),
            metrics={key: float(value) for key, value in data.get("metric_deltas", {}).items()},
            summary=str(data.get("summary", "")),
            task_reports=task_reports,
            failure_index=list(data.get("regressions", [])),
            artifacts=[EvalArtifact(**artifact) for artifact in data.get("artifacts", [])],
        )
    return AggregateReport(
        id=str(data["id"]),
        metrics={key: float(value) for key, value in data.get("metrics", {}).items()},
        summary=str(data.get("summary", "")),
        task_reports=list(data.get("task_reports", [])),
        failure_index=list(data.get("failure_index", [])),
        artifacts=[EvalArtifact(**artifact) for artifact in data.get("artifacts", [])],
    )


def _task_report_from_trace(
    trace: TraceRecord,
    run_output: EvalRunOutput,
    result: EvalResult,
    *,
    profile_label: str,
    role: str,
) -> dict[str, Any]:
    tool_sequence: list[str] = []
    tool_successes = 0
    tool_failures = 0
    for turn in trace.turns:
        for event in turn.events:
            if event.type == "tool_execution_start" and event.tool_name:
                tool_sequence.append(event.tool_name)
            if event.type == "tool_execution_end":
                if event.metadata.get("tool_result_error"):
                    tool_failures += 1
                else:
                    tool_successes += 1
    tool_total = tool_successes + tool_failures
    return {
        "task_id": result.task_id,
        "category": result.category,
        "profile_label": profile_label,
        "role": role,
        "passed": result.passed,
        "score": result.score,
        "reasons": list(result.reasons),
        "metrics": dict(result.metrics),
        "turn_count": len(trace.turns),
        "tool_sequence": tool_sequence,
        "tool_success_rate": (tool_successes / tool_total) if tool_total else 1.0,
        "retry_count": len(trace.retry_events),
        "abort_count": len(trace.abort_events),
        "abort_reasons": [event.error_message for event in trace.abort_events],
        "compaction_count": len(trace.compaction_events),
        "compaction_actions": [str(event.metadata.get("action", "")) for event in trace.compaction_events],
        "outcome_status": trace.outcome.status,
        "trace_json_path": run_output.trace_json_path,
        "replay_markdown_path": run_output.replay_markdown_path,
        "context_preview": trace.turns[0].context.messages_preview if trace.turns and trace.turns[0].context is not None else [],
    }


def _aggregate_artifacts(run_output: EvalRunOutput, result: EvalResult) -> list[EvalArtifact]:
    artifacts = list(result.artifacts)
    if run_output.trace_json_path:
        artifacts.append(EvalArtifact(name="trace_json", path=run_output.trace_json_path, description="trace file"))
    if run_output.replay_markdown_path:
        artifacts.append(EvalArtifact(name="replay_markdown", path=run_output.replay_markdown_path, description="replay file"))
    return artifacts
