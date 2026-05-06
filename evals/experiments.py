from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .aggregation import aggregate_batch_reports
from .benchmarks import _compute_suite_metrics
from .runner import run_eval_task_with_profile, score_eval_task
from .tasks import TASKS_BY_ID
from .types import EvalArtifact, ExperimentArmResult, ExperimentComparison, ExperimentSpec, EvalRuntimeProfile


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "memory_on_vs_off": ExperimentSpec(
        id="memory_on_vs_off",
        task_ids=["memory_profile_probe"],
        control_profile=EvalRuntimeProfile(name="memory-off", metadata={"memory_enabled": False}),
        candidate_profile=EvalRuntimeProfile(name="memory-on", metadata={"memory_enabled": True}),
        metrics=["pass_rate", "avg_score", "memory_delta", "context_retention_rate"],
        metric_directions={
            "pass_rate": "higher",
            "avg_score": "higher",
            "memory_delta": "higher",
            "context_retention_rate": "higher",
        },
        description="Compare recall behavior with and without memory-enabled runtime profile.",
    ),
    "compaction_on_vs_off": ExperimentSpec(
        id="compaction_on_vs_off",
        task_ids=["continuity_compaction_retains_fact"],
        control_profile=EvalRuntimeProfile(name="compaction-off"),
        candidate_profile=EvalRuntimeProfile(name="compaction-on", enable_compaction=True),
        metrics=["pass_rate", "avg_score", "context_retention_rate"],
        metric_directions={"pass_rate": "higher", "avg_score": "higher", "context_retention_rate": "higher"},
        description="Compare compaction governance before/after enabling compaction.",
    ),
}


async def compare_experiment(spec: ExperimentSpec, *, tmp_path: Path) -> ExperimentComparison:
    control = await _run_arm("control", spec.control_profile, spec.task_ids, tmp_path / "control")
    candidate = await _run_arm("candidate", spec.candidate_profile, spec.task_ids, tmp_path / "candidate")
    metric_deltas: dict[str, float] = {}
    regressions: list[str] = []
    for metric_name in spec.metrics:
        control_value = control.metrics.get(metric_name, 0.0)
        candidate_value = candidate.metrics.get(metric_name, 0.0)
        delta = candidate_value - control_value
        metric_deltas[metric_name] = delta
        direction = spec.metric_directions.get(metric_name, "higher")
        if direction == "higher" and delta < 0:
            regressions.append(f"{metric_name} regressed by {delta:.2f}")
        if direction == "lower" and delta > 0:
            regressions.append(f"{metric_name} regressed by {delta:.2f}")
    artifacts = list(control.artifacts) + list(candidate.artifacts)
    return ExperimentComparison(
        id=spec.id,
        passed=not regressions,
        control=control,
        candidate=candidate,
        metric_deltas=metric_deltas,
        summary=_render_experiment_summary(spec.id, metric_deltas, regressions),
        regressions=regressions,
        artifacts=artifacts,
    )


def serialize_experiment_comparison(comparison: ExperimentComparison, output_dir: str | Path) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "experiment.json"
    json_path.write_text(json.dumps(asdict(comparison), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = out_dir / "experiment.md"
    markdown_path.write_text(comparison.summary or _render_experiment_summary(comparison.id, comparison.metric_deltas, comparison.regressions), encoding="utf-8")
    return {"experiment_json": str(json_path), "experiment_markdown": str(markdown_path)}


async def _run_arm(
    label: str,
    profile: EvalRuntimeProfile,
    task_ids: list[str],
    tmp_path: Path,
) -> ExperimentArmResult:
    results = []
    entries = []
    artifacts: list[EvalArtifact] = []
    for task_id in task_ids:
        task = TASKS_BY_ID[task_id]
        run_output = await run_eval_task_with_profile(task, tmp_path=tmp_path / task.id, runtime_profile=profile)
        result = score_eval_task(task, run_output)
        results.append(result)
        entries.append((run_output, result, profile.name, label))
        artifacts.extend(result.artifacts)
    metrics = _compute_suite_metrics(results)
    aggregate = aggregate_batch_reports(f"{label}:{profile.name}", entries, extra_metrics=metrics)
    artifacts.extend(aggregate.artifacts)
    return ExperimentArmResult(label=label, profile=profile, results=results, metrics=metrics, artifacts=artifacts)


def _render_experiment_summary(experiment_id: str, metric_deltas: dict[str, float], regressions: list[str]) -> str:
    lines = [
        f"# Experiment: {experiment_id}",
        "",
        "## Metric Deltas",
    ]
    for key, value in sorted(metric_deltas.items()):
        lines.append(f"- {key}: `{value:.2f}`")
    lines.extend(["", "## Regressions"])
    if regressions:
        for item in regressions:
            lines.append(f"- {item}")
    else:
        lines.append("- (none)")
    return "\n".join(lines) + "\n"
