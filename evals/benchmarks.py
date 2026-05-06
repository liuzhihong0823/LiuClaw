from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .aggregation import aggregate_batch_reports, write_aggregate_report
from .runner import run_eval_task, score_eval_task
from .tasks import EVAL_TASKS, TASKS_BY_DOMAIN, TASKS_BY_ID
from .types import BenchmarkBaseline, BenchmarkResult, BenchmarkSuite, EvalArtifact, EvalResult, MetricValue


BASELINE_DIR = Path(__file__).with_name("baselines")


BENCHMARK_SUITES: dict[str, BenchmarkSuite] = {
    "coding_agent_v2": BenchmarkSuite(
        id="coding_agent_v2",
        task_ids=[task.id for task in EVAL_TASKS],
        critical_task_ids=[task.id for task in EVAL_TASKS],
        baseline_path=str(BASELINE_DIR / "coding_agent_v2.json"),
        description="Deterministic benchmark suite for coding_agent eval harness v2.",
    )
}


async def run_benchmark_suite(suite: BenchmarkSuite, *, tmp_path: Path) -> BenchmarkResult:
    results: list[EvalResult] = []
    entries: list[tuple] = []
    artifacts: list[EvalArtifact] = []
    for task_id in suite.task_ids:
        task = TASKS_BY_ID[task_id]
        run_output = await run_eval_task(task, tmp_path=tmp_path / task.id)
        result = score_eval_task(task, run_output)
        results.append(result)
        entries.append((run_output, result, task.runtime_profile.name, "benchmark"))
        artifacts.extend(result.artifacts)
    metrics = _compute_suite_metrics(results)
    aggregate = aggregate_batch_reports(suite.id, entries, extra_metrics=metrics)
    artifacts.extend(aggregate.artifacts)
    return BenchmarkResult(
        suite_id=suite.id,
        passed=all(result.passed for result in results),
        results=results,
        metrics=metrics,
        domain_scores=_compute_domain_scores(results),
        task_reports=aggregate.task_reports,
        summary=aggregate.summary,
        artifacts=artifacts,
    )


def load_benchmark_baseline(path: str | Path) -> BenchmarkBaseline:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = {name: MetricValue(**value) for name, value in data.get("metrics", {}).items()}
    return BenchmarkBaseline(suite_id=str(data["suite_id"]), metrics=metrics, description=str(data.get("description", "")))


def assert_benchmark_regression(result: BenchmarkResult, baseline: BenchmarkBaseline) -> list[str]:
    failures: list[str] = []
    for metric_name, expected in baseline.metrics.items():
        actual = result.metrics.get(metric_name)
        if actual is None:
            failures.append(f"missing metric {metric_name}")
            continue
        if expected.comparator == "min" and actual + expected.tolerance < float(expected.value):
            failures.append(f"{metric_name} regressed: expected >= {expected.value}, got {actual}")
        elif expected.comparator == "max" and actual - expected.tolerance > float(expected.value):
            failures.append(f"{metric_name} regressed: expected <= {expected.value}, got {actual}")
        elif expected.comparator == "eq" and abs(actual - float(expected.value)) > expected.tolerance:
            failures.append(f"{metric_name} regressed: expected == {expected.value}, got {actual}")
    for task_id in BENCHMARK_SUITES[result.suite_id].critical_task_ids:
        matched = next(item for item in result.results if item.task_id == task_id)
        if not matched.passed:
            failures.append(f"critical task failed: {task_id}")
    result.regression_failures[:] = failures
    result.passed = result.passed and not failures
    return failures


def serialize_benchmark_result(result: BenchmarkResult, output_dir: str | Path) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark.json"
    json_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    from .types import AggregateReport

    aggregate_path = write_aggregate_report(
        AggregateReport(
            id=result.suite_id,
            metrics=result.metrics,
            summary=result.summary,
            task_reports=result.task_reports,
            failure_index=list(result.regression_failures),
            artifacts=result.artifacts,
        ),
        out_dir,
    )
    return {"benchmark_json": str(json_path), **aggregate_path}


def _compute_suite_metrics(results: list[EvalResult]) -> dict[str, float]:
    size = max(len(results), 1)
    pass_rate = sum(1 for result in results if result.passed) / size
    avg_score = sum(result.score for result in results) / size
    tool_success_values = [float(result.metrics["tool_success"]) for result in results if "tool_success" in result.metrics]
    recovery_values = [float(result.metrics["recovery_success"]) for result in results if "recovery_success" in result.metrics]
    retention_values = [float(result.metrics["context_retained"]) for result in results if "context_retained" in result.metrics]
    memory_values = [float(result.metrics["memory_value"]) for result in results if "memory_value" in result.metrics]
    return {
        "pass_rate": pass_rate,
        "avg_score": avg_score,
        "tool_success_rate": sum(tool_success_values) / max(len(tool_success_values), 1),
        "recovery_success_rate": sum(recovery_values) / max(len(recovery_values), 1),
        "memory_delta": max(memory_values, default=0.0) - min(memory_values, default=0.0),
        "context_retention_rate": sum(retention_values) / max(len(retention_values), 1),
        "abort_rate": 0.0,
    }


def _compute_domain_scores(results: list[EvalResult]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for domain, tasks in TASKS_BY_DOMAIN.items():
        task_ids = {task.id for task in tasks}
        matched = [result.score for result in results if result.task_id in task_ids]
        scores[domain] = sum(matched) / max(len(matched), 1)
    return scores
