from __future__ import annotations

import pytest

from .aggregation import aggregate_run_artifacts, rebuild_aggregate_from_file
from .benchmarks import BENCHMARK_SUITES, assert_benchmark_regression, load_benchmark_baseline, run_benchmark_suite, serialize_benchmark_result
from .cli import main as evals_cli_main
from .experiments import EXPERIMENT_SPECS, compare_experiment
from .runner import run_eval_task, score_eval_task
from .tasks import TASKS_BY_ID
from .types import EvalRuntimeProfile, ExperimentSpec


@pytest.mark.asyncio
async def test_benchmark_suite_matches_baseline(tmp_path) -> None:
    suite = BENCHMARK_SUITES["coding_agent_v2"]
    result = await run_benchmark_suite(suite, tmp_path=tmp_path / "benchmark")
    baseline = load_benchmark_baseline(suite.baseline_path)
    failures = assert_benchmark_regression(result, baseline)
    assert not failures
    assert result.task_reports
    assert "Failure Index" in result.summary


@pytest.mark.asyncio
async def test_benchmark_regression_reports_specific_metric(tmp_path) -> None:
    result = await run_benchmark_suite(BENCHMARK_SUITES["coding_agent_v2"], tmp_path=tmp_path / "regression")
    result.metrics["pass_rate"] = 0.5
    baseline = load_benchmark_baseline(BENCHMARK_SUITES["coding_agent_v2"].baseline_path)
    failures = assert_benchmark_regression(result, baseline)
    assert any("pass_rate regressed" in item for item in failures)


@pytest.mark.asyncio
async def test_aggregate_run_artifacts_includes_trace_paths(tmp_path) -> None:
    task = TASKS_BY_ID["tool_invalid_args_recovery"]
    output = await run_eval_task(task, tmp_path=tmp_path / "single")
    result = score_eval_task(task, output)
    report = aggregate_run_artifacts(output, result, profile_label=task.runtime_profile.name, role="benchmark")
    assert report.task_reports[0]["trace_json_path"] == output.trace_json_path
    assert report.task_reports[0]["replay_markdown_path"] == output.replay_markdown_path
    assert report.metrics["tool_success_rate"] < 1.0


@pytest.mark.asyncio
async def test_experiment_comparison_and_regression(tmp_path) -> None:
    good = await compare_experiment(EXPERIMENT_SPECS["memory_on_vs_off"], tmp_path=tmp_path / "good")
    assert good.passed
    assert good.metric_deltas["context_retention_rate"] >= 0.0

    bad_spec = ExperimentSpec(
        id="memory_regression",
        task_ids=["memory_profile_probe"],
        control_profile=EvalRuntimeProfile(name="memory-on", metadata={"memory_enabled": True}),
        candidate_profile=EvalRuntimeProfile(name="memory-off", metadata={"memory_enabled": False}),
        metrics=["pass_rate", "avg_score", "context_retention_rate"],
        metric_directions={"pass_rate": "higher", "avg_score": "higher", "context_retention_rate": "higher"},
    )
    bad = await compare_experiment(bad_spec, tmp_path=tmp_path / "bad")
    assert not bad.passed
    assert bad.regressions


def test_cli_benchmark_and_aggregate_commands(tmp_path) -> None:
    benchmark_exit = evals_cli_main(
        [
            "benchmark",
            "run",
            "--suite",
            "coding_agent_v2",
            "--tmp-root",
            str(tmp_path / "benchmark-tmp"),
            "--output-dir",
            str(tmp_path / "benchmark-out"),
        ]
    )
    assert benchmark_exit == 0
    benchmark_json = tmp_path / "benchmark-out" / "benchmark.json"
    assert benchmark_json.exists()

    aggregate_exit = evals_cli_main(
        [
            "aggregate",
            str(benchmark_json),
            "--output-dir",
            str(tmp_path / "aggregate-out"),
        ]
    )
    assert aggregate_exit == 0
    report = rebuild_aggregate_from_file(tmp_path / "aggregate-out" / "aggregate.json")
    assert report.id == "coding_agent_v2"


@pytest.mark.asyncio
async def test_serialize_benchmark_result_writes_artifacts(tmp_path) -> None:
    suite = BENCHMARK_SUITES["coding_agent_v2"]
    result = await run_benchmark_suite(suite, tmp_path=tmp_path / "serialize")
    paths = serialize_benchmark_result(result, tmp_path / "out")
    assert (tmp_path / "out" / "benchmark.json").exists()
    assert paths["aggregate_json"].endswith("aggregate.json")
