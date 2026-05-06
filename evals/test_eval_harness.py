from __future__ import annotations

import pytest

from .runner import format_eval_failure, run_eval_task, score_eval_task, summarize_results
from .tasks import EVAL_TASKS, TASKS_BY_CATEGORY
from .types import EvalArtifact, EvalResult, EvalTask


@pytest.mark.asyncio
@pytest.mark.parametrize("task", EVAL_TASKS, ids=lambda task: task.id)
async def test_eval_task_passes(task: EvalTask, tmp_path) -> None:
    output = await run_eval_task(task, tmp_path=tmp_path)
    result = score_eval_task(task, output)
    assert result.passed, format_eval_failure(result)
    assert result.score == 1.0
    assert output.trace_json_path
    assert output.replay_markdown_path


@pytest.mark.asyncio
@pytest.mark.parametrize("category,tasks", sorted(TASKS_BY_CATEGORY.items()))
async def test_eval_category_summary(category: str, tasks: list[EvalTask], tmp_path, capsys) -> None:
    results = []
    for task in tasks:
        output = await run_eval_task(task, tmp_path=tmp_path)
        results.append(score_eval_task(task, output))
    summary = summarize_results(results)
    print(summary)
    captured = capsys.readouterr()
    assert category in summary
    assert all(result.passed for result in results), "\n".join(format_eval_failure(result) for result in results if not result.passed)
    assert captured.out.strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("task", EVAL_TASKS, ids=lambda task: task.id)
async def test_eval_task_is_repeatable(task: EvalTask, tmp_path) -> None:
    first = score_eval_task(task, await run_eval_task(task, tmp_path=tmp_path / "first"))
    second = score_eval_task(task, await run_eval_task(task, tmp_path=tmp_path / "second"))
    assert first.passed and second.passed
    assert first.score == second.score == 1.0
    assert first.reasons == second.reasons == []


@pytest.mark.asyncio
async def test_eval_failure_diagnostic_includes_task_reason_and_artifacts(tmp_path) -> None:
    task = EVAL_TASKS[0]
    output = await run_eval_task(task, tmp_path=tmp_path)
    failed = EvalResult(
        task_id=task.id,
        category=task.category,
        passed=False,
        score=0.0,
        reasons=["forced failure"],
        artifacts=[EvalArtifact(name="workspace", path=str(output.workspace), description="task workspace")],
    )
    text = format_eval_failure(failed)
    assert task.id in text
    assert task.category in text
    assert "forced failure" in text
    assert str(output.workspace) in text
