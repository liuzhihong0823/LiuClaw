from __future__ import annotations

import asyncio
from pathlib import Path

from ai import AssistantMessage, UserMessage
from agent_core.trace import TraceReplayLoader

from .runner import new_session_on_existing_file
from .types import EvalArtifact, EvalContext, EvalResult, EvalRunOutput, EvalRunStep, EvalRuntimeProfile, EvalTask


def _base_artifacts(output: EvalRunOutput) -> list[EvalArtifact]:
    return list(output.artifacts)


def _result(
    task: EvalTask,
    output: EvalRunOutput,
    checks: list[tuple[bool, str]],
    *,
    metrics: dict[str, float | str | bool] | None = None,
) -> EvalResult:
    reasons = [message for ok, message in checks if not ok]
    score = 1.0 if not reasons else 0.0
    return EvalResult(
        task_id=task.id,
        category=task.category,
        passed=not reasons,
        score=score,
        reasons=reasons,
        artifacts=_base_artifacts(output),
        metrics=dict(metrics or {}),
    )


def _final_equals(output: EvalRunOutput, expected: str) -> tuple[bool, str]:
    return output.final_message == expected, f"expected final message {expected!r}, got {output.final_message!r}"


def _final_contains(output: EvalRunOutput, expected: str) -> tuple[bool, str]:
    return expected in str(output.final_message or ""), f"expected final message to contain {expected!r}, got {output.final_message!r}"


def _tools_equal(output: EvalRunOutput, expected: list[str]) -> tuple[bool, str]:
    return output.tool_sequence == expected, f"expected tool sequence {expected!r}, got {output.tool_sequence!r}"


def _file_text(path: Path, expected: str) -> tuple[bool, str]:
    if not path.exists():
        return False, f"expected file {path} to exist"
    return path.read_text(encoding="utf-8") == expected, f"expected file {path} to equal {expected!r}"


async def _default_setup(context: EvalContext) -> None:
    return None


async def _setup_resume_task(context: EvalContext) -> None:
    def reload_session(current: EvalContext) -> None:
        new_session_on_existing_file(current)

    context.steps = [
        EvalRunStep(prompt="remember favorite_color=green"),
        EvalRunStep(prompt="what is my favorite_color=green?", pre_hook=reload_session),
    ]


async def _setup_branch_task(context: EvalContext) -> None:
    def save_fact_leaf(current: EvalContext) -> None:
        current.run_data["fact_leaf"] = current.session.leaf_id

    def branch_from_saved_leaf(current: EvalContext) -> None:
        new_session_on_existing_file(current, branch_id=current.run_data["fact_leaf"])

    context.steps = [
        EvalRunStep(prompt="remember branch_fact=delta", post_hook=save_fact_leaf),
        EvalRunStep(prompt="mainline changes"),
        EvalRunStep(prompt="what is branch_fact=delta?", pre_hook=branch_from_saved_leaf),
    ]


async def _setup_compaction_task(context: EvalContext) -> None:
    manager = context.session_manager
    parent_id = None
    for index in range(6):
        user = manager.append_message(UserMessage(content=f"history-{index} deployment_region=ap-southeast-1"), parent_id=parent_id)
        parent_id = user.id
        assistant = manager.append_message(AssistantMessage(content=f"ack-{index}"), parent_id=parent_id)
        parent_id = assistant.id
    context.session.leaf_id = manager.leaf_id
    context.session.branch_id = manager.branch_id
    context.session.resume_session()

    async def compact_before_prompt(current: EvalContext) -> None:
        current.run_data["compact_result"] = await current.session.compact()

    context.steps = [
        EvalRunStep(prompt="what is deployment_region=ap-southeast-1?", pre_hook=compact_before_prompt),
    ]


async def _setup_steer_follow_task(context: EvalContext) -> None:
    context.steps = [
        EvalRunStep(prompt="initial message"),
        EvalRunStep(prompt="steer message", mode="steer"),
        EvalRunStep(prompt="follow message", mode="follow_up"),
    ]


async def _setup_answer_file(context: EvalContext) -> None:
    (context.workspace / "answer.txt").write_text("42", encoding="utf-8")


async def _setup_team_runtime(context: EvalContext) -> None:
    if context.team_runtime is None:
        raise RuntimeError("team runtime expected")

    async def wait_for_plan(current: EvalContext) -> None:
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            pending = [item for item in current.team_runtime.list_protocol_requests() if item.status == "pending"]
            if pending:
                current.run_data["pending_request_id"] = pending[0].request_id
                return
            await asyncio.sleep(0.01)
        raise AssertionError("plan request not created")

    def inject_request_prompt(current: EvalContext) -> None:
        request_id = current.run_data["pending_request_id"]
        current.run_data["request_id"] = request_id

    context.steps = [
        EvalRunStep(prompt="delegate to a teammate", post_hook=wait_for_plan),
        EvalRunStep(prompt='review this request: {"request_id": "{request_id}"}', pre_hook=inject_request_prompt),
    ]


async def _setup_memory_profile_task(context: EvalContext) -> None:
    def reload_session(current: EvalContext) -> None:
        new_session_on_existing_file(current)

    context.steps = [
        EvalRunStep(prompt="remember durable_fact=violet"),
        EvalRunStep(prompt="what is durable_fact=violet after memory restore?", pre_hook=reload_session),
    ]


async def _setup_overflow_recovery_task(context: EvalContext) -> None:
    context.steps = [
        EvalRunStep(prompt="remember durable_fact=violet"),
        EvalRunStep(prompt="recover from overflow and answer durable_fact=violet"),
    ]


def _score_tool_single_write(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    checks = [
        _final_equals(output, "single-write complete"),
        _tools_equal(output, ["write"]),
        _file_text(output.workspace / "notes" / "alpha.txt", "alpha"),
    ]
    return _result(task, output, checks, metrics={"tool_calls": 1.0, "tool_success": 1.0})


def _score_tool_chain_write_read(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    checks = [
        _final_contains(output, "beta"),
        _tools_equal(output, ["write", "read"]),
        _file_text(output.workspace / "chain.txt", "beta"),
    ]
    return _result(task, output, checks, metrics={"tool_calls": 2.0, "tool_success": 1.0})


def _score_tool_invalid_recovery(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    checks = [
        _final_equals(output, "recovered after validation"),
        _tools_equal(output, ["write", "write"]),
        _file_text(output.workspace / "fixed.txt", "recovered"),
    ]
    return _result(task, output, checks, metrics={"tool_calls": 2.0, "tool_success": 1.0, "recovery_success": 1.0})


def _score_tool_runtime_error(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    checks = [
        _final_contains(output, "missing.txt"),
        _tools_equal(output, ["read"]),
    ]
    return _result(task, output, checks, metrics={"tool_calls": 1.0, "tool_success": 0.0})


def _score_resume(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    checks = [_final_equals(output, "favorite_color=green")]
    return _result(task, output, checks, metrics={"context_retained": 1.0, "recovery_success": 1.0})


def _score_branch(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    checks = [
        _final_equals(output, "branch_fact=delta"),
        (len(output.step_results) == 3, f"expected 3 steps, got {len(output.step_results)}"),
    ]
    return _result(task, output, checks, metrics={"context_retained": 1.0, "recovery_success": 1.0})


def _score_compaction(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    compact_result = output.run_data.get("compact_result")
    checks = [
        _final_contains(output, "ap-southeast-1"),
        (compact_result is not None and compact_result.compacted_count > 0, "expected compaction to compact at least one message"),
    ]
    return _result(task, output, checks, metrics={"context_retained": 1.0})


def _score_steer_follow(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    messages = [str(step.final_message or "").replace("[profile:default] ", "") for step in output.step_results]
    checks = [
        (messages == ["continuity-steer-follow:initial message", "continuity-steer-follow:steer message", "continuity-steer-follow:follow message"], f"unexpected step messages {messages!r}"),
    ]
    return _result(task, output, checks, metrics={"context_retained": 1.0})


def _score_multi_phase(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    checks = [
        _final_equals(output, "phase-1 complete -> final answer"),
        _tools_equal(output, ["write", "read"]),
    ]
    return _result(task, output, checks, metrics={"tool_calls": 2.0, "tool_success": 1.0})


def _score_answer_from_tool(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    checks = [
        _final_equals(output, "The tool result is 42."),
        _tools_equal(output, ["read"]),
    ]
    return _result(task, output, checks, metrics={"tool_calls": 1.0, "tool_success": 1.0})


def _score_file_artifact(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    checks = [
        _final_equals(output, "artifact report created"),
        _tools_equal(output, ["write"]),
        _file_text(output.workspace / "artifacts" / "report.md", "# Report\nstatus: ok\n"),
    ]
    return _result(task, output, checks, metrics={"tool_calls": 1.0, "tool_success": 1.0})


def _score_team_runtime(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    protocol_path = output.workspace / ".liuclaw" / "team" / "runtime" / "protocols.json"
    request_data = protocol_path.read_text(encoding="utf-8") if protocol_path.exists() else ""
    checks = [
        ("approved" in request_data, "expected protocol tracker to contain approved request"),
        ((output.workspace / ".liuclaw" / "team" / "config.json").exists(), "expected team config to exist"),
    ]
    return _result(task, output, checks, metrics={"recovery_success": 1.0})


def _score_memory_profile_probe(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    runtime_profile = output.run_data.get("runtime_profile", {})
    memory_enabled = bool(isinstance(runtime_profile, dict) and runtime_profile.get("metadata", {}).get("memory_enabled"))
    expected = "memory restored durable_fact=violet" if memory_enabled else "memory unavailable"
    checks = [_final_equals(output, expected)]
    return _result(
        task,
        output,
        checks,
        metrics={
            "memory_value": 1.0 if memory_enabled else 0.0,
            "context_retained": 1.0 if memory_enabled else 0.0,
        },
    )


def _score_overflow_recovery_probe(task: EvalTask, output: EvalRunOutput) -> EvalResult:
    trace = TraceReplayLoader.load(output.trace_json_path)
    recovered = any(event.metadata.get("action") == "recover_from_overflow" for event in trace.compaction_events)
    checks = [
        _final_equals(output, "overflow recovered durable_fact=violet"),
        (recovered, "expected trace to include recover_from_overflow compaction event"),
    ]
    return _result(task, output, checks, metrics={"recovery_success": 1.0 if recovered else 0.0, "context_retained": 1.0 if recovered else 0.0})


EVAL_TASKS: list[EvalTask] = [
    EvalTask(
        id="tool_single_write",
        category="tool_call_correctness",
        prompt="create a note file",
        setup=_default_setup,
        expected_outcomes={"tools": ["write"]},
        scorer=_score_tool_single_write,
        domains=("harness_regression", "tool_call_reliability"),
    ),
    EvalTask(
        id="tool_chain_write_read",
        category="tool_call_correctness",
        prompt="write and confirm file contents",
        setup=_default_setup,
        expected_outcomes={"tools": ["write", "read"]},
        scorer=_score_tool_chain_write_read,
        domains=("harness_regression", "tool_call_reliability"),
    ),
    EvalTask(
        id="tool_invalid_args_recovery",
        category="tool_call_correctness",
        prompt="recover from invalid tool arguments",
        setup=_default_setup,
        expected_outcomes={"tools": ["write", "write"]},
        scorer=_score_tool_invalid_recovery,
        domains=("harness_regression", "tool_call_reliability", "task_recovery_correctness"),
    ),
    EvalTask(
        id="tool_runtime_error_explained",
        category="tool_call_correctness",
        prompt="explain a runtime tool failure",
        setup=_default_setup,
        expected_outcomes={"tools": ["read"]},
        scorer=_score_tool_runtime_error,
        domains=("harness_regression", "tool_call_reliability"),
    ),
    EvalTask(
        id="continuity_resume_retains_fact",
        category="long_context_continuity",
        prompt="resume should preserve fact",
        setup=_setup_resume_task,
        expected_outcomes={"final": "favorite_color=green"},
        scorer=_score_resume,
        domains=("harness_regression", "long_context_governance", "task_recovery_correctness"),
    ),
    EvalTask(
        id="continuity_branch_retains_fact",
        category="long_context_continuity",
        prompt="branch should preserve fact",
        setup=_setup_branch_task,
        expected_outcomes={"final": "branch_fact=delta"},
        scorer=_score_branch,
        domains=("harness_regression", "long_context_governance", "task_recovery_correctness"),
    ),
    EvalTask(
        id="continuity_compaction_retains_fact",
        category="long_context_continuity",
        prompt="compaction should preserve fact",
        setup=_setup_compaction_task,
        expected_outcomes={"final": "ap-southeast-1"},
        scorer=_score_compaction,
        runtime_profile=EvalRuntimeProfile(name="compaction-on", enable_compaction=True),
        domains=("harness_regression", "long_context_governance"),
    ),
    EvalTask(
        id="continuity_steer_follow_up",
        category="long_context_continuity",
        prompt="exercise steer and follow-up",
        setup=_setup_steer_follow_task,
        expected_outcomes={"steps": 3},
        scorer=_score_steer_follow,
        domains=("harness_regression", "long_context_governance"),
    ),
    EvalTask(
        id="multi_step_plan_and_execute",
        category="multi_step_completion",
        prompt="complete a two-phase plan",
        setup=_default_setup,
        expected_outcomes={"tools": ["write", "read"]},
        scorer=_score_multi_phase,
        domains=("harness_regression",),
    ),
    EvalTask(
        id="multi_step_answer_from_tool_result",
        category="multi_step_completion",
        prompt="answer only from tool result",
        setup=_setup_answer_file,
        expected_outcomes={"final": "The tool result is 42."},
        scorer=_score_answer_from_tool,
        domains=("harness_regression",),
    ),
    EvalTask(
        id="multi_step_file_artifact_written",
        category="multi_step_completion",
        prompt="create a report artifact",
        setup=_default_setup,
        expected_outcomes={"artifact": "artifacts/report.md"},
        scorer=_score_file_artifact,
        domains=("harness_regression",),
    ),
    EvalTask(
        id="multi_step_team_runtime_review",
        category="multi_step_completion",
        prompt="delegate and review a plan",
        setup=_setup_team_runtime,
        expected_outcomes={"protocol_status": "approved"},
        scorer=_score_team_runtime,
        runtime_profile=EvalRuntimeProfile(name="team-runtime", attach_team_runtime=True),
        domains=("harness_regression",),
    ),
    EvalTask(
        id="memory_profile_probe",
        category="long_context_continuity",
        prompt="memory profile should influence recall",
        setup=_setup_memory_profile_task,
        expected_outcomes={"final": "profile dependent"},
        scorer=_score_memory_profile_probe,
        runtime_profile=EvalRuntimeProfile(name="memory-on", metadata={"memory_enabled": True}),
        domains=("memory_benefit", "long_context_governance"),
    ),
    EvalTask(
        id="overflow_recovery_profile_probe",
        category="long_context_continuity",
        prompt="overflow recovery should continue the task",
        setup=_setup_overflow_recovery_task,
        expected_outcomes={"final": "overflow recovered durable_fact=violet"},
        scorer=_score_overflow_recovery_probe,
        runtime_profile=EvalRuntimeProfile(name="overflow-recovery", metadata={"force_overflow_recovery": True}),
        domains=("task_recovery_correctness", "long_context_governance"),
    ),
]

TASKS_BY_CATEGORY: dict[str, list[EvalTask]] = {}
for task in EVAL_TASKS:
    TASKS_BY_CATEGORY.setdefault(task.category, []).append(task)

TASKS_BY_ID: dict[str, EvalTask] = {task.id: task for task in EVAL_TASKS}
TASKS_BY_DOMAIN: dict[str, list[EvalTask]] = {}
for task in EVAL_TASKS:
    for domain in task.domains:
        TASKS_BY_DOMAIN.setdefault(domain, []).append(task)
