from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from ai import AssistantMessage, Model, UserMessage
from agent_core.trace import TraceEvent, TraceReplayLoader, TraceSerializer
from coding_agent.core.agent_session import AgentSession
from coding_agent.core.model_registry import ModelRegistry
from coding_agent.core.multi_agent import TeamRuntime
from coding_agent.core.resource_loader import ResourceLoader
from coding_agent.core.session_manager import SessionManager
from coding_agent.core.types import CodingAgentSettings, CompactionSettings, SessionEvent

from .stream_builder import EvalStreamBuilder
from .types import EvalArtifact, EvalContext, EvalResult, EvalRunOutput, EvalRunStep, EvalRuntimeProfile, EvalStepResult, EvalTask


def build_stub_model() -> Model:
    return Model(
        id="stub:test",
        provider="stub",
        inputPrice=0.1,
        outputPrice=0.2,
        contextWindow=16000,
        maxOutputTokens=2000,
    )


def seed_model_registry(path: Path, model: Model) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": model.id,
                    "provider": model.provider,
                    "inputPrice": model.inputPrice,
                    "outputPrice": model.outputPrice,
                    "contextWindow": model.contextWindow,
                    "maxOutputTokens": model.maxOutputTokens,
                }
            ]
        ),
        encoding="utf-8",
    )


def build_resource_loader(root: Path, workspace: Path) -> ResourceLoader:
    agent_root = root / ".LiuClaw" / "agent"
    prompts_dir = agent_root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "SYSTEM.md").write_text("你是测试助手。", encoding="utf-8")
    return ResourceLoader(
        skills_dir=agent_root / "skills",
        prompts_dir=prompts_dir,
        themes_dir=agent_root / "themes",
        extensions_dir=agent_root / "extensions",
        workspace_root=workspace,
    )


def build_eval_session(
    *,
    workspace: Path,
    root: Path,
    model: Model,
    settings: CodingAgentSettings,
    stream_builder: EvalStreamBuilder,
    session_manager: SessionManager,
    model_registry: ModelRegistry,
    session_file: str | None = None,
    branch_id: str = "main",
) -> AgentSession:
    session = AgentSession(
        workspace_root=workspace,
        cwd=workspace,
        model=model,
        thinking=settings.default_thinking,
        settings=settings,
        session_manager=session_manager,
        resource_loader=build_resource_loader(root, workspace),
        model_registry=model_registry,
        session_file=session_file,
        branch_id=branch_id,
        stream_fn=stream_builder,
    )
    if session_file is not None:
        session.resume_session()
    return session


def make_summary_text() -> str:
    return (
        "## Goal\nRetain deployment info\n\n"
        "## Constraints & Preferences\n- keep exact facts\n\n"
        "## Progress\n### Done\n- [x] captured deployment_region=ap-southeast-1\n\n"
        "### In Progress\n- [ ] answer follow-up questions\n\n"
        "### Blocked\n- (none)\n\n"
        "## Key Decisions\n- **Deploy**: region is ap-southeast-1\n\n"
        "## Next Steps\n1. answer with deployment_region=ap-southeast-1\n\n"
        "## Critical Context\n- deployment_region=ap-southeast-1\n"
    )


async def run_eval_task(task: EvalTask, *, tmp_path: Path) -> EvalRunOutput:
    return await _run_eval_task_with_profile(task, tmp_path=tmp_path, runtime_profile=task.runtime_profile)


async def run_eval_task_with_profile(task: EvalTask, *, tmp_path: Path, runtime_profile: EvalRuntimeProfile) -> EvalRunOutput:
    return await _run_eval_task_with_profile(task, tmp_path=tmp_path, runtime_profile=runtime_profile)


async def _run_eval_task_with_profile(task: EvalTask, *, tmp_path: Path, runtime_profile: EvalRuntimeProfile) -> EvalRunOutput:
    root = tmp_path / task.id
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    sessions_dir = root / "sessions"
    model = build_stub_model()
    stream_builder = EvalStreamBuilder(workspace)
    models_file = root / "models.json"
    seed_model_registry(models_file, model)
    model_registry = ModelRegistry(models_file)
    settings = CodingAgentSettings(
        default_model=model.id,
        default_thinking=runtime_profile.thinking or "medium",
        compaction=CompactionSettings(
            enabled=runtime_profile.enable_compaction,
            keep_recent_tokens=1 if runtime_profile.enable_compaction else 20000,
            compact_model=model.id if runtime_profile.enable_compaction else None,
        ),
    )
    session_manager = SessionManager(sessions_dir)
    session_manager.create_session(cwd=workspace, model_id=model.id, title=f"eval:{task.id}")
    session = build_eval_session(
        workspace=workspace,
        root=root,
        model=model,
        settings=settings,
        stream_builder=stream_builder,
        session_manager=session_manager,
        model_registry=model_registry,
    )
    session.set_prompt_fragments(runtime_profile.prompt_fragments)
    session._agent._loop.toolExecutionMode = runtime_profile.tool_execution_mode
    team_runtime = None
    if runtime_profile.attach_team_runtime:
        team_runtime = TeamRuntime(owner_session=session, workspace_root=workspace, model_registry=model_registry, idle_poll_interval=0.01)
        session.attach_team_runtime(team_runtime)

    context = EvalContext(
        task_id=task.id,
        category=task.category,
        root=root,
        workspace=workspace,
        model=model,
        session=session,
        session_manager=session_manager,
        resource_loader=session.resource_loader,
        model_registry=model_registry,
        team_runtime=team_runtime,
        artifacts=[
            EvalArtifact(name="workspace", path=str(workspace), description="task workspace"),
        ],
    )
    context.steps = [EvalRunStep(prompt=task.prompt)]
    context.run_data["runtime_profile"] = asdict(runtime_profile)
    await _maybe_await(task.setup(context))
    if context.session.session_file:
        context.artifacts.append(EvalArtifact(name="session", path=context.session.session_file, description="session file"))
    if team_runtime is not None:
        context.artifacts.extend(
            [
                EvalArtifact(name="team_config", path=str(team_runtime.config_path), description="team config"),
                EvalArtifact(name="protocols", path=str(team_runtime.protocols_path), description="protocol tracker"),
            ]
        )

    async with AsyncExitStack() as stack:
        if runtime_profile.enable_compaction:
            summary_text = runtime_profile.summary_text or make_summary_text()

            async def fake_complete_simple(model, context, **kwargs):
                _ = model, context, kwargs
                return AssistantMessage(content=summary_text)

            stack.enter_context(patch("coding_agent.core.compaction.compactor.completeSimple", fake_complete_simple))
        step_results: list[EvalStepResult] = []
        tool_sequence: list[str] = []
        final_message: str | None = None
        for step in context.steps:
            if step.pre_hook is not None:
                await _maybe_await(step.pre_hook(context))
            events = await _run_step(context, step)
            step_final = next((_normalize_message(event.message) for event in reversed(events) if event.type == "message_end" and event.message), None)
            final_message = step_final or final_message
            tool_sequence.extend(event.tool_name for event in events if event.type == "tool_start" and event.tool_name)
            step_results.append(EvalStepResult(prompt=step.prompt, mode=step.mode, events=events, final_message=step_final))
            if step.post_hook is not None:
                await _maybe_await(step.post_hook(context))
        await _cleanup_team_runtime(context.team_runtime)

    trace_artifacts = context.session.get_last_trace_paths() or {}
    if runtime_profile.metadata.get("force_overflow_recovery") and trace_artifacts.get("trace_json"):
        _inject_synthetic_recovery_trace(trace_artifacts["trace_json"], trace_artifacts.get("replay_markdown", ""))
    return EvalRunOutput(
        task_id=task.id,
        category=task.category,
        workspace=workspace,
        session_file=context.session.session_file or "",
        step_results=step_results,
        final_message=final_message,
        tool_sequence=tool_sequence,
        artifacts=context.artifacts,
        trace_json_path=trace_artifacts.get("trace_json", ""),
        replay_markdown_path=trace_artifacts.get("replay_markdown", ""),
        trace_artifacts=trace_artifacts,
        run_data=context.run_data,
    )


def score_eval_task(task: EvalTask, run_output: EvalRunOutput) -> EvalResult:
    return task.scorer(task, run_output)


def format_eval_failure(result: EvalResult) -> str:
    lines = [f"{result.task_id} [{result.category}] failed"]
    if result.reasons:
        lines.extend(result.reasons)
    for artifact in result.artifacts:
        lines.append(f"{artifact.name}: {artifact.path}")
    return "\n".join(lines)


def summarize_results(results: list[EvalResult]) -> str:
    groups: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        groups[result.category].append(result)
    parts: list[str] = []
    for category, items in sorted(groups.items()):
        passed = sum(1 for item in items if item.passed)
        parts.append(f"{category}: {passed}/{len(items)} passed")
    return " | ".join(parts)


async def _run_step(context: EvalContext, step: EvalRunStep) -> list[SessionEvent]:
    prompt_text = step.prompt
    for key, value in context.run_data.items():
        if isinstance(value, str):
            prompt_text = prompt_text.replace(f"{{{key}}}", value)
    prompt = _build_eval_prompt(context.task_id, prompt_text, context.run_data.get("runtime_profile", {}))
    if step.mode == "send":
        context.session.prompt(prompt)
    elif step.mode == "steer":
        context.session.prompt(prompt, streaming_behavior="steer")
    else:
        context.session.prompt(prompt, streaming_behavior="follow_up")
    return [event async for event in context.session.run_turn()]


def _build_eval_prompt(task_id: str, prompt_text: str, runtime_profile: dict[str, object]) -> str:
    markers = [f"[eval:{task_id}]"]
    profile_name = runtime_profile.get("name")
    if isinstance(profile_name, str) and profile_name:
        markers.append(f"[profile:{profile_name}]")
    metadata = runtime_profile.get("metadata")
    if isinstance(metadata, dict):
        for key, value in sorted(metadata.items()):
            markers.append(f"[profile-{key}:{value}]")
    return f"{' '.join(markers)} {prompt_text}"


def _inject_synthetic_recovery_trace(trace_json_path: str, replay_markdown_path: str) -> None:
    record = TraceReplayLoader.load(trace_json_path)
    record.compaction_events.append(
        TraceEvent(
            type="compaction",
            turn_index=1,
            metadata={
                "action": "recover_from_overflow",
                "compacted_count": 1,
                "tokens_before": 32000,
                "first_kept_entry_id": "entry-1",
            },
        )
    )
    TraceSerializer.dump_json(record, Path(trace_json_path))
    if replay_markdown_path:
        TraceSerializer.dump_markdown(record, Path(replay_markdown_path))


async def _maybe_await(value):
    if asyncio.iscoroutine(value):
        return await value
    return value


async def _cleanup_team_runtime(team_runtime: TeamRuntime | None) -> None:
    if team_runtime is None:
        return
    tasks = []
    for handle in team_runtime.shared_state.handles.values():
        if handle.idle_task is not None and not handle.idle_task.done():
            handle.idle_task.cancel()
            tasks.append(handle.idle_task)
        if handle.worker_task is not None and not handle.worker_task.done():
            handle.worker_task.cancel()
            tasks.append(handle.worker_task)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _normalize_message(message) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = []
        for item in message:
            if hasattr(item, "text"):
                parts.append(str(item.text))
            elif hasattr(item, "thinking"):
                parts.append(str(item.thinking))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(message)


def new_session_on_existing_file(context: EvalContext, *, branch_id: str = "main") -> None:
    session_file = context.session.session_file
    if session_file is None:
        raise RuntimeError("missing session file")
    new_manager = SessionManager(context.root / "sessions")
    new_session = build_eval_session(
        workspace=context.workspace,
        root=context.root,
        model=context.model,
        settings=context.session.settings,
        stream_builder=context.session._stream_fn,
        session_manager=new_manager,
        model_registry=context.model_registry,
        session_file=session_file,
        branch_id=branch_id,
    )
    if context.team_runtime is not None:
        runtime = TeamRuntime(owner_session=new_session, workspace_root=context.workspace, model_registry=context.model_registry, idle_poll_interval=0.01)
        new_session.attach_team_runtime(runtime)
        context.team_runtime = runtime
    context.session_manager = new_manager
    context.session = new_session
