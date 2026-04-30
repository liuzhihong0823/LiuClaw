from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from ai import AssistantMessage, Model, StreamEvent, ToolCall, ToolResultMessage
from coding_agent.core.agent_session import AgentSession
from coding_agent.core.model_registry import ModelRegistry
from coding_agent.core.multi_agent import MessageBus, ProtocolTracker, TeamRuntime
from coding_agent.core.resource_loader import ResourceLoader
from coding_agent.core.session_manager import SessionManager
from coding_agent.core.types import CodingAgentSettings
from coding_agent.modes.interactive.controller import InteractiveController
from coding_agent.modes.interactive.state import InteractiveState


class FakeSession:
    """为测试提供最小流式会话对象。"""

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def consume(self) -> AsyncIterator[StreamEvent]:
        for event in self._events:
            yield event

    async def close(self) -> None:
        return None


@pytest.fixture
def stub_model() -> Model:
    return Model(
        id="stub:test",
        provider="stub",
        inputPrice=0.1,
        outputPrice=0.2,
        contextWindow=16000,
        maxOutputTokens=2000,
    )


def _build_resource_loader(root: Path, workspace: Path) -> ResourceLoader:
    """构造测试使用的 ResourceLoader。"""

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


async def _fake_stream(model, context, thinking, registry=None):
    """根据最新用户消息动态返回脚本化事件。"""

    _ = thinking, registry
    latest_user = next(
        (message.content for message in reversed(context.history) if getattr(message, "role", "") == "user"),
        "",
    )
    has_tool_result = any(isinstance(message, ToolResultMessage) for message in context.history)
    if has_tool_result:
        return FakeSession(
            [
                StreamEvent(type="start", provider=model.provider, model=model),
                StreamEvent(
                    type="done",
                    provider=model.provider,
                    model=model,
                    assistantMessage=AssistantMessage(content=f"handled:{latest_user}"),
                ),
            ]
        )
    if "shutdown_request" in str(latest_user):
        return FakeSession(
            [
                StreamEvent(type="start", provider=model.provider, model=model),
                StreamEvent(
                    type="done",
                    provider=model.provider,
                    model=model,
                    assistantMessage=AssistantMessage(
                        content="processing shutdown",
                        toolCalls=[
                            ToolCall(
                                id="shutdown-1",
                                name="respond_shutdown",
                                arguments=json.dumps({"request_id": _extract_request_id(str(latest_user)), "approve": True, "reason": "done"}),
                            )
                        ],
                    ),
                ),
            ]
        )
    if "slow-task" in str(latest_user):
        await asyncio.sleep(0.05)
        return FakeSession(
            [
                StreamEvent(type="start", provider=model.provider, model=model),
                StreamEvent(
                    type="done",
                    provider=model.provider,
                    model=model,
                    assistantMessage=AssistantMessage(content=f"ok:{latest_user}"),
                ),
            ]
        )
    return FakeSession(
        [
            StreamEvent(type="start", provider=model.provider, model=model),
            StreamEvent(
                type="done",
                provider=model.provider,
                model=model,
                assistantMessage=AssistantMessage(content=f"ok:{latest_user}"),
            ),
        ]
    )


def _extract_request_id(text: str) -> str:
    """从 inbox 提示里抽取 request_id。"""

    marker = '"request_id": "'
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = text.find('"', start)
    return text[start:end]


async def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    """等待直到断言条件成立。"""

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met before timeout")


def test_message_bus_send_peek_and_drain(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path / "inbox")

    bus.send("lead", "alice", "hello", message_type="message", request_id="req-1")
    bus.send("bob", "alice", "check", message_type="broadcast")

    peeked = bus.peek("alice")
    assert [item.sender for item in peeked] == ["lead", "bob"]
    assert peeked[0].request_id == "req-1"

    drained = bus.read_and_drain("alice")
    assert len(drained) == 2
    assert bus.peek("alice") == []


def test_protocol_tracker_create_and_update(tmp_path: Path) -> None:
    tracker = ProtocolTracker(tmp_path / "protocols.json")

    request = tracker.create_request(kind="plan", sender="alice", recipient="lead", content="refactor auth")
    assert request.status == "pending"
    loaded = tracker.get_request(request.request_id)
    assert loaded is not None
    assert loaded.content == "refactor auth"

    updated = tracker.update_request(request.request_id, status="approved", response="go ahead")
    assert updated.status == "approved"
    assert updated.response == "go ahead"
    assert tracker.list_requests()[0].request_id == request.request_id


@pytest.mark.asyncio
async def test_team_runtime_spawn_message_and_shutdown(tmp_path: Path, stub_model: Model) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resource_loader = _build_resource_loader(tmp_path, workspace)
    session_manager = SessionManager(tmp_path / "sessions")
    session = AgentSession(
        workspace_root=workspace,
        cwd=workspace,
        model=stub_model,
        thinking="medium",
        settings=CodingAgentSettings(default_model=stub_model.id),
        session_manager=session_manager,
        resource_loader=resource_loader,
        stream_fn=_fake_stream,
    )
    registry = ModelRegistry(tmp_path / "models.json")
    runtime = TeamRuntime(owner_session=session, workspace_root=workspace, model_registry=registry, idle_poll_interval=0.01)
    session.attach_team_runtime(runtime)

    result = runtime.spawn("alice", "coder", "请开始处理任务")
    assert result.name == "alice"
    await _wait_for(lambda: any(member.name == "alice" and member.status == "idle" for member in runtime.list_members()))

    runtime.send_message("alice", "请查看新的需求")
    handle = runtime.shared_state.handles["alice"]
    assert runtime.peek_inbox("alice")
    await _wait_for(lambda: handle.session.get_last_user_message() is not None and "<team_inbox>" in str(handle.session.get_last_user_message()))
    await _wait_for(lambda: runtime.peek_inbox("alice") == [])

    request = runtime.shutdown("alice")
    await _wait_for(lambda: any(item.request_id == request.request_id and item.status == "approved" for item in runtime.list_protocol_requests()))
    await _wait_for(lambda: any(member.name == "alice" and member.status == "shutdown" for member in runtime.list_members()))


@pytest.mark.asyncio
async def test_team_runtime_plan_submit_and_review(tmp_path: Path, stub_model: Model) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resource_loader = _build_resource_loader(tmp_path, workspace)
    session_manager = SessionManager(tmp_path / "sessions")
    session = AgentSession(
        workspace_root=workspace,
        cwd=workspace,
        model=stub_model,
        thinking="medium",
        settings=CodingAgentSettings(default_model=stub_model.id),
        session_manager=session_manager,
        resource_loader=resource_loader,
        stream_fn=_fake_stream,
    )
    registry = ModelRegistry(tmp_path / "models.json")
    runtime = TeamRuntime(owner_session=session, workspace_root=workspace, model_registry=registry, idle_poll_interval=0.01)
    session.attach_team_runtime(runtime)
    runtime.spawn("alice", "coder", "请处理任务")
    await _wait_for(lambda: any(member.name == "alice" and member.status == "idle" for member in runtime.list_members()))

    handle = runtime.shared_state.handles["alice"]
    submit_tool = next(tool for tool in handle.session.runtime.tools if tool.name == "submit_plan")
    request_payload = json.loads(await submit_tool.execute(json.dumps({"plan": "重构认证模块"}), None))
    assert request_payload["status"] == "pending"
    assert runtime.peek_inbox("lead")[0].message_type == "plan_submit"

    review_tool = next(tool for tool in session.runtime.tools if tool.name == "review_plan")
    reviewed = json.loads(await review_tool.execute(json.dumps({"request_id": request_payload["request_id"], "approve": True, "feedback": "可以开始"}), None))
    assert reviewed["status"] == "approved"
    await _wait_for(lambda: any(item.request_id == request_payload["request_id"] and item.status == "approved" for item in runtime.list_protocol_requests()))


@pytest.mark.asyncio
async def test_interactive_team_commands(tmp_path: Path, stub_model: Model) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resource_loader = _build_resource_loader(tmp_path, workspace)
    session_manager = SessionManager(tmp_path / "sessions")
    session = AgentSession(
        workspace_root=workspace,
        cwd=workspace,
        model=stub_model,
        thinking="medium",
        settings=CodingAgentSettings(default_model=stub_model.id),
        session_manager=session_manager,
        resource_loader=resource_loader,
        stream_fn=_fake_stream,
    )
    registry = ModelRegistry(tmp_path / "models.json")
    runtime = TeamRuntime(owner_session=session, workspace_root=workspace, model_registry=registry, idle_poll_interval=0.01)
    session.attach_team_runtime(runtime)
    runtime.spawn("alice", "coder", "请处理任务")
    await _wait_for(lambda: any(member.name == "alice" and member.status == "idle" for member in runtime.list_members()))

    state = InteractiveState.from_session(session)
    renderer = type(
        "RendererStub",
        (),
        {
            "__init__": lambda self: setattr(self, "input_buffer", type("BufferStub", (), {"text": "", "completer": None, "history": None})()) or setattr(self, "application", None),
            "invalidate": lambda self: None,
            "sync_transcript": lambda self, state=None: True,
            "sync_transcript_content": lambda self, state=None: True,
            "reconcile_viewport_after_content_change": lambda self: None,
            "focus_input_if_idle": lambda self: None,
            "refresh_style": lambda self: None,
        },
    )()
    controller = InteractiveController(session, registry, renderer, state)

    await controller.handle_command("/team")
    await controller.handle_command("/team inbox alice")
    await controller.handle_command("/team requests")

    assert any("alice | role=coder | status=idle(polling)" in item for item in state.status_timeline)


@pytest.mark.asyncio
async def test_worker_consumes_inbox_only_after_current_work_finishes(tmp_path: Path, stub_model: Model) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resource_loader = _build_resource_loader(tmp_path, workspace)
    session_manager = SessionManager(tmp_path / "sessions")
    session = AgentSession(
        workspace_root=workspace,
        cwd=workspace,
        model=stub_model,
        thinking="medium",
        settings=CodingAgentSettings(default_model=stub_model.id),
        session_manager=session_manager,
        resource_loader=resource_loader,
        stream_fn=_fake_stream,
    )
    registry = ModelRegistry(tmp_path / "models.json")
    runtime = TeamRuntime(owner_session=session, workspace_root=workspace, model_registry=registry, idle_poll_interval=0.01)
    session.attach_team_runtime(runtime)

    runtime.spawn("alice", "coder", "slow-task")
    await _wait_for(lambda: any(member.name == "alice" and member.status == "working" for member in runtime.list_members()))
    runtime.send_message("alice", "第二条任务")
    assert runtime.peek_inbox("alice")
    await _wait_for(lambda: any(member.name == "alice" and member.status == "idle" for member in runtime.list_members()))
    await _wait_for(lambda: runtime.peek_inbox("alice") == [])
    handle = runtime.shared_state.handles["alice"]
    await _wait_for(lambda: "<team_inbox>" in str(handle.session.get_last_user_message() or ""))
