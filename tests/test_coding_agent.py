from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import pytest
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from ai import AssistantMessage, Model, StreamEvent, ToolCall, UserMessage
from coding_agent.cli import parse_args
from coding_agent.config.paths import build_agent_paths, find_project_settings_file
from coding_agent.core.agent_session import AgentSession
from coding_agent.core.compaction import SessionCompactor
from coding_agent.core.compaction import compactor as compactor_module
from coding_agent.core.model_registry import ModelRegistry
from coding_agent.core.resource_loader import ResourceLoader
from coding_agent.core.system_prompt import build_system_prompt
from coding_agent.core.session_manager import SessionManager
from coding_agent.core.settings_manager import SettingsManager
from coding_agent.core.tools import build_default_tools, build_tool_registry
from coding_agent.core.runtime_assembly import build_session_context
from coding_agent.core.types import CodingAgentSettings, CompactionSettings, SessionEvent, ToolPolicy
from coding_agent.modes.interactive.controller import InteractiveController
from coding_agent.modes.interactive.renderer import InteractiveRenderer
from coding_agent.modes.interactive.state import InteractiveState, TranscriptBlock, TranscriptTurn

main_module = importlib.import_module("coding_agent.main")


class FakeSession:
    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def consume(self):
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
        contextWindow=10000,
        maxOutputTokens=1000,
    )


def test_paths_and_settings_merge(tmp_path: Path) -> None:
    paths = build_agent_paths(tmp_path)
    paths.ensure_exists()
    workspace = tmp_path / "workspace"
    project_dir = workspace / ".LiuClaw"
    project_dir.mkdir(parents=True)
    project_file = find_project_settings_file(workspace)
    paths.settings_file.write_text(
        json.dumps({"default_model": "openai:gpt-5", "tool_policy": {"allow_bash": False}}),
        encoding="utf-8",
    )
    project_file.write_text(json.dumps({"default_thinking": "high", "theme": "sunrise"}), encoding="utf-8")

    settings = SettingsManager(paths.settings_file, project_file).load()

    assert settings.default_model == "openai:gpt-5"
    assert settings.default_thinking == "high"
    assert settings.theme == "sunrise"
    assert settings.tool_policy.allow_bash is False


def test_resource_loader_and_conflict_detection(tmp_path: Path) -> None:
    root = tmp_path / "root"
    paths = build_agent_paths(root)
    paths.ensure_exists()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (paths.skills_dir / "demo").mkdir(parents=True)
    (paths.skills_dir / "demo" / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\n# Skill\nbody\n", encoding="utf-8")
    (paths.prompts_dir / "SYSTEM.md").write_text("custom prompt", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("project context", encoding="utf-8")

    bundle = ResourceLoader(
        skills_dir=paths.skills_dir,
        prompts_dir=paths.prompts_dir,
        themes_dir=paths.themes_dir,
        extensions_dir=paths.extensions_dir,
        workspace_root=workspace,
    ).load()

    assert bundle.skills[0].name == "demo"
    assert bundle.skills[0].description == "Demo skill"
    assert bundle.prompts["SYSTEM"].content == "custom prompt"
    assert bundle.agents_context == "project context"

    (paths.prompts_dir / "demo.md").write_text("conflict", encoding="utf-8")
    with pytest.raises(ValueError, match="Resource name conflict"):
        ResourceLoader(
            skills_dir=paths.skills_dir,
            prompts_dir=paths.prompts_dir,
            themes_dir=paths.themes_dir,
            extensions_dir=paths.extensions_dir,
            workspace_root=workspace,
        ).load()


def test_resource_loader_skips_invalid_skills(tmp_path: Path) -> None:
    root = tmp_path / "root"
    paths = build_agent_paths(root)
    paths.ensure_exists()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (paths.skills_dir / "valid-skill").mkdir(parents=True)
    (paths.skills_dir / "valid-skill" / "SKILL.md").write_text(
        "---\nname: valid-skill\ndescription: Valid skill\n---\n# Valid\n",
        encoding="utf-8",
    )
    (paths.skills_dir / "missing-description").mkdir(parents=True)
    (paths.skills_dir / "missing-description" / "SKILL.md").write_text("---\nname: missing-description\n---\n# Invalid\n", encoding="utf-8")
    (paths.skills_dir / "mismatch").mkdir(parents=True)
    (paths.skills_dir / "mismatch" / "SKILL.md").write_text(
        "---\nname: different-name\ndescription: mismatch\n---\n# Invalid\n",
        encoding="utf-8",
    )

    bundle = ResourceLoader(
        skills_dir=paths.skills_dir,
        prompts_dir=paths.prompts_dir,
        themes_dir=paths.themes_dir,
        extensions_dir=paths.extensions_dir,
        workspace_root=workspace,
    ).load()

    assert [skill.name for skill in bundle.skills] == ["valid-skill"]


@pytest.mark.asyncio
async def test_session_manager_and_compaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_model: Model) -> None:
    manager = SessionManager(tmp_path / "sessions")
    manager.create_session(cwd=tmp_path, model_id="openai:gpt-5")
    parent_id = None
    for index in range(4):
        user = manager.append_message(
            message=UserMessage(content=f"user-{index}"),
            parent_id=parent_id,
        )
        parent_id = user.id
        assistant = manager.append_message(
            message=AssistantMessage(content=f"assistant-{index}"),
            parent_id=parent_id,
        )
        parent_id = assistant.id

    calls: list[dict] = []

    async def fake_complete_simple(model, context, **kwargs):
        calls.append({"model": model, "context": context, "kwargs": kwargs})
        return AssistantMessage(
            content=(
                "任务目标\n- 总结历史\n"
                "关键上下文\n- 已存在历史\n"
                "已完成事项\n- 完成若干轮对话\n"
                "未完成事项\n- 继续处理最新问题\n"
                "风险与注意点\n- 注意历史可能被压缩"
            )
        )

    monkeypatch.setattr(compactor_module, "completeSimple", fake_complete_simple)
    runtime = compactor_module.CompactionRuntime(
        model=stub_model,
        thinking="medium",
        settings=CodingAgentSettings(
            default_model=stub_model.id,
            compact_model="stub:test",
            compaction=CompactionSettings(keep_recent_tokens=1),
        ),
        model_resolver=lambda model_id: Model(
            id=model_id,
            provider="stub",
            inputPrice=0.1,
            outputPrice=0.2,
            contextWindow=10000,
            maxOutputTokens=1000,
        ),
    )

    result = await SessionCompactor(manager, runtime).compact_session(manager.session_id)
    messages = manager.build_context_messages()

    assert result.compacted_count >= 6
    assert messages[0].metadata["summary"] is True
    assert "任务目标" in str(messages[0].content)
    contents = [str(message.content) for message in messages]
    assert "user-0" not in contents
    assert "assistant-3" in contents
    assert calls[0]["model"].id == "stub:test"
    assert "structured checkpoints" in calls[0]["context"].systemPrompt


@pytest.mark.asyncio
async def test_compaction_returns_noop_when_summary_generation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model: Model,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    manager.create_session(cwd=tmp_path, model_id=stub_model.id)
    parent_id = None
    for index in range(3):
        user = manager.append_message(
            message=UserMessage(content=f"user-{index}"),
            parent_id=parent_id,
        )
        parent_id = user.id
        assistant = manager.append_message(
            message=AssistantMessage(content=f"assistant-{index}"),
            parent_id=parent_id,
        )
        parent_id = assistant.id

    async def fake_complete_simple(model, context, **kwargs):
        _ = model, context, kwargs
        raise RuntimeError("summary failed")

    monkeypatch.setattr(compactor_module, "completeSimple", fake_complete_simple)
    runtime = compactor_module.CompactionRuntime(
        model=stub_model,
        thinking="medium",
        settings=CodingAgentSettings(default_model=stub_model.id, compaction=CompactionSettings(keep_recent_tokens=1)),
    )

    result = await SessionCompactor(manager, runtime).compact_session(manager.session_id)

    assert result.compacted_count == 0
    assert all(not getattr(message, "metadata", {}).get("summary") for message in manager.build_context_messages())


@pytest.mark.asyncio
async def test_tools_and_agent_session_flow(tmp_path: Path, stub_model: Model) -> None:
    settings = CodingAgentSettings(default_model=stub_model.id, tool_policy=ToolPolicy(max_read_chars=1000))
    tools = {tool.name: tool for tool in build_default_tools(tmp_path, settings)}
    await tools["write"].execute(json.dumps({"path": "a.txt", "content": "hello"}), None)
    assert await tools["read"].execute(json.dumps({"path": "a.txt"}), None) == "hello"
    await tools["edit"].execute(json.dumps({"path": "a.txt", "old": "hello", "new": "world"}), None)
    assert await tools["read"].execute(json.dumps({"path": "a.txt"}), None) == "world"
    assert "a.txt" in await tools["find"].execute(json.dumps({"pattern": "a.txt"}), None)
    assert "a.txt" in await tools["ls"].execute(json.dumps({"path": "."}), None)
    assert "a.txt:1:world" in await tools["grep"].execute(json.dumps({"pattern": "world", "path": "."}), None)
    bash_output = await tools["bash"].execute(json.dumps({"command": "printf world"}), None)
    assert "exit_code: 0" in bash_output

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "b.txt"
    await tools["write"].execute(json.dumps({"path": str(outside_file), "content": "outside"}), None)
    assert await tools["read"].execute(json.dumps({"path": str(outside_file)}), None) == "outside"
    assert str(outside_file) in await tools["find"].execute(json.dumps({"pattern": "b.txt", "path": str(outside_dir)}), None)

    home_root = tmp_path / "home"
    paths = build_agent_paths(home_root)
    paths.ensure_exists()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resource_loader = ResourceLoader(
        skills_dir=paths.skills_dir,
        prompts_dir=paths.prompts_dir,
        themes_dir=paths.themes_dir,
        extensions_dir=paths.extensions_dir,
        workspace_root=workspace,
    )
    session_manager = SessionManager(paths.sessions_dir)

    async def fake_stream(model, context, thinking, registry=None):
        latest_user = next(message.content for message in reversed(context.history) if getattr(message, "role", "") == "user")
        return FakeSession(
            [
                StreamEvent(type="start", provider="stub", model=stub_model),
                StreamEvent(type="text_delta", provider="stub", model=stub_model, text="echo:"),
                StreamEvent(
                    type="done",
                    provider="stub",
                    model=stub_model,
                    assistantMessage=AssistantMessage(content=f"echo:{latest_user}"),
                ),
            ]
        )

    agent_session = AgentSession(
        workspace_root=workspace,
        cwd=workspace,
        model=stub_model,
        thinking="medium",
        settings=settings,
        session_manager=session_manager,
        resource_loader=resource_loader,
        stream_fn=fake_stream,
    )
    agent_session.send_user_message("hello")
    events = [event async for event in agent_session.run_turn()]
    restored = session_manager.build_context_messages()

    assert any(event.type == "message_delta" for event in events)
    assert restored[-1].content == "echo:hello"


def test_system_prompt_advertises_skill_descriptions_only(tmp_path: Path, stub_model: Model) -> None:
    root = tmp_path / "root"
    paths = build_agent_paths(root)
    paths.ensure_exists()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (paths.skills_dir / "demo").mkdir(parents=True)
    skill_path = paths.skills_dir / "demo" / "SKILL.md"
    skill_path.write_text(
        "---\nname: demo\ndescription: Query API specs\n---\n# Demo Skill\nsecret body\n",
        encoding="utf-8",
    )
    resources = ResourceLoader(
        skills_dir=paths.skills_dir,
        prompts_dir=paths.prompts_dir,
        themes_dir=paths.themes_dir,
        extensions_dir=paths.extensions_dir,
        workspace_root=workspace,
    ).load()
    registry = build_tool_registry(workspace, workspace, CodingAgentSettings(default_model=stub_model.id))
    registry.activate_all()
    context = build_session_context(
        workspace_root=workspace,
        cwd=workspace,
        model=stub_model,
        thinking="medium",
        settings=CodingAgentSettings(default_model=stub_model.id),
        resources=resources,
        tool_registry=registry,
    )

    prompt = build_system_prompt(context)

    assert "demo: Query API specs" in prompt
    assert str(skill_path) in prompt
    assert "secret body" not in prompt
    assert "再使用 read 工具读取对应 SKILL.md" in prompt


@pytest.mark.asyncio
async def test_agent_session_ends_naturally_after_tool_results(tmp_path: Path, stub_model: Model) -> None:
    settings = CodingAgentSettings(default_model=stub_model.id)
    home_root = tmp_path / "home"
    paths = build_agent_paths(home_root)
    paths.ensure_exists()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resource_loader = ResourceLoader(
        skills_dir=paths.skills_dir,
        prompts_dir=paths.prompts_dir,
        themes_dir=paths.themes_dir,
        extensions_dir=paths.extensions_dir,
        workspace_root=workspace,
    )
    session_manager = SessionManager(paths.sessions_dir)

    async def fake_stream(model, context, thinking, registry=None):
        if any(getattr(message, "role", "") == "tool" for message in context.history):
            return FakeSession(
                [
                    StreamEvent(type="start", provider="stub", model=stub_model),
                    StreamEvent(
                        type="done",
                        provider="stub",
                        model=stub_model,
                        assistantMessage=AssistantMessage(content="draft answer"),
                    ),
                ]
            )
        return FakeSession(
            [
                StreamEvent(type="start", provider="stub", model=stub_model),
                StreamEvent(
                    type="done",
                    provider="stub",
                    model=stub_model,
                    assistantMessage=AssistantMessage(
                        content="need tool",
                        toolCalls=[ToolCall(id="call_1", name="ls", arguments='{"path":"."}')],
                    ),
                ),
            ]
        )

    agent_session = AgentSession(
        workspace_root=workspace,
        cwd=workspace,
        model=stub_model,
        thinking="medium",
        settings=settings,
        session_manager=session_manager,
        resource_loader=resource_loader,
        stream_fn=fake_stream,
    )
    agent_session.send_user_message("inspect workspace")
    events = [event async for event in agent_session.run_turn()]
    restored = session_manager.build_context_messages()
    raw_events = session_manager.iter_events()

    assert all(event.type != "status" for event in events)
    assert events[-1].message == "draft answer"
    assert [message.content for message in restored if getattr(message, "role", "") == "user"] == ["inspect workspace"]
    assert all(event["type"] != "control" for event in raw_events)


@pytest.mark.asyncio
async def test_agent_session_can_queue_steering_and_follow_up_messages(tmp_path: Path, stub_model: Model) -> None:
    settings = CodingAgentSettings(default_model=stub_model.id)
    home_root = tmp_path / "home"
    paths = build_agent_paths(home_root)
    paths.ensure_exists()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resource_loader = ResourceLoader(
        skills_dir=paths.skills_dir,
        prompts_dir=paths.prompts_dir,
        themes_dir=paths.themes_dir,
        extensions_dir=paths.extensions_dir,
        workspace_root=workspace,
    )
    session_manager = SessionManager(paths.sessions_dir)

    async def fake_stream(model, context, thinking, registry=None):
        latest_user = next(message.content for message in reversed(context.history) if getattr(message, "role", "") == "user")
        return FakeSession(
            [
                StreamEvent(type="start", provider="stub", model=stub_model),
                StreamEvent(
                    type="done",
                    provider="stub",
                    model=stub_model,
                    assistantMessage=AssistantMessage(content=f"echo:{latest_user}"),
                ),
            ]
        )

    agent_session = AgentSession(
        workspace_root=workspace,
        cwd=workspace,
        model=stub_model,
        thinking="medium",
        settings=settings,
        session_manager=session_manager,
        resource_loader=resource_loader,
        stream_fn=fake_stream,
    )
    agent_session.send_user_message("hello")
    first_events = [event async for event in agent_session.run_turn()]

    agent_session.steer("queued steer")
    steer_events = [event async for event in agent_session.run_turn()]
    agent_session.follow_up("queued follow")
    follow_events = [event async for event in agent_session.run_turn()]
    restored = session_manager.build_context_messages()

    assert first_events[-1].message == "echo:hello"
    assert steer_events[-1].message == "echo:queued steer"
    assert follow_events[-1].message == "echo:queued follow"
    assert [message.content for message in restored if getattr(message, "role", "") == "user"] == [
        "hello",
        "queued steer",
        "queued follow",
    ]


@pytest.mark.asyncio
async def test_session_manager_lists_recent_sessions(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "sessions")
    first = manager.create_session(cwd=tmp_path / "a", model_id="openai:gpt-5", title="first")
    second = manager.create_session(cwd=tmp_path / "b", model_id="openai:gpt-5", title="second")
    manager.append_message(message=UserMessage(content="hello second"), parent_id=None)
    recent = manager.list_recent_sessions(limit=10)

    assert recent[0]["session_id"] == second.session_id
    assert any(item["session_id"] == first.session_id for item in recent)


@pytest.mark.asyncio
async def test_interactive_controller_commands(tmp_path: Path, stub_model: Model) -> None:
    settings = CodingAgentSettings(default_model=stub_model.id)
    home_root = tmp_path / "home"
    paths = build_agent_paths(home_root)
    paths.ensure_exists()
    paths.models_file.write_text(
        json.dumps(
            [
                {
                    "id": "stub:test",
                    "provider": "stub",
                    "inputPrice": 0.1,
                    "outputPrice": 0.2,
                    "contextWindow": 10000,
                    "maxOutputTokens": 1000,
                }
            ]
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resource_loader = ResourceLoader(
        skills_dir=paths.skills_dir,
        prompts_dir=paths.prompts_dir,
        themes_dir=paths.themes_dir,
        extensions_dir=paths.extensions_dir,
        workspace_root=workspace,
    )
    session_manager = SessionManager(paths.sessions_dir)
    registry = ModelRegistry(paths.models_file)

    async def fake_stream(model, context, thinking, registry=None):
        latest_user = next(message.content for message in reversed(context.history) if getattr(message, "role", "") == "user")
        return FakeSession(
            [
                StreamEvent(type="start", provider="stub", model=stub_model),
                StreamEvent(type="text_delta", provider="stub", model=stub_model, text="ok:"),
                StreamEvent(type="done", provider="stub", model=stub_model, assistantMessage=AssistantMessage(content=f"ok:{latest_user}")),
            ]
        )

    session = AgentSession(
        workspace_root=workspace,
        cwd=workspace,
        model=stub_model,
        thinking="medium",
        settings=settings,
        session_manager=session_manager,
        resource_loader=resource_loader,
        stream_fn=fake_stream,
    )
    state = InteractiveState.from_session(session)
    renderer = type(
        "RendererStub",
        (),
        {
            "__init__": lambda self: setattr(self, "input_buffer", type("BufferStub", (), {"text": "", "completer": None, "history": None})()) or setattr(self, "application", None) or setattr(self, "focused_back", False),
            "invalidate": lambda self: None,
            "sync_transcript": lambda self, state=None: True,
            "sync_transcript_content": lambda self, state=None: True,
            "reconcile_viewport_after_content_change": lambda self: None,
            "focus_input_if_idle": lambda self: setattr(self, "focused_back", True),
            "refresh_style": lambda self: None,
        },
    )()
    controller = InteractiveController(session, registry, renderer, state)

    await controller.handle_command("/model stub:test")
    await controller.handle_command("/thinking high")
    await controller.handle_command("/theme default")
    await controller.handle_command("/pwd")
    await controller.handle_text("hello")
    await controller.handle_command("/retry")
    await controller.handle_command("/sessions")

    assert controller.session.model.id == "stub:test"
    assert state.thinking == "high"
    assert state.theme == "default"
    assert any("workspace" in item for item in state.status_timeline)
    assert "[User]\nhello" in state.transcript_text
    assert "[Assistant]\nok:hello" in state.transcript_text
    assert any("stub:test" in item for item in state.status_timeline)
    assert renderer.focused_back is True
    await controller.handle_command("/clear")
    assert not state.transcript_blocks


def test_renderer_keeps_full_history_and_scroll_api() -> None:
    state = InteractiveState(
        session_id="s1",
        model_id="stub:test",
        thinking="medium",
        cwd=Path("/tmp"),
        theme="default",
    )
    for index in range(30):
        state.transcript_turns.append(
            TranscriptTurn(
                turn_id=f"turn-{index}",
                user_prompt_preview=f"question-{index}",
                user_block=TranscriptBlock(id=f"user-{index}", kind="user", title="User", body=f"question-{index}"),
                assistant_block=TranscriptBlock(
                    id=f"assistant-{index}",
                    kind="assistant",
                    title="Assistant",
                    body=f"body-{index}",
                ),
                completed=True,
            )
        )
    state.rebuild_transcript()
    renderer = InteractiveRenderer(state)
    renderer.sync_transcript()

    assert "body-0" in renderer.transcript_buffer.text
    assert "body-29" in renderer.transcript_buffer.text
    renderer.scroll_main(3)
    assert renderer.main_window.vertical_scroll == 3


def test_interactive_state_tracks_output_follow_and_unseen_updates() -> None:
    state = InteractiveState(
        session_id="s1",
        model_id="stub:test",
        thinking="medium",
        cwd=Path("/tmp"),
        theme="default",
    )
    state.start_user_turn("hello", "turn-1")
    state.apply_event(SessionEvent(type="message_start", message_id="m1"))
    state.apply_event(SessionEvent(type="message_delta", delta="hello", message_id="m1"))
    assert state.auto_follow_output is True
    assert state.unseen_output_updates == 0
    assert state.transcript_text == "[User]\nhello\n\n[Assistant]\nhello\n"

    state.mark_history_view()
    state.apply_event(SessionEvent(type="message_delta", delta=" world", message_id="m1"))
    assert state.auto_follow_output is False
    assert state.unseen_output_updates == 1
    assert state.transcript_text == "[User]\nhello\n\n[Assistant]\nhello world\n"

    state.mark_jumped_to_latest()
    assert state.auto_follow_output is True
    assert state.unseen_output_updates == 0


def test_agent_session_does_not_create_blank_assistant_cards_for_user_message_events(tmp_path: Path, stub_model: Model) -> None:
    paths = build_agent_paths(tmp_path / "home")
    paths.ensure_exists()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resource_loader = ResourceLoader(
        skills_dir=paths.skills_dir,
        prompts_dir=paths.prompts_dir,
        themes_dir=paths.themes_dir,
        extensions_dir=paths.extensions_dir,
        workspace_root=workspace,
    )
    session_manager = SessionManager(paths.sessions_dir)
    session = AgentSession(
        workspace_root=workspace,
        cwd=workspace,
        model=stub_model,
        thinking="medium",
        settings=CodingAgentSettings(default_model=stub_model.id),
        session_manager=session_manager,
        resource_loader=resource_loader,
    )
    user_start_events = session._map_event(type("E", (), {"type": "message_start", "message": UserMessage(content="hello")})())
    assistant_start_events = session._map_event(type("E", (), {"type": "message_start", "message": None})())

    assert user_start_events == []
    assert len(assistant_start_events) == 1


def test_renderer_jump_to_bottom_and_follow_logic() -> None:
    state = InteractiveState(
        session_id="s1",
        model_id="stub:test",
        thinking="medium",
        cwd=Path("/tmp"),
        theme="default",
    )
    state.start_user_turn("scroll me", "turn-1")
    state.apply_event(
        SessionEvent(
            type="message_end",
            message_id="assistant-1",
            turn_id="turn-1",
            message="\n".join(f"line-{i}" for i in range(120)),
        )
    )
    state.rebuild_transcript()
    renderer = InteractiveRenderer(state)
    renderer.sync_transcript()
    renderer.main_window.render_info = type("FakeRenderInfo", (), {"window_height": 20})()
    renderer.main_window.vertical_scroll = 10
    state.mark_history_view()
    renderer.scroll_main_to_bottom()

    assert renderer.main_window.vertical_scroll == state.transcript_line_count - 20
    assert state.auto_follow_output is True
    assert state.unseen_output_updates == 0


def test_renderer_follow_after_render_uses_real_render_info() -> None:
    state = InteractiveState(
        session_id="s1",
        model_id="stub:test",
        thinking="medium",
        cwd=Path("/tmp"),
        theme="default",
    )
    state.start_user_turn("scroll me", "turn-1")
    state.apply_event(
        SessionEvent(
            type="message_end",
            message_id="assistant-1",
            turn_id="turn-1",
            message="\n".join(f"line-{i}" for i in range(120)),
        )
    )
    state.rebuild_transcript()
    renderer = InteractiveRenderer(state)
    renderer.sync_transcript()
    renderer.main_window.render_info = type("FakeRenderInfo", (), {"window_height": 20})()
    renderer.follow_output_if_needed()
    renderer.update_scroll_after_render()

    assert renderer.main_window.vertical_scroll == state.transcript_line_count - 20


def test_interactive_state_builds_transcript_without_blank_assistant_blocks() -> None:
    state = InteractiveState(
        session_id="s1",
        model_id="stub:test",
        thinking="medium",
        cwd=Path("/tmp"),
        theme="default",
    )
    state.start_user_turn("inspect file", "turn-1")
    state.apply_event(SessionEvent(type="status", message="准备执行工具", is_transient=True, panel="status", source="steering", turn_id="turn-1"))
    state.apply_event(SessionEvent(type="message_start", message_id="a1", turn_id="turn-1"))
    state.apply_event(SessionEvent(type="message_delta", message_id="a1", delta="first", turn_id="turn-1"))
    state.apply_event(SessionEvent(type="message_delta", message_id="a1", delta=" second", turn_id="turn-1"))
    state.apply_event(SessionEvent(type="thinking", message="先分析文件结构", message_id="a1", turn_id="turn-1"))
    state.apply_event(SessionEvent(type="tool_start", tool_name="read", tool_arguments='{"path":"a.txt"}', message_id="t1", turn_id="turn-1"))
    state.apply_event(SessionEvent(type="tool_end", tool_name="read", message="ok", tool_output_preview="ok", message_id="t1", turn_id="turn-1"))

    assert state.transcript_text.count("[Assistant]") == 1
    assert "[Thinking]\n先分析文件结构" in state.transcript_text
    assert "[Tool:read] success\nargs: {\"path\":\"a.txt\"}\nok" in state.transcript_text
    assert state.transcript_text.index("[User]") < state.transcript_text.index("[Thinking]") < state.transcript_text.index("[Tool:read] success") < state.transcript_text.index("[Assistant]")


def test_renderer_applies_theme_styles_and_focus_returns_to_input() -> None:
    state = InteractiveState(
        session_id="s1",
        model_id="stub:test",
        thinking="medium",
        cwd=Path("/tmp"),
        theme="default",
        theme_styles={
            "user": "ansimagenta bold",
            "assistant_header": "ansicyan bold",
            "assistant_body": "ansicyan",
            "thinking_header": "ansiblue bold",
            "thinking_body": "ansiblue",
            "tool_header": "ansiyellow bold",
            "tool_running": "ansiyellow",
            "tool_success": "ansigreen",
            "tool_error": "ansired",
            "status": "ansiwhite",
            "error": "ansired bold",
            "status_bar": "reverse",
            "input_prompt": "ansimagenta",
        },
    )
    state.start_user_turn("hello", "turn-1")
    state.apply_event(SessionEvent(type="thinking", message="plan", turn_id="turn-1"))
    state.apply_event(SessionEvent(type="tool_start", tool_name="ls", turn_id="turn-1", message_id="tool-1"))
    state.apply_event(SessionEvent(type="tool_end", tool_name="ls", turn_id="turn-1", message_id="tool-1", message="done"))
    state.apply_event(SessionEvent(type="message_end", message="answer", turn_id="turn-1", message_id="assistant-1"))
    renderer = InteractiveRenderer(state)
    controller = type(
        "ControllerStub",
        (),
        {
            "submit_current_buffer": lambda self: None,
            "cancel_current": lambda self: None,
            "clear_output": lambda self: None,
            "autocomplete_buffer": lambda self: None,
            "show_help": lambda self, message: None,
            "scroll_main_page_up": lambda self: None,
            "scroll_main_page_down": lambda self: None,
            "jump_to_latest": lambda self: None,
            "jump_to_oldest": lambda self: None,
            "scroll_main_up": lambda self: None,
            "scroll_main_down": lambda self: None,
            "toggle_focus": lambda self: None,
            "focus_input": lambda self: None,
        },
    )()
    app = renderer.build_application(controller)
    assert app.layout.current_window is renderer.input_window
    renderer.focus_main()
    assert renderer.focused_on_input() is False
    renderer.focus_input_if_idle()
    assert renderer.focused_on_input() is True
    assert renderer._build_style().style_rules
    assert state.transcript_line_styles[0] == "user"


def test_renderer_application_enables_mouse_support() -> None:
    state = InteractiveState(
        session_id="s1",
        model_id="stub:test",
        thinking="medium",
        cwd=Path("/tmp"),
        theme="default",
    )
    renderer = InteractiveRenderer(state)
    controller = type(
        "ControllerStub",
        (),
        {
            "submit_current_buffer": lambda self: None,
            "cancel_current": lambda self: None,
            "clear_output": lambda self: None,
            "autocomplete_buffer": lambda self: None,
            "show_help": lambda self, message: None,
            "scroll_main_page_up": lambda self: None,
            "scroll_main_page_down": lambda self: None,
            "jump_to_latest": lambda self: None,
            "jump_to_oldest": lambda self: None,
            "scroll_main_up": lambda self: None,
            "scroll_main_down": lambda self: None,
            "toggle_focus": lambda self: None,
        },
    )()
    app = renderer.build_application(controller)

    assert app.mouse_support()


def test_main_output_mouse_wheel_uses_renderer_scroll_state_machine() -> None:
    state = InteractiveState(
        session_id="s1",
        model_id="stub:test",
        thinking="medium",
        cwd=Path("/tmp"),
        theme="default",
    )
    state.start_user_turn("scroll me", "turn-1")
    state.apply_event(
        SessionEvent(
            type="message_end",
            message_id="assistant-1",
            turn_id="turn-1",
            message="\n".join(f"line-{i}" for i in range(200)),
        )
    )
    renderer = InteractiveRenderer(state)

    class FakeRenderInfo:
        content_height = 220
        window_height = 20
        vertical_scroll = 200

    renderer.main_window.render_info = FakeRenderInfo()
    scroll_calls: list[int] = []
    original_scroll_main_lines = renderer.scroll_main_lines

    def tracked_scroll(delta: int) -> None:
        scroll_calls.append(delta)
        original_scroll_main_lines(delta)

    renderer.scroll_main_lines = tracked_scroll

    event_up = MouseEvent(position=Point(x=0, y=0), event_type=MouseEventType.SCROLL_UP, button=MouseButton.NONE, modifiers=frozenset())
    event_down = MouseEvent(position=Point(x=0, y=0), event_type=MouseEventType.SCROLL_DOWN, button=MouseButton.NONE, modifiers=frozenset())

    assert renderer.main_control.mouse_handler(event_up) is None
    assert scroll_calls[-1] == -3
    assert state.main_view_mode == "history"

    FakeRenderInfo.vertical_scroll = 197
    assert renderer.main_control.mouse_handler(event_down) is None
    assert scroll_calls[-1] == 3


def test_main_output_mouse_wheel_does_not_require_focus_change() -> None:
    state = InteractiveState(
        session_id="s1",
        model_id="stub:test",
        thinking="medium",
        cwd=Path("/tmp"),
        theme="default",
    )
    renderer = InteractiveRenderer(state)
    controller = type(
        "ControllerStub",
        (),
        {
            "submit_current_buffer": lambda self: None,
            "cancel_current": lambda self: None,
            "clear_output": lambda self: None,
            "autocomplete_buffer": lambda self: None,
            "show_help": lambda self, message: None,
            "scroll_main_page_up": lambda self: None,
            "scroll_main_page_down": lambda self: None,
            "jump_to_latest": lambda self: None,
            "jump_to_oldest": lambda self: None,
            "scroll_main_up": lambda self: None,
            "scroll_main_down": lambda self: None,
            "toggle_focus": lambda self: None,
            "focus_input": lambda self: None,
        },
    )()
    app = renderer.build_application(controller)
    assert app.layout.current_window is renderer.input_window
    called: list[str] = []
    renderer.handle_main_mouse_scroll = lambda direction: called.append(direction)
    event_up = MouseEvent(position=Point(x=0, y=0), event_type=MouseEventType.SCROLL_UP, button=MouseButton.NONE, modifiers=frozenset())

    renderer.main_control.mouse_handler(event_up)

    assert called == ["up"]
    assert app.layout.current_window is renderer.input_window


def test_model_registry_cli_and_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home_root = tmp_path / "home"
    paths = build_agent_paths(home_root)
    paths.ensure_exists()
    paths.models_file.write_text(
        json.dumps(
            [
                {
                    "id": "stub:test",
                    "provider": "stub",
                    "inputPrice": 0.1,
                    "outputPrice": 0.2,
                    "contextWindow": 10000,
                    "maxOutputTokens": 1000,
                }
            ]
        ),
        encoding="utf-8",
    )
    registry = ModelRegistry(paths.models_file)
    assert registry.get("stub:test").provider == "stub"

    args = parse_args(["--model", "stub:test", "--thinking", "low", "--cwd", str(tmp_path)])
    assert args.model == "stub:test"
    assert args.thinking == "low"

    run_state: dict[str, object] = {}

    async def fake_run(self) -> int:
        run_state["model"] = self.session.model.id
        run_state["thinking"] = self.session.thinking
        return 0

    monkeypatch.setattr(main_module, "build_agent_paths", lambda: paths)
    monkeypatch.setattr("coding_agent.modes.interactive.app.InteractiveApp.run", fake_run)

    exit_code = main_module.main(["--model", "stub:test", "--thinking", "low", "--cwd", str(tmp_path)])

    assert exit_code == 0
    assert run_state == {"model": "stub:test", "thinking": "low"}


def test_resource_loader_loads_python_extension_runtime_and_tool_registry(tmp_path: Path, stub_model: Model) -> None:
    root = tmp_path / "root"
    paths = build_agent_paths(root)
    paths.ensure_exists()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    extension_dir = paths.extensions_dir / "demo_ext"
    extension_dir.mkdir(parents=True)
    (extension_dir / "extension.json").write_text(json.dumps({"module": "extension.py"}), encoding="utf-8")
    (extension_dir / "extension.py").write_text(
        "\n".join(
            [
                "from agent_core import AgentTool",
                "",
                "def register(api):",
                "    api.extend_system_prompt('EXT PROMPT')",
                "    api.register_command('demo', description='demo command')",
                "    api.register_provider('demo_provider', lambda **kwargs: None)",
                "    async def execute(arguments, context):",
                "        return 'demo-result'",
                "    api.register_tool(AgentTool(name='demo_tool', description='demo', inputSchema={}, execute=execute))",
            ]
        ),
        encoding="utf-8",
    )

    bundle = ResourceLoader(
        skills_dir=paths.skills_dir,
        prompts_dir=paths.prompts_dir,
        themes_dir=paths.themes_dir,
        extensions_dir=paths.extensions_dir,
        workspace_root=workspace,
    ).load()
    registry = build_tool_registry(workspace, workspace, CodingAgentSettings(default_model=stub_model.id))
    for tool in bundle.extension_runtime.tools:
        registry.register_tool(tool, source=tool.metadata.get("source", "extension"))

    assert bundle.extension_runtime.prompt_fragments == ["EXT PROMPT"]
    assert bundle.extension_runtime.commands[0].name == "demo"
    assert "demo_provider" in bundle.extension_runtime.provider_factories
    assert any(tool.name == "demo_tool" for tool in registry.active_tools)


def test_session_manager_persists_only_message_events(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "sessions")
    manager.create_session(cwd=tmp_path, model_id="openai:gpt-5")
    manager.append_message(message=UserMessage(content="hello"), parent_id=None)

    messages = manager.build_context_messages()
    events = manager.iter_events()

    assert [message.content for message in messages] == ["hello"]
    assert [event["type"] for event in events] == ["session", "message"]
