from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ai import AssistantMessage, Model, StreamEvent, ToolCall, UserMessage
from coding_agent.config.paths import build_agent_paths
from coding_agent.core.resource_loader import ResourceLoader
from coding_agent.core.session_manager import SessionManager
from coding_agent.core.settings_manager import SettingsManager
from coding_agent.core.types import CodingAgentSettings, SessionEvent
from mom.context_sync import sync_channel_log_to_session
from mom.events import EventsWatcher
from mom.feishu import FeishuBotTransport, FeishuConfig
from mom.main import MomApp, MomConfig
from mom.runner import MomRunner
from mom.store import MomStore
from mom.types import ChatAttachment, ChatContext, ChatEvent, ChatInfo, ChatUser, MomRenderConfig, RunResult


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
        id="stub:test-mom",
        provider="stub",
        inputPrice=0.1,
        outputPrice=0.2,
        contextWindow=10000,
        maxOutputTokens=1000,
    )


def make_store(tmp_path: Path) -> MomStore:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return MomStore(workspace)


def test_sync_channel_log_to_session_uses_message_ids(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    session_manager = SessionManager(store.paths.sessions_dir)
    ref = store.get_or_create_session_ref("chat-1", session_manager, "openai:gpt-5")
    event = ChatEvent(
        platform="feishu",
        chat_id="chat-1",
        message_id="m-1",
        sender_id="u-1",
        sender_name="alice",
        text="hello world",
        attachments=[ChatAttachment(original_name="a.txt", local_path="workspace/.mom/channels/chat-1/attachments/m-1_a.txt")],
        is_direct=True,
        is_trigger=True,
    )
    store.log_event(event)

    inserted = sync_channel_log_to_session(session_manager, ref, store.channel_dir("chat-1"))
    messages = session_manager.build_context_messages(ref.leaf_id)

    assert inserted == 1
    assert ref.synced_message_ids == ["m-1"]
    assert "hello world" in messages[0].text
    assert "attachments" in messages[0].text

    inserted_again = sync_channel_log_to_session(session_manager, ref, store.channel_dir("chat-1"))
    assert inserted_again == 0


@pytest.mark.asyncio
async def test_mom_runner_logs_detail_and_main_messages(tmp_path: Path, stub_model: Model) -> None:
    store = make_store(tmp_path)
    home_root = tmp_path / "home"
    agent_paths = build_agent_paths(home_root)
    agent_paths.ensure_exists()
    settings = CodingAgentSettings(default_model=stub_model.id)
    ref = store.get_or_create_session_ref("chat-2", SessionManager(store.paths.sessions_dir), stub_model.id)
    store.log_event(
        ChatEvent(
            platform="feishu",
            chat_id="chat-2",
            message_id="history-1",
            sender_id="u-1",
            sender_name="alice",
            text="before trigger",
        )
    )

    async def fake_stream(model, context, thinking, registry=None):
        if any(getattr(message, "role", "") == "tool" for message in context.history):
            return FakeSession(
                [
                    StreamEvent(type="start", provider="stub", model=stub_model),
                    StreamEvent(
                        type="done",
                        provider="stub",
                        model=stub_model,
                        assistantMessage=AssistantMessage(content="final answer"),
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
                        content="draft",
                        toolCalls=[ToolCall(id="call_1", name="ls", arguments='{"path":"."}')],
                    ),
                ),
            ]
        )

    runner = MomRunner(
        platform_name="飞书",
        chat_id="chat-2",
        chat_dir=store.channel_dir("chat-2"),
        store=store,
        model=stub_model,
        settings=settings,
        agent_paths=agent_paths,
        session_ref=ref,
        stream_fn=fake_stream,
    )

    messages: list[tuple[str, str]] = []

    async def respond(text: str, _: bool = True) -> str | None:
        messages.append(("main", text))
        return "main-1"

    async def replace_message(text: str) -> str | None:
        messages.append(("main", text))
        return "main-1"

    async def respond_detail(text: str) -> str | None:
        messages.append(("detail", text))
        return f"detail-{len([item for item in messages if item[0] == 'detail'])}"

    async def upload_file(_: str, __: str | None = None) -> str | None:
        return None

    async def noop(_: bool | None = None) -> None:
        return None

    ctx = ChatContext(
        message=ChatEvent(
            platform="feishu",
            chat_id="chat-2",
            message_id="trigger-1",
            sender_id="u-2",
            sender_name="bob",
            text="please help",
            attachments=[ChatAttachment(original_name="spec.md", local_path=".mom/channels/chat-2/attachments/trigger-1_spec.md")],
            is_direct=True,
            is_trigger=True,
            chat_name="product",
        ),
        chat_name="product",
        users=[ChatUser(id="u-2", name="bob")],
        chats=[ChatInfo(id="chat-2", name="product")],
        respond=respond,
        replace_message=replace_message,
        respond_detail=respond_detail,
        upload_file=upload_file,
        set_working=noop,
        delete_message=noop,
    )

    result = await runner.run(ctx, store)
    log_entries = store.read_log_entries("chat-2")

    assert result.stop_reason == "completed"
    assert [text for kind, text in messages if kind == "main"][-1] == "final answer"
    assert all(kind != "detail" for kind, _ in messages)
    assert result.final_text == "final answer"
    assert result.detail_count == 0
    assert any(entry["is_bot"] and entry["text"] == "final answer" for entry in log_entries)
    assert "history-1" in ref.synced_message_ids


@pytest.mark.asyncio
async def test_mom_runner_buffers_deltas_into_single_final_message(tmp_path: Path, stub_model: Model) -> None:
    store = make_store(tmp_path)
    home_root = tmp_path / "home"
    agent_paths = build_agent_paths(home_root)
    agent_paths.ensure_exists()
    settings = CodingAgentSettings(default_model=stub_model.id)
    ref = store.get_or_create_session_ref("chat-delta", SessionManager(store.paths.sessions_dir), stub_model.id)

    runner = MomRunner(
        platform_name="飞书",
        chat_id="chat-delta",
        chat_dir=store.channel_dir("chat-delta"),
        store=store,
        model=stub_model,
        settings=settings,
        agent_paths=agent_paths,
        session_ref=ref,
    )

    async def fake_run_turn():
        yield SessionEvent(type="message_start")
        yield SessionEvent(type="message_delta", delta="你")
        yield SessionEvent(type="message_delta", delta="好")
        yield SessionEvent(type="message_delta", delta="呀")
        yield SessionEvent(type="message_end", message="你好呀")

    runner.session.run_turn = fake_run_turn  # type: ignore[method-assign]

    calls: list[tuple[str, str]] = []

    async def respond(text: str, _: bool = True) -> str | None:
        calls.append(("respond", text))
        return "main-1"

    async def replace_message(text: str) -> str | None:
        calls.append(("replace", text))
        return "main-1"

    async def respond_detail(text: str) -> str | None:
        calls.append(("detail", text))
        return "detail-1"

    async def upload_file(_: str, __: str | None = None) -> str | None:
        return None

    async def noop(_: bool | None = None) -> None:
        return None

    ctx = ChatContext(
        message=ChatEvent(
            platform="feishu",
            chat_id="chat-delta",
            message_id="trigger-delta",
            sender_id="u-1",
            sender_name="alice",
            text="hi",
            is_direct=True,
            is_trigger=True,
        ),
        chat_name=None,
        users=[],
        chats=[],
        respond=respond,
        replace_message=replace_message,
        respond_detail=respond_detail,
        upload_file=upload_file,
        set_working=noop,
        delete_message=noop,
    )

    result = await runner.run(ctx, store)

    assert result.stop_reason == "completed"
    assert result.final_text == "你好呀"
    assert result.suppressed_events_count == 3
    assert calls == [("respond", "处理中…"), ("replace", "你好呀")]


@pytest.mark.asyncio
async def test_mom_runner_suppresses_internal_events_by_default(tmp_path: Path, stub_model: Model) -> None:
    store = make_store(tmp_path)
    home_root = tmp_path / "home"
    agent_paths = build_agent_paths(home_root)
    agent_paths.ensure_exists()
    settings = CodingAgentSettings(default_model=stub_model.id)
    ref = store.get_or_create_session_ref("chat-quiet", SessionManager(store.paths.sessions_dir), stub_model.id)

    runner = MomRunner(
        platform_name="飞书",
        chat_id="chat-quiet",
        chat_dir=store.channel_dir("chat-quiet"),
        store=store,
        model=stub_model,
        settings=settings,
        agent_paths=agent_paths,
        session_ref=ref,
    )

    async def fake_run_turn():
        yield SessionEvent(type="thinking", message="内部思考")
        yield SessionEvent(type="tool_start", tool_name="ls")
        yield SessionEvent(type="status", message="follow-up")
        yield SessionEvent(type="tool_end", tool_name="ls", message="done")
        yield SessionEvent(type="message_end", message="最终答复")

    runner.session.run_turn = fake_run_turn  # type: ignore[method-assign]

    calls: list[tuple[str, str]] = []

    async def respond(text: str, _: bool = True) -> str | None:
        calls.append(("respond", text))
        return "main-1"

    async def replace_message(text: str) -> str | None:
        calls.append(("replace", text))
        return "main-1"

    async def respond_detail(text: str) -> str | None:
        calls.append(("detail", text))
        return "detail-1"

    async def upload_file(_: str, __: str | None = None) -> str | None:
        return None

    async def noop(_: bool | None = None) -> None:
        return None

    ctx = ChatContext(
        message=ChatEvent(
            platform="feishu",
            chat_id="chat-quiet",
            message_id="trigger-quiet",
            sender_id="u-1",
            sender_name="alice",
            text="hi",
            is_direct=True,
            is_trigger=True,
        ),
        chat_name=None,
        users=[],
        chats=[],
        respond=respond,
        replace_message=replace_message,
        respond_detail=respond_detail,
        upload_file=upload_file,
        set_working=noop,
        delete_message=noop,
    )

    result = await runner.run(ctx, store)

    assert result.stop_reason == "completed"
    assert result.detail_count == 0
    assert result.suppressed_events_count == 4
    assert all(kind != "detail" for kind, _ in calls)
    assert calls[-1] == ("replace", "最终答复")


@pytest.mark.asyncio
async def test_mom_runner_emits_tool_details_only_in_debug_mode(tmp_path: Path, stub_model: Model) -> None:
    store = make_store(tmp_path)
    home_root = tmp_path / "home"
    agent_paths = build_agent_paths(home_root)
    agent_paths.ensure_exists()
    settings = CodingAgentSettings(default_model=stub_model.id)
    ref = store.get_or_create_session_ref("chat-debug", SessionManager(store.paths.sessions_dir), stub_model.id)

    runner = MomRunner(
        platform_name="飞书",
        chat_id="chat-debug",
        chat_dir=store.channel_dir("chat-debug"),
        store=store,
        model=stub_model,
        settings=settings,
        agent_paths=agent_paths,
        session_ref=ref,
        render_config=MomRenderConfig(show_tool_details=True),
    )

    async def fake_run_turn():
        yield SessionEvent(type="tool_end", tool_name="ls", tool_arguments='{"path":"."}', message="done")
        yield SessionEvent(type="message_end", message="ok")

    runner.session.run_turn = fake_run_turn  # type: ignore[method-assign]

    calls: list[tuple[str, str]] = []

    async def respond(text: str, _: bool = True) -> str | None:
        calls.append(("respond", text))
        return "main-1"

    async def replace_message(text: str) -> str | None:
        calls.append(("replace", text))
        return "main-1"

    async def respond_detail(text: str) -> str | None:
        calls.append(("detail", text))
        return "detail-1"

    async def upload_file(_: str, __: str | None = None) -> str | None:
        return None

    async def noop(_: bool | None = None) -> None:
        return None

    ctx = ChatContext(
        message=ChatEvent(
            platform="feishu",
            chat_id="chat-debug",
            message_id="trigger-debug",
            sender_id="u-1",
            sender_name="alice",
            text="hi",
            is_direct=True,
            is_trigger=True,
        ),
        chat_name=None,
        users=[],
        chats=[],
        respond=respond,
        replace_message=replace_message,
        respond_detail=respond_detail,
        upload_file=upload_file,
        set_working=noop,
        delete_message=noop,
    )

    result = await runner.run(ctx, store)

    assert result.detail_count == 1
    assert any(kind == "detail" and "工具: ls" in text for kind, text in calls)


def test_feishu_transport_parse_event_strips_mentions() -> None:
    transport = FeishuBotTransport(handler=None, config=FeishuConfig(app_id="app", app_secret="secret"))
    payload = {
        "event": {
            "message": {
                "chat_id": "oc_123",
                "message_id": "om_123",
                "chat_type": "group",
                "content": json.dumps({"text": "@mom 请处理这个问题"}),
            },
            "sender": {"sender_id": {"open_id": "ou_123"}, "sender_name": "Alice"},
            "mentions": [{"name": "@mom"}],
            "chat_name": "研发群",
        }
    }

    event = transport.parse_event(payload)

    assert event is not None
    assert event.chat_id == "oc_123"
    assert event.text == "请处理这个问题"
    assert event.is_trigger is True
    assert event.is_direct is False


@pytest.mark.asyncio
async def test_feishu_transport_serve_uses_long_connection_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Handler:
        async def handle_chat_event(self, event: ChatEvent) -> None:
            _ = event

    transport = FeishuBotTransport(handler=Handler(), config=FeishuConfig(app_id="app", app_secret="secret"))
    store = make_store(tmp_path)

    async def fake_long_connection(s: MomStore) -> None:
        _ = s
        calls.append("long")

    async def fake_webhook(s: MomStore) -> None:
        _ = s
        calls.append("webhook")

    monkeypatch.setattr(transport, "_serve_long_connection", fake_long_connection)
    monkeypatch.setattr(transport, "_serve_webhook", fake_webhook)
    await transport.serve(store)

    assert calls == ["long"]


@pytest.mark.asyncio
async def test_feishu_transport_long_connection_dispatches_message_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[ChatEvent] = []

    class Handler:
        async def handle_chat_event(self, event: ChatEvent) -> None:
            received.append(event)

    transport = FeishuBotTransport(handler=Handler(), config=FeishuConfig(app_id="app", app_secret="secret"))
    store = make_store(tmp_path)

    class FakeClient:
        def __init__(self, app_id, app_secret, event_handler):
            self.app_id = app_id
            self.app_secret = app_secret
            self.event_handler = event_handler

        def start(self) -> None:
            self.event_handler.do_without_validation(
                json.dumps(
                    {
                        "schema": "2.0",
                        "header": {
                            "event_id": "evt_1",
                            "token": "",
                            "create_time": "0",
                            "event_type": "im.message.receive_v1",
                            "tenant_key": "tenant",
                            "app_id": "app",
                        },
                        "event": {
                            "message": {
                                "chat_id": "oc_long_1",
                                "message_id": "om_long_1",
                                "chat_type": "p2p",
                                "content": json.dumps({"text": "long connection hello"}),
                            },
                            "sender": {"sender_id": {"open_id": "ou_long_1"}, "sender_name": "Long User"},
                            "mentions": [],
                            "chat_name": "LongChat",
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
            )

    real_lark = __import__("lark_oapi")
    fake_lark = SimpleNamespace(
        ws=SimpleNamespace(Client=FakeClient),
        EventDispatcherHandler=real_lark.EventDispatcherHandler,
        JSON=real_lark.JSON,
    )
    monkeypatch.setitem(sys.modules, "lark_oapi", fake_lark)

    await transport._serve_long_connection(store)
    logs = store.read_log_entries("oc_long_1")

    assert len(received) == 1
    assert received[0].text == "long connection hello"
    assert logs and logs[0]["text"] == "long connection hello"


@pytest.mark.asyncio
async def test_events_watcher_dispatches_immediate_and_periodic(tmp_path: Path) -> None:
    dispatched: list[ChatEvent] = []

    async def dispatch(event: ChatEvent) -> None:
        dispatched.append(event)

    events_dir = tmp_path / "events"
    events_dir.mkdir()
    (events_dir / "now.json").write_text(
        json.dumps({"type": "immediate", "channelId": "chat-1", "text": "ping"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (events_dir / "periodic.json").write_text(
        json.dumps({"type": "periodic", "channelId": "chat-1", "text": "tick", "interval_seconds": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (events_dir / "future.json").write_text(
        json.dumps(
            {
                "type": "one-shot",
                "channelId": "chat-1",
                "text": "later",
                "at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    watcher = EventsWatcher(events_dir, dispatch)
    await watcher.scan_once()

    assert [item.text for item in dispatched] == ["ping", "tick"]
    assert not (events_dir / "now.json").exists()
    assert (events_dir / "future.json").exists()


@pytest.mark.asyncio
async def test_mom_app_handles_busy_and_stop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home_root = tmp_path / "home"
    agent_paths = build_agent_paths(home_root)
    agent_paths.ensure_exists()
    settings = SettingsManager(agent_paths.settings_file, tmp_path / "workspace" / ".LiuClaw" / "settings.json").load()

    class FakeTransport:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def create_context(self, event: ChatEvent, store: MomStore) -> ChatContext:
            async def respond(text: str, _: bool = True) -> str | None:
                self.messages.append(text)
                return "msg"

            async def replace_message(text: str) -> str | None:
                self.messages.append(text)
                return "msg"

            async def respond_detail(text: str) -> str | None:
                self.messages.append(text)
                return "detail"

            async def upload_file(_: str, __: str | None = None) -> str | None:
                return None

            async def noop(_: bool | None = None) -> None:
                return None

            return ChatContext(
                message=event,
                chat_name=event.chat_name,
                users=[],
                chats=[],
                respond=respond,
                replace_message=replace_message,
                respond_detail=respond_detail,
                upload_file=upload_file,
                set_working=noop,
                delete_message=noop,
            )

    class FakeRunner:
        def __init__(self) -> None:
            self.abort_called = False
            self.started = asyncio.Event()

        def abort(self) -> None:
            self.abort_called = True

        async def run(self, ctx: ChatContext, store: MomStore) -> RunResult:
            self.started.set()
            while not self.abort_called:
                await asyncio.sleep(0.01)
            return RunResult(stop_reason="aborted")

    fake_runner = FakeRunner()
    fake_transport = FakeTransport()
    monkeypatch.setattr("mom.main.build_agent_paths", lambda: agent_paths)
    monkeypatch.setattr("mom.main.get_or_create_runner", lambda *args, **kwargs: fake_runner)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = MomApp(
        MomConfig(workspace_root=workspace, feishu=FeishuConfig(app_id="app", app_secret="secret"), model_id="openai:gpt-5"),
        transport=fake_transport,
    )

    trigger = ChatEvent(
        platform="feishu",
        chat_id="chat-5",
        message_id="m-1",
        sender_id="u-1",
        sender_name="alice",
        text="start",
        is_direct=True,
        is_trigger=True,
    )
    busy = ChatEvent(
        platform="feishu",
        chat_id="chat-5",
        message_id="m-2",
        sender_id="u-2",
        sender_name="bob",
        text="another",
        is_direct=True,
        is_trigger=True,
    )
    stop = ChatEvent(
        platform="feishu",
        chat_id="chat-5",
        message_id="m-3",
        sender_id="u-2",
        sender_name="bob",
        text="stop",
        is_direct=True,
        is_trigger=True,
    )

    task = asyncio.create_task(app.handle_chat_event(trigger))
    await fake_runner.started.wait()
    await app.handle_chat_event(busy)
    await app.handle_chat_event(stop)
    await task

    assert fake_runner.abort_called is True
    assert any("正在处理当前频道任务" in item for item in fake_transport.messages)
    assert any("正在停止当前任务" in item for item in fake_transport.messages)


@pytest.mark.asyncio
async def test_mom_app_resets_running_state_after_runner_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home_root = tmp_path / "home"
    agent_paths = build_agent_paths(home_root)
    agent_paths.ensure_exists()

    class FakeTransport:
        def create_context(self, event: ChatEvent, store: MomStore) -> ChatContext:
            async def respond(text: str, _: bool = True) -> str | None:
                return "msg"

            async def replace_message(text: str) -> str | None:
                return "msg"

            async def respond_detail(text: str) -> str | None:
                return "detail"

            async def upload_file(_: str, __: str | None = None) -> str | None:
                return None

            async def noop(_: bool | None = None) -> None:
                return None

            return ChatContext(
                message=event,
                chat_name=event.chat_name,
                users=[],
                chats=[],
                respond=respond,
                replace_message=replace_message,
                respond_detail=respond_detail,
                upload_file=upload_file,
                set_working=noop,
                delete_message=noop,
            )

    class ErrorRunner:
        def abort(self) -> None:
            return None

        async def run(self, ctx: ChatContext, store: MomStore) -> RunResult:
            _ = ctx, store
            raise RuntimeError("boom")

    fake_transport = FakeTransport()
    monkeypatch.setattr("mom.main.build_agent_paths", lambda: agent_paths)
    monkeypatch.setattr("mom.main.get_or_create_runner", lambda *args, **kwargs: ErrorRunner())

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = MomApp(
        MomConfig(workspace_root=workspace, feishu=FeishuConfig(app_id="app", app_secret="secret"), model_id="openai:gpt-5"),
        transport=fake_transport,
    )
    event = ChatEvent(
        platform="feishu",
        chat_id="chat-err",
        message_id="m-1",
        sender_id="u-1",
        sender_name="alice",
        text="hello",
        is_direct=True,
        is_trigger=True,
    )

    await app.handle_chat_event(event)

    assert app.get_state("chat-err").running is False


@pytest.mark.asyncio
async def test_mom_app_dedupes_same_incoming_message_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home_root = tmp_path / "home"
    agent_paths = build_agent_paths(home_root)
    agent_paths.ensure_exists()

    class FakeTransport:
        def create_context(self, event: ChatEvent, store: MomStore) -> ChatContext:
            _ = store

            async def respond(text: str, _: bool = True) -> str | None:
                return "msg"

            async def replace_message(text: str) -> str | None:
                return "msg"

            async def respond_detail(text: str) -> str | None:
                return "detail"

            async def upload_file(_: str, __: str | None = None) -> str | None:
                return None

            async def noop(_: bool | None = None) -> None:
                return None

            return ChatContext(
                message=event,
                chat_name=event.chat_name,
                users=[],
                chats=[],
                respond=respond,
                replace_message=replace_message,
                respond_detail=respond_detail,
                upload_file=upload_file,
                set_working=noop,
                delete_message=noop,
            )

    class CountingRunner:
        def __init__(self) -> None:
            self.calls = 0

        def abort(self) -> None:
            return None

        async def run(self, ctx: ChatContext, store: MomStore) -> RunResult:
            _ = ctx, store
            self.calls += 1
            return RunResult(stop_reason="completed", final_text="ok")

    fake_runner = CountingRunner()
    fake_transport = FakeTransport()
    monkeypatch.setattr("mom.main.build_agent_paths", lambda: agent_paths)
    monkeypatch.setattr("mom.main.get_or_create_runner", lambda *args, **kwargs: fake_runner)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = MomApp(
        MomConfig(workspace_root=workspace, feishu=FeishuConfig(app_id="app", app_secret="secret"), model_id="openai:gpt-5"),
        transport=fake_transport,
    )

    event = ChatEvent(
        platform="feishu",
        chat_id="chat-dedupe",
        message_id="same-msg",
        sender_id="u-1",
        sender_name="alice",
        text="你好",
        is_direct=True,
        is_trigger=True,
    )

    await app.handle_chat_event(event)
    await app.handle_chat_event(event)

    assert fake_runner.calls == 1


def test_store_log_event_is_idempotent_for_same_message_id(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    event = ChatEvent(
        platform="feishu",
        chat_id="chat-log-dedupe",
        message_id="msg-1",
        sender_id="u-1",
        sender_name="alice",
        text="hello",
        is_direct=True,
        is_trigger=True,
    )

    store.log_event(event)
    store.log_event(event)

    logs = store.read_log_entries("chat-log-dedupe")
    assert len(logs) == 1
