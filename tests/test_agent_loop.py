from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from ai import AssistantMessage, Model, StreamEvent, ToolCall, ToolResultMessage, UserMessage
from ai.options import Options
from ai.providers.base import Provider
from ai.registry import ProviderRegistry
from agent_core import (
    AgentTraceCollector,
    AfterToolCallPass,
    AfterToolCallReplace,
    AgentLoopConfig,
    AgentState,
    AgentTool,
    BeforeToolCallAllow,
    BeforeToolCallError,
    BeforeToolCallSkip,
    RetryDecision,
    agentLoop,
    agentLoopContinue,
)
from agent_core import agent_loop as agent_loop_module


class ScriptedProvider(Provider):
    name = "stub"

    def __init__(self, *, fail: bool = False, multi_tool: bool = False, tool_only: bool = False) -> None:
        self.fail = fail
        self.multi_tool = multi_tool
        self.tool_only = tool_only

    def supports(self, model: Model) -> bool:
        return model.provider == self.name

    async def stream(self, model: Model, context, options: Options) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="start", provider=model.provider, model=model)
        if self.fail:
            yield StreamEvent(type="error", provider=model.provider, model=model, error="boom")
            return

        has_tool_result = any(isinstance(message, ToolResultMessage) for message in context.messages)
        if has_tool_result:
            yield StreamEvent(type="text_start", provider=model.provider, model=model)
            yield StreamEvent(type="text_delta", provider=model.provider, model=model, text="done")
            yield StreamEvent(
                type="done",
                provider=model.provider,
                model=model,
                assistantMessage=AssistantMessage(content="done"),
            )
            return

        if self.multi_tool:
            yield StreamEvent(
                type="done",
                provider=model.provider,
                model=model,
                assistantMessage=AssistantMessage(
                    content="",
                    toolCalls=[
                        ToolCall(id="call_1", name="slow_lookup", arguments='{"q":"slow"}'),
                        ToolCall(id="call_2", name="fast_lookup", arguments='{"q":"fast"}'),
                    ],
                ),
            )
            return

        wants_tool = any(getattr(message, "content", "") == "tool" for message in context.messages)
        if wants_tool:
            if not self.tool_only:
                yield StreamEvent(type="text_start", provider=model.provider, model=model)
                yield StreamEvent(type="text_delta", provider=model.provider, model=model, text="checking")
            yield StreamEvent(
                type="done",
                provider=model.provider,
                model=model,
                assistantMessage=AssistantMessage(
                    content="checking" if not self.tool_only else "",
                    toolCalls=[ToolCall(id="call_1", name="lookup", arguments='{"q":"spec"}')],
                ),
            )
            return

        latest_user = next(
            (message.content for message in reversed(context.messages) if getattr(message, "role", "") == "user"),
            "hello",
        )
        yield StreamEvent(type="text_start", provider=model.provider, model=model)
        yield StreamEvent(type="text_delta", provider=model.provider, model=model, text=f"echo:{latest_user}")
        yield StreamEvent(
            type="done",
            provider=model.provider,
            model=model,
            assistantMessage=AssistantMessage(content=f"echo:{latest_user}"),
        )


class FakeSession:
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
        id="stub:test-model",
        provider="stub",
        inputPrice=0.1,
        outputPrice=0.2,
        contextWindow=128000,
        maxOutputTokens=4096,
    )


async def collect_events(session) -> list:
    events = []
    async for event in session.consume():
        events.append(event)
    return events


def make_loop(stub_model: Model, **kwargs) -> AgentLoopConfig:
    return AgentLoopConfig(model=stub_model, **kwargs)


@pytest.mark.asyncio
async def test_agent_loop_returns_session_immediately(stub_model: Model) -> None:
    session = await agentLoop(make_loop(stub_model), initialMessages=[UserMessage(content="hello")])
    assert hasattr(session, "queue")
    assert hasattr(session, "consume")
    assert hasattr(session, "producer_task")


@pytest.mark.asyncio
async def test_agent_loop_defaults_to_stream_simple(stub_model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"streamSimple": 0}

    async def fake_stream_simple(model, context, *, reasoning=None, registry=None, **kwargs):
        called["streamSimple"] += 1
        return FakeSession(
            [
                StreamEvent(type="start", provider="stub", model=stub_model),
                StreamEvent(type="text_start", provider="stub", model=stub_model),
                StreamEvent(type="text_delta", provider="stub", model=stub_model, text="simple"),
                StreamEvent(
                    type="done",
                    provider="stub",
                    model=stub_model,
                    assistantMessage=AssistantMessage(content="simple"),
                ),
            ]
        )

    monkeypatch.setattr(agent_loop_module, "streamSimple", fake_stream_simple)

    session = await agentLoop(make_loop(stub_model), initialMessages=[UserMessage(content="hello")])
    events = await collect_events(session)

    assert called["streamSimple"] == 1
    assert events[-1].state.history[-1].content == "simple"


@pytest.mark.asyncio
async def test_custom_stream_fn_takes_precedence(stub_model: Model) -> None:
    called = {"custom": 0}

    async def custom_stream(model, context, thinking, registry=None):
        called["custom"] += 1
        return FakeSession(
            [
                StreamEvent(type="start", provider="stub", model=stub_model),
                StreamEvent(
                    type="done",
                    provider="stub",
                    model=stub_model,
                    assistantMessage=AssistantMessage(content="custom"),
                ),
            ]
        )

    session = await agentLoop(
        make_loop(stub_model, stream=custom_stream),
        initialMessages=[UserMessage(content="hello")],
    )
    events = await collect_events(session)

    assert called["custom"] == 1
    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]


@pytest.mark.asyncio
async def test_agent_loop_emits_basic_message_lifecycle(stub_model: Model) -> None:
    registry = ProviderRegistry([ScriptedProvider()])

    async def custom_stream(model, context, thinking, registry=None):
        return await agent_loop_module.streamSimple(
            model,
            {"systemPrompt": context.systemPrompt, "messages": context.history, "tools": context.tools},
            reasoning=thinking,
            registry=registry,
        )

    session = await agentLoop(
        make_loop(stub_model, stream=custom_stream, registry=registry),
        initialMessages=[UserMessage(content="hello")],
    )
    events = await collect_events(session)

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_update",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert events[7].message.content == "echo:hello"


@pytest.mark.asyncio
async def test_agent_loop_trace_collector_records_retry(stub_model: Model) -> None:
    attempts = {"count": 0}

    async def flaky_stream(model, context, thinking, registry=None, signal=None):
        _ = model, context, thinking, registry, signal
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary failure")
        return FakeSession(
            [
                StreamEvent(type="start", provider="stub", model=stub_model),
                StreamEvent(
                    type="done",
                    provider="stub",
                    model=stub_model,
                    assistantMessage=AssistantMessage(content="ok"),
                ),
            ]
        )

    loop = make_loop(
        stub_model,
        stream=flaky_stream,
        retryPolicy=lambda context: RetryDecision(shouldRetry=context.attempt == 1, delaySeconds=0.0),
    )
    collector = AgentTraceCollector(loop)
    loop.traceRecorder = collector
    session = await agentLoop(loop, initialMessages=[UserMessage(content="hello")])
    await collect_events(session)
    record = collector.finalize()

    assert record.retry_events
    assert record.retry_events[0].error_message == "temporary failure"
    assert record.outcome.status == "completed"


@pytest.mark.asyncio
async def test_agent_loop_trace_collector_records_before_tool_error(stub_model: Model) -> None:
    async def custom_stream(model, context, thinking, registry=None):
        _ = model, context, thinking, registry
        has_tool_result = any(isinstance(message, ToolResultMessage) for message in context.history)
        if has_tool_result:
            return FakeSession(
                [
                    StreamEvent(type="start", provider="stub", model=stub_model),
                    StreamEvent(
                        type="done",
                        provider="stub",
                        model=stub_model,
                        assistantMessage=AssistantMessage(content="done"),
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
                        content="checking",
                        toolCalls=[ToolCall(id="call_1", name="lookup", arguments='{\"q\":\"spec\"}')],
                    ),
                ),
            ]
        )

    loop = make_loop(
        stub_model,
        stream=custom_stream,
        beforeToolCall=lambda context: BeforeToolCallError(error="blocked"),
        tools=[AgentTool(name="lookup", description="demo", inputSchema={"type": "object"}, execute=lambda *args: "unused")],
    )
    collector = AgentTraceCollector(loop)
    loop.traceRecorder = collector
    session = await agentLoop(loop, initialMessages=[UserMessage(content="tool")])
    await collect_events(session)
    record = collector.finalize()

    assert any(event.type == "before_tool_result" and event.metadata["outcome"] == "error" for turn in record.turns for event in turn.events)


@pytest.mark.asyncio
async def test_agent_loop_continue_requires_history(stub_model: Model) -> None:
    with pytest.raises(ValueError, match="agentLoopContinue requires existing history"):
        await agentLoopContinue(AgentState(model=stub_model))


@pytest.mark.asyncio
async def test_agent_loop_continue_rejects_assistant_tail(stub_model: Model) -> None:
    state = AgentState(model=stub_model, history=[AssistantMessage(content="hello")])
    with pytest.raises(ValueError, match="agentLoopContinue cannot continue from an assistant message"):
        await agentLoopContinue(state)


@pytest.mark.asyncio
async def test_agent_loop_continue_preserves_history(stub_model: Model) -> None:
    registry = ProviderRegistry([ScriptedProvider()])

    async def custom_stream(model, context, thinking, registry=None):
        return await agent_loop_module.streamSimple(
            model,
            {"systemPrompt": context.systemPrompt, "messages": context.history, "tools": context.tools},
            reasoning=thinking,
            registry=registry,
        )

    state = AgentState(model=stub_model, history=[UserMessage(content="history")])
    session = await agentLoopContinue(
        state,
        loop=make_loop(stub_model, stream=custom_stream, registry=registry),
    )
    events = await collect_events(session)

    assert events[-1].state.history[0].content == "history"
    assert events[-1].state.history[-1].content == "echo:history"


@pytest.mark.asyncio
async def test_before_tool_call_allow_runs_tool_normally(stub_model: Model) -> None:
    registry = ProviderRegistry([ScriptedProvider()])

    async def custom_stream(model, context, thinking, registry=None):
        return await agent_loop_module.streamSimple(
            model,
            {"systemPrompt": context.systemPrompt, "messages": context.history, "tools": context.tools},
            reasoning=thinking,
            registry=registry,
        )

    async def lookup(arguments: str, context) -> str:
        return f"tool:{arguments}"

    session = await agentLoop(
        make_loop(
            stub_model,
            stream=custom_stream,
            registry=registry,
            tools=[AgentTool(name="lookup", description="Lookup", execute=lookup)],
            beforeToolCall=lambda context: BeforeToolCallAllow(),
        ),
        initialMessages=[UserMessage(content="tool")],
    )
    events = await collect_events(session)

    tool_end = next(event for event in events if event.type == "tool_execution_end")
    assert tool_end.toolResult.content == 'tool:{"q":"spec"}'


@pytest.mark.asyncio
async def test_before_tool_call_skip_replaces_real_execution(stub_model: Model) -> None:
    registry = ProviderRegistry([ScriptedProvider(tool_only=True)])
    called = {"tool": 0}

    async def custom_stream(model, context, thinking, registry=None):
        return await agent_loop_module.streamSimple(
            model,
            {"systemPrompt": context.systemPrompt, "messages": context.history, "tools": context.tools},
            reasoning=thinking,
            registry=registry,
        )

    async def lookup(arguments: str, context) -> str:
        called["tool"] += 1
        return "should-not-run"

    session = await agentLoop(
        make_loop(
            stub_model,
            stream=custom_stream,
            registry=registry,
            tools=[AgentTool(name="lookup", description="Lookup", execute=lookup)],
            beforeToolCall=lambda context: BeforeToolCallSkip(result={"content": "skipped-result"}),
        ),
        initialMessages=[UserMessage(content="tool")],
    )
    events = await collect_events(session)

    tool_end = next(event for event in events if event.type == "tool_execution_end")
    assert called["tool"] == 0
    assert tool_end.toolResult.content == "skipped-result"


@pytest.mark.asyncio
async def test_before_tool_call_error_blocks_execution(stub_model: Model) -> None:
    registry = ProviderRegistry([ScriptedProvider(tool_only=True)])

    async def custom_stream(model, context, thinking, registry=None):
        return await agent_loop_module.streamSimple(
            model,
            {"systemPrompt": context.systemPrompt, "messages": context.history, "tools": context.tools},
            reasoning=thinking,
            registry=registry,
        )

    async def lookup(arguments: str, context) -> str:
        raise AssertionError("tool should not run")

    session = await agentLoop(
        make_loop(
            stub_model,
            stream=custom_stream,
            registry=registry,
            tools=[AgentTool(name="lookup", description="Lookup", execute=lookup)],
            beforeToolCall=lambda context: BeforeToolCallError(error="blocked"),
        ),
        initialMessages=[UserMessage(content="tool")],
    )
    events = await collect_events(session)

    tool_end = next(event for event in events if event.type == "tool_execution_end")
    assert tool_end.toolResult.metadata["error"] is True
    assert tool_end.toolResult.content == "blocked"


@pytest.mark.asyncio
async def test_after_tool_call_can_replace_result(stub_model: Model) -> None:
    registry = ProviderRegistry([ScriptedProvider(tool_only=True)])

    async def custom_stream(model, context, thinking, registry=None):
        return await agent_loop_module.streamSimple(
            model,
            {"systemPrompt": context.systemPrompt, "messages": context.history, "tools": context.tools},
            reasoning=thinking,
            registry=registry,
        )

    async def lookup(arguments: str, context) -> str:
        return "original"

    session = await agentLoop(
        make_loop(
            stub_model,
            stream=custom_stream,
            registry=registry,
            tools=[AgentTool(name="lookup", description="Lookup", execute=lookup)],
            afterToolCall=lambda context: AfterToolCallReplace(result={"content": "replaced"}),
        ),
        initialMessages=[UserMessage(content="tool")],
    )
    events = await collect_events(session)

    tool_end = next(event for event in events if event.type == "tool_execution_end")
    assert tool_end.toolResult.content == "replaced"


@pytest.mark.asyncio
async def test_after_tool_call_pass_keeps_result(stub_model: Model) -> None:
    registry = ProviderRegistry([ScriptedProvider(tool_only=True)])

    async def custom_stream(model, context, thinking, registry=None):
        return await agent_loop_module.streamSimple(
            model,
            {"systemPrompt": context.systemPrompt, "messages": context.history, "tools": context.tools},
            reasoning=thinking,
            registry=registry,
        )

    async def lookup(arguments: str, context) -> str:
        return "original"

    session = await agentLoop(
        make_loop(
            stub_model,
            stream=custom_stream,
            registry=registry,
            tools=[AgentTool(name="lookup", description="Lookup", execute=lookup)],
            afterToolCall=lambda context: AfterToolCallPass(),
        ),
        initialMessages=[UserMessage(content="tool")],
    )
    events = await collect_events(session)

    tool_end = next(event for event in events if event.type == "tool_execution_end")
    assert tool_end.toolResult.content == "original"


@pytest.mark.asyncio
async def test_parallel_tool_execution_keeps_result_order(stub_model: Model) -> None:
    registry = ProviderRegistry([ScriptedProvider(multi_tool=True)])

    async def custom_stream(model, context, thinking, registry=None):
        return await agent_loop_module.streamSimple(
            model,
            {"systemPrompt": context.systemPrompt, "messages": context.history, "tools": context.tools},
            reasoning=thinking,
            registry=registry,
        )

    async def slow_lookup(arguments: str, context) -> str:
        await asyncio.sleep(0.02)
        return "slow-result"

    async def fast_lookup(arguments: str, context) -> str:
        await asyncio.sleep(0.001)
        return "fast-result"

    session = await agentLoop(
        make_loop(
            stub_model,
            stream=custom_stream,
            registry=registry,
            toolExecutionMode="parallel",
            tools=[
                AgentTool(name="slow_lookup", description="Slow", execute=slow_lookup),
                AgentTool(name="fast_lookup", description="Fast", execute=fast_lookup),
            ],
        ),
        initialMessages=[UserMessage(content="parallel")],
    )
    events = await collect_events(session)

    tool_results = [message for message in events[-1].state.history if isinstance(message, ToolResultMessage)]
    assert [result.toolName for result in tool_results] == ["slow_lookup", "fast_lookup"]
    assert [result.content for result in tool_results] == ["slow-result", "fast-result"]


@pytest.mark.asyncio
async def test_steer_runs_before_follow_up(stub_model: Model) -> None:
    registry = ProviderRegistry([ScriptedProvider()])
    calls: list[str] = []

    async def custom_stream(model, context, thinking, registry=None):
        return await agent_loop_module.streamSimple(
            model,
            {"systemPrompt": context.systemPrompt, "messages": context.history, "tools": context.tools},
            reasoning=thinking,
            registry=registry,
        )

    def steer(state: AgentState):
        calls.append("steer")
        if len([m for m in state.history if getattr(m, "role", "") == "assistant"]) == 1:
            return [UserMessage(content="steered")]
        return []

    def follow_up(state: AgentState):
        calls.append("followUp")
        return []

    session = await agentLoop(
        make_loop(
            stub_model,
            stream=custom_stream,
            registry=registry,
            steer=steer,
            followUp=follow_up,
        ),
        initialMessages=[UserMessage(content="hello")],
    )
    events = await collect_events(session)

    assert events[-1].state.history[-1].content == "echo:steered"
    assert calls[:2] == ["steer", "steer"]


@pytest.mark.asyncio
async def test_follow_up_runs_after_inner_loop_exits(stub_model: Model) -> None:
    registry = ProviderRegistry([ScriptedProvider()])
    follow_up_calls = 0

    async def custom_stream(model, context, thinking, registry=None):
        return await agent_loop_module.streamSimple(
            model,
            {"systemPrompt": context.systemPrompt, "messages": context.history, "tools": context.tools},
            reasoning=thinking,
            registry=registry,
        )

    def follow_up(state: AgentState):
        nonlocal follow_up_calls
        follow_up_calls += 1
        if len([m for m in state.history if getattr(m, "role", "") == "assistant"]) == 1:
            return [UserMessage(content="follow-up")]
        return []

    session = await agentLoop(
        make_loop(
            stub_model,
            stream=custom_stream,
            registry=registry,
            followUp=follow_up,
        ),
        initialMessages=[UserMessage(content="hello")],
    )
    events = await collect_events(session)

    assert [event.type for event in events].count("turn_start") == 2
    assert events[-1].state.history[-1].content == "echo:follow-up"
    assert follow_up_calls >= 1


@pytest.mark.asyncio
async def test_state_tracks_current_message_running_tool_and_error(stub_model: Model) -> None:
    registry = ProviderRegistry([ScriptedProvider(tool_only=True)])

    async def custom_stream(model, context, thinking, registry=None):
        return await agent_loop_module.streamSimple(
            model,
            {"systemPrompt": context.systemPrompt, "messages": context.history, "tools": context.tools},
            reasoning=thinking,
            registry=registry,
        )

    def before_tool_call(context):
        assert context.state.currentMessage is not None
        assert context.state.runningToolCall is None
        return BeforeToolCallError(error="danger")

    session = await agentLoop(
        make_loop(
            stub_model,
            stream=custom_stream,
            registry=registry,
            tools=[AgentTool(name="lookup", description="Lookup", execute=lambda a, c: "noop")],
            beforeToolCall=before_tool_call,
        ),
        initialMessages=[UserMessage(content="tool")],
    )
    events = await collect_events(session)

    tool_end = next(event for event in events if event.type == "tool_execution_end")
    assert tool_end.state.currentMessage is not None
    assert tool_end.error == "danger"
    assert events[-1].state.error == "danger"


@pytest.mark.asyncio
async def test_pure_tool_call_still_emits_message_start_and_message_end(stub_model: Model) -> None:
    registry = ProviderRegistry([ScriptedProvider(tool_only=True)])

    async def custom_stream(model, context, thinking, registry=None):
        return await agent_loop_module.streamSimple(
            model,
            {"systemPrompt": context.systemPrompt, "messages": context.history, "tools": context.tools},
            reasoning=thinking,
            registry=registry,
        )

    async def lookup(arguments: str, context) -> str:
        return "ok"

    session = await agentLoop(
        make_loop(
            stub_model,
            stream=custom_stream,
            registry=registry,
            tools=[AgentTool(name="lookup", description="Lookup", execute=lookup)],
        ),
        initialMessages=[UserMessage(content="tool")],
    )
    events = await collect_events(session)

    assert [event.type for event in events[:8]] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_update",
    ]
