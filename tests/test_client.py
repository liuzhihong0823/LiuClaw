from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from ai import (
    AssistantMessage,
    Context,
    Model,
    StreamEvent,
    Tool,
    ToolCall,
    UserMessage,
    complete,
    completeSimple,
    stream,
    streamSimple,
)
from ai.errors import ProviderResponseError
from ai.options import Options
from ai.providers.base import Provider
from ai.registry import ProviderRegistry


class StubProvider(Provider):
    name = "stub"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.seen_context: Context | None = None
        self.seen_model: Model | None = None
        self.seen_options: Options | None = None

    def supports(self, model: Model) -> bool:
        return model.provider == self.name

    async def stream(
        self,
        model: Model,
        context: Context,
        options: Options,
    ) -> AsyncIterator[StreamEvent]:
        self.seen_model = model
        self.seen_context = context
        self.seen_options = options
        yield StreamEvent(type="start", provider=model.provider, model=model)
        if self.fail:
            yield StreamEvent(type="error", provider=model.provider, model=model, error="boom")
            return
        yield StreamEvent(type="text_start", provider=model.provider, model=model)
        yield StreamEvent(type="text_delta", provider=model.provider, model=model, text="hel")
        yield StreamEvent(type="text_delta", provider=model.provider, model=model, text="lo")
        yield StreamEvent(type="text_end", provider=model.provider, model=model, text="hello")
        yield StreamEvent(type="thinking_start", provider=model.provider, model=model)
        yield StreamEvent(
            type="thinking_delta",
            provider=model.provider,
            model=model,
            thinking="need lookup",
        )
        yield StreamEvent(type="thinking_end", provider=model.provider, model=model)
        yield StreamEvent(
            type="toolcall_start",
            provider=model.provider,
            model=model,
            toolCallId="call_1",
            toolName="lookup_spec",
        )
        yield StreamEvent(
            type="toolcall_delta",
            provider=model.provider,
            model=model,
            toolCallId="call_1",
            toolName="lookup_spec",
            argumentsDelta='{"query":',
        )
        yield StreamEvent(
            type="toolcall_delta",
            provider=model.provider,
            model=model,
            toolCallId="call_1",
            toolName="lookup_spec",
            argumentsDelta='"llm"}',
        )
        yield StreamEvent(
            type="toolcall_end",
            provider=model.provider,
            model=model,
            toolCallId="call_1",
            toolName="lookup_spec",
            arguments='{"query":"llm"}',
        )
        yield StreamEvent(
            type="done",
            provider=model.provider,
            model=model,
            assistantMessage=AssistantMessage(
                content="hello",
                thinking="need lookup",
                toolCalls=[
                    ToolCall(
                        id="call_1",
                        name="lookup_spec",
                        arguments='{"query":"llm"}',
                    )
                ],
            ),
        )


class RaisingProvider(Provider):
    name = "stub"

    def supports(self, model: Model) -> bool:
        return model.provider == self.name

    async def stream(
        self,
        model: Model,
        context: Context,
        options: Options,
    ) -> AsyncIterator[StreamEvent]:
        raise RuntimeError("provider crashed")
        yield StreamEvent(type="done", provider=model.provider, model=model)  # pragma: no cover


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


@pytest.fixture
def sample_context() -> Context:
    return Context(
        systemPrompt="你是测试助手。",
        messages=[UserMessage(content="Hi")],
        tools=[
            Tool(
                name="lookup_spec",
                description="查询规格说明",
                inputSchema={"type": "object"},
            )
        ],
    )


@pytest.mark.asyncio
async def test_stream_returns_session_with_queue(
    stub_model: Model,
    sample_context: Context,
) -> None:
    registry = ProviderRegistry([StubProvider()])

    session = await stream(stub_model, sample_context, registry=registry)

    assert hasattr(session, "queue")
    assert hasattr(session, "producer_task")
    assert hasattr(session, "consume")
    assert hasattr(session, "close")
    assert session.queue.maxsize > 0


@pytest.mark.asyncio
async def test_stream_queue_yields_done_event(
    stub_model: Model,
    sample_context: Context,
) -> None:
    registry = ProviderRegistry([StubProvider()])

    session = await stream(stub_model, sample_context, registry=registry)
    events = []
    async for event in session.consume():
        events.append(event)

    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    assert events[-1].assistantMessage is not None
    assert events[-1].assistantMessage.content == "hello"


@pytest.mark.asyncio
async def test_stream_queue_exposes_error_event(
    stub_model: Model,
    sample_context: Context,
) -> None:
    registry = ProviderRegistry([StubProvider(fail=True)])

    session = await stream(stub_model, sample_context, registry=registry)
    events = []
    async for event in session.consume():
        events.append(event)
        if event.type == "error":
            break

    assert events[-1].type == "error"
    assert events[-1].error == "boom"


@pytest.mark.asyncio
async def test_stream_simple_returns_bounded_queue_session(
    stub_model: Model,
    sample_context: Context,
) -> None:
    provider = StubProvider()
    registry = ProviderRegistry([provider])

    session = await streamSimple(
        stub_model,
        sample_context,
        reasoning="high",
        temperature=0.2,
        max_tokens=321,
        metadata={"trace_id": "abc"},
        timeout=5,
        registry=registry,
    )

    assert hasattr(session, "queue")
    assert session.queue.maxsize > 0
    assert provider.seen_model == stub_model
    assert provider.seen_context == sample_context
    assert provider.seen_options is not None
    assert provider.seen_options.reasoning == "high"
    assert provider.seen_options.temperature == 0.2
    assert provider.seen_options.maxTokens == 321
    assert provider.seen_options.metadata == {"trace_id": "abc"}
    assert provider.seen_options.timeout == 5
    assert not hasattr(provider.seen_options, "tools")


@pytest.mark.asyncio
async def test_stream_clamps_reasoning_to_model_capability(
    sample_context: Context,
) -> None:
    provider = StubProvider()
    registry = ProviderRegistry([provider])
    limited_model = Model(
        id="stub:test-model-limited",
        provider="stub",
        inputPrice=0.1,
        outputPrice=0.2,
        contextWindow=128000,
        maxOutputTokens=4096,
        supports_reasoning_levels=("off", "minimal", "low"),
    )

    session = await streamSimple(
        limited_model,
        sample_context,
        reasoning="xhigh",
        registry=registry,
    )
    async for _ in session.consume():
        pass

    assert provider.seen_options is not None
    assert provider.seen_options.reasoning == "low"
    assert provider.seen_options.metadata["_requestedReasoning"] == "xhigh"
    assert provider.seen_options.metadata["_clampedReasoning"] == "low"


@pytest.mark.asyncio
async def test_complete_aggregates_done_message_from_queue(
    stub_model: Model,
    sample_context: Context,
) -> None:
    registry = ProviderRegistry([StubProvider()])

    message = await complete(stub_model, sample_context, registry=registry)

    assert message.content == "hello"
    assert message.thinking == "need lookup"
    assert message.toolCalls[0].name == "lookup_spec"
    assert message.toolCalls[0].arguments == '{"query":"llm"}'


@pytest.mark.asyncio
async def test_complete_raises_on_error_event_from_queue(
    stub_model: Model,
    sample_context: Context,
) -> None:
    registry = ProviderRegistry([StubProvider(fail=True)])

    with pytest.raises(ProviderResponseError, match="boom"):
        await complete(stub_model, sample_context, registry=registry)


@pytest.mark.asyncio
async def test_complete_converts_provider_exception_to_error_event(
    stub_model: Model,
    sample_context: Context,
) -> None:
    registry = ProviderRegistry([RaisingProvider()])

    with pytest.raises(ProviderResponseError, match="provider crashed"):
        await complete(stub_model, sample_context, registry=registry)


@pytest.mark.asyncio
async def test_complete_simple_uses_done_result_from_queue(
    stub_model: Model,
    sample_context: Context,
) -> None:
    registry = ProviderRegistry([StubProvider()])

    message = await completeSimple(
        stub_model,
        sample_context,
        reasoning="low",
        registry=registry,
    )

    assert message.content == "hello"
    assert message.thinking == "need lookup"
