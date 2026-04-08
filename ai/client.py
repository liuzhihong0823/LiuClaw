from __future__ import annotations

import asyncio
from typing import Any

from .converters import convert_context_for_provider
from .errors import ProviderResponseError
from .options import Options, ensure_options
from .model_registry import DEFAULT_MODEL_REGISTRY, ModelRegistry
from .reasoning import merge_reasoning_metadata
from .registry import ProviderRegistry
from .session import StreamSession
from .types import AssistantMessage, Context, Model, StreamEvent, ensure_context, ensure_model
from .utils.context_window import ensure_context_fits_window, truncate_context_to_window
from .utils.streaming import StreamAccumulator
from .utils.unicode import sanitize_unicode_context

DEFAULT_REGISTRY = ProviderRegistry()



def _prepare_context(
    model: Model,
    context: Context | dict[str, Any],
    options: Options,
) -> Context:
    """规范化、清理并转换上下文，使其适配当前目标 provider。"""

    normalized = ensure_context(context)
    sanitized = sanitize_unicode_context(normalized)
    converted = convert_context_for_provider(model, sanitized)
    if options.contextOverflowStrategy == "truncate_oldest":
        return truncate_context_to_window(model, converted, options)
    return converted



def _prepare_options(model: Model, options: Options | None) -> Options:
    """规范化 options，并附加 provider 专用 reasoning 元数据。"""

    resolved = ensure_options(options)
    clamped_reasoning = model.clamp_reasoning(resolved.reasoning)
    metadata = merge_reasoning_metadata(resolved.metadata, model, clamped_reasoning)
    if clamped_reasoning != resolved.reasoning:
        metadata["_requestedReasoning"] = resolved.reasoning
        metadata["_clampedReasoning"] = clamped_reasoning
    return Options(
        reasoning=clamped_reasoning,
        temperature=resolved.temperature,
        maxTokens=resolved.maxTokens,
        metadata=metadata,
        timeout=resolved.timeout,
        includeRawProviderEvents=resolved.includeRawProviderEvents,
        streamQueueMaxSize=resolved.streamQueueMaxSize,
        streamPutTimeout=resolved.streamPutTimeout,
        contextOverflowStrategy=resolved.contextOverflowStrategy,
        debug=dict(resolved.debug),
        provider=dict(resolved.provider),
    )


async def _put_event(
    queue: asyncio.Queue[StreamEvent],
    event: StreamEvent,
    *,
    put_timeout: float | None,
) -> None:
    """将事件写入队列，并在配置了超时时间时施加等待上限。"""

    if put_timeout is None:
        await queue.put(event)
        return
    await asyncio.wait_for(queue.put(event), timeout=put_timeout)


async def _produce_events(
    *,
    model: Model,
    context: Context,
    options: Options,
    provider_registry: ProviderRegistry,
    queue: asyncio.Queue[StreamEvent],
) -> None:
    """桥接 provider 的 async iterator，并把事件转发到队列中。"""

    provider = provider_registry.resolve(model)
    try:
        async for event in provider.stream(model, context, options):
            await _put_event(queue, event, put_timeout=options.streamPutTimeout)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error_event = StreamEvent(
            type="error",
            lifecycle="error",
            itemType="message",
            model=model,
            provider=getattr(model, "provider", None),
            error=str(exc),
            details={"source": "provider", "exception_type": type(exc).__name__},
            metadata={"debugOnlyRawEvents": options.includeRawProviderEvents},
        )
        await _put_event(queue, error_event, put_timeout=options.streamPutTimeout)


async def stream(
    model: Model | str,
    context: Context | dict[str, Any],
    options: Options | None = None,
    *,
    registry: ProviderRegistry | None = None,
    model_registry: ModelRegistry | None = None,
) -> StreamSession:
    """创建流式队列会话，并在后台生产统一事件。"""

    if isinstance(model, str):
        normalized_model = (model_registry or DEFAULT_MODEL_REGISTRY).get_model(model)
    else:
        normalized_model = ensure_model(model)
    prepared_options = _prepare_options(normalized_model, options)
    prepared_context = _prepare_context(normalized_model, context, prepared_options)
    ensure_context_fits_window(normalized_model, prepared_context, prepared_options)

    queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=prepared_options.streamQueueMaxSize)
    provider_registry = registry or DEFAULT_REGISTRY
    producer_task = asyncio.create_task(
        _produce_events(
            model=normalized_model,
            context=prepared_context,
            options=prepared_options,
            provider_registry=provider_registry,
            queue=queue,
        )
    )
    await asyncio.sleep(0)
    return StreamSession(model=normalized_model, queue=queue, producer_task=producer_task)


async def complete(
    model: Model | str,
    context: Context | dict[str, Any],
    options: Options | None = None,
    *,
    registry: ProviderRegistry | None = None,
    model_registry: ModelRegistry | None = None,
) -> AssistantMessage:
    """消费流式队列会话，并聚合出最终 `AssistantMessage`。"""

    session = await stream(model, context, options, registry=registry, model_registry=model_registry)
    accumulator = StreamAccumulator()
    try:
        async for event in session.consume():
            if event.type == "error":
                raise ProviderResponseError(event.error or "provider emitted error event")
            final_message = accumulator.apply(event)
            if final_message is not None:
                return final_message
        return accumulator.assistant_message
    finally:
        await session.close()


async def streamSimple(
    model: Model | str,
    context: Context | dict[str, Any],
    *,
    reasoning: Any = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
    include_raw_provider_events: bool = False,
    stream_queue_max_size: int = 64,
    stream_put_timeout: float | None = None,
    registry: ProviderRegistry | None = None,
    model_registry: ModelRegistry | None = None,
) -> StreamSession:
    """兼容入口：用简化参数构造 `Options`，再返回流式队列会话。"""

    options = Options(
        reasoning=reasoning,
        temperature=temperature,
        maxTokens=max_tokens,
        metadata=dict(metadata or {}),
        timeout=timeout,
        includeRawProviderEvents=include_raw_provider_events,
        streamQueueMaxSize=stream_queue_max_size,
        streamPutTimeout=stream_put_timeout,
    )
    return await stream(model, context, options, registry=registry, model_registry=model_registry)


async def completeSimple(
    model: Model | str,
    context: Context | dict[str, Any],
    *,
    reasoning: Any = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
    include_raw_provider_events: bool = False,
    stream_queue_max_size: int = 64,
    stream_put_timeout: float | None = None,
    registry: ProviderRegistry | None = None,
    model_registry: ModelRegistry | None = None,
) -> AssistantMessage:
    """兼容入口：用简化参数构造 `Options`，再消费队列会话得到最终回复。"""

    options = Options(
        reasoning=reasoning,
        temperature=temperature,
        maxTokens=max_tokens,
        metadata=dict(metadata or {}),
        timeout=timeout,
        includeRawProviderEvents=include_raw_provider_events,
        streamQueueMaxSize=stream_queue_max_size,
        streamPutTimeout=stream_put_timeout,
    )
    return await complete(model, context, options, registry=registry, model_registry=model_registry)
