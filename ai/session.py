from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Callable, Generic, TypeVar

from .types import Model, StreamEvent

TEvent = TypeVar("TEvent")


class StreamSession(Generic[TEvent]):
    """表示一次流式调用的队列会话。"""

    def __init__(
        self,
        *,
        model: Model,
        queue: asyncio.Queue[TEvent],
        producer_task: asyncio.Task[None],
        should_stop: Callable[[TEvent], bool] | None = None,
    ) -> None:
        """初始化会话并保存模型、队列与生产者任务。"""

        self.model = model  # 当前流式会话绑定的模型。
        self.queue = queue  # 生产者写入、消费者读取的事件队列。
        self.producer_task = producer_task  # 后台生产事件的任务。
        self._should_stop = should_stop or self._default_should_stop  # 判断流何时结束的回调。

    async def consume(self) -> AsyncIterator[TEvent]:
        """持续从队列中取事件，并在 `done/error` 后结束。"""

        while True:
            event = await self.queue.get()
            yield event
            if self._should_stop(event):
                await self._wait_producer()
                break

    async def close(self) -> None:
        """取消生产者任务并等待其结束。"""

        if self.producer_task.done():
            await self._wait_producer()
            return
        self.producer_task.cancel()
        await self._wait_producer()

    async def wait_closed(self) -> None:
        """等待生产者任务自然结束。"""

        await self._wait_producer()

    async def _wait_producer(self) -> None:
        """等待生产者任务结束并吞掉取消异常。"""

        try:
            await self.producer_task
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _default_should_stop(event: TEvent) -> bool:
        """为默认流事件协议提供结束判断。"""

        event_type = getattr(event, "type", None)
        return event_type in {"done", "error"}
