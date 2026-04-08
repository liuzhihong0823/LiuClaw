from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ai.config import ProviderConfig
from ai.options import Options
from ai.types import StreamEvent


class Provider(ABC):
    """统一 provider 适配器抽象。"""

    name: str

    def __init__(self, *, config: ProviderConfig | None = None) -> None:
        """初始化 provider，并注入 provider 级配置。"""

        self.config = config

    def set_config(self, config: ProviderConfig | None) -> None:
        """更新 provider 级配置。"""

        self.config = config

    @abstractmethod
    def supports(self, model: Any) -> bool:
        """判断当前 provider 是否支持给定 `Model` 或模型 ID。"""
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        model: Any,
        context: Any,
        options: Options,
    ) -> AsyncIterator[StreamEvent]:
        """将统一 `Context` 映射到厂商请求，并输出统一事件流。"""
        raise NotImplementedError
