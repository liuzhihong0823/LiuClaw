from __future__ import annotations

import asyncio

from ...core import ModelRegistry
from ...core.agent_session import AgentSession
from .controller import InteractiveController
from .renderer import InteractiveRenderer
from .state import InteractiveState


class InteractiveApp:
    """基于终端输入输出的交互式运行界面。"""

    def __init__(self, session: AgentSession, model_registry: ModelRegistry | None = None) -> None:
        """保存当前正在交互的会话对象。"""

        self.session = session  # 当前交互使用的会话对象。
        self.model_registry = model_registry  # 可选的模型注册表。

    async def run(self) -> int:
        """启动交互循环，优先使用 `prompt_toolkit`。"""

        try:
            from prompt_toolkit.application import Application
        except ImportError:
            return await self._fallback_loop()
        state = InteractiveState.from_session(self.session)
        renderer = InteractiveRenderer(state)
        controller = InteractiveController(self.session, self.model_registry, renderer, state)
        app = renderer.build_application(controller)
        return await app.run_async()

    async def _fallback_loop(self) -> int:
        """在缺少 `prompt_toolkit` 时退回到标准输入循环。"""

        state = InteractiveState.from_session(self.session)
        renderer = type("FallbackRenderer", (), {"input_buffer": type("B", (), {"text": ""})(), "invalidate": lambda self: None, "application": None})()
        controller = InteractiveController(self.session, self.model_registry, renderer, state)
        while True:
            text = await asyncio.to_thread(input, "> ")
            text = text.strip()
            if not text:
                continue
            await controller.handle_text(text)
            self.session = controller.session
            if text == "/exit" and not state.is_running:
                return 0

    @staticmethod
    def _render_event(event) -> None:
        """把会话事件渲染到终端。"""

        if event.type == "message_delta":
            print(event.delta, end="", flush=True)
        elif event.type == "thinking":
            print(f"\n[thinking]\n{event.message}")
        elif event.type == "status":
            print(f"\n[status] {event.message}")
        elif event.type == "message_end":
            print()
        elif event.type == "tool_start":
            print(f"\n[tool] {event.tool_name}")
        elif event.type == "tool_end":
            print(f"\n[tool:{event.tool_name}] {event.message}")
        elif event.type == "error":
            print(f"\n[error] {event.message}")
