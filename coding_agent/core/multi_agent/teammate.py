from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .team_runtime import TeamRuntime


@dataclass(slots=True)
class TeammateHandle:
    """保存单个 worker 的常驻后台运行句柄。"""

    name: str  # 成员名称。
    role: str  # 成员角色。
    session: "AgentSession"  # 成员使用的独立会话。
    runtime: "TeamRuntime"  # 当前成员绑定的团队运行时。
    worker_task: asyncio.Task[None] | None = None  # 常驻 worker 主循环。
    idle_task: asyncio.Task[str | None] | None = None  # 当前 idle 阶段的轮询任务。
    pending_shutdown: bool = False  # 是否已收到并批准关停请求。
    last_error: str = ""  # 最近一次后台运行错误。
    is_polling: bool = False  # 当前是否处于 idle 轮询阶段。

    @property
    def is_running(self) -> bool:
        """返回常驻 worker 主循环是否仍在运行。"""

        return self.worker_task is not None and not self.worker_task.done()

    def start_worker(self, prompt: str) -> asyncio.Task[None]:
        """启动常驻 worker 主循环。"""

        if self.is_running:
            raise RuntimeError(f"Teammate '{self.name}' is already running")
        self.pending_shutdown = False
        self.worker_task = asyncio.create_task(self._worker_loop(prompt))
        return self.worker_task

    def request_shutdown_after_current_turn(self) -> None:
        """标记 worker 在当前轮完成后进入 shutdown。"""

        self.pending_shutdown = True
        if self.idle_task is not None and not self.idle_task.done():
            self.idle_task.cancel()

    async def _worker_loop(self, prompt: str) -> None:
        """在 WORK / IDLE 两阶段之间循环切换。"""

        next_prompt: str | None = prompt
        while next_prompt is not None:
            completed = await self._work_once(next_prompt)
            if not completed:
                return
            if self.pending_shutdown:
                self.runtime._set_member_status(self.name, "shutdown")
                self.is_polling = False
                return
            self.runtime._set_member_status(self.name, "idle")
            next_prompt = await self._idle_poll_loop()

    async def _work_once(self, prompt: str) -> bool:
        """执行一轮 worker 工作。"""

        self.runtime._set_member_status(self.name, "working")
        self.is_polling = False
        self.session.send_user_message(prompt)
        try:
            async for _event in self.session.run_turn():
                # 这里主动消费事件，但不把 worker 的全部中间态灌给主界面。
                continue
        except Exception as exc:
            self.last_error = str(exc)
            self.runtime._set_member_error(self.name, self.last_error)
            self.runtime._set_member_status(self.name, "idle")
            self.is_polling = False
            return False
        return True

    async def _idle_poll_loop(self) -> str | None:
        """在 idle 状态下轮询 inbox，直到拿到新消息或收到关停请求。"""

        self.is_polling = True
        while True:
            self.runtime._mark_member_polled(self.name)
            self.idle_task = asyncio.create_task(self._wait_for_inbox_prompt())
            try:
                prompt = await self.idle_task
            except asyncio.CancelledError:
                self.is_polling = False
                if self.pending_shutdown:
                    return None
                raise
            finally:
                self.idle_task = None
            if prompt is not None:
                self.is_polling = False
                return prompt

    async def _wait_for_inbox_prompt(self) -> str | None:
        """等待直到收件箱出现新消息。"""

        while True:
            messages = self.runtime.drain_inbox_for_worker(self.name)
            if messages:
                return self.runtime.format_idle_resume_prompt(self.name, self.role, messages)
            if self.pending_shutdown:
                return None
            await asyncio.sleep(self.runtime.idle_poll_interval)


if TYPE_CHECKING:
    from ..agent_session import AgentSession
