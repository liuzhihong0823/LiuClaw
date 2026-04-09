from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from coding_agent.config.paths import build_agent_paths
from coding_agent.core import ModelRegistry, SettingsManager

from .events import EventsWatcher
from .feishu import FeishuBotTransport, FeishuConfig
from .runner import get_or_create_runner
from .store import MomStore
from .types import ChannelState, ChatEvent, MomRenderConfig


@dataclass(slots=True)
class MomConfig:
    """mom 应用配置，负责承载工作区与飞书接入参数。"""

    workspace_root: Path  # mom 工作区根目录。
    feishu: FeishuConfig  # 飞书接入配置。
    model_id: str | None = None  # 指定使用的模型 ID，可为空。

    @classmethod
    def from_env(cls) -> "MomConfig":
        """从环境变量读取 mom 启动配置。"""
        workspace_root = Path(os.environ["MOM_WORKDIR"]).resolve()
        return cls(
            workspace_root=workspace_root,
            feishu=FeishuConfig(
                app_id=os.environ.get("MOM_FEISHU_APP_ID", ""),
                app_secret=os.environ.get("MOM_FEISHU_APP_SECRET", ""),
                connection_mode=os.environ.get("MOM_FEISHU_CONNECTION_MODE", "long_connection"),
                verification_token=os.environ.get("MOM_FEISHU_VERIFICATION_TOKEN", ""),
                encrypt_key=os.environ.get("MOM_FEISHU_ENCRYPT_KEY", ""),
                bind_host=os.environ.get("MOM_BIND_HOST", "127.0.0.1"),
                bind_port=int(os.environ.get("MOM_BIND_PORT", "8123")),
            ),
            model_id=os.environ.get("MOM_MODEL"),
        )


class MomApp:
    def __init__(self, config: MomConfig, *, transport: FeishuBotTransport | None = None, stream_fn=None) -> None:
        """初始化 mom 主应用，装配存储、模型、传输层和事件监听器。"""
        self.config = config
        self.stream_fn = stream_fn
        self.store = MomStore(config.workspace_root)
        self.agent_paths = build_agent_paths()
        self.agent_paths.ensure_exists()
        self.settings = SettingsManager(self.agent_paths.settings_file, config.workspace_root / ".LiuClaw" / "settings.json").load()
        self.model = ModelRegistry(self.agent_paths.models_file).get(config.model_id or self.settings.default_model)
        self.transport = transport or FeishuBotTransport(self, config.feishu)
        self.channel_states: dict[str, ChannelState] = {}
        self.events = EventsWatcher(self.store.paths.events_dir, self.handle_chat_event)
        self.render_config = self._load_render_config()

    def _load_render_config(self) -> MomRenderConfig:
        """读取渲染配置，优先使用 mom 本地设置，其次读取环境变量。"""
        settings = self.store.load_settings()
        return MomRenderConfig(
            render_mode=str(settings.get("render_mode") or os.environ.get("MOM_RENDER_MODE") or "final_only"),
            placeholder_text=str(
                settings.get("placeholder_text") or os.environ.get("MOM_PLACEHOLDER_TEXT") or "处理中…"
            ),
            show_intermediate_updates=self._read_bool(
                os.environ.get("MOM_SHOW_INTERMEDIATE_UPDATES"),
                settings.get("show_intermediate_updates", False),
            ),
            show_tool_details=self._read_bool(
                os.environ.get("MOM_SHOW_TOOL_DETAILS"),
                settings.get("show_tool_details", False),
            ),
            show_thinking=self._read_bool(
                os.environ.get("MOM_SHOW_THINKING"),
                settings.get("show_thinking", False),
            ),
        )

    @staticmethod
    def _read_bool(raw: str | None, default: bool) -> bool:
        """把字符串型开关值解析为布尔值。"""
        if raw is None:
            return bool(default)
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def get_state(self, chat_id: str) -> ChannelState:
        """获取频道运行态；如果不存在则为该频道创建一份状态对象。"""
        state = self.channel_states.get(chat_id)
        if state is None:
            state = ChannelState(store=self.store)
            self.channel_states[chat_id] = state
        return state

    @staticmethod
    def _should_dedupe(event: ChatEvent) -> bool:
        """判断当前事件是否需要做去重处理。"""
        return not bool(event.metadata.get("synthetic"))

    @staticmethod
    def _remember_message_id(state: ChannelState, message_id: str) -> None:
        """记录最近处理过的消息 ID，避免重复消费飞书事件。"""
        if not message_id:
            return
        state.recent_incoming_message_ids = [
            item for item in state.recent_incoming_message_ids if item != message_id
        ]
        state.recent_incoming_message_ids.append(message_id)
        if len(state.recent_incoming_message_ids) > 200:
            state.recent_incoming_message_ids = state.recent_incoming_message_ids[-200:]

    async def handle_chat_event(self, event: ChatEvent) -> None:
        """处理单条聊天事件，包括去重、停止指令和串行调度控制。"""
        state = self.get_state(event.chat_id)
        if self._should_dedupe(event) and event.message_id in state.recent_incoming_message_ids:
            return
        if self._should_dedupe(event):
            self._remember_message_id(state, event.message_id)
        ctx = self.transport.create_context(event, self.store)

        if event.text.strip().lower() == "stop":
            if state.running and state.runner is not None:
                state.stop_requested = True
                state.runner.abort()
                state.stop_message_id = await ctx.respond("正在停止当前任务…", False)
            else:
                await ctx.respond("当前没有正在运行的任务。", False)
            return

        if not event.is_trigger:
            return

        synthetic = bool(event.metadata.get("synthetic"))
        if state.running and not synthetic:
            await ctx.respond("正在处理当前频道任务，可发送 stop 中断。", False)
            return
        if state.running and synthetic:
            state.queued_events.append(event)
            return
        await self._run_event(event, state)

    async def _run_event(self, event: ChatEvent, state: ChannelState) -> None:
        """真正执行一次频道任务，并在结束后继续消费排队事件。"""
        state.running = True
        state.stop_requested = False
        ctx = self.transport.create_context(event, self.store)
        try:
            session_ref = self.store.get_or_create_session_ref(event.chat_id, self.store.sessions_manager(), self.model.id)
            runner = get_or_create_runner(
                event.chat_id,
                platform_name="飞书",
                chat_dir=self.store.channel_dir(event.chat_id),
                store=self.store,
                model=self.model,
                settings=self.settings,
                agent_paths=self.agent_paths,
                session_ref=session_ref,
                render_config=self.render_config,
                stream_fn=self.stream_fn,
            )
            state.runner = runner
            try:
                result = await runner.run(ctx, self.store)
            except Exception as exc:
                await ctx.respond_detail(f"错误: {exc}")
                return
            if result.stop_reason == "aborted" and state.stop_requested:
                await ctx.respond("已停止。", False)
            elif result.stop_reason == "error" and result.error_message:
                await ctx.respond_detail(f"错误: {result.error_message}")
        finally:
            state.running = False

        if state.queued_events:
            next_event = state.queued_events.pop(0)
            await self._run_event(next_event, state)

    async def run(self) -> None:
        """启动事件监听并进入飞书服务主循环。"""
        self.events.start()
        await self.transport.serve(self.store)


def main() -> int:
    """命令行入口：构造应用并启动事件循环。"""
    config = MomConfig.from_env()
    asyncio.run(MomApp(config).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
