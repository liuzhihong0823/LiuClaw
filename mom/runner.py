from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ai import Model
from agent_core import AfterToolCallPass, BeforeToolCallAllow
from coding_agent.config.paths import AgentPaths
from coding_agent.core import AgentSession, ResourceLoader, SessionManager
from coding_agent.core.runtime_assembly import build_session_context
from coding_agent.core.types import CodingAgentSettings, SessionEvent

from .context_sync import sync_channel_log_to_session
from .prompt import build_mom_system_prompt
from .store import MomStore
from .types import ChatAttachment, ChatContext, ChatInfo, ChatUser, MomRenderConfig, RunResult, SessionRef


def _read_memory(path: Path) -> str:
    """读取频道记忆文件内容。"""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _format_event_message(sender_name: str, text: str, attachments: list[ChatAttachment]) -> str:
    """把当前触发消息格式化成注入 Agent 的用户输入文本。"""
    body = f"[{sender_name}]: {text}".strip()
    if attachments:
        lines = ["", "<attachments>"]
        for item in attachments:
            lines.append(f"- {item.original_name}: {item.local_path}")
        lines.append("</attachments>")
        body = "\n".join([body, *lines])
    return body


class MomAgentSession(AgentSession):
    def __init__(
        self,
        *,
        platform_name: str,
        mom_store: MomStore,
        chat_id: str,
        chat_name: str | None,
        users: list[ChatUser],
        chats: list[ChatInfo],
        **kwargs,
    ) -> None:
        """扩展 AgentSession，为 mom 场景注入平台与频道上下文。"""
        self.platform_name = platform_name  # 平台显示名。
        self.mom_store = mom_store  # mom 存储层对象。
        self.chat_id = chat_id  # 当前频道 ID。
        self.chat_name = chat_name  # 当前频道名。
        self.chat_users = users  # 已知用户列表。
        self.chat_infos = chats  # 已知频道列表。
        super().__init__(**kwargs)

    def update_chat_directory(self, chat_name: str | None, users: list[ChatUser], chats: list[ChatInfo]) -> None:
        """更新频道上下文，并重建系统提示词。"""
        self.chat_name = chat_name
        self.chat_users = users
        self.chat_infos = chats
        self._agent.setSystemPrompt(self._build_system_prompt())

    def _build_system_prompt(self) -> str:
        """构造当前频道专用的系统提示词。"""
        context = build_session_context(
            workspace_root=self.workspace_root,
            cwd=self.cwd,
            model=self.model,
            thinking=self.thinking,
            settings=self.settings,
            resources=self.runtime.resources,
            tool_registry=self.runtime.tool_registry,
        )
        return build_mom_system_prompt(
            context,
            workspace_root=self.workspace_root,
            mom_root=self.mom_store.paths.root,
            chat_id=self.chat_id,
            chat_name=self.chat_name,
            platform_name=self.platform_name,
            users=self.chat_users,
            chats=self.chat_infos,
            channel_memory=_read_memory(self.mom_store.channel_memory_path(self.chat_id)),
        )

    async def _before_tool_call(self, context):  # type: ignore[override]
        """记录本轮发生过工具调用，允许工具继续执行。"""
        self._tool_activity_in_run = True
        _ = context
        return BeforeToolCallAllow()

    async def _after_tool_call(self, context):  # type: ignore[override]
        """记录工具调用结束事件，不额外打断会话流程。"""
        self._tool_activity_in_run = True
        _ = context
        return AfterToolCallPass()

    async def _follow_up(self, state, signal=None):  # type: ignore[override]
        """禁用默认 follow-up 逻辑，避免群聊机器人产生额外尾随消息。"""
        _ = state, signal
        return []


class MomRunner:
    def __init__(
        self,
        *,
        platform_name: str,
        chat_id: str,
        chat_dir: Path,
        store: MomStore,
        model: Model,
        settings: CodingAgentSettings,
        agent_paths: AgentPaths,
        session_ref: SessionRef,
        render_config: MomRenderConfig | None = None,
        stream_fn=None,
    ) -> None:
        """初始化频道级 runner，负责驱动 Agent 会话并把结果回写聊天平台。"""
        self.platform_name = platform_name  # 平台显示名。
        self.chat_id = chat_id  # 频道 ID。
        self.chat_dir = chat_dir  # 频道工作目录。
        self.store = store  # mom 存储层对象。
        self.model = model  # 当前模型。
        self.settings = settings  # 生效设置。
        self.agent_paths = agent_paths  # agent 级路径集合。
        self.render_config = render_config or MomRenderConfig()  # mom 输出渲染配置。
        self.session_manager = SessionManager(store.paths.sessions_dir)  # 会话管理器。
        self.session_ref = session_ref  # 频道与会话的持久化引用。
        self.resource_loader = ResourceLoader(
            skills_dir=agent_paths.skills_dir,
            prompts_dir=agent_paths.prompts_dir,
            themes_dir=agent_paths.themes_dir,
            extensions_dir=agent_paths.extensions_dir,
            workspace_root=chat_dir,
        )  # 频道上下文下的资源加载器。
        self.session = MomAgentSession(
            platform_name=platform_name,
            mom_store=store,
            chat_id=chat_id,
            chat_name=None,
            users=[],
            chats=[],
            workspace_root=chat_dir,
            cwd=chat_dir,
            model=model,
            thinking=settings.default_thinking,
            settings=settings,
            session_manager=self.session_manager,
            resource_loader=self.resource_loader,
            session_id=session_ref.session_id,
            branch_id=session_ref.branch_id,
            stream_fn=stream_fn,
        )  # 复用的频道 AgentSession。

    def abort(self) -> None:
        """中断当前 Agent 执行。"""
        self.session.cancel()

    async def run(self, ctx: ChatContext, store: MomStore) -> RunResult:
        """执行一次聊天请求，包括同步历史、运行 Agent 和输出结果。"""
        self.session.update_chat_directory(ctx.chat_name, ctx.users, ctx.chats)
        sync_channel_log_to_session(self.session_manager, self.session_ref, self.chat_dir, exclude_message_id=ctx.message.message_id)
        store.save_session_ref(self.chat_id, self.session_ref)
        self.session.resume_session()
        self.session.send_user_message(_format_event_message(ctx.message.sender_name, ctx.message.text, ctx.message.attachments))

        main_text = ""
        main_message_id: str | None = None
        final_text = ""
        detail_count = 0
        suppressed_events_count = 0
        try:
            await ctx.set_working(True)
            if self.render_config.placeholder_text:
                main_message_id = await ctx.respond(self.render_config.placeholder_text, False)
            async for event in self.session.run_turn():
                main_text, main_message_id, final_text, detail_count, suppressed_events_count = await self._handle_event(
                    ctx,
                    event,
                    main_text,
                    main_message_id,
                    final_text,
                    detail_count,
                    suppressed_events_count,
                )
            await ctx.set_working(False)
        except asyncio.CancelledError:
            await ctx.set_working(False)
            return RunResult(
                stop_reason="aborted",
                main_message_id=main_message_id,
                final_text=final_text,
                detail_count=detail_count,
                suppressed_events_count=suppressed_events_count,
            )
        except Exception as exc:
            await ctx.set_working(False)
            message_id = await ctx.respond_detail(f"运行失败: {exc}")
            if message_id:
                store.log_bot_message(self.chat_id, message_id=message_id, text=f"运行失败: {exc}", response_kind="detail")
                detail_count += 1
            return RunResult(
                stop_reason="error",
                error_message=str(exc),
                main_message_id=main_message_id,
                final_text=final_text,
                detail_count=detail_count,
                suppressed_events_count=suppressed_events_count,
            )
        return RunResult(
            stop_reason="completed",
            main_message_id=main_message_id,
            final_text=final_text,
            detail_count=detail_count,
            suppressed_events_count=suppressed_events_count,
        )

    async def _handle_event(
        self,
        ctx: ChatContext,
        event: SessionEvent,
        main_text: str,
        main_message_id: str | None,
        final_text: str,
        detail_count: int,
        suppressed_events_count: int,
    ) -> tuple[str, str | None, str, int, int]:
        """处理 SessionEvent，并根据渲染配置决定是否对外展示。"""
        if event.type == "message_delta":
            main_text += event.delta
            if self.render_config.render_mode == "streaming" and self.render_config.show_intermediate_updates:
                updated_id = await ctx.replace_message(main_text)
                return main_text, updated_id or main_message_id, final_text, detail_count, suppressed_events_count
            return main_text, main_message_id, final_text, detail_count, suppressed_events_count + 1
        if event.type == "thinking":
            if self.render_config.show_thinking and event.message.strip():
                detail_id = await ctx.respond_detail(event.message)
                if detail_id:
                    self.store.log_bot_message(self.chat_id, message_id=detail_id, text=event.message, response_kind="detail")
                    detail_count += 1
                return main_text, main_message_id, final_text, detail_count, suppressed_events_count
            return main_text, main_message_id, final_text, detail_count, suppressed_events_count + 1
        if event.type in {"tool_start", "tool_end", "tool_update", "status"}:
            if event.type == "tool_end" and self.render_config.show_tool_details:
                detail = "\n".join(
                    [
                        f"工具: {event.tool_name}",
                        f"参数: {event.tool_arguments or '{}'}",
                        "结果:",
                        str(event.message),
                    ]
                )
                detail_id = await ctx.respond_detail(detail)
                if detail_id:
                    self.store.log_bot_message(self.chat_id, message_id=detail_id, text=detail, response_kind="detail")
                    detail_count += 1
                return main_text, main_message_id, final_text, detail_count, suppressed_events_count
            return main_text, main_message_id, final_text, detail_count, suppressed_events_count + 1
        if event.type == "message_end":
            text = event.message.strip()
            if text == "[SILENT]":
                await ctx.delete_message()
                return "", main_message_id, "", detail_count, suppressed_events_count
            final_id = await ctx.replace_message(text)
            if final_id:
                self.store.log_bot_message(self.chat_id, message_id=final_id, text=text, response_kind="message")
            return text, final_id or main_message_id, text, detail_count, suppressed_events_count
        if event.type == "error":
            text = event.message or event.error or "未知错误"
            error_id = await ctx.replace_message(text) if main_message_id else await ctx.respond(text, False)
            if error_id:
                self.store.log_bot_message(
                    self.chat_id,
                    message_id=error_id,
                    text=text,
                    response_kind="message",
                    metadata={"error": True},
                )
            return main_text, error_id or main_message_id, final_text, detail_count, suppressed_events_count
        return main_text, main_message_id, final_text, detail_count, suppressed_events_count


_RUNNERS: dict[str, MomRunner] = {}


def get_or_create_runner(
    chat_id: str,
    *,
    platform_name: str,
    chat_dir: Path,
    store: MomStore,
    model: Model,
    settings: CodingAgentSettings,
    agent_paths: AgentPaths,
    session_ref: SessionRef,
    render_config: MomRenderConfig | None = None,
    stream_fn=None,
) -> MomRunner:
    """按频道复用 runner，避免同一频道重复创建会话执行器。"""
    runner = _RUNNERS.get(chat_id)
    if runner is not None:
        runner.session_ref = session_ref
        if render_config is not None:
            runner.render_config = render_config
        return runner
    runner = MomRunner(
        platform_name=platform_name,
        chat_id=chat_id,
        chat_dir=chat_dir,
        store=store,
        model=model,
        settings=settings,
        agent_paths=agent_paths,
        session_ref=session_ref,
        render_config=render_config,
        stream_fn=stream_fn,
    )
    _RUNNERS[chat_id] = runner
    return runner
