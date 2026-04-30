from __future__ import annotations

import asyncio
import shlex
from collections.abc import Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory

from ...core import AgentSession, ModelRegistry
from ...core.multi_agent import TeamRuntime
from ...core.types import SessionEvent
from .state import InteractiveState


class CommandCompleter(Completer):
    """为斜杠命令和部分参数提供补全。"""

    def __init__(self, controller: "InteractiveController") -> None:
        """初始化补全器。"""

        self.controller = controller  # 所属控制器。

    def get_completions(self, document, complete_event):
        """根据当前输入生成补全项。"""

        text = document.text_before_cursor
        commands = [
            "/new",
            "/resume",
            "/fork",
            "/branch",
            "/label",
            "/model",
            "/thinking",
            "/compact",
            "/theme",
            "/pwd",
            "/exit",
            "/sessions",
            "/help",
            "/clear",
            "/retry",
            "/bottom",
            "/top",
            "/team",
        ]
        if not text.startswith("/"):
            return
        parts = text.split()
        if len(parts) <= 1:
            for command in commands:
                if command.startswith(text):
                    yield Completion(command, start_position=-len(text))
            return
        command = parts[0]
        current = parts[-1]
        candidates: Iterable[str] = []
        if command == "/model":
            candidates = [model.id for model in self.controller.model_registry.list()]
        elif command == "/resume":
            recent = self.controller.session_manager.list_recent_sessions(limit=20, cwd=self.controller.session.cwd)
            candidates = [item["session_file"] for item in recent] + [item["session_id"] for item in recent]
        elif command == "/branch":
            if self.controller.session.session_file:
                self.controller.session_manager.set_session_file(self.controller.session.session_file)
                candidates = [entry.id for entry in self.controller.session_manager.get_entries()]
        elif command == "/theme":
            candidates = list(self.controller.session.resource_loader.load().themes.keys())
        elif command == "/thinking":
            candidates = ["low", "medium", "high"]
        elif command == "/team":
            candidates = ["inbox", "requests", "shutdown"]
        for candidate in candidates:
            if candidate.startswith(current):
                yield Completion(candidate, start_position=-len(current))


class InteractiveController:
    """负责连接输入、会话运行与界面状态。"""

    def __init__(self, session: AgentSession, model_registry: ModelRegistry, renderer, state: InteractiveState) -> None:
        """初始化控制器与状态引用。"""

        self.session = session  # 当前会话对象。
        self.model_registry = model_registry  # 模型注册表。
        self.session_manager = session.session_manager  # 会话管理器。
        self.renderer = renderer  # UI 渲染器。
        self.state = state  # 交互状态。
        self.history = InMemoryHistory()  # 输入历史。
        self.completer = CommandCompleter(self)  # 输入补全器。
        self._current_task: asyncio.Task[int] | None = None  # 当前正在执行的异步任务。
        if hasattr(self.renderer, "input_buffer"):
            self.renderer.input_buffer.completer = self.completer
            self.renderer.input_buffer.history = self.history

    def submit_current_buffer(self) -> None:
        """提交输入缓冲区中的内容。"""

        text = self.renderer.input_buffer.text.strip()
        if not text or self.state.is_running:
            return
        self.renderer.input_buffer.text = ""
        self._current_task = asyncio.create_task(self.handle_text(text))

    async def handle_text(self, text: str) -> int:
        """处理一条用户输入或斜杠命令。"""

        if text.startswith("/"):
            await self.handle_command(text)
            self.renderer.invalidate()
            return 0
        self.state.is_running = True
        self.state.last_error = ""
        self.state.add_status("发送用户消息")
        self.renderer.invalidate()
        try:
            self.session.send_user_message(text)
            self.state.start_user_turn(text, self.session.current_turn_id)
            if hasattr(self.renderer, "sync_transcript_content"):
                self.renderer.sync_transcript_content(self.state)
                self.renderer.reconcile_viewport_after_content_change()
            async for event in self.session.run_turn():
                visible_output = self.state.apply_event(event)
                if visible_output and hasattr(self.renderer, "sync_transcript_content"):
                    self.renderer.sync_transcript_content(self.state)
                    self.renderer.reconcile_viewport_after_content_change()
                self.renderer.invalidate()
                if visible_output and hasattr(self.renderer, "follow_output_if_needed"):
                    self.renderer.follow_output_if_needed()
        except Exception as exc:
            self.state.last_error = str(exc)
            visible_output = self.state.apply_event(
                SessionEvent(type="error", message=str(exc), error=str(exc), status_level="error")
            )
            if visible_output and hasattr(self.renderer, "sync_transcript_content"):
                self.renderer.sync_transcript_content(self.state)
                self.renderer.reconcile_viewport_after_content_change()
        finally:
            self.state.is_running = False
            self.state.sync_from_session(self.session)
            self.renderer.invalidate()
            if hasattr(self.renderer, "update_scroll_after_render"):
                self.renderer.update_scroll_after_render()
            if hasattr(self.renderer, "focus_input_if_idle"):
                self.renderer.focus_input_if_idle()
        return 0

    async def handle_command(self, text: str) -> None:
        """执行斜杠命令。"""

        parts = shlex.split(text)
        command = parts[0]
        args = parts[1:]
        if command == "/new":
            previous_team_runtime = getattr(self.session, "team_runtime", None)
            self.session = AgentSession(
                workspace_root=self.session.workspace_root,
                cwd=self.session.cwd,
                model=self.session.model,
                thinking=self.session.thinking,
                settings=self.session.settings,
                session_manager=self.session_manager,
                resource_loader=self.session.resource_loader,
                model_registry=self.session.model_registry,
                session_file=None,
            )
            self.session.attach_team_runtime(
                TeamRuntime(
                    owner_session=self.session,
                    workspace_root=self.session.workspace_root,
                    model_registry=self.model_registry,
                    shared_state=previous_team_runtime.shared_state if previous_team_runtime is not None else None,
                )
            )
            self.state.clear_output()
            if hasattr(self.renderer, "sync_transcript_content"):
                self.renderer.sync_transcript_content(self.state)
            self.state.add_status(f"新会话已创建: {self.session.session_id}")
        elif command == "/resume":
            session_ref = args[0] if args else self._default_resume_session_ref()
            if session_ref is None:
                self.state.last_error = "没有可恢复的会话"
                return
            self.session_manager.set_session_file(session_ref)
            self.session.session_file = str(self.session_manager.session_file or "")
            self.session.leaf_id = self.session_manager.get_leaf_id()
            self.session.resume_session()
            self.state.clear_output()
            if hasattr(self.renderer, "sync_transcript_content"):
                self.renderer.sync_transcript_content(self.state)
            self.state.add_status(f"已恢复会话 {self.session_manager.session_id}")
        elif command == "/fork":
            source = args[0] if args else self._default_resume_session_ref()
            if source is None:
                self.state.last_error = "没有可 fork 的会话"
                return
            source_file = self.session_manager.resolve_session_file(source)
            if source_file is None:
                self.state.last_error = f"未知会话: {source}"
                return
            self.session_manager.set_session_file(source_file)
            leaf_id = self.session_manager.get_leaf_id()
            if leaf_id is None:
                self.state.last_error = "源会话没有可 fork 的历史"
                return
            new_file = self.session_manager.create_branched_session(leaf_id)
            self.session_manager.set_session_file(new_file)
            self.session.session_file = str(self.session_manager.session_file or "")
            self.session.leaf_id = self.session_manager.get_leaf_id()
            self.session.resume_session()
            self.state.clear_output()
            if hasattr(self.renderer, "sync_transcript_content"):
                self.renderer.sync_transcript_content(self.state)
            self.state.add_status(f"已 fork 到新会话 {self.session_manager.session_id}")
        elif command == "/branch":
            if not args:
                self.show_help("用法: /branch <entry_id>")
                return
            if not self.session.session_file:
                self.state.last_error = "当前没有活动会话"
                return
            self.session_manager.set_session_file(self.session.session_file)
            self.session_manager.branch(args[0])
            self.session.leaf_id = self.session_manager.get_leaf_id()
            self.session.branch_id = self.session.leaf_id or "main"
            self.session.resume_session()
            self.state.clear_output()
            if hasattr(self.renderer, "sync_transcript_content"):
                self.renderer.sync_transcript_content(self.state)
            self.state.add_status(f"已切换到分支节点 {args[0]}")
        elif command == "/label":
            if len(args) < 2:
                self.show_help("用法: /label <entry_id> <name>")
                return
            if not self.session.session_file:
                self.state.last_error = "当前没有活动会话"
                return
            self.session_manager.set_session_file(self.session.session_file)
            self.session_manager.append_label_change(args[0], " ".join(args[1:]))
            self.state.add_status(f"已标记 {args[0]}")
        elif command == "/model":
            if not args:
                self.show_help("用法: /model <model_id>")
                return
            model = self.model_registry.get(args[0])
            self.session.switch_model(model)
            self.state.add_status(f"模型已切换为 {model.id}")
        elif command == "/thinking":
            if not args:
                self.show_help("用法: /thinking <low|medium|high>")
                return
            self.session.set_thinking(args[0])
            self.state.add_status(f"思考等级已切换为 {args[0]}")
        elif command == "/compact":
            result = await self.session.compact()
            self.state.add_status(f"已压缩 {result.compacted_count} 条历史消息")
        elif command == "/theme":
            if not args:
                self.show_help("用法: /theme <theme_name>")
                return
            theme_name = args[0]
            themes = self.session.resource_loader.load().themes
            if theme_name not in themes:
                raise ValueError(f"Unknown theme '{theme_name}'")
            self.session.settings.theme = theme_name
            self.state.theme = theme_name
            self.state.theme_styles = self.state._resolve_theme_styles(self.session)
            self.state.add_status(f"主题已切换为 {theme_name}")
            if hasattr(self.renderer, "refresh_style"):
                self.renderer.refresh_style()
        elif command == "/pwd":
            self.state.add_status(str(self.session.cwd))
        elif command == "/sessions":
            recent = self.session_manager.list_recent_sessions(limit=10)
            if not recent:
                self.state.add_status("暂无最近会话")
            for item in recent:
                self.state.add_status(
                    f"{item['session_id']} | {item.get('title', '')} | {item.get('model_id', '')} | {item.get('session_file', '')}"
                )
        elif command == "/help":
            self.show_help(
                "Commands: /new /resume [session] /fork [session] /branch <entry_id> /label <entry_id> <name> /model <id> /thinking <level> /compact /theme <name> /pwd /sessions /team /clear /retry /exit"
            )
        elif command == "/clear":
            self.clear_output()
        elif command == "/retry":
            last_message = self.session.get_last_user_message()
            if not last_message:
                self.state.add_status("没有可重试的用户消息")
            else:
                await self.handle_text(last_message)
        elif command == "/exit":
            if self.state.is_running:
                self.state.add_status("正在运行中，请先 Ctrl-C 取消再退出")
            else:
                if self.renderer.application is not None:
                    self.renderer.application.exit(result=0)
        elif command == "/bottom":
            self.jump_to_latest()
        elif command == "/top":
            self.jump_to_oldest()
        elif command == "/team":
            await self.handle_team_command(args)
        else:
            self.state.last_error = f"unknown command: {command}"
        self.state.sync_from_session(self.session)

    async def handle_team_command(self, args: list[str]) -> None:
        """处理团队协作相关命令。"""

        team_runtime = getattr(self.session, "team_runtime", None)
        if team_runtime is None:
            self.state.add_status("当前会话未启用 multi-agent 团队运行时")
            return
        if not args:
            members = team_runtime.list_members()
            if not members:
                self.state.add_status("当前团队暂无成员")
                return
            for member in members:
                status = member.status
                handle = team_runtime.shared_state.handles.get(member.name)
                if status == "idle" and handle is not None and handle.is_polling:
                    status = "idle(polling)"
                self.state.add_status(
                    f"{member.name} | role={member.role} | status={status} | session={member.session_id or '-'}"
                )
            return
        subcommand = args[0]
        if subcommand == "inbox":
            target = args[1] if len(args) > 1 else team_runtime.owner_name
            messages = team_runtime.peek_inbox(target)
            if not messages:
                self.state.add_status(f"{target} 收件箱为空")
                return
            for message in messages:
                self.state.add_status(
                    f"{target} <- {message.sender} | type={message.message_type} | request_id={message.request_id or '-'} | {message.content}"
                )
            return
        if subcommand == "requests":
            requests = team_runtime.list_protocol_requests()
            if not requests:
                self.state.add_status("当前没有协议请求")
                return
            for request in requests:
                self.state.add_status(
                    f"{request.request_id} | kind={request.kind} | status={request.status} | from={request.sender} | to={request.recipient}"
                )
            return
        if subcommand == "shutdown":
            if len(args) < 2:
                self.show_help("用法: /team shutdown <name>")
                return
            request = team_runtime.shutdown(args[1])
            self.state.add_status(f"已向 {args[1]} 发起关停请求: {request.request_id}")
            return
        self.show_help("用法: /team [inbox [name]|requests|shutdown <name>]")

    def cancel_current(self) -> None:
        """取消当前会话执行任务。"""

        if self._current_task is not None and not self._current_task.done():
            self.session.cancel()
            self._current_task.cancel()
            self.state.is_running = False
            self.state.add_status("已取消当前运行")
            self.renderer.invalidate()

    def clear_output(self) -> None:
        """清空输出区域但保留会话状态。"""

        self.state.clear_output()
        if hasattr(self.renderer, "sync_transcript"):
            self.renderer.sync_transcript_content(self.state)
        self.state.add_status("已清空输出面板")
        self.renderer.invalidate()

    def autocomplete_buffer(self) -> None:
        """触发输入缓冲区补全。"""

        if self.renderer.application is not None:
            self.renderer.application.current_buffer.start_completion(select_first=False)

    def show_help(self, message: str) -> None:
        """在状态栏中显示帮助说明。"""

        self.state.add_status(message)
        self.renderer.invalidate()

    def _default_resume_session_ref(self) -> str | None:
        """返回默认用于恢复的最近会话引用。"""

        recent = self.session.list_recent_sessions(limit=1)
        if not recent:
            return None
        return recent[0].get("session_file") or recent[0]["session_id"]

    def scroll_main_up(self) -> None:
        """向上滚动主输出区一行。"""

        self.renderer.scroll_main_lines(-1)

    def scroll_main_down(self) -> None:
        """向下滚动主输出区一行。"""

        self.renderer.scroll_main_lines(1)

    def scroll_main_page_up(self) -> None:
        """向上滚动主输出区一页。"""

        self.renderer.scroll_main_pages(-1)

    def scroll_main_page_down(self) -> None:
        """向下滚动主输出区一页。"""

        self.renderer.scroll_main_pages(1)

    def jump_to_latest(self) -> None:
        """跳转到最新消息并恢复自动跟随。"""

        self.renderer.scroll_main_to_bottom(mark_mode="jumped")
        self.state.add_status("已回到最新消息")
        self.renderer.invalidate()

    def jump_to_oldest(self) -> None:
        """跳转到最早消息并进入历史浏览模式。"""

        if hasattr(self.renderer, "scroll_main_to_top"):
            self.renderer.scroll_main_to_top()
        self.state.add_status("已跳到最早消息")
        self.renderer.invalidate()

    def toggle_focus(self) -> None:
        """在主输出区和输入区之间切换焦点。"""

        if self.renderer.focused_on_input():
            self.renderer.focus_main()
            self.state.add_status("焦点已切换到主输出区")
        else:
            self.renderer.focus_input()
            self.state.add_status("焦点已切换到输入区")
        self.renderer.invalidate()

    def focus_input(self) -> None:
        """把焦点直接切回输入区。"""

        self.renderer.focus_input()
        self.state.add_status("焦点已回到输入区")
        self.renderer.invalidate()
