from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ...core.agent_session import AgentSession
from ...core.types import SessionEvent
from ...core.themes_loader import DEFAULT_THEME_DATA


@dataclass(slots=True)
class MessageCard:
    """表示兼容旧状态结构保留的一张消息卡片。"""

    id: str  # 卡片 ID。
    title: str  # 卡片标题。
    body: str = ""  # 卡片正文。
    style: str = "assistant"  # 卡片样式类型。


@dataclass(slots=True)
class ToolCard:
    """表示兼容旧状态结构保留的一张工具卡片。"""

    id: str  # 卡片 ID。
    tool_name: str  # 工具名。
    status: str  # 工具状态。
    arguments: str = ""  # 工具参数摘要。
    output_preview: str = ""  # 工具输出摘要。


@dataclass(slots=True)
class TranscriptBlock:
    """表示 transcript 中的一个稳定块。"""

    id: str  # 块唯一 ID。
    kind: Literal["user", "assistant", "thinking", "tool", "status", "error"]  # 块类型。
    title: str  # 块标题。
    body: str = ""  # 块正文。
    status: str = ""  # 工具或状态块的状态值。
    tool_name: str = ""  # 关联工具名。
    source: str = ""  # 块来源。
    render_order: int = 0  # 渲染排序权重。
    collapsed: bool = False  # 是否折叠显示。


@dataclass(slots=True)
class TranscriptTurn:
    """表示一次用户问题及其后续完整回答流程。"""

    turn_id: str  # turn 唯一 ID。
    user_prompt_preview: str  # 用户输入摘要。
    user_block: TranscriptBlock  # 用户输入块。
    thinking_blocks: list[TranscriptBlock] = field(default_factory=list)  # thinking 块列表。
    tool_blocks: list[TranscriptBlock] = field(default_factory=list)  # 工具块列表。
    status_blocks: list[TranscriptBlock] = field(default_factory=list)  # 状态块列表。
    assistant_block: TranscriptBlock | None = None  # assistant 主回复块。
    started_at: str = ""  # turn 开始时间。
    completed: bool = False  # turn 是否已完成。


@dataclass(slots=True)
class InteractiveState:
    """保存交互界面的全部可见状态。"""

    session_id: str  # 当前会话 ID。
    model_id: str  # 当前模型 ID。
    thinking: str | None  # 当前思考等级。
    cwd: Path  # 当前工作目录。
    theme: str  # 当前主题名。
    theme_styles: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_THEME_DATA))  # 当前主题样式映射。
    submit_on_enter: bool = True  # 回车是否直接提交。
    is_running: bool = False  # 当前是否正在运行请求。
    last_error: str = ""  # 最近一次错误消息。
    status_message: str = ""  # 状态栏当前文案。
    current_tool: str = ""  # 当前正在执行的工具名。
    output_cards: list[MessageCard] = field(default_factory=list)  # 旧输出卡片结构。
    thinking_cards: list[MessageCard] = field(default_factory=list)  # 旧 thinking 卡片结构。
    tool_cards: list[ToolCard] = field(default_factory=list)  # 旧工具卡片结构。
    transcript_turns: list[TranscriptTurn] = field(default_factory=list)  # transcript turn 列表。
    transcript_orphans: list[TranscriptBlock] = field(default_factory=list)  # 未归属 turn 的块。
    transcript_text: str = "No output yet.\n"  # 主输出区全文本。
    transcript_line_count: int = 1  # 主输出区总行数。
    transcript_line_styles: dict[int, str] = field(default_factory=dict)  # 行号到样式的映射。
    transcript_blocks: list[TranscriptBlock] = field(default_factory=list)  # 展平后的块序列。
    last_visible_block_id: str = ""  # 最后一个可见块 ID。
    transcript_revision: int = 0  # transcript 修订号。
    last_rendered_revision: int = -1  # 最近一次已渲染修订号。
    last_known_bottom_line: int = 0  # 最近一次已知底部行号。
    status_timeline: list[str] = field(default_factory=list)  # 近期状态消息时间线。
    recent_sessions: list[dict] = field(default_factory=list)  # 最近会话列表。
    auto_follow_output: bool = True  # 是否自动跟随最新输出。
    unseen_output_updates: int = 0  # 未读输出更新数。
    scroll_anchor: Literal["latest", "history", "jumped_latest"] = "latest"  # 当前滚动锚点。
    main_view_mode: Literal["latest", "history"] = "latest"  # 主区域查看模式。
    main_view_top_display_line: int = 0  # 主区域顶部显示行号。
    main_last_rendered_content_height: int = 0  # 上次渲染内容高度。
    main_pending_jump_to_bottom: bool = False  # 是否待跳到底部。
    last_output_event_id: str = ""  # 最近一次可见输出事件 ID。

    @classmethod
    def from_session(cls, session: AgentSession) -> "InteractiveState":
        """从当前会话对象构造界面初始状态。"""

        return cls(
            session_id=session.session_id,
            model_id=session.model.id,
            thinking=session.thinking,
            cwd=session.cwd,
            theme=session.settings.theme,
            theme_styles=cls._resolve_theme_styles(session),
            recent_sessions=session.list_recent_sessions(),
        )

    def sync_from_session(self, session: AgentSession) -> None:
        """把会话对象中的最新信息同步到界面状态。"""

        self.session_id = session.session_id
        self.model_id = session.model.id
        self.thinking = session.thinking
        self.cwd = session.cwd
        self.theme = session.settings.theme
        self.theme_styles = self._resolve_theme_styles(session)
        self.recent_sessions = session.list_recent_sessions()

    def clear_output(self) -> None:
        """清空主输出区、思考区和工具区内容。"""

        self.output_cards.clear()
        self.thinking_cards.clear()
        self.tool_cards.clear()
        self.transcript_turns.clear()
        self.transcript_orphans.clear()
        self.transcript_blocks.clear()
        self.transcript_text = "No output yet.\n"
        self.transcript_line_count = 1
        self.transcript_line_styles.clear()
        self.last_visible_block_id = ""
        self.transcript_revision += 1
        self.last_rendered_revision = -1
        self.last_known_bottom_line = 0
        self.auto_follow_output = True
        self.unseen_output_updates = 0
        self.scroll_anchor = "latest"
        self.main_view_mode = "latest"
        self.main_view_top_display_line = 0
        self.main_last_rendered_content_height = 0
        self.main_pending_jump_to_bottom = False
        self.last_output_event_id = ""

    def add_status(self, message: str) -> None:
        """向状态时间线追加一条消息。"""

        self.status_message = message
        self.status_timeline.append(message)
        self.status_timeline = self.status_timeline[-20:]

    def mark_history_view(self) -> None:
        """标记用户正在查看历史内容。"""

        self.auto_follow_output = False
        self.scroll_anchor = "history"
        self.main_view_mode = "history"

    def mark_latest_view(self) -> None:
        """标记界面当前跟随最新消息。"""

        self.auto_follow_output = True
        self.unseen_output_updates = 0
        self.scroll_anchor = "latest"
        self.main_view_mode = "latest"
        self.main_pending_jump_to_bottom = False

    def mark_jumped_to_latest(self) -> None:
        """标记用户刚主动跳到了最新消息。"""

        self.auto_follow_output = True
        self.unseen_output_updates = 0
        self.scroll_anchor = "jumped_latest"
        self.main_view_mode = "latest"
        self.main_pending_jump_to_bottom = False

    def register_output_update(self, event: SessionEvent) -> None:
        """登记一次新的可见输出更新。"""

        self.last_output_event_id = event.message_id or self.last_output_event_id
        if self.auto_follow_output:
            self.scroll_anchor = "latest"
            return
        self.unseen_output_updates += 1
        self.status_message = f"有新消息未显示，按 End 跳到最新 ({self.unseen_output_updates})"

    def start_user_turn(self, user_text: str, turn_id: str) -> None:
        """在真实用户发送问题时创建新的 turn。"""

        preview = user_text.strip() or "(empty)"
        user_block = TranscriptBlock(
            id=f"{turn_id}-user",
            kind="user",
            title="User",
            body=preview,
            source="user",
            render_order=100,
        )
        turn = TranscriptTurn(
            turn_id=turn_id,
            user_prompt_preview=preview,
            user_block=user_block,
        )
        self.transcript_turns.append(turn)
        self.rebuild_transcript()

    def apply_event(self, event: SessionEvent) -> bool:
        """根据会话事件更新 UI 状态，并返回是否产生了可见输出。"""

        visible_output = False
        turn = self._ensure_turn(event.turn_id)
        if event.type == "message_start":
            if turn is None:
                return False
            self._ensure_assistant_block(turn, event.message_id)
            visible_output = True
        elif event.type == "message_delta":
            if turn is None:
                return False
            block = self._ensure_assistant_block(turn, event.message_id)
            block.body += event.delta
            self._sync_legacy_assistant_card(block)
            visible_output = True
        elif event.type == "message_end":
            if turn is None:
                return False
            block = self._ensure_assistant_block(turn, event.message_id)
            if event.message:
                block.body = event.message
            self._sync_legacy_assistant_card(block)
            turn.completed = True
            visible_output = True
        elif event.type == "thinking":
            if turn is None:
                return False
            block = TranscriptBlock(
                id=event.message_id or f"{turn.turn_id}-thinking-{len(turn.thinking_blocks) + 1}",
                kind="thinking",
                title="Thinking",
                body=event.message,
                source=event.source or "thinking",
                render_order=event.render_order or 200,
            )
            turn.thinking_blocks.append(block)
            self.thinking_cards.append(MessageCard(id=block.id, title=block.title, body=block.body, style="thinking"))
            visible_output = True
        elif event.type == "status":
            self.add_status(event.message)
            if event.source in {"steering", "follow_up"} and turn is not None and event.message:
                block = TranscriptBlock(
                    id=event.message_id or f"{turn.turn_id}-status-{len(turn.status_blocks) + 1}",
                    kind="status",
                    title="Status",
                    body=event.message,
                    source=event.source or "status",
                    render_order=event.render_order or 230,
                )
                turn.status_blocks.append(block)
                visible_output = True
        elif event.type == "tool_start":
            if turn is None:
                return False
            self.current_tool = event.tool_name
            block = TranscriptBlock(
                id=event.message_id or f"{turn.turn_id}-tool-{len(turn.tool_blocks) + 1}",
                kind="tool",
                title=f"Tool:{event.tool_name}",
                status="running",
                tool_name=event.tool_name,
                body=self._render_tool_body(event.tool_arguments, ""),
                source=event.source or "tool",
                render_order=event.render_order or 250,
            )
            turn.tool_blocks.append(block)
            self.tool_cards.append(
                ToolCard(
                    id=block.id,
                    tool_name=event.tool_name,
                    status="running",
                    arguments=event.tool_arguments,
                )
            )
            self.add_status(f"工具开始: {event.tool_name}")
            visible_output = True
        elif event.type == "tool_update":
            self.current_tool = event.tool_name
            self.add_status(event.message)
        elif event.type == "tool_end":
            self.current_tool = ""
            if turn is None:
                return False
            self._finalize_tool_block(turn, event)
            self.add_status(f"工具完成: {event.tool_name}")
            visible_output = True
        elif event.type == "error":
            self.last_error = event.message
            self.add_status(event.message)
            block = TranscriptBlock(
                id=event.message_id or f"error-{len(self.transcript_orphans) + 1}",
                kind="error",
                title="Error",
                body=event.message,
                source=event.source or "error",
                render_order=event.render_order or 500,
            )
            if turn is None:
                self.transcript_orphans.append(block)
            else:
                turn.status_blocks.append(block)
            visible_output = True
        if visible_output:
            self.register_output_update(event)
            self.rebuild_transcript()
        return visible_output

    def rebuild_transcript(self) -> None:
        """根据 turn 结构重建主输出区文本与样式映射。"""

        if not self.transcript_turns and not self.transcript_orphans:
            self.transcript_text = "No output yet.\n"
            self.transcript_line_count = 1
            self.transcript_line_styles = {0: "status"}
            self.last_visible_block_id = ""
            self.transcript_revision += 1
            self.last_known_bottom_line = 0
            self.transcript_blocks = []
            return
        lines: list[str] = []
        line_styles: dict[int, str] = {}
        flattened: list[TranscriptBlock] = []
        for turn in self.transcript_turns:
            block_sequence = [turn.user_block]
            block_sequence.extend(turn.thinking_blocks)
            middle_blocks = sorted(
                [*turn.status_blocks, *turn.tool_blocks],
                key=lambda block: (block.render_order, block.id),
            )
            block_sequence.extend(middle_blocks)
            if turn.assistant_block is not None:
                block_sequence.append(turn.assistant_block)
            for block in block_sequence:
                flattened.append(block)
                self._append_block_lines(lines, line_styles, block)
        for block in self.transcript_orphans:
            flattened.append(block)
            self._append_block_lines(lines, line_styles, block)
        self.transcript_text = "\n".join(lines).rstrip() + "\n"
        self.transcript_line_count = self.transcript_text.count("\n") or 1
        self.transcript_line_styles = line_styles
        self.transcript_blocks = flattened
        self.last_visible_block_id = flattened[-1].id if flattened else ""
        self.transcript_revision += 1
        self.last_known_bottom_line = max(0, self.transcript_line_count - 1)

    def _ensure_turn(self, turn_id: str) -> TranscriptTurn | None:
        """根据 turn_id 查找或创建 turn。"""

        if not turn_id:
            if self.transcript_turns:
                return self.transcript_turns[-1]
            turn_id = "implicit-turn-1"
        for turn in reversed(self.transcript_turns):
            if turn.turn_id == turn_id:
                return turn
        placeholder = TranscriptTurn(
            turn_id=turn_id,
            user_prompt_preview="",
            user_block=TranscriptBlock(
                id=f"{turn_id}-user",
                kind="user",
                title="User",
                body="",
                source="user",
                render_order=100,
            ),
        )
        self.transcript_turns.append(placeholder)
        return placeholder

    def _ensure_assistant_block(self, turn: TranscriptTurn, message_id: str) -> TranscriptBlock:
        """确保当前 turn 的 assistant block 已存在。"""

        block_id = message_id or f"{turn.turn_id}-assistant"
        if turn.assistant_block is not None:
            return turn.assistant_block
        block = TranscriptBlock(
            id=block_id,
            kind="assistant",
            title="Assistant",
            source="assistant",
            render_order=400,
        )
        turn.assistant_block = block
        self._sync_legacy_assistant_card(block)
        return block

    def _sync_legacy_assistant_card(self, block: TranscriptBlock) -> None:
        """同步 assistant block 到旧卡片结构，兼容现有测试和调用。"""

        for card in self.output_cards:
            if card.id == block.id:
                card.body = block.body
                card.title = block.title
                card.style = "assistant"
                return
        self.output_cards.append(MessageCard(id=block.id, title=block.title, body=block.body, style="assistant"))

    def _finalize_tool_block(self, turn: TranscriptTurn, event: SessionEvent) -> None:
        """把工具结束事件回写到对应工具块和旧工具卡片中。"""

        status = "error" if event.status_level == "error" else "success"
        preview = event.tool_output_preview or event.message
        arguments = event.tool_arguments
        block = None
        for candidate in reversed(turn.tool_blocks):
            if candidate.id == event.message_id or candidate.tool_name == event.tool_name:
                block = candidate
                break
        if block is None:
            block = TranscriptBlock(
                id=event.message_id or f"{turn.turn_id}-tool-{len(turn.tool_blocks) + 1}",
                kind="tool",
                title=f"Tool:{event.tool_name}",
                tool_name=event.tool_name,
                source=event.source or "tool",
                render_order=event.render_order or 270,
            )
            turn.tool_blocks.append(block)
        if not arguments and block.body:
            first_line = block.body.splitlines()[0]
            if first_line.startswith("args: "):
                arguments = first_line[len("args: ") :]
        block.status = status
        block.body = self._render_tool_body(arguments, preview)
        for card in reversed(self.tool_cards):
            if card.id == block.id or card.tool_name == event.tool_name:
                card.status = status
                if arguments:
                    card.arguments = arguments
                card.output_preview = preview
                return
        self.tool_cards.append(
            ToolCard(
                id=block.id,
                tool_name=event.tool_name,
                status=status,
                arguments=arguments,
                output_preview=preview,
            )
        )

    def _append_block_lines(self, lines: list[str], line_styles: dict[int, str], block: TranscriptBlock) -> None:
        """把一个块追加到 transcript 文本，并记录每一行的样式。"""

        if lines:
            lines.append("")
        header_style, body_style = self._styles_for_block(block)
        header = self._header_for_block(block)
        header_index = len(lines)
        lines.append(header)
        line_styles[header_index] = header_style
        if block.body:
            for body_line in block.body.rstrip().splitlines():
                body_index = len(lines)
                lines.append(body_line)
                line_styles[body_index] = body_style

    @staticmethod
    def _render_tool_body(arguments: str, preview: str) -> str:
        """把工具参数和输出摘要拼成稳定文本。"""

        lines: list[str] = []
        if arguments:
            lines.append(f"args: {arguments}")
        if preview:
            lines.append(preview)
        return "\n".join(lines)

    @staticmethod
    def _header_for_block(block: TranscriptBlock) -> str:
        """生成 transcript 块头部。"""

        if block.kind == "tool":
            suffix = f" {block.status}" if block.status else ""
            return f"[Tool:{block.tool_name or block.title}]{suffix}"
        return f"[{block.title}]"

    @staticmethod
    def _styles_for_block(block: TranscriptBlock) -> tuple[str, str]:
        """返回块头部和正文对应的样式类。"""

        if block.kind == "user":
            return ("user", "user")
        if block.kind == "assistant":
            return ("assistant_header", "assistant_body")
        if block.kind == "thinking":
            return ("thinking_header", "thinking_body")
        if block.kind == "tool":
            body_style = "tool_error" if block.status == "error" else ("tool_success" if block.status == "success" else "tool_running")
            return ("tool_header", body_style)
        if block.kind == "error":
            return ("error", "error")
        return ("status", "status")

    @staticmethod
    def _resolve_theme_styles(session: AgentSession) -> dict[str, str]:
        """从当前会话资源中解析活动主题样式，并补全默认值。"""

        styles = dict(DEFAULT_THEME_DATA)
        theme_resource = session.resources.themes.get(session.settings.theme) or session.resources.themes.get("default")
        if theme_resource is not None:
            styles.update(theme_resource.data)
        return styles
