from __future__ import annotations

from ai.types import AssistantMessage, ConversationMessage, UserMessage, ensure_message

from ..session_manager import SessionManager
from ..types import CompactResult


class SessionCompactor:
    """负责选择旧消息并生成会话摘要。"""

    def __init__(self, session_manager: SessionManager, keep_turns: int = 4) -> None:
        """初始化压缩器并配置保留的最近对话轮数。"""

        self.session_manager = session_manager  # 会话管理器。
        self.keep_turns = keep_turns  # 压缩时保留的最近 turn 数。

    def compact_session(self, session_id: str, branch_id: str | None = None) -> CompactResult:
        """压缩指定会话分支的旧消息并写入摘要事件。"""

        snapshot = self.session_manager.load_session(session_id)
        active_branch = branch_id or snapshot.branch_id
        nodes = [node for node in snapshot.nodes if node.branch_id == active_branch]
        keep_count = self.keep_turns * 2
        compacted = nodes[:-keep_count] if len(nodes) > keep_count else []
        if not compacted:
            return CompactResult(summary="", compacted_count=0, branch_id=active_branch, node_ids=[])
        summary = self._summarize_nodes(compacted)
        node_ids = [node.id for node in compacted]
        self.session_manager.append_summary(session_id, branch_id=active_branch, summary=summary, node_ids=node_ids)
        return CompactResult(summary=summary, compacted_count=len(compacted), branch_id=active_branch, node_ids=node_ids)

    @staticmethod
    def _summarize_nodes(nodes) -> str:
        """把一组旧节点转换成可回填上下文的文本摘要。"""

        lines: list[str] = ["历史摘要："]
        for node in nodes:
            prefix = {"user": "用户", "assistant": "助手", "tool": "工具"}.get(node.role, node.role)
            message = ensure_message(
                {
                    "role": node.role,
                    "content": node.content,
                    "toolCallId": node.tool_call_id,
                    "toolName": node.tool_name,
                    "thinking": node.thinking,
                    "toolCalls": node.tool_calls,
                    "metadata": node.metadata,
                }
            )
            content = message.content.strip().replace("\n", " ")
            if len(content) > 120:
                content = content[:120] + "..."
            lines.append(f"- {prefix}: {content}")
        return "\n".join(lines)
