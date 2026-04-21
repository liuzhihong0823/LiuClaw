from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from ai import Context, Model, ProviderRegistry, UserMessage, completeSimple
from ai.utils.context_window import estimate_context_tokens
from ai.types import AssistantMessage, ConversationMessage, ToolResultMessage

from ..session_manager import SessionManager
from ..types import CompactResult, CompactionSettings, SessionEntry, SessionMessageEntry

SUMMARY_SYSTEM_PROMPT = """You compress earlier agent history into a checkpoint summary for continued work.

Use this EXACT format:

## Goal
[Current objective]

## Constraints & Preferences
- [constraints]

## Progress
### Done
- [x] completed work

### In Progress
- [ ] work underway

### Blocked
- [issues blocking progress]

## Key Decisions
- **Decision**: rationale

## Next Steps
1. next action

## Critical Context
- important files, errors, tool results, or references

Keep it concise and preserve exact file paths, identifiers, and errors when important."""

TURN_PREFIX_PROMPT = """Summarize the earlier prefix of a split turn so the retained suffix still makes sense.

Use this exact format:

## Original Request
[what the user asked in this turn]

## Early Progress
- key work completed in the discarded prefix

## Context for Suffix
- context needed to understand the retained suffix
"""


@dataclass(slots=True)
class CompactionRuntime:
    """定义压缩流程运行时所依赖的模型与配置。"""

    model: Model  # 当前默认压缩所使用的模型。
    thinking: str | None  # 压缩调用时使用的 thinking 级别。
    settings: Any  # 生效中的全局设置对象。
    registry: ProviderRegistry | None = None  # provider 注册表。
    model_resolver: Any | None = None  # 可选的模型解析器。


@dataclass(slots=True)
class CutPointResult:
    """描述压缩切点以及是否切开了一个未完成 turn。"""

    first_kept_index: int  # 压缩后第一个保留条目的索引。
    turn_start_index: int  # 若切开 turn，则该 turn 起点索引。
    is_split_turn: bool  # 是否切开了一个 turn。


@dataclass(slots=True)
class CompactionPreparation:
    """保存一次压缩前分析得到的中间结果。"""

    first_kept_entry_id: str  # 压缩边界之后第一个保留条目 ID。
    messages_to_summarize: list[ConversationMessage]  # 需要主摘要压缩的消息。
    turn_prefix_messages: list[ConversationMessage]  # 若切开 turn，需要额外摘要的前缀消息。
    is_split_turn: bool  # 是否发生了 turn 级别切分。
    tokens_before: int  # 压缩前估算的 token 数。
    previous_summary: str | None  # 之前已有的压缩摘要。
    details: dict[str, Any]  # 附加细节，如读写文件信息。


class SessionCompactor:
    """负责选择旧消息并生成会话摘要。"""

    def __init__(self, session_manager: SessionManager, runtime: CompactionRuntime) -> None:
        """初始化压缩器，并绑定会话管理器与运行时配置。"""

        self.session_manager = session_manager
        self.runtime = runtime

    async def compact_session(self, session_ref: str, leaf_id: str | None = None, custom_instructions: str | None = None) -> CompactResult:
        """对指定会话分支执行一次完整的上下文压缩。"""

        self.session_manager.set_session_file(session_ref)
        path_entries = self.session_manager.get_branch(leaf_id)
        preparation = self.prepare_compaction(path_entries, self.runtime.settings.compaction)
        if preparation is None:
            return CompactResult(summary="", compacted_count=0, first_kept_entry_id="", tokens_before=0)
        try:
            summary = await self._generate_compaction_summary(preparation, custom_instructions=custom_instructions)
        except Exception:
            return CompactResult(summary="", compacted_count=0, first_kept_entry_id="", tokens_before=preparation.tokens_before)
        if not summary:
            return CompactResult(summary="", compacted_count=0, first_kept_entry_id="", tokens_before=preparation.tokens_before)
        self.session_manager.append_compaction(
            summary=summary,
            first_kept_entry_id=preparation.first_kept_entry_id,
            tokens_before=preparation.tokens_before,
            details=preparation.details,
        )
        compacted_count = len(preparation.messages_to_summarize) + len(preparation.turn_prefix_messages)
        return CompactResult(
            summary=summary,
            compacted_count=compacted_count,
            first_kept_entry_id=preparation.first_kept_entry_id,
            tokens_before=preparation.tokens_before,
            details=preparation.details,
        )

    def prepare_compaction(self, path_entries: list[SessionEntry], settings: CompactionSettings) -> CompactionPreparation | None:
        """分析当前分支，决定压缩边界与待摘要消息。"""

        if path_entries and path_entries[-1].type == "compaction":
            return None

        previous_summary: str | None = None
        boundary_start = 0
        for index in range(len(path_entries) - 1, -1, -1):
            entry = path_entries[index]
            if entry.type == "compaction":
                previous_summary = getattr(entry, "summary", "")
                first_kept_id = getattr(entry, "first_kept_entry_id", "")
                boundary_start = next((i for i, item in enumerate(path_entries) if item.id == first_kept_id), index + 1)
                break

        messages = self._messages_from_entries(path_entries)
        tokens_before = estimate_context_tokens(Context(messages=messages))
        cut_point = self.find_cut_point(path_entries, boundary_start, len(path_entries), settings.keep_recent_tokens)
        first_kept_entry = path_entries[cut_point.first_kept_index] if 0 <= cut_point.first_kept_index < len(path_entries) else None
        if first_kept_entry is None:
            return None

        history_end = cut_point.turn_start_index if cut_point.is_split_turn else cut_point.first_kept_index
        messages_to_summarize = self._messages_from_entries(path_entries[boundary_start:history_end], include_compactions=False)
        turn_prefix_messages = self._messages_from_entries(path_entries[cut_point.turn_start_index:cut_point.first_kept_index], include_compactions=False) if cut_point.is_split_turn else []
        if not messages_to_summarize and not turn_prefix_messages:
            return None
        details = self._extract_file_details(messages_to_summarize + turn_prefix_messages)
        return CompactionPreparation(
            first_kept_entry_id=first_kept_entry.id,
            messages_to_summarize=messages_to_summarize,
            turn_prefix_messages=turn_prefix_messages,
            is_split_turn=cut_point.is_split_turn,
            tokens_before=tokens_before,
            previous_summary=previous_summary,
            details=details,
        )

    def find_cut_point(self, entries: list[SessionEntry], start_index: int, end_index: int, keep_recent_tokens: int) -> CutPointResult:
        """根据保留 token 预算计算压缩切点。"""

        message_indexes = [index for index in range(start_index, end_index) if entries[index].type == "message"]
        if not message_indexes:
            return CutPointResult(first_kept_index=start_index, turn_start_index=-1, is_split_turn=False)

        accumulated_tokens = 0
        cut_index = message_indexes[0]
        for index in range(end_index - 1, start_index - 1, -1):
            entry = entries[index]
            if entry.type != "message":
                continue
            accumulated_tokens += self._estimate_entry_tokens(entry)
            if accumulated_tokens >= keep_recent_tokens:
                cut_index = next((item for item in message_indexes if item >= index), message_indexes[0])
                break

        while cut_index > start_index and entries[cut_index - 1].type != "message" and entries[cut_index - 1].type != "compaction":
            cut_index -= 1

        is_user_message = isinstance(getattr(entries[cut_index], "message", None), UserMessage)
        turn_start_index = -1 if is_user_message else self._find_turn_start(entries, cut_index, start_index)
        return CutPointResult(first_kept_index=cut_index, turn_start_index=turn_start_index, is_split_turn=not is_user_message and turn_start_index != -1)

    async def _generate_compaction_summary(self, preparation: CompactionPreparation, custom_instructions: str | None = None) -> str:
        """根据压缩准备结果生成最终摘要文本。"""

        history_summary = await self._summarize_messages(preparation.messages_to_summarize, custom_instructions, preparation.previous_summary)
        if preparation.is_split_turn and preparation.turn_prefix_messages:
            turn_prefix = await self._summarize_turn_prefix(preparation.turn_prefix_messages)
            summary = f"{history_summary}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix}"
        else:
            summary = history_summary
        return summary + self._format_file_details(preparation.details)

    async def _summarize_messages(
        self,
        messages: list[ConversationMessage],
        custom_instructions: str | None,
        previous_summary: str | None,
    ) -> str:
        """调用摘要模型，将旧消息压缩成结构化 checkpoint。"""

        if not messages and not previous_summary:
            return "## Goal\n暂无\n\n## Constraints & Preferences\n- (none)\n\n## Progress\n### Done\n- [x] (none)\n\n### In Progress\n- [ ] (none)\n\n### Blocked\n- (none)\n\n## Key Decisions\n- **None**: no prior decisions recorded\n\n## Next Steps\n1. Continue from the retained recent context\n\n## Critical Context\n- (none)"
        history_text = self._serialize_messages(messages)
        prompt = f"<conversation>\n{history_text}\n</conversation>\n\n"
        if previous_summary:
            prompt += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\nUpdate the previous summary with the new messages while preserving relevant information.\n\n"
        if custom_instructions:
            prompt += f"Additional focus: {custom_instructions}\n\n"
        prompt += SUMMARY_SYSTEM_PROMPT
        message = await completeSimple(
            self._resolve_summary_model(),
            Context(systemPrompt="You summarize conversation history into structured checkpoints.", messages=[UserMessage(content=prompt)], tools=[]),
            reasoning=self.runtime.thinking,
            registry=self.runtime.registry,
        )
        return str(message.content).strip()

    async def _summarize_turn_prefix(self, messages: list[ConversationMessage]) -> str:
        """为被截断 turn 的前半段单独生成补充上下文摘要。"""

        prompt = f"<conversation>\n{self._serialize_messages(messages)}\n</conversation>\n\n{TURN_PREFIX_PROMPT}"
        message = await completeSimple(
            self._resolve_summary_model(),
            Context(systemPrompt="You summarize turn prefixes for compaction.", messages=[UserMessage(content=prompt)], tools=[]),
            reasoning=self.runtime.thinking,
            registry=self.runtime.registry,
        )
        return str(message.content).strip()

    def _resolve_summary_model(self) -> Model:
        """解析压缩专用模型；若未配置则回退到当前模型。"""

        compact_model = self.runtime.settings.compaction.compact_model or self.runtime.settings.compact_model
        if compact_model and self.runtime.model_resolver is not None:
            resolver = self.runtime.model_resolver
            return resolver.get(compact_model) if hasattr(resolver, "get") else resolver(compact_model)
        return self.runtime.model

    def _serialize_messages(self, messages: list[ConversationMessage]) -> str:
        """将消息列表序列化为适合送给摘要模型的文本。"""

        lines: list[str] = []
        for message in messages:
            if isinstance(message, UserMessage):
                lines.append(f"[USER]\n{message.text}")
            elif isinstance(message, AssistantMessage):
                lines.append(f"[ASSISTANT]\n{message.text}")
                if message.thinking:
                    lines.append(f"[THINKING]\n{message.thinking}")
                for tool_call in message.toolCalls:
                    lines.append(f"[TOOL_CALL:{tool_call.name}]\n{tool_call.arguments_text}")
            elif isinstance(message, ToolResultMessage):
                lines.append(f"[TOOL_RESULT:{message.toolName}]\n{message.text}")
        return "\n---\n".join(lines)

    def _messages_from_entries(self, entries: list[SessionEntry], include_compactions: bool = True) -> list[ConversationMessage]:
        """将会话条目转换为消息列表，可选择包含摘要占位消息。"""

        messages: list[ConversationMessage] = []
        for entry in entries:
            if isinstance(entry, SessionMessageEntry):
                messages.append(entry.message)
            elif include_compactions and entry.type == "compaction":
                messages.append(UserMessage(content=f"[Session Summary]\n{getattr(entry, 'summary', '')}", metadata={"summary": True}))
            elif include_compactions and entry.type == "branch_summary":
                messages.append(UserMessage(content=f"[Branch Summary]\n{getattr(entry, 'summary', '')}", metadata={"branch_summary": True}))
        return messages

    def _estimate_entry_tokens(self, entry: SessionEntry) -> int:
        """粗略估算单个条目的 token 占用。"""

        if not isinstance(entry, SessionMessageEntry):
            return max(1, len(json.dumps(entry.details if hasattr(entry, "details") else asdict(entry), ensure_ascii=False)) // 3)  # type: ignore[name-defined]
        return estimate_context_tokens(Context(messages=[entry.message]))

    def _find_turn_start(self, entries: list[SessionEntry], cut_index: int, start_index: int) -> int:
        """从切点向前寻找当前 turn 的起始用户消息。"""

        for index in range(cut_index, start_index - 1, -1):
            entry = entries[index]
            if isinstance(getattr(entry, "message", None), UserMessage):
                return index
        return -1

    def _extract_file_details(self, messages: list[ConversationMessage]) -> dict[str, Any]:
        """从工具调用历史中提取读写文件明细。"""

        read_files: set[str] = set()
        modified_files: set[str] = set()
        for message in messages:
            if not isinstance(message, AssistantMessage):
                continue
            for tool_call in message.toolCalls:
                if tool_call.name in {"read", "open"}:
                    path = tool_call.arguments.get("path") if isinstance(tool_call.arguments, dict) else None
                    if path:
                        read_files.add(str(path))
                if tool_call.name in {"write", "edit", "truncate"}:
                    path = tool_call.arguments.get("path") if isinstance(tool_call.arguments, dict) else None
                    if path:
                        modified_files.add(str(path))
        return {"readFiles": sorted(read_files), "modifiedFiles": sorted(modified_files)}

    def _format_file_details(self, details: dict[str, Any]) -> str:
        """将文件读写明细格式化到摘要尾部。"""

        read_files = details.get("readFiles", [])
        modified_files = details.get("modifiedFiles", [])
        parts: list[str] = []
        if read_files:
            parts.append("\n\nRead files:\n" + "\n".join(f"- {item}" for item in read_files))
        if modified_files:
            parts.append("\n\nModified files:\n" + "\n".join(f"- {item}" for item in modified_files))
        return "".join(parts)
