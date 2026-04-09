from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, TypeAlias

ReasoningLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]
StreamEventType = Literal[
    "start",
    "update",
    "done",
    "error",
    "text_start",
    "text_delta",
    "text_end",
    "thinking_start",
    "thinking_delta",
    "thinking_end",
    "toolcall_start",
    "toolcall_delta",
    "toolcall_end",
    "tool_result",
]
ContentBlockType = Literal["text", "thinking", "tool_call", "image", "tool_result_content"]
StreamLifecycle = Literal["start", "update", "done", "error"]
StreamItemType = Literal["message", "text", "thinking", "tool_call", "tool_result", "image"]

_LEGACY_STREAM_EVENT_ALIASES: dict[str, tuple[StreamLifecycle, StreamItemType]] = {
    "text_start": ("start", "text"),
    "text_delta": ("update", "text"),
    "text_end": ("done", "text"),
    "thinking_start": ("start", "thinking"),
    "thinking_delta": ("update", "thinking"),
    "thinking_end": ("done", "thinking"),
    "toolcall_start": ("start", "tool_call"),
    "toolcall_delta": ("update", "tool_call"),
    "toolcall_end": ("done", "tool_call"),
    "tool_result": ("update", "tool_result"),
}
_REASONING_ORDER: tuple[ReasoningLevel, ...] = ("off", "minimal", "low", "medium", "high", "xhigh")


@dataclass(slots=True)
class TextContent:
    """统一文本内容块。"""

    type: Literal["text"] = "text"  # 内容块类型标记。
    text: str = ""  # 文本正文。


@dataclass(slots=True)
class ThinkingContent:
    """统一思考内容块。"""

    type: Literal["thinking"] = "thinking"  # 内容块类型标记。
    thinking: str = ""  # 思考文本。


@dataclass(slots=True)
class ImageContent:
    """统一图片内容块。"""

    type: Literal["image"] = "image"  # 内容块类型标记。
    data: str = ""  # 图片数据，通常是 base64 或可传输内容。
    mimeType: str = "image/png"  # 图片 MIME 类型。
    detail: str | None = None  # 图片细节等级或渲染提示。
    metadata: dict[str, Any] = field(default_factory=dict)  # 附加图片元数据。


@dataclass(slots=True)
class ToolResultContent:
    """统一工具结果内容块。"""

    type: Literal["tool_result_content"] = "tool_result_content"  # 内容块类型标记。
    text: str = ""  # 工具结果的人类可读文本。
    data: str | None = None  # 原始结构化结果或附件引用。
    mimeType: str | None = None  # 原始结果 MIME 类型。
    metadata: dict[str, Any] = field(default_factory=dict)  # 工具结果附加元信息。


@dataclass(slots=True)
class Tool:
    """定义可供模型调用的工具。"""

    name: str  # 工具名。
    description: str | None = None  # 工具描述。
    inputSchema: dict[str, Any] = field(default_factory=dict)  # 工具参数 JSON Schema。
    metadata: dict[str, Any] = field(default_factory=dict)  # 工具内部元信息。
    renderMetadata: dict[str, Any] = field(default_factory=dict)  # 工具展示层元信息。


def serialize_tool_arguments(arguments: Any) -> str:
    """把工具参数稳定序列化为字符串。"""

    if isinstance(arguments, str):
        return arguments
    if arguments in (None, ""):
        return ""
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_tool_arguments(arguments: Any) -> Any:
    """尽可能把工具参数解析为结构化对象。"""

    if isinstance(arguments, (dict, list)):
        return arguments
    if arguments in (None, ""):
        return {}
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


@dataclass(slots=True)
class ToolCall:
    """表示 assistant 发起的一次工具调用。"""

    id: str  # 工具调用唯一 ID。
    name: str  # 被调用工具名。
    arguments: Any = field(default_factory=dict)  # 工具参数，归一化后可为字典或字符串。
    metadata: dict[str, Any] = field(default_factory=dict)  # 工具调用元信息。

    def __post_init__(self) -> None:
        self.arguments = parse_tool_arguments(self.arguments)

    @property
    def arguments_text(self) -> str:
        """返回兼容旧实现的字符串参数。"""

        return serialize_tool_arguments(self.arguments)


@dataclass(slots=True)
class ToolCallContent:
    """统一工具调用内容块。"""

    type: Literal["tool_call"] = "tool_call"  # 内容块类型标记。
    id: str = ""  # 工具调用 ID。
    name: str = ""  # 工具名。
    arguments: Any = field(default_factory=dict)  # 工具参数内容。
    metadata: dict[str, Any] = field(default_factory=dict)  # 工具调用元信息。

    def __post_init__(self) -> None:
        self.arguments = parse_tool_arguments(self.arguments)

    @property
    def arguments_text(self) -> str:
        """返回兼容旧实现的字符串参数。"""

        return serialize_tool_arguments(self.arguments)


MessageContentBlock: TypeAlias = TextContent | ThinkingContent | ToolCallContent | ImageContent | ToolResultContent
AssistantContentBlock: TypeAlias = TextContent | ThinkingContent | ToolCallContent | ImageContent
UserContentBlock: TypeAlias = TextContent | ImageContent
ToolResultContentBlock: TypeAlias = TextContent | ImageContent | ToolResultContent


class ContentBlocks(list[MessageContentBlock]):
    """统一内容块列表，兼容列表访问与文本视图读取。"""

    @property
    def text(self) -> str:
        """返回内容块中的纯文本拼接结果。"""

        return extract_text_from_blocks(self)

    @property
    def thinking(self) -> str:
        """返回内容块中的 thinking 拼接结果。"""

        return extract_thinking_from_blocks(self)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """返回内容块中的工具调用。"""

        return extract_tool_calls_from_blocks(self)

    def append_text(self, text: str) -> None:
        """追加一个文本块。"""

        if text:
            self.append(TextContent(text=text))

    def append_thinking(self, thinking: str) -> None:
        """追加一个思考块。"""

        if thinking:
            self.append(ThinkingContent(thinking=thinking))

    def append_tool_call(self, tool_call: ToolCall | ToolCallContent | dict[str, Any]) -> None:
        """追加一个工具调用块。"""

        if isinstance(tool_call, ToolCallContent):
            self.append(tool_call)
            return
        normalized = ensure_tool_call(tool_call)
        self.append(
            ToolCallContent(
                id=normalized.id,
                name=normalized.name,
                arguments=normalized.arguments,
                metadata=dict(normalized.metadata),
            )
        )

    def __str__(self) -> str:
        return self.text

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.text == other
        return super().__eq__(other)

    def __iadd__(self, other: object):
        if isinstance(other, str):
            self.append_text(other)
            return self
        if isinstance(other, Iterable):
            for item in other:
                self.append(item)
            return self
        return NotImplemented

    def strip(self, chars: str | None = None) -> str:
        """兼容旧式字符串读取。"""

        return self.text.strip(chars)

    def replace(self, old: str, new: str, count: int = -1) -> str:
        """兼容旧式字符串读取。"""

        return self.text.replace(old, new, count)

    def splitlines(self, keepends: bool = False) -> list[str]:
        """兼容旧式字符串读取。"""

        return self.text.splitlines(keepends)

    def rstrip(self, chars: str | None = None) -> str:
        """兼容旧式字符串读取。"""

        return self.text.rstrip(chars)

    def lstrip(self, chars: str | None = None) -> str:
        """兼容旧式字符串读取。"""

        return self.text.lstrip(chars)

    def startswith(self, prefix: str | tuple[str, ...], start: int = 0, end: int | None = None) -> bool:
        """兼容旧式字符串读取。"""

        text = self.text if end is None else self.text[:end]
        return text.startswith(prefix, start)

    def endswith(self, suffix: str | tuple[str, ...], start: int = 0, end: int | None = None) -> bool:
        """兼容旧式字符串读取。"""

        text = self.text[slice(start, end)]
        return text.endswith(suffix)


def _now_ms() -> int:
    """返回当前时间戳毫秒值。"""

    return int(time.time() * 1000)


@dataclass(slots=True)
class UserMessage:
    """表示用户输入消息。"""

    role: Literal["user"] = "user"  # 消息角色。
    content: ContentBlocks = field(default_factory=ContentBlocks)  # 用户消息内容块。
    metadata: dict[str, Any] = field(default_factory=dict)  # 用户消息元信息。
    timestamp: int = field(default_factory=_now_ms)  # 消息创建时间戳（毫秒）。

    def __post_init__(self) -> None:
        self.content = ContentBlocks(normalize_user_content_blocks(self.content))

    @property
    def text(self) -> str:
        """返回消息中的纯文本。"""

        return self.content.text


@dataclass(init=False, slots=True)
class AssistantMessage:
    """表示 assistant 消息，也是 `complete()` 的最终结果对象。"""

    role: Literal["assistant"] = "assistant"  # 消息角色。
    content: ContentBlocks = field(default_factory=ContentBlocks)  # assistant 输出内容块。
    metadata: dict[str, Any] = field(default_factory=dict)  # assistant 消息元信息。
    usage: dict[str, Any] | None = None  # token 使用量等统计数据。
    stopReason: str | None = None  # provider 返回的停止原因。
    responseId: str | None = None  # provider 侧响应 ID。
    errorMessage: str | None = None  # 错误场景下记录的错误消息。
    timestamp: int = field(default_factory=_now_ms)  # 消息创建时间戳（毫秒）。

    def __init__(
        self,
        *,
        role: Literal["assistant"] = "assistant",
        content: Any = None,
        thinking: str | None = None,
        toolCalls: list[ToolCall | dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        stopReason: str | None = None,
        responseId: str | None = None,
        errorMessage: str | None = None,
        timestamp: int | None = None,
    ) -> None:
        self.role = role
        self.content = ContentBlocks(
            normalize_assistant_content_blocks(
                content,
                thinking=thinking or "",
                tool_calls=toolCalls or [],
            )
        )
        self.metadata = dict(metadata or {})
        self.usage = usage
        self.stopReason = stopReason
        self.responseId = responseId
        self.errorMessage = errorMessage
        self.timestamp = _now_ms() if timestamp is None else timestamp

    @property
    def text(self) -> str:
        """返回 assistant 的最终文本内容。"""

        return self.content.text

    @property
    def thinking_text(self) -> str:
        """返回 assistant 的思考文本。"""

        return self.content.thinking

    @property
    def thinking(self) -> str:
        """兼容旧实现读取 thinking。"""

        return self.content.thinking

    @property
    def toolCalls(self) -> list[ToolCall]:
        """兼容旧实现读取工具调用列表。"""

        return self.content.tool_calls


@dataclass(slots=True)
class ToolResultMessage:
    """表示工具执行结果消息，供上层回填给模型。"""

    role: Literal["tool"] = "tool"  # 消息角色。
    toolCallId: str = ""  # 对应的工具调用 ID。
    toolName: str = ""  # 工具名。
    content: ContentBlocks = field(default_factory=ContentBlocks)  # 工具执行结果内容块。
    metadata: dict[str, Any] = field(default_factory=dict)  # 工具结果元信息。
    isError: bool = False  # 当前结果是否表示错误。
    details: Any | None = None  # 附加调试细节或原始结果。
    timestamp: int = field(default_factory=_now_ms)  # 消息创建时间戳（毫秒）。

    def __post_init__(self) -> None:
        self.content = ContentBlocks(normalize_tool_result_content_blocks(self.content))

    @property
    def text(self) -> str:
        """返回工具结果中的纯文本。"""

        return self.content.text


ConversationMessage: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage


@dataclass(slots=True)
class Model:
    """描述一个可供统一接入层使用的模型元数据。"""

    id: str  # 模型唯一 ID。
    provider: str  # 归属 provider 名称。
    inputPrice: float  # 输入 token 单价。
    outputPrice: float  # 输出 token 单价。
    contextWindow: int  # 模型上下文窗口大小。
    maxOutputTokens: int  # 单次输出最大 token 数。
    metadata: dict[str, Any] = field(default_factory=dict)  # 模型附加元信息。
    supports_reasoning_levels: tuple[ReasoningLevel, ...] = ("off", "minimal", "low", "medium", "high")  # 支持的 reasoning 等级。
    supports_images: bool = False  # 是否支持图片输入/输出相关能力。
    supports_prompt_cache: bool = False  # 是否支持 prompt cache。
    supports_session: bool = False  # 是否支持 provider 原生会话。
    providerConfig: dict[str, Any] = field(default_factory=dict)  # provider 专属配置。

    def clamp_reasoning(self, reasoning: ReasoningLevel | None) -> ReasoningLevel | None:
        """按模型能力收敛 reasoning 级别。"""

        if reasoning is None:
            return None
        if reasoning in self.supports_reasoning_levels:
            return reasoning
        if not self.supports_reasoning_levels:
            return None

        requested_index = _REASONING_ORDER.index(reasoning)
        supported_indexes = [_REASONING_ORDER.index(level) for level in self.supports_reasoning_levels]
        lower_or_equal = [index for index in supported_indexes if index <= requested_index]
        if lower_or_equal:
            clamped_index = max(lower_or_equal)
            return _REASONING_ORDER[clamped_index]
        return self.supports_reasoning_levels[0]


@dataclass(slots=True)
class Context:
    """描述一次模型调用的统一上下文。"""

    systemPrompt: str | None = None  # 系统提示词。
    messages: list[ConversationMessage | dict[str, Any]] = field(default_factory=list)  # 对话消息历史。
    tools: list[Tool | dict[str, Any]] = field(default_factory=list)  # 可用工具列表。

    def __post_init__(self) -> None:
        self.messages = [ensure_message(item) for item in self.messages]
        self.tools = [ensure_tool(item) for item in self.tools]


@dataclass(slots=True)
class StreamEvent:
    """定义统一流式事件对象。"""

    type: StreamEventType  # 事件类型。
    lifecycle: StreamLifecycle | None = None  # 事件在当前流项中的生命周期阶段。
    itemType: StreamItemType | None = None  # 事件对应的流项种类。
    messageId: str | None = None  # 所属消息 ID。
    model: Model | None = None  # 产生该事件的模型。
    provider: str | None = None  # 产生该事件的 provider。
    text: str | None = None  # 文本内容或文本更新。
    thinking: str | None = None  # thinking 内容或更新。
    delta: str | None = None  # 通用增量字段。
    toolCallId: str | None = None  # 工具调用 ID。
    toolName: str | None = None  # 工具名称。
    argumentsDelta: str | None = None  # 工具参数流式增量。
    arguments: Any | None = None  # 工具参数完整值。
    assistantMessage: AssistantMessage | None = None  # done 事件中的完整 assistant 消息。
    toolResultMessage: ToolResultMessage | None = None  # 工具结果消息对象。
    usage: dict[str, Any] | None = None  # token 用量等统计。
    stopReason: str | None = None  # 终止原因。
    responseId: str | None = None  # provider 响应 ID。
    error: str | None = None  # 错误文本。
    details: Any | None = None  # 错误或事件附加细节。
    metadata: dict[str, Any] = field(default_factory=dict)  # 通用元信息。
    providerMetadata: dict[str, Any] = field(default_factory=dict)  # provider 专属元信息。
    rawEvent: Any | None = None  # 原始 provider 事件对象。
    timestamp: int = field(default_factory=_now_ms)  # 事件时间戳（毫秒）。

    def __post_init__(self) -> None:
        alias = _LEGACY_STREAM_EVENT_ALIASES.get(self.type)
        if alias is not None:
            lifecycle, item_type = alias
            if self.lifecycle is None:
                self.lifecycle = lifecycle
            if self.itemType is None:
                self.itemType = item_type
        else:
            if self.lifecycle is None:
                self.lifecycle = self.type if self.type in {"start", "update", "done", "error"} else "update"
            if self.itemType is None:
                if self.toolResultMessage is not None:
                    self.itemType = "tool_result"
                elif self.toolCallId is not None:
                    self.itemType = "tool_call"
                elif self.thinking is not None:
                    self.itemType = "thinking"
                elif self.text is not None:
                    self.itemType = "text"
                else:
                    self.itemType = "message"
        if self.delta is None:
            if self.argumentsDelta is not None:
                self.delta = self.argumentsDelta
            elif self.text is not None and self.lifecycle == "update":
                self.delta = self.text
            elif self.thinking is not None and self.lifecycle == "update":
                self.delta = self.thinking

    @property
    def is_terminal(self) -> bool:
        """返回当前事件是否是流的终止事件。"""

        return self.lifecycle in {"done", "error"} and self.itemType == "message"


def normalize_user_content_blocks(value: Any) -> list[UserContentBlock]:
    """归一化用户消息内容块。"""

    if isinstance(value, ContentBlocks):
        return [ensure_user_content_block(item) for item in value]
    if isinstance(value, list):
        return [ensure_user_content_block(item) for item in value]
    if value in (None, ""):
        return []
    return [TextContent(text=str(value))]


def normalize_assistant_content_blocks(
    value: Any,
    *,
    thinking: str = "",
    tool_calls: list[ToolCall | dict[str, Any]] | None = None,
) -> list[AssistantContentBlock]:
    """归一化 assistant 内容块。"""

    if isinstance(value, ContentBlocks):
        blocks = [ensure_assistant_content_block(item) for item in value]
    elif isinstance(value, list):
        blocks = [ensure_assistant_content_block(item) for item in value]
    elif value in (None, ""):
        blocks = []
    else:
        blocks = [TextContent(text=str(value))]

    if thinking:
        blocks.append(ThinkingContent(thinking=thinking))
    for item in tool_calls or []:
        tool_call = ensure_tool_call(item)
        blocks.append(
            ToolCallContent(
                id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                metadata=dict(tool_call.metadata),
            )
        )
    return blocks


def normalize_tool_result_content_blocks(value: Any) -> list[ToolResultContentBlock]:
    """归一化工具结果内容块。"""

    if isinstance(value, ContentBlocks):
        return [ensure_tool_result_content_block(item) for item in value]
    if isinstance(value, list):
        return [ensure_tool_result_content_block(item) for item in value]
    if value in (None, ""):
        return []
    return [TextContent(text=str(value))]


def extract_text_from_blocks(blocks: Iterable[Any]) -> str:
    """从内容块中提取纯文本。"""

    parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolResultContent) and block.text:
            parts.append(block.text)
    return "".join(parts)


def extract_thinking_from_blocks(blocks: Iterable[Any]) -> str:
    """从内容块中提取思考文本。"""

    return "".join(block.thinking for block in blocks if isinstance(block, ThinkingContent))


def extract_tool_calls_from_blocks(blocks: Iterable[Any]) -> list[ToolCall]:
    """从内容块中提取工具调用。"""

    tool_calls: list[ToolCall] = []
    for block in blocks:
        if isinstance(block, ToolCallContent):
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.arguments,
                    metadata=dict(block.metadata),
                )
            )
    return tool_calls


def ensure_message(value: ConversationMessage | dict[str, Any]) -> ConversationMessage:
    """将输入值规范化为统一消息对象。"""

    if isinstance(value, (UserMessage, AssistantMessage, ToolResultMessage)):
        return value
    if not isinstance(value, dict):
        raise TypeError("messages entries must be a message object or dict")

    role = value.get("role")
    if role == "user":
        blocks = value.get("content")
        if "contentBlocks" in value:
            blocks = value["contentBlocks"]
        return UserMessage(
            content=blocks,
            metadata=dict(value.get("metadata", {})),
            timestamp=int(value.get("timestamp", _now_ms())),
        )
    if role == "assistant":
        blocks = value.get("content")
        if "contentBlocks" in value:
            blocks = value["contentBlocks"]
        return AssistantMessage(
            content=blocks,
            thinking=value.get("thinking"),
            toolCalls=value.get("toolCalls"),
            metadata=dict(value.get("metadata", {})),
            usage=value.get("usage"),
            stopReason=value.get("stopReason"),
            responseId=value.get("responseId"),
            errorMessage=value.get("errorMessage"),
            timestamp=int(value.get("timestamp", _now_ms())),
        )
    if role in {"tool", "toolResult"}:
        blocks = value.get("content")
        if "contentBlocks" in value:
            blocks = value["contentBlocks"]
        return ToolResultMessage(
            toolCallId=str(value.get("toolCallId", "")),
            toolName=str(value.get("toolName", "")),
            content=blocks,
            metadata=dict(value.get("metadata", {})),
            isError=bool(value.get("isError", value.get("metadata", {}).get("error", False))),
            details=value.get("details"),
            timestamp=int(value.get("timestamp", _now_ms())),
        )
    raise ValueError("message role must be one of: user, assistant, tool")


def ensure_user_content_block(value: UserContentBlock | dict[str, Any] | Any) -> UserContentBlock:
    """将输入值规范化为用户内容块。"""

    if isinstance(value, (TextContent, ImageContent)):
        return value
    if not isinstance(value, dict):
        return TextContent(text=str(value))
    if value.get("type") == "image":
        return ImageContent(
            data=str(value.get("data", "")),
            mimeType=str(value.get("mimeType", "image/png")),
            detail=value.get("detail"),
            metadata=dict(value.get("metadata", {})),
        )
    return TextContent(text=str(value.get("text", value.get("content", ""))))


def ensure_assistant_content_block(value: AssistantContentBlock | dict[str, Any] | Any) -> AssistantContentBlock:
    """将输入值规范化为 assistant 内容块。"""

    if isinstance(value, (TextContent, ThinkingContent, ImageContent, ToolCallContent)):
        return value
    if isinstance(value, ToolCall):
        return ToolCallContent(id=value.id, name=value.name, arguments=value.arguments, metadata=dict(value.metadata))
    if not isinstance(value, dict):
        return TextContent(text=str(value))

    block_type = value.get("type")
    if block_type == "thinking":
        return ThinkingContent(thinking=str(value.get("thinking", "")))
    if block_type == "image":
        return ImageContent(
            data=str(value.get("data", "")),
            mimeType=str(value.get("mimeType", "image/png")),
            detail=value.get("detail"),
            metadata=dict(value.get("metadata", {})),
        )
    if block_type in {"tool_call", "toolCall"}:
        return ToolCallContent(
            id=str(value.get("id", "")),
            name=str(value.get("name", "")),
            arguments=value.get("arguments", {}),
            metadata=dict(value.get("metadata", {})),
        )
    return TextContent(text=str(value.get("text", value.get("content", ""))))


def ensure_tool_result_content_block(value: ToolResultContentBlock | dict[str, Any] | Any) -> ToolResultContentBlock:
    """将输入值规范化为工具结果内容块。"""

    if isinstance(value, (TextContent, ImageContent, ToolResultContent)):
        return value
    if not isinstance(value, dict):
        return TextContent(text=str(value))
    block_type = value.get("type")
    if block_type == "image":
        return ImageContent(
            data=str(value.get("data", "")),
            mimeType=str(value.get("mimeType", "image/png")),
            detail=value.get("detail"),
            metadata=dict(value.get("metadata", {})),
        )
    if block_type == "tool_result_content":
        return ToolResultContent(
            text=str(value.get("text", "")),
            data=value.get("data"),
            mimeType=value.get("mimeType"),
            metadata=dict(value.get("metadata", {})),
        )
    return TextContent(text=str(value.get("text", value.get("content", ""))))


def ensure_tool(value: Tool | dict[str, Any]) -> Tool:
    """将输入值规范化为工具定义。"""

    if isinstance(value, Tool):
        return value
    if not isinstance(value, dict):
        raise TypeError("tools entries must be Tool or dict")
    return Tool(
        name=str(value.get("name", "")),
        description=value.get("description"),
        inputSchema=dict(value.get("inputSchema", value.get("input_schema", {}))),
        metadata=dict(value.get("metadata", {})),
        renderMetadata=dict(value.get("renderMetadata", value.get("render_metadata", {}))),
    )


def ensure_tool_call(value: ToolCall | dict[str, Any]) -> ToolCall:
    """将输入值规范化为工具调用对象。"""

    if isinstance(value, ToolCall):
        return value
    if not isinstance(value, dict):
        raise TypeError("toolCalls entries must be ToolCall or dict")
    return ToolCall(
        id=str(value.get("id", "")),
        name=str(value.get("name", "")),
        arguments=value.get("arguments", {}),
        metadata=dict(value.get("metadata", {})),
    )


def ensure_context(value: Context | dict[str, Any]) -> Context:
    """将输入值规范化为统一的 Context 对象。"""

    if isinstance(value, Context):
        return value
    if not isinstance(value, dict):
        raise TypeError("context must be Context or dict")
    return Context(
        systemPrompt=value.get("systemPrompt"),
        messages=[ensure_message(item) for item in value.get("messages", [])],
        tools=[ensure_tool(item) for item in value.get("tools", [])],
    )


def ensure_model(value: Model | str) -> Model:
    """将模型对象或模型 ID 规范化为 `Model`。"""

    if isinstance(value, Model):
        return value
    if not isinstance(value, str):
        raise TypeError("model must be a Model or model id string")
    from .models import get_model

    return get_model(value)
