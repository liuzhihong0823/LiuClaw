from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

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
ContentBlockType = Literal["text", "thinking", "image", "tool_call", "tool_result_content"]
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


@dataclass(slots=True)
class TextContent:
    """统一文本内容块。"""

    type: Literal["text"] = "text"
    text: str = ""


@dataclass(slots=True)
class ThinkingContent:
    """统一思考内容块。"""

    type: Literal["thinking"] = "thinking"
    thinking: str = ""


@dataclass(slots=True)
class ImageContent:
    """统一图片内容块。"""

    type: Literal["image"] = "image"
    data: str = ""
    mimeType: str = "image/png"


@dataclass(slots=True)
class ToolResultContent:
    """统一工具结果内容块。"""

    type: Literal["tool_result_content"] = "tool_result_content"
    text: str = ""
    data: str | None = None
    mimeType: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


MessageContentBlock: TypeAlias = TextContent | ThinkingContent | ImageContent | ToolResultContent


@dataclass(slots=True)
class Tool:
    """定义可供模型调用的工具。"""

    name: str
    description: str | None = None
    inputSchema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    renderMetadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    """表示 assistant 发起的一次工具调用。"""

    id: str
    name: str
    arguments: Any = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def arguments_text(self) -> str:
        """返回兼容旧实现的字符串参数。"""

        return serialize_tool_arguments(self.arguments)


@dataclass(slots=True)
class ToolCallContent:
    """统一工具调用内容块。"""

    type: Literal["tool_call"] = "tool_call"
    id: str = ""
    name: str = ""
    arguments: Any = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def arguments_text(self) -> str:
        """返回兼容旧实现的字符串参数。"""

        return serialize_tool_arguments(self.arguments)


AssistantContentBlock: TypeAlias = TextContent | ThinkingContent | ImageContent | ToolCallContent
UserContentBlock: TypeAlias = TextContent | ImageContent
ToolResultContentBlock: TypeAlias = TextContent | ImageContent | ToolResultContent


@dataclass(slots=True)
class UserMessage:
    """表示用户输入消息。"""

    role: Literal["user"] = "user"
    content: Any = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    contentBlocks: list[UserContentBlock] = field(default_factory=list)
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if self.contentBlocks:
            self.content = extract_text_from_blocks(self.contentBlocks)
        else:
            self.contentBlocks = normalize_user_content_blocks(self.content)
            self.content = extract_text_from_blocks(self.contentBlocks)


@dataclass(slots=True)
class AssistantMessage:
    """表示 assistant 消息，也是 `complete()` 的最终结果对象。"""

    role: Literal["assistant"] = "assistant"
    content: Any = ""
    thinking: str = ""
    toolCalls: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    contentBlocks: list[AssistantContentBlock] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    stopReason: str | None = None
    responseId: str | None = None
    errorMessage: str | None = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        self.toolCalls = [ensure_tool_call(item) for item in self.toolCalls]
        if self.contentBlocks:
            self.content = extract_text_from_blocks(self.contentBlocks)
            self.thinking = extract_thinking_from_blocks(self.contentBlocks) or self.thinking
            self.toolCalls = extract_tool_calls_from_blocks(self.contentBlocks) or self.toolCalls
        else:
            self.contentBlocks = normalize_assistant_content_blocks(
                self.content,
                thinking=self.thinking,
                tool_calls=self.toolCalls,
            )
            self.content = extract_text_from_blocks(self.contentBlocks)
            self.thinking = extract_thinking_from_blocks(self.contentBlocks)

    @property
    def text(self) -> str:
        """返回 assistant 的最终文本内容。"""

        return self.content


@dataclass(slots=True)
class ToolResultMessage:
    """表示工具执行结果消息，供上层回填给模型。"""

    role: Literal["tool"] = "tool"
    toolCallId: str = ""
    toolName: str = ""
    content: Any = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    contentBlocks: list[ToolResultContentBlock] = field(default_factory=list)
    isError: bool = False
    details: Any | None = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def __post_init__(self) -> None:
        if self.contentBlocks:
            self.content = extract_text_from_blocks(self.contentBlocks)
        else:
            self.contentBlocks = normalize_tool_result_content_blocks(self.content)
            self.content = extract_text_from_blocks(self.contentBlocks)


ConversationMessage: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage


@dataclass(slots=True)
class Model:
    """描述一个可供统一接入层使用的模型元数据。"""

    id: str
    provider: str
    inputPrice: float
    outputPrice: float
    contextWindow: int
    maxOutputTokens: int
    metadata: dict[str, Any] = field(default_factory=dict)
    supports_reasoning_levels: tuple[ReasoningLevel, ...] = ("off", "minimal", "low", "medium", "high")
    supports_images: bool = False
    supports_prompt_cache: bool = False
    supports_session: bool = False
    providerConfig: dict[str, Any] = field(default_factory=dict)

    def clamp_reasoning(self, reasoning: ReasoningLevel | None) -> ReasoningLevel | None:
        """按模型能力收敛 reasoning 级别。"""

        if reasoning is None:
            return None
        if reasoning in self.supports_reasoning_levels:
            return reasoning
        if not self.supports_reasoning_levels:
            return None
        return self.supports_reasoning_levels[-1]


@dataclass(slots=True)
class Context:
    """描述一次模型调用的统一上下文。"""

    systemPrompt: str | None = None
    messages: list[ConversationMessage | dict[str, Any]] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.messages = [ensure_message(item) for item in self.messages]
        self.tools = [ensure_tool(item) for item in self.tools]


@dataclass(slots=True)
class StreamEvent:
    """定义统一流式事件对象。"""

    type: StreamEventType
    lifecycle: StreamLifecycle | None = None
    itemType: StreamItemType | None = None
    messageId: str | None = None
    model: Model | None = None
    provider: str | None = None
    text: str | None = None
    thinking: str | None = None
    delta: str | None = None
    toolCallId: str | None = None
    toolName: str | None = None
    argumentsDelta: str | None = None
    arguments: Any | None = None
    assistantMessage: AssistantMessage | None = None
    toolResultMessage: ToolResultMessage | None = None
    usage: dict[str, Any] | None = None
    stopReason: str | None = None
    responseId: str | None = None
    error: str | None = None
    details: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    providerMetadata: dict[str, Any] = field(default_factory=dict)
    rawEvent: Any | None = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

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


def serialize_tool_arguments(arguments: Any) -> str:
    """把工具参数稳定序列化为字符串。"""

    if isinstance(arguments, str):
        return arguments
    if arguments in (None, ""):
        return ""
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True)


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


def normalize_user_content_blocks(value: Any) -> list[UserContentBlock]:
    """归一化用户消息内容块。"""

    blocks: list[UserContentBlock] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (TextContent, ImageContent)):
                blocks.append(item)
            elif isinstance(item, dict) and item.get("type") == "image":
                blocks.append(ImageContent(data=str(item.get("data", "")), mimeType=str(item.get("mimeType", "image/png"))))
            elif isinstance(item, dict):
                blocks.append(TextContent(text=str(item.get("text", item.get("content", "")))))
            else:
                blocks.append(TextContent(text=str(item)))
        return blocks
    if value not in (None, ""):
        blocks.append(TextContent(text=str(value)))
    return blocks


def normalize_assistant_content_blocks(
    value: Any,
    *,
    thinking: str = "",
    tool_calls: list[ToolCall] | None = None,
) -> list[AssistantContentBlock]:
    """归一化 assistant 内容块。"""

    blocks: list[AssistantContentBlock] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (TextContent, ThinkingContent, ImageContent, ToolCallContent)):
                blocks.append(item)
            elif isinstance(item, ToolCall):
                blocks.append(ToolCallContent(id=item.id, name=item.name, arguments=item.arguments, metadata=dict(item.metadata)))
            elif isinstance(item, dict):
                block_type = item.get("type")
                if block_type == "thinking":
                    blocks.append(ThinkingContent(thinking=str(item.get("thinking", ""))))
                elif block_type == "image":
                    blocks.append(ImageContent(data=str(item.get("data", "")), mimeType=str(item.get("mimeType", "image/png"))))
                elif block_type in {"tool_call", "toolCall"}:
                    blocks.append(
                        ToolCallContent(
                            id=str(item.get("id", "")),
                            name=str(item.get("name", "")),
                            arguments=parse_tool_arguments(item.get("arguments", {})),
                            metadata=dict(item.get("metadata", {})),
                        )
                    )
                else:
                    blocks.append(TextContent(text=str(item.get("text", item.get("content", "")))))
            else:
                blocks.append(TextContent(text=str(item)))
    elif value not in (None, ""):
        blocks.append(TextContent(text=str(value)))

    if thinking:
        blocks.append(ThinkingContent(thinking=thinking))
    for tool_call in tool_calls or []:
        normalized_tool_call = ensure_tool_call(tool_call)
        blocks.append(
            ToolCallContent(
                id=normalized_tool_call.id,
                name=normalized_tool_call.name,
                arguments=normalized_tool_call.arguments,
                metadata=dict(normalized_tool_call.metadata),
            )
        )
    return blocks


def normalize_tool_result_content_blocks(value: Any) -> list[ToolResultContentBlock]:
    """归一化工具结果内容块。"""

    blocks: list[ToolResultContentBlock] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (TextContent, ImageContent, ToolResultContent)):
                blocks.append(item)
            elif isinstance(item, dict) and item.get("type") == "image":
                blocks.append(ImageContent(data=str(item.get("data", "")), mimeType=str(item.get("mimeType", "image/png"))))
            elif isinstance(item, dict) and item.get("type") == "tool_result_content":
                blocks.append(
                    ToolResultContent(
                        text=str(item.get("text", "")),
                        data=item.get("data"),
                        mimeType=item.get("mimeType"),
                        metadata=dict(item.get("metadata", {})),
                    )
                )
            elif isinstance(item, dict):
                blocks.append(TextContent(text=str(item.get("text", item.get("content", "")))))
            else:
                blocks.append(TextContent(text=str(item)))
        return blocks
    if value not in (None, ""):
        blocks.append(TextContent(text=str(value)))
    return blocks


def extract_text_from_blocks(blocks: list[Any]) -> str:
    """从内容块中提取文本。"""

    parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolResultContent) and block.text:
            parts.append(block.text)
    return "".join(parts)


def extract_thinking_from_blocks(blocks: list[Any]) -> str:
    """从内容块中提取思考文本。"""

    return "".join(block.thinking for block in blocks if isinstance(block, ThinkingContent))


def extract_tool_calls_from_blocks(blocks: list[Any]) -> list[ToolCall]:
    """从内容块中提取工具调用。"""

    tool_calls: list[ToolCall] = []
    for block in blocks:
        if isinstance(block, ToolCallContent):
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.arguments, metadata=dict(block.metadata)))
    return tool_calls


def rebuild_assistant_content_blocks(message: AssistantMessage) -> list[AssistantContentBlock]:
    """从旧字段重建 assistant 内容块。"""

    return normalize_assistant_content_blocks(message.content, thinking=message.thinking, tool_calls=message.toolCalls)


def rebuild_user_content_blocks(message: UserMessage) -> list[UserContentBlock]:
    """从旧字段重建 user 内容块。"""

    return normalize_user_content_blocks(message.content)


def rebuild_tool_result_content_blocks(message: ToolResultMessage) -> list[ToolResultContentBlock]:
    """从旧字段重建 tool result 内容块。"""

    return normalize_tool_result_content_blocks(message.content)


def ensure_message(value: ConversationMessage | dict[str, Any]) -> ConversationMessage:
    """将输入值规范化为新的消息类型。"""

    if isinstance(value, (UserMessage, AssistantMessage, ToolResultMessage)):
        return value
    if not isinstance(value, dict):
        raise TypeError("messages entries must be a message object or dict")

    role = value.get("role")
    if role == "user":
        return UserMessage(
            content=value.get("content", ""),
            metadata=dict(value.get("metadata", {})),
            contentBlocks=[ensure_content_block(item) for item in value.get("contentBlocks", [])],
            timestamp=int(value.get("timestamp", int(time.time() * 1000))),
        )
    if role == "assistant":
        tool_calls = [ensure_tool_call(item) for item in value.get("toolCalls", [])]
        return AssistantMessage(
            content=value.get("content", ""),
            thinking=str(value.get("thinking", "")),
            toolCalls=tool_calls,
            metadata=dict(value.get("metadata", {})),
            contentBlocks=[ensure_assistant_content_block(item) for item in value.get("contentBlocks", [])],
            usage=value.get("usage"),
            stopReason=value.get("stopReason"),
            responseId=value.get("responseId"),
            errorMessage=value.get("errorMessage"),
            timestamp=int(value.get("timestamp", int(time.time() * 1000))),
        )
    if role in {"tool", "toolResult"}:
        return ToolResultMessage(
            toolCallId=str(value.get("toolCallId", "")),
            toolName=str(value.get("toolName", "")),
            content=value.get("content", ""),
            metadata=dict(value.get("metadata", {})),
            contentBlocks=[ensure_tool_result_content_block(item) for item in value.get("contentBlocks", [])],
            isError=bool(value.get("isError", value.get("metadata", {}).get("error", False))),
            details=value.get("details"),
            timestamp=int(value.get("timestamp", int(time.time() * 1000))),
        )
    raise ValueError("message role must be one of: user, assistant, tool")


def ensure_content_block(value: MessageContentBlock | dict[str, Any]) -> MessageContentBlock:
    """将输入值规范化为内容块。"""

    if isinstance(value, (TextContent, ThinkingContent, ImageContent, ToolResultContent)):
        return value
    if not isinstance(value, dict):
        return TextContent(text=str(value))
    block_type = value.get("type")
    if block_type == "thinking":
        return ThinkingContent(thinking=str(value.get("thinking", "")))
    if block_type == "image":
        return ImageContent(data=str(value.get("data", "")), mimeType=str(value.get("mimeType", "image/png")))
    if block_type == "tool_result_content":
        return ToolResultContent(
            text=str(value.get("text", "")),
            data=value.get("data"),
            mimeType=value.get("mimeType"),
            metadata=dict(value.get("metadata", {})),
        )
    return TextContent(text=str(value.get("text", value.get("content", ""))))


def ensure_assistant_content_block(value: AssistantContentBlock | dict[str, Any]) -> AssistantContentBlock:
    """将输入值规范化为 assistant 内容块。"""

    if isinstance(value, ToolCallContent):
        return value
    if isinstance(value, (TextContent, ThinkingContent, ImageContent)):
        return value
    if not isinstance(value, dict):
        return TextContent(text=str(value))
    if value.get("type") in {"tool_call", "toolCall"}:
        return ToolCallContent(
            id=str(value.get("id", "")),
            name=str(value.get("name", "")),
            arguments=parse_tool_arguments(value.get("arguments", {})),
            metadata=dict(value.get("metadata", {})),
        )
    return ensure_content_block(value)


def ensure_tool_result_content_block(value: ToolResultContentBlock | dict[str, Any]) -> ToolResultContentBlock:
    """将输入值规范化为 tool result 内容块。"""

    block = ensure_content_block(value)
    if isinstance(block, ThinkingContent):
        return TextContent(text=block.thinking)
    return block


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
        arguments=parse_tool_arguments(value.get("arguments", "")),
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
