from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from ai import AssistantMessage, Context, Model, Tool, ToolCall, ToolResultMessage, UserMessage
from ai.options import Options
from ai.providers.zhipu import ZhipuProvider


MODEL = Model(
    id="zhipu:glm-4.7",
    provider="zhipu",
    inputPrice=0.0,
    outputPrice=0.0,
    contextWindow=200000,
    maxOutputTokens=128000,
)


def test_zhipu_build_request_maps_messages_tools_and_reasoning() -> None:
    provider = ZhipuProvider()
    context = Context(
        systemPrompt="你是测试助手。",
        messages=[
            UserMessage(content="你好"),
            AssistantMessage(
                content="先查一下",
                thinking="我需要调用工具",
                toolCalls=[ToolCall(id="call_1", name="lookup", arguments='{"q":"glm"}')],
            ),
            ToolResultMessage(toolCallId="call_1", toolName="lookup", content='{"answer":"ok"}'),
        ],
        tools=[
            Tool(
                name="lookup",
                description="查询资料",
                inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ],
    )
    options = Options(
        maxTokens=2048,
        temperature=0.3,
        metadata={"_providerReasoning": {"thinking": {"type": "enabled"}, "clear_thinking": False}},
    )

    request = provider._build_request(MODEL, context, options)

    assert request["model"] == "glm-4.7"
    assert request["stream"] is True
    assert request["tool_stream"] is True
    assert request["thinking"] == {"type": "enabled"}
    assert request["clear_thinking"] is False
    assert request["messages"][0] == {"role": "system", "content": "你是测试助手。"}
    assert request["messages"][2]["reasoning_content"] == "我需要调用工具"
    assert request["messages"][2]["tool_calls"][0]["function"]["name"] == "lookup"
    assert request["messages"][3]["role"] == "tool"
    assert request["messages"][3]["tool_call_id"] == "call_1"
    assert request["tools"][0]["function"]["parameters"]["type"] == "object"


@pytest.mark.asyncio
async def test_zhipu_stream_maps_text_thinking_and_toolcalls() -> None:
    provider = ZhipuProvider()
    context = Context(messages=[UserMessage(content="你好")], tools=[Tool(name="lookup", inputSchema={"type": "object"})])
    options = Options()

    async def fake_chunks(*args, **kwargs) -> AsyncIterator[dict]:
        yield {
            "id": "resp_1",
            "choices": [
                {
                    "delta": {"reasoning_content": "先分析"},
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": "resp_1",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"q":"g'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": "resp_1",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"arguments": 'lm"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        yield {
            "id": "resp_1",
            "choices": [
                {
                    "delta": {"content": "最终答案"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        }

    provider._iter_sse_chunks = fake_chunks  # type: ignore[method-assign]

    events = [event async for event in provider.stream(MODEL, context, options)]

    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "thinking_end",
        "toolcall_end",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]
    assert events[-1].assistantMessage is not None
    assert events[-1].assistantMessage.content == "最终答案"
    assert events[-1].assistantMessage.thinking == "先分析"
    assert events[-1].assistantMessage.toolCalls[0].arguments == '{"q":"glm"}'
    assert events[-1].usage == {"prompt_tokens": 12, "completion_tokens": 6}
    assert events[-1].responseId == "resp_1"
    assert events[-1].providerMetadata["request_model"] == "glm-4.7"
