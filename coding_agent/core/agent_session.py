from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from ai.registry import ProviderRegistry
from ai.types import AssistantMessage, ConversationMessage, ToolResultMessage, UserMessage
from agent_core import (
    AfterToolCallContext,
    AfterToolCallPass,
    Agent,
    AgentEvent,
    AgentLoopConfig,
    AgentOptions,
    BeforeToolCallAllow,
    BeforeToolCallContext,
)

from .compaction import CompactionCoordinator
from .resource_loader import ResourceLoader
from .runtime_assembly import SessionRuntimeAssembly, assemble_session_runtime, build_runtime_context_messages, build_session_context
from .session_manager import SessionManager
from .types import CodingAgentSettings, ControlMessage, SessionEvent


class AgentSession:
    """面向产品层的会话编排对象。"""

    def __init__(
        self,
        *,
        workspace_root: Path,
        cwd: Path,
        model,
        thinking: str | None,
        settings: CodingAgentSettings,
        session_manager: SessionManager,
        resource_loader: ResourceLoader,
        session_id: str | None = None,
        branch_id: str = "main",
        stream_fn=None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        """初始化一次 coding-agent 会话运行时。"""

        self.workspace_root = workspace_root  # 工作区根目录。
        self.cwd = cwd  # 当前会话目录。
        self.model = model  # 当前模型对象。
        self.thinking = thinking  # 当前思考等级。
        self.settings = settings  # 会话生效设置。
        self.session_manager = session_manager  # 会话持久化管理器。
        self.resource_loader = resource_loader  # 资源加载器。
        self.session_id = session_id  # 当前会话 ID。
        self.branch_id = branch_id  # 当前分支 ID。
        self.last_node_id: str | None = None  # 最近写入的会话节点 ID。
        self._event_message_counter = 0  # 生成事件消息 ID 的计数器。
        self._current_assistant_message_id = ""  # 当前 assistant 消息 ID。
        self._turn_counter = 0  # turn 计数器。
        self._current_turn_id = ""  # 当前处理中的 turn ID。
        self._pending_control_messages: list[ControlMessage] = []  # 待注入的控制消息。
        self._tool_activity_in_run = False  # 本轮是否发生过工具调用。
        self._follow_up_sent = False  # 本轮是否已经发出 follow-up。
        self.runtime = self._assemble_runtime(provider_registry=registry)  # 当前 session 装配出的运行时组件。
        self.compaction: CompactionCoordinator = self.runtime.compaction  # 上下文压缩协调器。
        self._stream_fn = stream_fn  # 自定义底层流式函数。
        self._agent = self._build_agent()  # 底层高层 Agent 封装对象。
        if self.session_id is None:
            snapshot = self.session_manager.create_session(cwd=cwd, model_id=model.id)
            self.session_id = snapshot.session_id
            self.branch_id = snapshot.branch_id

    def _assemble_runtime(self, provider_registry: ProviderRegistry | None = None) -> SessionRuntimeAssembly:
        """装配当前 session 运行时依赖。"""

        return assemble_session_runtime(
            workspace_root=self.workspace_root,
            cwd=self.cwd,
            model=self.model,
            thinking=self.thinking,
            settings=self.settings,
            session_manager=self.session_manager,
            resource_loader=self.resource_loader,
            provider_registry=provider_registry,
        )

    def _build_agent(self) -> Agent:
        """构造底层 agent_core.Agent。"""

        loop = AgentLoopConfig(
            model=self.model,
            thinking=self.thinking,
            systemPrompt=self._build_system_prompt(),
            tools=self.runtime.tools,
            stream=self._stream_fn,
            convert_to_llm=self._convert_to_llm,
            steer=self._steer,
            followUp=self._follow_up,
            beforeToolCall=self._before_tool_call,
            afterToolCall=self._after_tool_call,
            registry=self.runtime.provider_registry,
        )
        return Agent(AgentOptions(loop=loop, listeners=self.runtime.listeners))

    def _build_system_prompt(self) -> str:
        """根据当前模型、目录、资源和工具构建系统提示。"""

        context = build_session_context(
            workspace_root=self.workspace_root,
            cwd=self.cwd,
            model=self.model,
            thinking=self.thinking,
            settings=self.settings,
            resources=self.runtime.resources,
            tool_registry=self.runtime.tool_registry,
        )
        return self.runtime.prompt_builder.build(context)

    @property
    def resources(self):
        """兼容旧实现读取当前资源包。"""

        return self.runtime.resources

    def _convert_to_llm(self, messages: list[Any], state) -> list[ConversationMessage]:
        """把 runtime 消息列表转换成真正送给模型的对话上下文。"""

        llm_messages: list[ConversationMessage] = []
        for message in messages:
            if isinstance(message, ControlMessage):
                metadata = {"control": True, "kind": message.kind, **dict(message.metadata)}
                if message.kind == "follow_up":
                    metadata["follow_up"] = True
                if message.kind == "steering":
                    metadata["steering"] = True
                llm_messages.append(UserMessage(content=message.content, metadata=metadata))
                continue
            if isinstance(message, (AssistantMessage, ToolResultMessage, UserMessage)):
                llm_messages.append(message)
        return llm_messages

    def send_user_message(self, content: str) -> None:
        """接收用户输入，写入会话存储并加入待发送队列。"""

        self._turn_counter += 1
        self._current_turn_id = f"turn-{self._turn_counter}"
        message = UserMessage(content=content)
        self.session_manager.append_message(
            self.session_id,
            message=message,
            branch_id=self.branch_id,
            parent_id=self.last_node_id,
        )
        self.last_node_id = self.session_manager.load_session(self.session_id).nodes[-1].id
        self._agent.enqueue(message)

    @property
    def current_turn_id(self) -> str:
        """返回当前正在处理的用户轮次 ID。"""

        return self._current_turn_id

    def resume_session(self) -> None:
        """从持久化会话恢复历史上下文并同步到底层 Agent。"""

        history = self.session_manager.build_context_messages(self.session_id, self.branch_id)
        self._agent.setState(
            replace(
                self._agent.getState(),
                history=history,
                systemPrompt=self._build_system_prompt(),
                model=self.model,
                thinking=self.thinking,
                tools=self.runtime.tools,
            )
        )
        snapshot = self.session_manager.load_session(self.session_id)
        if snapshot.nodes:
            self.last_node_id = snapshot.nodes[-1].id

    def switch_model(self, model) -> None:
        """切换当前会话使用的模型，并更新系统提示。"""

        self.model = model
        self.runtime = self._assemble_runtime(provider_registry=self.runtime.provider_registry)
        self.compaction = self.runtime.compaction
        self._agent.setModel(model)
        self._agent.setTools(self.runtime.tools)
        self._agent.setSystemPrompt(self._build_system_prompt())
        meta = self.session_manager._read_meta(self.session_id)
        meta["model_id"] = model.id
        self.session_manager._write_meta(self.session_id, meta)

    def set_thinking(self, thinking: str | None) -> None:
        """调整当前会话的思考等级，并更新系统提示。"""

        self.thinking = thinking
        self._agent.setThinking(thinking)
        self._agent.setSystemPrompt(self._build_system_prompt())

    def compact(self):
        """手动触发当前会话分支的上下文压缩。"""

        return self.compaction.compact_manual(self.session_id, self.branch_id)

    def cancel(self) -> None:
        """取消当前运行中的 agent 循环。"""

        self._agent.cancel()

    def list_recent_sessions(self, limit: int = 10) -> list[dict]:
        """列出当前工作区的最近会话。"""

        return self.session_manager.list_recent_sessions(limit=limit, cwd=self.cwd)

    def get_last_user_message(self) -> str | None:
        """返回当前会话最近一条真实用户输入。"""

        messages = self.session_manager.build_context_messages(self.session_id, self.branch_id)
        for message in reversed(messages):
            if isinstance(message, UserMessage):
                return message.content
        return None

    async def run_turn(self) -> AsyncIterator[SessionEvent]:
        """运行一轮会话，并把底层事件映射为 UI 事件流。"""

        self._pending_control_messages.clear()
        self._tool_activity_in_run = False
        self._follow_up_sent = False
        attempt = 0
        while True:
            try:
                async for event in self._run_turn_once():
                    overflow = self._extract_overflow_error(event)
                    if overflow and attempt == 0:
                        recovered = self.compaction.recover_from_overflow(self.session_id, self.branch_id)
                        if recovered is not None:
                            attempt += 1
                            self.resume_session()
                            break
                    yield event
                else:
                    return
            except Exception as exc:
                if attempt == 0 and self._is_context_overflow_error(str(exc)):
                    recovered = self.compaction.recover_from_overflow(self.session_id, self.branch_id)
                    if recovered is not None:
                        attempt += 1
                        self.resume_session()
                        continue
                raise

    async def _run_turn_once(self) -> AsyncIterator[SessionEvent]:
        """执行单次 turn，不包含 overflow 恢复重试。"""

        if not self._agent.pendingMessages and self._agent.state.history:
            session = await self._agent.continueConversation()
        else:
            self.resume_session()
            self._maybe_auto_compact()
            session = await self._agent.run()

        async for event in session.consume():
            for mapped in self._map_event(event):
                yield mapped
            if event.type == "message_end" and isinstance(event.message, AssistantMessage):
                self._persist_assistant(event.message)
            if event.type == "message_end" and isinstance(event.message, ControlMessage):
                self._persist_control_message(event.message)
            if event.type == "tool_execution_end" and event.toolResult is not None:
                self._persist_tool_result(event.toolResult)

    def _persist_assistant(self, message: AssistantMessage) -> None:
        """把 assistant 消息持久化为会话节点。"""

        node = self.session_manager.append_message(
            self.session_id,
            message=message,
            branch_id=self.branch_id,
            parent_id=self.last_node_id,
        )
        self.last_node_id = node.id

    def _persist_tool_result(self, message: ToolResultMessage) -> None:
        """把工具结果消息持久化为会话节点。"""

        node = self.session_manager.append_message(
            self.session_id,
            message=message,
            branch_id=self.branch_id,
            parent_id=self.last_node_id,
        )
        self.last_node_id = node.id

    def _persist_control_message(self, message: ControlMessage) -> None:
        """把 steering 或 follow-up 控制消息作为显式 control 事件持久化。"""

        node = self.session_manager.append_control(
            self.session_id,
            message=message,
            branch_id=self.branch_id,
            parent_id=self.last_node_id,
        )
        self.last_node_id = node.id

    def _map_event(self, event: AgentEvent) -> list[SessionEvent]:
        """把 `agent_core` 事件转换成交互层事件。"""

        events: list[SessionEvent] = []
        if event.type == "message_start":
            if isinstance(event.message, ControlMessage):
                return events
            if event.message is not None and not isinstance(event.message, AssistantMessage):
                return events
            self._event_message_counter += 1
            self._current_assistant_message_id = f"assistant-{self._event_message_counter}"
            return [
                SessionEvent(
                    type="message_start",
                    message="",
                    panel="main",
                    message_id=self._current_assistant_message_id,
                    turn_id=self._current_turn_id,
                    source="assistant",
                    render_group="assistant",
                    render_order=400,
                )
            ]
        if event.type == "message_update":
            return [
                SessionEvent(
                    type="message_delta",
                    delta=event.messageDelta or "",
                    panel="main",
                    is_transient=True,
                    message_id=self._current_assistant_message_id,
                    turn_id=self._current_turn_id,
                    source="assistant",
                    render_group="assistant",
                    render_order=400,
                )
            ]
        if event.type == "message_end" and isinstance(event.message, ControlMessage):
            source = "follow_up" if event.message.kind == "follow_up" else "steering"
            message = event.message.content if event.message.kind != "follow_up" else "正在基于工具结果继续整理最终答复"
            return [
                SessionEvent(
                    type="status",
                    message=message,
                    panel="status",
                    status_level="info",
                    turn_id=self._current_turn_id,
                    source=source,
                    render_group="pre_assistant",
                    render_order=320 if source == "follow_up" else 220,
                    payload={"source": source},
                )
            ]
        if event.type == "message_end" and isinstance(event.message, AssistantMessage):
            if event.message.thinking:
                events.append(
                    SessionEvent(
                        type="thinking",
                        message=event.message.thinking,
                        panel="thinking",
                        message_id=self._current_assistant_message_id,
                        turn_id=self._current_turn_id,
                        source="thinking",
                        render_group="thinking",
                        render_order=200,
                    )
                )
            events.append(
                SessionEvent(
                    type="message_end",
                    message=event.message.content,
                    panel="main",
                    message_id=self._current_assistant_message_id,
                    turn_id=self._current_turn_id,
                    source="assistant",
                    render_group="assistant",
                    render_order=400,
                )
            )
            return events
        if event.type == "tool_execution_start":
            return [
                SessionEvent(
                    type="tool_start",
                    tool_name=event.toolCall.name if event.toolCall else "",
                    panel="tool",
                    status_level="info",
                    tool_arguments=event.toolCall.arguments_text if event.toolCall else "",
                    message_id=event.toolCall.id if event.toolCall else "",
                    turn_id=self._current_turn_id,
                    source="tool",
                    render_group="pre_assistant",
                    render_order=250,
                )
            ]
        if event.type == "tool_execution_update":
            name = event.toolCall.name if event.toolCall else ""
            return [
                SessionEvent(
                    type="tool_update",
                    tool_name=name,
                    message=f"running {name}",
                    panel="tool",
                    status_level="info",
                    is_transient=True,
                    tool_arguments=event.toolCall.arguments_text if event.toolCall else "",
                    message_id=event.toolCall.id if event.toolCall else "",
                    turn_id=self._current_turn_id,
                    source="tool",
                    render_group="pre_assistant",
                    render_order=260,
                )
            ]
        if event.type == "tool_execution_end":
            tool_name = event.toolResult.toolName if event.toolResult else ""
            tool_message = event.toolResult.content if event.toolResult else ""
            return [
                SessionEvent(
                    type="tool_end",
                    tool_name=tool_name,
                    message=tool_message,
                    panel="tool",
                    status_level="error" if event.error else "success",
                    tool_output_preview=str(tool_message)[:300],
                    message_id=event.toolResult.toolCallId if event.toolResult else "",
                    turn_id=self._current_turn_id,
                    source="tool",
                    render_group="pre_assistant",
                    render_order=270,
                )
            ]
        if event.type == "agent_end" and event.error:
            return [
                SessionEvent(
                    type="error",
                    error=event.error,
                    message=event.error,
                    panel="error",
                    status_level="error",
                    turn_id=self._current_turn_id,
                    source="error",
                    render_group="error",
                    render_order=500,
                )
            ]
        return events

    def _maybe_auto_compact(self) -> None:
        """在发送请求前根据上下文大小决定是否自动压缩。"""

        context = build_runtime_context_messages(
            self.session_manager,
            self.session_id,
            self.branch_id,
            self.model,
            self._build_system_prompt(),
            self.runtime.tools,
        )
        self.compaction.maybe_compact_for_threshold(self.session_id, self.branch_id, self.model, context)

    async def _steer(self, state, signal=None) -> list[Any]:
        """在工具执行轮次之间注入 steering 控制消息。"""

        _ = state, signal
        messages = [replace(message) for message in self._pending_control_messages if message.kind == "steering"]
        self._pending_control_messages = [message for message in self._pending_control_messages if message.kind != "steering"]
        return messages

    async def _follow_up(self, state, signal=None) -> list[Any]:
        """在 AI 暂时完成工作后注入 follow-up 控制消息。"""

        _ = signal
        pending = [replace(message) for message in self._pending_control_messages if message.kind == "follow_up"]
        if pending:
            self._pending_control_messages = [message for message in self._pending_control_messages if message.kind != "follow_up"]
            self._follow_up_sent = True
            return pending
        if not self._tool_activity_in_run or self._follow_up_sent:
            return []
        last_message = state.messages[-1] if state.messages else None
        if not isinstance(last_message, AssistantMessage) or last_message.toolCalls:
            return []
        self._follow_up_sent = True
        return [
            ControlMessage(
                kind="follow_up",
                content="请基于当前工具执行结果给出最终答复；如果任务尚未完成，请继续推进并明确下一步。",
                metadata={"phase": "follow_up"},
            )
        ]

    async def _before_tool_call(self, context: BeforeToolCallContext):
        """在工具调用前排入 steering 控制消息。"""

        self._tool_activity_in_run = True
        self._pending_control_messages.append(
            ControlMessage(
                kind="steering",
                content=f"[Steering] 即将执行工具 `{context.toolCall.name}`，参数如下：\n{context.arguments}",
                metadata={"phase": "before_tool", "tool_name": context.toolCall.name},
            )
        )
        return BeforeToolCallAllow()

    async def _after_tool_call(self, context: AfterToolCallContext):
        """在工具调用后排入 steering 控制消息，并为最终总结启用 follow-up。"""

        self._tool_activity_in_run = True
        self._pending_control_messages.append(
            ControlMessage(
                kind="steering",
                content=f"[Steering] 工具 `{context.toolCall.name}` 已执行完成，请结合工具结果继续工作。",
                metadata={"phase": "after_tool", "tool_name": context.toolCall.name},
            )
        )
        self._follow_up_sent = False
        return AfterToolCallPass()

    @staticmethod
    def _is_context_overflow_error(message: str) -> bool:
        """判断错误文本是否表示上下文溢出。"""

        lowered = message.lower()
        return "context" in lowered and ("overflow" in lowered or "length" in lowered or "window" in lowered)

    def _extract_overflow_error(self, event: SessionEvent) -> str | None:
        """从 UI 事件中提取上下文溢出错误。"""

        if event.type != "error":
            return None
        if self._is_context_overflow_error(event.message):
            return event.message
        return None
