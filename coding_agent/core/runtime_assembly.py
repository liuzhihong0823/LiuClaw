from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ai.registry import ProviderRegistry
from ai.types import Context, Model
from agent_core import AgentTool

from .compaction import CompactionCoordinator
from .resource_loader import ResourceLoader
from .session_manager import SessionManager
from .system_prompt_builder import SystemPromptBuilder
from .tools import ToolRegistry, build_tool_registry
from .types import CodingAgentSettings, ResourceBundle, SessionContext


@dataclass(slots=True)
class SessionRuntimeAssembly:
    """表示一次 session runtime 装配后的稳定组件集合。"""

    resources: ResourceBundle  # 已加载资源集合。
    tool_registry: ToolRegistry  # 工具注册表。
    tools: list[AgentTool]  # 当前激活的工具列表。
    provider_registry: ProviderRegistry  # provider 注册表。
    prompt_builder: SystemPromptBuilder  # 系统提示构造器。
    compaction: CompactionCoordinator  # 上下文压缩协调器。
    listeners: list[Callable[[Any], Any]]  # 扩展或运行时注册的监听器。


def assemble_session_runtime(
    *,
    workspace_root: Path,
    cwd: Path,
    model: Model,
    thinking: str | None,
    settings: CodingAgentSettings,
    session_manager: SessionManager,
    resource_loader: ResourceLoader,
    provider_registry: ProviderRegistry | None = None,
) -> SessionRuntimeAssembly:
    """装配 AgentSession 需要的 runtime 组件。"""

    resources = resource_loader.load()
    tool_registry = build_tool_registry(workspace_root, cwd, settings)
    tools = tool_registry.activate_all()
    for tool in resources.extension_runtime.tools:
        tool_registry.register_tool(tool, source=tool.metadata.get("source", "extension"), group=tool.metadata.get("group", "extension"), mode=tool.metadata.get("mode", "workspace-write"))
    tools = tool_registry.active_tools
    registry = provider_registry or ProviderRegistry()
    for name, factory in resources.extension_runtime.provider_factories.items():
        registry.register_factory(name, factory)
    prompt_builder = SystemPromptBuilder()
    compaction = CompactionCoordinator(session_manager, settings)
    listeners = list(resources.extension_runtime.event_listeners)
    _ = SessionContext(
        workspace_root=workspace_root,
        cwd=cwd,
        model=model,
        thinking=thinking,
        settings=settings,
        resources=resources,
        tools_markdown=tool_registry.render_markdown(),
        extra_prompt_fragments=list(resources.extension_runtime.prompt_fragments),
    )
    return SessionRuntimeAssembly(
        resources=resources,
        tool_registry=tool_registry,
        tools=tools,
        provider_registry=registry,
        prompt_builder=prompt_builder,
        compaction=compaction,
        listeners=listeners,
    )


def build_session_context(
    *,
    workspace_root: Path,
    cwd: Path,
    model: Model,
    thinking: str | None,
    settings: CodingAgentSettings,
    resources: ResourceBundle,
    tool_registry: ToolRegistry,
) -> SessionContext:
    """构造系统提示构建所需上下文。"""

    return SessionContext(
        workspace_root=workspace_root,
        cwd=cwd,
        model=model,
        thinking=thinking,
        settings=settings,
        resources=resources,
        tools_markdown=tool_registry.render_markdown(),
        extra_prompt_fragments=list(resources.extension_runtime.prompt_fragments),
    )


def build_runtime_context_messages(session_manager: SessionManager, session_id: str, branch_id: str, model: Model, system_prompt: str, tools: list[AgentTool]) -> Context:
    """构造压缩判断等运行时需要的上下文对象。"""

    return Context(systemPrompt=system_prompt, messages=session_manager.build_context_messages(session_id, branch_id), tools=tools)
