from __future__ import annotations

import json
from pathlib import Path

from agent_core import AgentTool

from ..types import CodingAgentSettings, ToolDefinition, ToolExecutionContext, ToolSecurityPolicy


class ToolRegistry:
    """维护工具定义、激活状态与构建后的工具实例。"""

    def __init__(
        self,
        workspace_root: Path,
        cwd: Path,
        settings: CodingAgentSettings,
        *,
        security_policy: ToolSecurityPolicy | None = None,
    ) -> None:
        """初始化工具注册表。"""

        self.workspace_root = workspace_root  # 工作区根目录。
        self.cwd = cwd  # 当前工具执行目录。
        self.settings = settings  # 生效设置。
        self.security_policy = security_policy  # 工具安全策略。
        self._definitions: dict[str, ToolDefinition] = {}  # 已注册工具定义。
        self._active_tools: list[AgentTool] = []  # 当前激活工具实例。

    @property
    def definitions(self) -> list[ToolDefinition]:
        """返回全部已注册工具定义。"""

        return list(self._definitions.values())

    @property
    def active_tools(self) -> list[AgentTool]:
        """返回当前已激活工具。"""

        return list(self._active_tools)

    def register_definition(self, definition: ToolDefinition) -> None:
        """注册一个工具定义。"""

        self._definitions[definition.name] = definition

    def register_tool(self, tool: AgentTool, *, source: str = "extension", group: str = "extension", mode: str = "workspace-write") -> None:
        """直接注册一个已经构造好的工具实例。"""

        tool.metadata = dict(tool.metadata)
        tool.metadata.setdefault("group", group)
        tool.metadata.setdefault("source", source)
        tool.metadata.setdefault("mode", mode)
        self._active_tools.append(tool)

    def activate_all(self) -> list[AgentTool]:
        """构造并激活全部工具定义。"""

        self._active_tools = [self._wrap_tool(definition, definition.builder(self.workspace_root, self.cwd, self.settings)) for definition in self.definitions]
        return self.active_tools

    def render_markdown(self) -> str:
        """把当前激活工具渲染成系统提示文本。"""

        lines: list[str] = []
        for tool in self._active_tools:
            group = tool.metadata.get("group", "general")
            source = tool.metadata.get("source", "core")
            lines.append(f"- {tool.name}: {tool.description or ''} [group={group}, source={source}]")
        return "\n".join(lines)

    def _wrap_tool(self, definition: ToolDefinition, tool: AgentTool) -> AgentTool:
        """为工具执行包上统一安全策略与渲染元信息。"""

        original_execute = tool.execute
        policy = self.security_policy
        metadata = dict(tool.metadata)
        metadata.update(
            {
                "group": definition.group,
                "built_in": definition.built_in,
                "source": definition.source,
                "mode": definition.mode,
            }
        )
        render_metadata = dict(tool.renderMetadata)
        render_metadata.setdefault("group", definition.group)
        render_metadata.setdefault("source", definition.source)

        async def execute(arguments: str, context):
            payload = json.loads(arguments or "{}") if isinstance(arguments, str) else dict(arguments or {})
            execution_context = ToolExecutionContext(
                tool_name=tool.name,
                workspace_root=self.workspace_root,
                cwd=self.cwd,
                arguments=payload,
                mode=definition.mode,
            )
            if policy is not None and policy.before_execute is not None:
                policy.before_execute(execution_context)
            result = await original_execute(arguments, context)
            if policy is not None and policy.after_execute is not None:
                result = policy.after_execute(execution_context, result)
            return result

        return AgentTool(
            name=tool.name,
            description=tool.description,
            inputSchema=dict(tool.inputSchema),
            metadata=metadata,
            renderMetadata=render_metadata,
            execute=execute,
        )
