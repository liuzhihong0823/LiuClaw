from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core import AgentTool

from .types import ExtensionResource, ExtensionRuntime


@dataclass(slots=True)
class ExtensionApi:
    """提供给扩展模块注册贡献能力的最小 API。"""

    source: str  # 当前扩展来源标识。
    tools: list[AgentTool] = field(default_factory=list)  # 扩展注册的工具。
    provider_factories: dict[str, Any] = field(default_factory=dict)  # 扩展提供的 provider 工厂。
    event_listeners: list[Any] = field(default_factory=list)  # 扩展订阅的事件监听器。
    prompt_fragments: list[str] = field(default_factory=list)  # 扩展追加的系统提示片段。

    def register_tool(self, tool: AgentTool) -> None:
        """注册一个扩展工具。"""

        tool.metadata = dict(tool.metadata)
        tool.metadata.setdefault("source", self.source)
        self.tools.append(tool)

    def register_provider(self, name: str, factory) -> None:
        """注册一个 provider 工厂。"""

        self.provider_factories[name] = factory

    def subscribe(self, listener) -> None:
        """订阅 agent/session 事件。"""

        self.event_listeners.append(listener)

    def extend_system_prompt(self, fragment: str) -> None:
        """追加系统提示附加片段。"""

        if fragment.strip():
            self.prompt_fragments.append(fragment.strip())


def scan_extensions(extensions_dir: Path) -> list[ExtensionResource]:
    """扫描扩展目录并解析基础元信息。"""

    if not extensions_dir.exists():
        return []
    extensions: list[ExtensionResource] = []
    for path in sorted(extensions_dir.iterdir()):
        if path.name.startswith("."):
            continue
        metadata: dict[str, object] = {}
        manifest = path / "extension.json" if path.is_dir() else path
        if manifest.is_file() and manifest.suffix == ".json":
            try:
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        module_name = str(metadata.get("module", "extension.py"))
        module_path = path / module_name if path.is_dir() else None
        extensions.append(
            ExtensionResource(
                name=path.stem,
                path=path,
                module_path=module_path if module_path and module_path.exists() else None,
                source=str(module_path if module_path else path),
                metadata=metadata,
            )
        )
    return extensions


def load_extension_runtime(extensions: list[ExtensionResource]) -> ExtensionRuntime:
    """加载扩展模块并汇总贡献能力。"""

    runtime = ExtensionRuntime()
    for extension in extensions:
        if extension.module_path is None:
            continue
        api = ExtensionApi(source=extension.source or str(extension.module_path))
        _load_single_extension(extension.module_path, api)
        runtime.tools.extend(api.tools)
        runtime.provider_factories.update(api.provider_factories)
        runtime.event_listeners.extend(api.event_listeners)
        runtime.prompt_fragments.extend(api.prompt_fragments)
    return runtime


def _load_single_extension(module_path: Path, api: ExtensionApi) -> None:
    """加载单个扩展模块并执行 register 钩子。"""

    spec = importlib.util.spec_from_file_location(f"coding_agent_extension_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load extension module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    register = getattr(module, "register", None)
    if callable(register):
        register(api)
