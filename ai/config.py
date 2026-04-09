from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProviderConfig:
    """描述一个 provider 的运行时配置。"""

    name: str  # provider 名称，用作注册表中的唯一键。
    baseUrl: str | None = None  # provider 的基础 API 地址。
    apiKey: str | None = None  # 显式写在配置里的 API Key。
    apiKeyEnv: str | None = None  # 存放 API Key 的环境变量名。
    headers: dict[str, str] = field(default_factory=dict)  # 额外附加到请求上的固定请求头。
    providerOverrides: dict[str, Any] = field(default_factory=dict)  # 对 provider 级行为的覆盖配置。
    modelOverrides: dict[str, dict[str, Any]] = field(default_factory=dict)  # 针对具体模型的字段覆盖。
    capabilities: dict[str, Any] = field(default_factory=dict)  # 补充声明 provider 或模型能力。
    sdk: str | None = None  # 指定优先使用的 SDK 类型或接入方式。

    def resolve_api_key(self) -> str | None:
        """返回配置中的 API Key，优先使用显式值，其次读取环境变量。"""

        if self.apiKey:
            return self.apiKey
        if self.apiKeyEnv:
            return os.getenv(self.apiKeyEnv)
        return None


@dataclass(slots=True)
class AIConfig:
    """描述本地 AI 配置中心。"""

    providers: dict[str, ProviderConfig] = field(default_factory=dict)  # provider 配置映射。
    models: dict[str, dict[str, Any]] = field(default_factory=dict)  # 模型覆盖或自定义模型配置。


def load_ai_config(config_path: str | Path | None = None) -> AIConfig:
    """从环境变量或本地文件读取 AI 配置。"""

    resolved_path = _resolve_config_path(config_path)
    if resolved_path is None or not resolved_path.exists():
        return AIConfig()

    raw = json.loads(resolved_path.read_text(encoding="utf-8"))
    providers = {
        name: ProviderConfig(name=name, **payload)
        for name, payload in raw.get("providers", {}).items()
    }
    return AIConfig(providers=providers, models=dict(raw.get("models", {})))


def _resolve_config_path(config_path: str | Path | None) -> Path | None:
    """解析配置文件路径。"""

    if config_path is not None:
        return Path(config_path)

    env_path = os.getenv("AI_CONFIG_FILE")
    if env_path:
        return Path(env_path)

    default_path = Path("ai.config.json")
    if default_path.exists():
        return default_path
    return None
