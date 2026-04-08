from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProviderConfig:
    """描述一个 provider 的运行时配置。"""

    name: str
    baseUrl: str | None = None
    apiKey: str | None = None
    apiKeyEnv: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    providerOverrides: dict[str, Any] = field(default_factory=dict)
    modelOverrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    sdk: str | None = None

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

    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)


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
