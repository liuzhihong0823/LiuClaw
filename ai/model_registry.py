from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import AIConfig, ProviderConfig, load_ai_config
from .errors import ProviderNotFoundError
from .types import Model


def _default_model_catalog() -> dict[str, Model]:
    """返回内置模型目录。"""

    from .models import _MODEL_CATALOG

    return {model_id: replace(model) for model_id, model in _MODEL_CATALOG.items()}


class ModelRegistry:
    """模型与 provider 配置中心。"""

    def __init__(
        self,
        *,
        models: dict[str, Model] | None = None,
        provider_configs: dict[str, ProviderConfig] | None = None,
        ai_config: AIConfig | None = None,
    ) -> None:
        self._models = dict(models or _default_model_catalog())
        self._provider_configs = dict(provider_configs or {})
        if ai_config is not None:
            self.merge_ai_config(ai_config)

    @property
    def provider_configs(self) -> dict[str, ProviderConfig]:
        """返回 provider 配置副本。"""

        return dict(self._provider_configs)

    def merge_ai_config(self, ai_config: AIConfig) -> None:
        """合并本地配置文件中的 provider 与模型配置。"""

        self._provider_configs.update(ai_config.providers)
        for model_id, payload in ai_config.models.items():
            if model_id in self._models:
                self._models[model_id] = self._merge_model(self._models[model_id], payload)
            else:
                self._models[model_id] = Model(**payload)

    def register_model(self, model: Model) -> None:
        """注册一个模型定义。"""

        self._models[model.id] = model

    def register_provider_config(self, config: ProviderConfig) -> None:
        """注册一个 provider 配置。"""

        self._provider_configs[config.name] = config

    def get_model(self, model_id: str) -> Model:
        """根据模型 ID 返回模型定义，并应用 provider/model override。"""

        try:
            model = self._models[model_id]
        except KeyError as exc:
            raise ProviderNotFoundError(f"Unknown model '{model_id}'") from exc
        provider_config = self._provider_configs.get(model.provider)
        if provider_config is None:
            return model
        return self._apply_provider_config(model, provider_config)

    def list_models(self, provider: str | None = None) -> list[Model]:
        """列出模型目录，可按 provider 过滤。"""

        models = [self.get_model(model_id) for model_id in self._models]
        if provider is None:
            return models
        return [model for model in models if model.provider == provider]

    def get_provider_config(self, provider_name: str) -> ProviderConfig | None:
        """返回指定 provider 的配置。"""

        return self._provider_configs.get(provider_name)

    def load_local_config(self, config_path: str | None = None) -> None:
        """从本地配置文件加载 provider 与模型覆盖配置。"""

        self.merge_ai_config(load_ai_config(config_path))

    def _apply_provider_config(self, model: Model, config: ProviderConfig) -> Model:
        """把 provider/model override 应用到模型定义。"""

        payload = {
            "metadata": {**model.metadata, **config.providerOverrides.get("metadata", {})},
            "providerConfig": {
                **model.providerConfig,
                "baseUrl": config.baseUrl,
                "apiKeyEnv": config.apiKeyEnv,
                "headers": dict(config.headers),
                **config.providerOverrides,
            },
        }
        model_override = config.modelOverrides.get(model.id) or config.modelOverrides.get(model.id.split(":", 1)[-1], {})
        merged = self._merge_model(model, {**payload, **model_override})
        capabilities = config.capabilities
        if capabilities:
            merged = self._merge_model(merged, capabilities)
        return merged

    @staticmethod
    def _merge_model(model: Model, overrides: dict[str, Any]) -> Model:
        """以不可变方式应用模型覆盖字段。"""

        payload = {
            "id": overrides.get("id", model.id),
            "provider": overrides.get("provider", model.provider),
            "inputPrice": overrides.get("inputPrice", model.inputPrice),
            "outputPrice": overrides.get("outputPrice", model.outputPrice),
            "contextWindow": overrides.get("contextWindow", model.contextWindow),
            "maxOutputTokens": overrides.get("maxOutputTokens", model.maxOutputTokens),
            "metadata": dict(overrides.get("metadata", model.metadata)),
            "supports_reasoning_levels": tuple(overrides.get("supports_reasoning_levels", model.supports_reasoning_levels)),
            "supports_images": overrides.get("supports_images", model.supports_images),
            "supports_prompt_cache": overrides.get("supports_prompt_cache", model.supports_prompt_cache),
            "supports_session": overrides.get("supports_session", model.supports_session),
            "providerConfig": dict(overrides.get("providerConfig", model.providerConfig)),
        }
        return Model(**payload)


DEFAULT_MODEL_REGISTRY = ModelRegistry()
DEFAULT_MODEL_REGISTRY.load_local_config()
