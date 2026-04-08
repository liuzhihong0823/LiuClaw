from __future__ import annotations

from ai import Model
from ai.config import ProviderConfig
from ai.providers.base import Provider
from ai.registry import ProviderRegistry


class ConfigAwareProvider(Provider):
    name = "stubcfg"

    def supports(self, model: Model) -> bool:
        return model.provider == self.name

    async def stream(self, model: Model, context, options):  # pragma: no cover - config test only
        raise NotImplementedError


def test_registry_injects_provider_config_into_factory() -> None:
    registry = ProviderRegistry(
        factories={"stubcfg": ConfigAwareProvider},
        provider_configs={
            "stubcfg": ProviderConfig(name="stubcfg", baseUrl="https://example.com", apiKeyEnv="STUB_KEY")
        },
    )
    model = Model(
        id="stubcfg:model",
        provider="stubcfg",
        inputPrice=0.0,
        outputPrice=0.0,
        contextWindow=10,
        maxOutputTokens=10,
    )

    provider = registry.resolve(model)

    assert provider.config is not None
    assert provider.config.baseUrl == "https://example.com"
    assert provider.config.apiKeyEnv == "STUB_KEY"
