from __future__ import annotations

import json

from ai import ModelRegistry, ProviderConfig, load_ai_config
from ai.utils.context_window import truncate_context_to_window
from ai import Context, Model, Options, TextContent, ImageContent, UserMessage


def test_model_registry_applies_provider_override() -> None:
    registry = ModelRegistry(
        provider_configs={
            "openai_compatible": ProviderConfig(
                name="openai_compatible",
                baseUrl="https://example.com/v1",
                apiKeyEnv="EXAMPLE_API_KEY",
                headers={"x-test": "1"},
                capabilities={"supports_images": True},
            )
        }
    )
    registry.register_model(
        Model(
            id="openai_compatible:test-model",
            provider="openai_compatible",
            inputPrice=0.1,
            outputPrice=0.2,
            contextWindow=1000,
            maxOutputTokens=100,
        )
    )

    model = registry.get_model("openai_compatible:test-model")

    assert model.providerConfig["baseUrl"] == "https://example.com/v1"
    assert model.providerConfig["apiKeyEnv"] == "EXAMPLE_API_KEY"
    assert model.providerConfig["headers"]["x-test"] == "1"
    assert model.supports_images is True


def test_load_ai_config_from_file(tmp_path) -> None:
    config_file = tmp_path / "ai.config.json"
    config_file.write_text(
        json.dumps(
            {
                "providers": {
                    "openai_compatible": {
                        "baseUrl": "https://example.com/v1",
                        "apiKeyEnv": "EXAMPLE_API_KEY",
                    }
                },
                "models": {
                    "openai_compatible:test-model": {
                        "id": "openai_compatible:test-model",
                        "provider": "openai_compatible",
                        "inputPrice": 0.1,
                        "outputPrice": 0.2,
                        "contextWindow": 1000,
                        "maxOutputTokens": 100,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_ai_config(config_file)

    assert "openai_compatible" in config.providers
    assert config.providers["openai_compatible"].baseUrl == "https://example.com/v1"
    assert "openai_compatible:test-model" in config.models


def test_truncate_context_to_window_removes_oldest_messages() -> None:
    model = Model(
        id="stub:model",
        provider="stub",
        inputPrice=0.0,
        outputPrice=0.0,
        contextWindow=20,
        maxOutputTokens=4,
    )
    context = Context(
        messages=[
            UserMessage(content="a" * 30),
            UserMessage(content="b" * 30),
            UserMessage(content="ok"),
        ]
    )

    truncated = truncate_context_to_window(model, context, Options(maxTokens=4))

    assert len(truncated.messages) < len(context.messages)
    assert truncated.messages[-1].content == "ok"


def test_capability_clamp_replaces_image_with_text() -> None:
    model = Model(
        id="stub:model",
        provider="stub",
        inputPrice=0.0,
        outputPrice=0.0,
        contextWindow=100,
        maxOutputTokens=10,
        supports_images=False,
    )
    from ai.converters.capabilities import apply_model_capabilities

    context = Context(messages=[UserMessage(content=[ImageContent(data="abc"), TextContent(text="hi")])])

    converted = apply_model_capabilities(model, context)

    assert converted.messages[0].content[0].text == "[image omitted by capability clamp]"
