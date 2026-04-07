from ai import Model
from ai.reasoning import build_reasoning_config


OPENAI_MODEL = Model(
    id="openai:gpt-5",
    provider="openai",
    inputPrice=1.25,
    outputPrice=10.0,
    contextWindow=272000,
    maxOutputTokens=8192,
)
ANTHROPIC_MODEL = Model(
    id="anthropic:claude-sonnet-4",
    provider="anthropic",
    inputPrice=3.0,
    outputPrice=15.0,
    contextWindow=200000,
    maxOutputTokens=8192,
)
ZHIPU_MODEL = Model(
    id="zhipu:glm-5",
    provider="zhipu",
    inputPrice=0.0,
    outputPrice=0.0,
    contextWindow=200000,
    maxOutputTokens=128000,
)
ZHIPU_GLm46_MODEL = Model(
    id="zhipu:glm-4.6",
    provider="zhipu",
    inputPrice=0.0,
    outputPrice=0.0,
    contextWindow=200000,
    maxOutputTokens=128000,
)


def test_reasoning_mapping_for_openai() -> None:
    config = build_reasoning_config(OPENAI_MODEL, "high")

    assert config == {"reasoning": {"effort": "high"}}


def test_reasoning_mapping_for_anthropic() -> None:
    config = build_reasoning_config(ANTHROPIC_MODEL, "medium")

    assert config == {"thinking": {"type": "enabled", "budget_tokens": 4096}}


def test_reasoning_mapping_for_anthropic_xhigh() -> None:
    config = build_reasoning_config(ANTHROPIC_MODEL, "xhigh")

    assert config == {"thinking": {"type": "enabled", "budget_tokens": 16384}}


def test_reasoning_mapping_for_zhipu_high() -> None:
    config = build_reasoning_config(ZHIPU_MODEL, "high")

    assert config == {"thinking": {"type": "enabled"}, "clear_thinking": False}


def test_reasoning_mapping_for_zhipu_glm46_degrades_high() -> None:
    config = build_reasoning_config(ZHIPU_GLm46_MODEL, "high")

    assert config == {"thinking": {"type": "enabled"}}
