# `ai` 模块说明

## 1. 模块职责

`ai` 是本项目的统一大模型接入层。它不是某一家 SDK 的轻包装，而是把模型、消息、工具调用、流式事件、provider 配置和上下文预处理统一成一套稳定接口，供上层模块直接依赖。

从代码结构看，这个模块主要解决六件事：

1. 定义统一的数据协议，包括 `Context`、消息类型、内容块、`Model`、`StreamEvent`。
2. 提供统一调用入口，包括 `stream()`、`complete()` 以及简化版 `streamSimple()`、`completeSimple()`。
3. 维护模型目录和 provider 配置覆盖，把静态模型定义和本地配置合并到运行时模型对象中。
4. 维护 provider 注册与懒实例化，把不同厂商的流式输出转成统一事件流。
5. 在调用前完成上下文清理、能力裁剪、thinking 兼容处理、工具定义转换和窗口检测。
6. 提供通用工具能力，包括上下文窗口估算、Unicode 清理、JSON Schema 参数校验、流式聚合。

当前内置 provider 有四类：

- `openai`
- `openai_compatible`
- `anthropic`
- `zhipu`

对外导出入口位于 [ai/__init__.py](/Users/admin/PyCharmProject/LiuClaw/ai/__init__.py)。

## 2. 模块分层

`ai/` 目录可以按职责分成下面几层：

### 2.1 协议与核心类型

- [ai/types.py](/Users/admin/PyCharmProject/LiuClaw/ai/types.py)
- [ai/options.py](/Users/admin/PyCharmProject/LiuClaw/ai/options.py)
- [ai/errors.py](/Users/admin/PyCharmProject/LiuClaw/ai/errors.py)

这里定义统一对象模型，包括消息、内容块、工具、模型、流事件、调用选项和错误类型。上层与 provider 层都围绕这套协议交互。

### 2.2 调用入口

- [ai/client.py](/Users/admin/PyCharmProject/LiuClaw/ai/client.py)
- [ai/session.py](/Users/admin/PyCharmProject/LiuClaw/ai/session.py)

`client.py` 负责把一次调用串起来：规范化模型和上下文、附加 reasoning 元数据、转换上下文、检测窗口、启动 provider 流并返回 `StreamSession`，或者进一步聚合出 `AssistantMessage`。

### 2.3 模型与配置中心

- [ai/models.py](/Users/admin/PyCharmProject/LiuClaw/ai/models.py)
- [ai/model_registry.py](/Users/admin/PyCharmProject/LiuClaw/ai/model_registry.py)
- [ai/config.py](/Users/admin/PyCharmProject/LiuClaw/ai/config.py)
- [ai/reasoning.py](/Users/admin/PyCharmProject/LiuClaw/ai/reasoning.py)

这一层负责内置模型目录、本地 `ai.config.json` 读取、provider 配置注入、模型能力覆盖，以及统一 reasoning 等级到厂商参数的映射。

### 2.4 Provider 适配层

- [ai/registry.py](/Users/admin/PyCharmProject/LiuClaw/ai/registry.py)
- [ai/providers/base.py](/Users/admin/PyCharmProject/LiuClaw/ai/providers/base.py)
- [ai/providers/openai.py](/Users/admin/PyCharmProject/LiuClaw/ai/providers/openai.py)
- [ai/providers/anthropic.py](/Users/admin/PyCharmProject/LiuClaw/ai/providers/anthropic.py)
- [ai/providers/zhipu.py](/Users/admin/PyCharmProject/LiuClaw/ai/providers/zhipu.py)

这一层把统一 `Context` 转成厂商请求，并把厂商流式输出转成统一 `StreamEvent`。

### 2.5 转换与清理层

- [ai/converters/messages.py](/Users/admin/PyCharmProject/LiuClaw/ai/converters/messages.py)
- [ai/converters/tools.py](/Users/admin/PyCharmProject/LiuClaw/ai/converters/tools.py)
- [ai/converters/capabilities.py](/Users/admin/PyCharmProject/LiuClaw/ai/converters/capabilities.py)
- [ai/converters/thinking.py](/Users/admin/PyCharmProject/LiuClaw/ai/converters/thinking.py)
- [ai/utils/context_window.py](/Users/admin/PyCharmProject/LiuClaw/ai/utils/context_window.py)
- [ai/utils/unicode.py](/Users/admin/PyCharmProject/LiuClaw/ai/utils/unicode.py)
- [ai/utils/schema_validation.py](/Users/admin/PyCharmProject/LiuClaw/ai/utils/schema_validation.py)
- [ai/utils/streaming.py](/Users/admin/PyCharmProject/LiuClaw/ai/utils/streaming.py)

这里集中处理跨 provider 的兼容问题和调用前后的公共逻辑。

## 3. 统一协议

## 3.1 内容块模型

`ai/types.py` 不再把消息内容限定为单一字符串，而是采用内容块模型：

- `TextContent`
- `ThinkingContent`
- `ImageContent`
- `ToolCallContent`
- `ToolResultContent`

`ContentBlocks` 是对内容块列表的包装，额外提供这些兼容视图：

- `.text`：拼接文本块和工具结果文本
- `.thinking`：拼接 thinking 块
- `.tool_calls`：提取 `ToolCall`

这意味着上层既可以按结构化内容块消费，也可以继续通过 `.text`、`.thinking` 读取兼容视图。

## 3.2 消息类型

统一消息分为三种：

- `UserMessage`
- `AssistantMessage`
- `ToolResultMessage`

其中：

- `UserMessage.content` 只接受用户可发送的内容块，当前是文本和图片。
- `AssistantMessage.content` 可以包含文本、thinking、工具调用和图片。
- `ToolResultMessage` 通过 `toolCallId`、`toolName` 关联某次工具调用，`content` 中可放文本、图片或 `ToolResultContent`。

`AssistantMessage` 仍保留若干兼容属性：

- `text`
- `thinking`
- `toolCalls`

这些属性来自 `content` 的投影，不是独立状态源。

## 3.3 Tool 与 ToolCall

工具定义由 `Tool` 表示，核心字段是：

- `name`
- `description`
- `inputSchema`
- `metadata`
- `renderMetadata`

工具调用由 `ToolCall` 表示，核心字段是：

- `id`
- `name`
- `arguments`
- `metadata`

`ToolCall` 和 `ToolCallContent` 在初始化时都会调用 `parse_tool_arguments()`，因此既兼容字符串 JSON，也兼容已经解析好的字典。对应地，`arguments_text` 会用稳定 JSON 序列化输出，方便不同 provider 发送请求。

## 3.4 Model

`Model` 是整个统一层的运行时模型对象，除了基础计费和窗口信息，还内置能力描述：

- `supports_reasoning_levels`
- `supports_images`
- `supports_prompt_cache`
- `supports_session`
- `providerConfig`

其中 `clamp_reasoning()` 很关键。它会把用户请求的 reasoning 等级收敛到该模型真正支持的等级。例如测试覆盖了“请求 `high`，但模型只支持到 `medium`”的场景，最终会自动降级。

## 3.5 StreamEvent

`StreamEvent` 是跨 provider 的统一流式协议。事件既保留旧式 `type`，也显式记录：

- `lifecycle`：`start` / `update` / `done` / `error`
- `itemType`：`message` / `text` / `thinking` / `tool_call` / `tool_result` / `image`

模块仍兼容旧式事件名：

- `text_start`
- `text_delta`
- `text_end`
- `thinking_start`
- `thinking_delta`
- `thinking_end`
- `toolcall_start`
- `toolcall_delta`
- `toolcall_end`
- `tool_result`

`StreamEvent.__post_init__()` 会把这些旧事件名映射成标准 `lifecycle` 和 `itemType`，因此上层既能继续按旧事件类型处理，也能统一按生命周期处理。

## 4. 调用入口与主流程

主要入口在 [ai/client.py](/Users/admin/PyCharmProject/LiuClaw/ai/client.py)。

### 4.1 `stream()`

`stream()` 的职责是创建一次流式队列会话。核心流程如下：

1. 解析 `model`。
   - 如果传入字符串，就通过 `ModelRegistry.get_model()` 找到运行时模型。
   - 如果传入的是 `Model`，则直接规范化。
2. 调用 `_prepare_options()`。
   - `ensure_options()` 补默认值。
   - `Model.clamp_reasoning()` 收敛 reasoning。
   - `merge_reasoning_metadata()` 把厂商专用 reasoning 参数写进 `options.metadata["_providerReasoning"]`。
   - 如果发生等级钳制，还会记录 `_requestedReasoning` 和 `_clampedReasoning`。
3. 调用 `_prepare_context()`。
   - `ensure_context()` 把输入转成统一 `Context`。
   - `sanitize_unicode_context()` 清洗 Unicode。
   - `convert_context_for_provider()` 完成 provider 前转换。
   - 若 `contextOverflowStrategy == "truncate_oldest"`，则调用 `truncate_context_to_window()` 预裁剪旧消息。
4. 调用 `ensure_context_fits_window()` 做窗口校验。
5. 创建有界 `asyncio.Queue[StreamEvent]`。
6. 启动后台任务 `_produce_events()`。
   - 通过 `ProviderRegistry.resolve()` 找到 provider。
   - 消费 provider 的 `stream()` 输出。
   - 按配置把事件放入队列。
   - 如果 provider 抛异常，则包装成统一 `error` 事件。
7. 返回 `StreamSession`。

### 4.2 `complete()`

`complete()` 并不直接走另一套逻辑，而是：

1. 先调用 `stream()` 创建 `StreamSession`。
2. 用 `StreamAccumulator` 消费 `session.consume()`。
3. 遇到 `done` 时返回最终 `AssistantMessage`。
4. 遇到 `error` 时抛出 `ProviderResponseError`。
5. 在 `finally` 中关闭会话，确保后台 producer 结束。

这也是当前实现的一个重要特点：非流式接口其实是流式接口的聚合包装，因此两者的行为基线一致。

### 4.3 `StreamSession`

[ai/session.py](/Users/admin/PyCharmProject/LiuClaw/ai/session.py) 中的 `StreamSession` 封装了：

- `model`
- `queue`
- `producer_task`

它提供三个核心方法：

- `consume()`：持续从队列取事件，收到 `done/error` 后结束
- `close()`：取消 producer 并等待结束
- `wait_closed()`：等待 producer 自然结束

默认停止条件由 `_default_should_stop()` 决定，即 `event.type in {"done", "error"}`。

## 5. 配置、模型目录与注册表

## 5.1 静态模型目录

[ai/models.py](/Users/admin/PyCharmProject/LiuClaw/ai/models.py) 内置了一组模型定义，目前包括：

- `openai:gpt-5`
- `openai:gpt-5-mini`
- `anthropic:claude-sonnet-4`
- `anthropic:claude-haiku-3-5`
- `zhipu:glm-5`
- `zhipu:glm-5-turbo`
- `zhipu:glm-4.7`
- `zhipu:glm-4.6`

`get_model()` 和 `list_models()` 实际上都委托给默认的 `DEFAULT_MODEL_REGISTRY`，因此它们返回的是“应用过本地配置覆盖后的运行时模型”，而不是硬编码原始副本。

## 5.2 `AIConfig` 与本地配置文件

[ai/config.py](/Users/admin/PyCharmProject/LiuClaw/ai/config.py) 定义：

- `ProviderConfig`
- `AIConfig`
- `load_ai_config()`

配置文件查找顺序如下：

1. 调用方显式传入的 `config_path`
2. 环境变量 `AI_CONFIG_FILE`
3. 当前目录下的 `ai.config.json`

如果都不存在，则返回空配置。

`ProviderConfig` 不只是 API 地址和 key，还包含：

- `headers`
- `providerOverrides`
- `modelOverrides`
- `capabilities`
- `sdk`

`resolve_api_key()` 会优先使用显式 `apiKey`，否则从 `apiKeyEnv` 指向的环境变量读取。

## 5.3 `ModelRegistry`

[ai/model_registry.py](/Users/admin/PyCharmProject/LiuClaw/ai/model_registry.py) 负责管理模型目录和 provider 配置。

主要职责：

- 以 `_MODEL_CATALOG` 为基础构建运行时模型目录。
- 合并本地 `AIConfig`。
- 返回已经应用 provider/model override 的 `Model`。
- 维护 provider 配置表。

`get_model()` 是最关键的方法。它会：

1. 先按模型 ID 取出基础模型。
2. 查找同名 provider 的 `ProviderConfig`。
3. 如果存在配置，则调用 `_apply_provider_config()` 叠加：
   - `baseUrl`
   - `apiKeyEnv`
   - `headers`
   - `providerOverrides`
   - `modelOverrides`
   - `capabilities`

测试 [tests/test_model_registry.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_model_registry.py) 验证了：

- provider 配置中的 `baseUrl`、`apiKeyEnv`、`headers` 会写入 `model.providerConfig`
- `capabilities` 能直接影响模型能力，例如把 `supports_images` 置为 `True`
- `load_ai_config()` 能从 `ai.config.json` 正确加载 provider 和模型定义

## 5.4 `ProviderRegistry`

[ai/registry.py](/Users/admin/PyCharmProject/LiuClaw/ai/registry.py) 负责 provider 工厂和实例缓存。

它维护三类状态：

- `_factories`：provider 工厂映射
- `_instances`：已经实例化的 provider
- `_provider_configs`：provider 级配置

默认工厂由 `_default_factories()` 提供，包含：

- `openai`
- `openai_compatible`
- `anthropic`
- `zhipu`

`resolve()` 最终走 `get_provider()`，解析逻辑是：

1. 如果 `model.provider` 已明确，优先按名称拿 provider。
2. 否则如果模型 ID 形如 `provider:model-name`，按前缀推断 provider。
3. 再不行就遍历全部 provider，看谁的 `supports()` 返回真。

`ProviderRegistry` 采用懒实例化策略。测试 [tests/test_registry.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_registry.py) 验证了：

- 工厂不会在注册时立即执行
- 首次 `resolve()` 才会创建实例
- 同一 provider 会复用缓存实例
- 未知 provider 会抛 `ProviderNotFoundError`

测试 [tests/test_registry_config.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_registry_config.py) 还验证了：如果工厂构造器支持 `config=` 参数，注册表会在实例化时直接注入 `ProviderConfig`。

## 6. Provider 适配层

所有 provider 都继承 [ai/providers/base.py](/Users/admin/PyCharmProject/LiuClaw/ai/providers/base.py) 中的 `Provider` 抽象类，只要求实现两个接口：

- `supports(model)`
- `stream(model, context, options)`

这意味着 provider 层只需要处理“是否支持”和“如何把厂商流映射成统一流”，不需要关心上层队列会话、上下文裁剪和最终聚合。

## 6.1 OpenAIProvider

[ai/providers/openai.py](/Users/admin/PyCharmProject/LiuClaw/ai/providers/openai.py) 适配 OpenAI Responses API。

关键点：

- `supports()` 支持显式 `provider == "openai"`，也兼容模型名前缀 `openai:`、`gpt-`、`o`
- `_runtime_model_name()` 会把 `openai:gpt-5` 转成实际请求模型名 `gpt-5`
- `_client_kwargs()` 使用 `ProviderConfig.baseUrl` 或环境变量 `OPENAI_BASE_URL`
- API key 优先来自 `ProviderConfig.resolve_api_key()`，其次读取 `OPENAI_API_KEY`
- `_build_request()` 使用 `input` 字段组织系统提示和历史消息，工具定义映射为 Responses API 的 `tools`
- `options.metadata["_providerReasoning"]` 会直接写入请求体

流式事件映射方面：

- `response.output_text.delta` -> `text_delta`
- `response.reasoning_text.delta` / `response.reasoning_summary_text.delta` -> `thinking_delta`
- `response.function_call_arguments.delta` -> `toolcall_delta`
- `response.function_call_arguments.done` -> `toolcall_end`

最终通过 `create_done_event()` 产出统一 `done` 事件，并把原始最终响应对象塞进 `final_message.metadata["response"]`。

`OpenAICompatibleProvider` 只是复用 `OpenAIProvider`，把 `name` 改成 `openai_compatible`，并放宽模型前缀识别。

## 6.2 AnthropicProvider

[ai/providers/anthropic.py](/Users/admin/PyCharmProject/LiuClaw/ai/providers/anthropic.py) 适配 Anthropic Messages API。

关键点：

- `_build_request()` 使用 `messages` 和可选 `system`
- 默认 `max_tokens` 取 `options.maxTokens`，否则退回 `model.maxOutputTokens`
- 工具定义映射为 `tools: [{name, description, input_schema}]`
- 工具结果消息映射为 `role="user"` 且 `content=[{type: "tool_result", ...}]`
- assistant 历史工具调用会映射成 `tool_use`

流式阶段主要处理这些事件：

- `content_block_start`
- `content_block_delta`
- `content_block_stop`

其中：

- `text_delta` 映射为统一文本事件
- `thinking_delta` 映射为统一思考事件
- `input_json_delta` 用来增量拼接工具参数

块结束时，如果该块是工具调用块，会补 `toolcall_end`；如果是 thinking 或 text 块，则分别补 `thinking_end` 或 `text_end`。

## 6.3 ZhipuProvider

[ai/providers/zhipu.py](/Users/admin/PyCharmProject/LiuClaw/ai/providers/zhipu.py) 适配智谱的 SSE 风格 `chat/completions` 接口，是当前三家里实现差异最大的一家。

关键点：

- 默认基地址是 `https://open.bigmodel.cn/api/paas/v4`
- 会从 `ProviderConfig.baseUrl` 或环境变量 `ZHIPU_BASE_URL` 覆盖
- API key 读取顺序是 `ProviderConfig` -> `ZHIPU_API_KEY` -> `ZHIPUAI_API_KEY`
- 请求头由 `_headers()` 手工构造，不依赖厂商 SDK
- `_iter_sse_chunks()` 使用 `httpx.AsyncClient.stream()` 逐行消费 SSE，并手工解析 `data:` 负载

消息映射方面有两个智谱特有点：

1. assistant 历史 thinking 会写入 `reasoning_content`
2. 对 `glm-4.6` 和 `glm-4.7`，如果请求里包含工具，还会追加 `tool_stream=True`

流式输出时：

- `delta.reasoning_content` -> thinking 事件
- `delta.content` -> text 事件
- `delta.tool_calls[*].function.arguments` -> 工具参数增量

当 `finish_reason` 变成 `tool_calls`、`stop` 或 `length` 时，provider 会把尚未收尾的工具调用统一补成 `toolcall_end`，然后在最终 `done` 事件中附带：

- `usage`
- `responseId`
- `providerMetadata["request_model"]`

测试 [tests/test_zhipu_provider.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_zhipu_provider.py) 覆盖了两件关键事情：

- `_build_request()` 会正确映射 system prompt、assistant thinking、tool results、工具定义和 reasoning 配置
- `stream()` 会正确产出 thinking、tool call、text 和最终 `done` 事件，并把工具参数字符串解析成结构化对象

## 7. 转换层

转换入口在 [ai/converters/messages.py](/Users/admin/PyCharmProject/LiuClaw/ai/converters/messages.py) 的 `convert_context_for_provider()`。

它的处理顺序很明确：

1. `ensure_context()` 规范化
2. `apply_model_capabilities()` 根据模型能力裁剪
3. `convert_thinking_for_provider()` 处理 thinking 兼容
4. `convert_messages_for_provider()` 转历史消息
5. `convert_tools_for_provider()` 转工具定义

最终返回的仍然是统一 `Context`，只是其中消息和工具的元数据更接近目标 provider 所需格式。

## 7.1 能力裁剪

[ai/converters/capabilities.py](/Users/admin/PyCharmProject/LiuClaw/ai/converters/capabilities.py) 当前重点处理图片能力。

如果模型 `supports_images=False`，则图片块不会直接保留，而是替换成文本：

`[image omitted by capability clamp]`

这一行为有测试覆盖，见 [tests/test_model_registry.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_model_registry.py)。

## 7.2 Thinking 兼容

[ai/converters/thinking.py](/Users/admin/PyCharmProject/LiuClaw/ai/converters/thinking.py) 不直接删除历史 thinking，而是把 assistant 消息里的思考文本额外写入 `metadata["historicalThinking"]`，方便目标 provider 或上层调试逻辑读取。

## 7.3 消息与工具转换

[ai/converters/messages.py](/Users/admin/PyCharmProject/LiuClaw/ai/converters/messages.py) 和 [ai/converters/tools.py](/Users/admin/PyCharmProject/LiuClaw/ai/converters/tools.py) 本身转换得比较保守：

- 不会把消息转成厂商原始 JSON
- 而是复制统一对象，并通过 `metadata["targetProvider"]` 标记目标 provider
- 工具定义默认补 `metadata["schemaDialect"] = "jsonschema"`

这种设计说明：真正的“最后一跳厂商请求构造”仍然留在 provider 文件里，converter 层主要负责统一对象的轻量兼容和能力约束。

测试 [tests/test_converters.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_converters.py) 验证了：

- 历史消息在跨 provider 转换中不会丢失
- 工具 schema 会被保留

## 8. Reasoning 映射

[ai/reasoning.py](/Users/admin/PyCharmProject/LiuClaw/ai/reasoning.py) 把统一 reasoning 等级映射到各 provider 自己的参数格式。

统一等级定义在 `types.py`：

- `off`
- `minimal`
- `low`
- `medium`
- `high`
- `xhigh`

当前映射规则如下：

### 8.1 OpenAI

- `off` -> 不附加 reasoning 参数
- 其他等级 -> `{"reasoning": {"effort": level}}`

### 8.2 Anthropic

- `off` / `minimal` -> 不启用 thinking
- `low` -> `budget_tokens = 1024`
- `medium` -> `budget_tokens = 4096`
- `high` -> `budget_tokens = 8192`
- `xhigh` -> `budget_tokens = 16384`

### 8.3 Zhipu

- `off` / `minimal` / `low` -> `{"thinking": {"type": "disabled"}}`
- `medium` -> `{"thinking": {"type": "enabled"}}`
- `high` / `xhigh`
  - 如果模型是 `glm-4.6` -> `{"thinking": {"type": "enabled"}}`
  - 其他模型 -> `{"thinking": {"type": "enabled"}, "clear_thinking": False}`

`merge_reasoning_metadata()` 会把映射结果写到 `Options.metadata["_providerReasoning"]`。provider 自己只负责读取这个结果，不重复做一遍映射。

测试 [tests/test_reasoning.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_reasoning.py) 对这几类映射都有覆盖。

## 9. 工具类与通用基础设施

## 9.1 上下文窗口检测

[ai/utils/context_window.py](/Users/admin/PyCharmProject/LiuClaw/ai/utils/context_window.py) 提供：

- `estimate_context_tokens()`
- `detect_context_overflow()`
- `ensure_context_fits_window()`
- `truncate_context_to_window()`

实现不是 tokenizer 级精确计数，而是保守估算：

- 文本按 `max(1, (len(text) + 2) // 3)` 估算
- 消息、工具定义、工具参数、图片元数据都会参与预算

如果配置 `Options.contextOverflowStrategy == "truncate_oldest"`，则会不断删除最旧消息，直到预算重新落回窗口。

测试验证了“按最旧优先裁剪消息”的行为。

## 9.2 Unicode 清理

[ai/utils/unicode.py](/Users/admin/PyCharmProject/LiuClaw/ai/utils/unicode.py) 在进入 provider 之前对上下文做 NFKC 规范化，并移除危险控制字符。

它会清理：

- `systemPrompt`
- 各类消息里的文本字段
- `Tool`
- `ToolCall`

测试 [tests/test_utils.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_utils.py) 验证了零宽字符会被移除。

## 9.3 工具参数校验

[ai/utils/schema_validation.py](/Users/admin/PyCharmProject/LiuClaw/ai/utils/schema_validation.py) 实现了一个简化版 JSON Schema 校验器，当前支持：

- `object`
- `array`
- `string`
- `integer`
- `number`
- `boolean`
- `null`

同时支持：

- `required`
- `additionalProperties`
- `minItems` / `maxItems`
- `minLength` / `maxLength`
- `enum`
- `minimum` / `maximum`

这部分目前是同步校验逻辑，定位是工具调用前的轻量参数守卫。

## 9.4 流式辅助

[ai/utils/streaming.py](/Users/admin/PyCharmProject/LiuClaw/ai/utils/streaming.py) 提供了一组与 provider 无关的流式工具：

- `EventBuilder`
- `StreamAccumulator`
- `create_event_queue()`
- `enqueue_event()`
- `consume_queue()`
- `drain_queue_to_accumulator()`
- `forward_stream_to_queue()`
- `finalize_producer_error()`
- `cancel_producer_task()`
- `create_done_event()`

其中最重要的是两个类：

### `EventBuilder`

负责快速构造统一 `StreamEvent`，自动填充模型和 provider 默认值，还能通过 `build_error()` 统一生成错误事件。

### `StreamAccumulator`

负责把一串 `StreamEvent` 聚合成 `AssistantMessage`：

- 文本增量会追加成 `TextContent`
- thinking 增量会追加成 `ThinkingContent`
- 工具调用会先建占位，再逐步拼接参数
- 收到 `done` 时，如果事件里已经带完整 `assistantMessage`，则直接以该消息为准

也正因为有这层，`complete()` 才能复用 `stream()` 的结果，而不用每个 provider 各自实现一套“非流式完整返回”逻辑。

## 10. 错误模型

[ai/errors.py](/Users/admin/PyCharmProject/LiuClaw/ai/errors.py) 定义了统一错误体系：

- `AIError`
- `ProviderNotFoundError`
- `AuthenticationError`
- `UnsupportedFeatureError`
- `ProviderResponseError`

这些错误在模块里的职责边界比较明确：

- 找不到 provider 或模型映射异常时，用 `ProviderNotFoundError`
- 缺少 API key 或认证失败时，用 `AuthenticationError`
- 某 provider 无法支持某能力映射时，用 `UnsupportedFeatureError`
- provider 返回格式错误、SDK 异常或流式处理失败时，用 `ProviderResponseError`

需要注意的是，provider 层多数时候不会直接把异常抛到上层，而是先转成统一 `error` 事件；最终 `complete()` 在消费到 `error` 事件时，再抛出 `ProviderResponseError`。

## 11. 测试反映出的设计重点

从 `tests/` 里的相关用例可以看到，这个模块当前最看重以下能力：

### 11.1 流式协议稳定

[tests/test_client.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_client.py) 验证了：

- `stream()` 返回的是带队列和后台任务的 `StreamSession`
- `complete()` 能正确从流式事件聚合出最终消息
- provider 发出 `error` 事件时，上层能感知失败
- 简化接口 `streamSimple()` 会正确构造 `Options`
- reasoning 会按模型能力自动钳制

### 11.2 注册与配置可扩展

[tests/test_registry.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_registry.py)、[tests/test_registry_config.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_registry_config.py)、[tests/test_model_registry.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_model_registry.py) 验证了：

- provider 是懒加载的
- provider 配置可以从注册表注入
- 模型定义和 provider 配置可以通过本地文件覆盖

### 11.3 兼容多 provider 的中间层设计

[tests/test_converters.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_converters.py)、[tests/test_zhipu_provider.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_zhipu_provider.py)、[tests/test_reasoning.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_reasoning.py) 表明：

- 历史消息不能在转换中丢失
- 工具 schema 需要保留
- thinking 要能跨 provider 传递
- 不同厂商的 reasoning 参数必须在统一层完成收敛

## 12. 一次完整调用的真实链路

把代码串起来，一次 `await complete(model, context, options)` 的实际链路如下：

1. 上层传入模型 ID 或 `Model`、统一 `Context` 和 `Options`。
2. `client.complete()` 调用 `client.stream()`。
3. `stream()` 从 `ModelRegistry` 解析运行时模型。
4. `stream()` 调用 `_prepare_options()`，完成 reasoning 钳制和 provider 元数据附加。
5. `stream()` 调用 `_prepare_context()`，完成上下文规范化、Unicode 清理、能力裁剪、thinking 兼容、消息和工具转换。
6. `stream()` 检测上下文窗口，必要时按策略裁剪旧消息。
7. `stream()` 通过 `ProviderRegistry.resolve()` 拿到目标 provider。
8. provider 构造厂商请求并输出统一 `StreamEvent`。
9. `_produce_events()` 把事件写入有界队列。
10. `StreamSession.consume()` 从队列对外提供统一事件流。
11. `complete()` 用 `StreamAccumulator` 聚合事件。
12. 收到 `done` 时返回 `AssistantMessage`；收到 `error` 时抛错。

这个链路体现出 `ai` 模块的真正边界：它不是“SDK 适配器集合”，而是“围绕统一协议组织的大模型运行时中间层”。
