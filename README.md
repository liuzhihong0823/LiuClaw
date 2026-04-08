# LiuClaw

LiuClaw 是一个分层清晰的终端智能体项目，核心由三部分组成：

- `ai`：统一大模型接入层，负责模型协议、provider 适配、流式事件、上下文预处理与工具参数兼容。
- `agent_core`：Agent 运行时内核，负责多轮循环、工具调用、状态管理、事件流、中断与重试。
- `coding_agent`：产品编排层，负责 CLI、TUI、会话持久化、工具注册、资源加载、扩展机制与上下文压缩。

如果把整个仓库看成一条运行链路，可以理解为：

`coding_agent` 负责把应用装起来，`agent_core` 负责把 Agent 跑起来，`ai` 负责把模型调起来。

## 项目结构

```text
LiuClaw/
├── ai/                  # 统一 LLM 接入层
├── agent_core/          # Agent 运行时内核
├── coding_agent/        # 终端产品层与交互入口
├── tests/               # 模块级行为测试
├── examples/            # 基础调用示例
├── pyproject.toml
└── README.md
```

## 分层说明

### 1. `ai`

`ai` 模块把不同厂商的大模型能力统一成一致接口，核心能力包括：

- 统一类型体系：`Context`、`Model`、`AssistantMessage`、`ToolCall`、`StreamEvent`
- 统一调用入口：`stream()`、`complete()`、`streamSimple()`、`completeSimple()`
- provider 注册与懒加载：`ProviderRegistry`
- 模型目录与配置覆盖：`ModelRegistry`、`AIConfig`
- provider 前转换与清理：messages/tools/thinking/capabilities/context-window/unicode
- 统一流式聚合与错误模型

更详细说明见 [ai/ai模块.md](/Users/admin/PyCharmProject/LiuClaw/ai/ai模块.md)。

### 2. `agent_core`

`agent_core` 建立在 `ai` 之上，负责把“模型输出 + 工具执行 + 多轮会话”组织成可持续推进的 Agent 循环，核心能力包括：

- 低层循环入口：`agentLoop()`、`agentLoopContinue()`
- 高层封装：`Agent`
- 状态模型：`AgentState`、`AgentRuntimeFlags`
- 统一事件：`AgentEvent`
- 工具前后钩子：`beforeToolCall`、`afterToolCall`
- steering / follow-up / retry / abort 等运行控制

更详细说明见 [agent_core/agent-core模块.md](/Users/admin/PyCharmProject/LiuClaw/agent_core/agent-core模块.md)。

### 3. `coding_agent`

`coding_agent` 是面向终端使用的产品层。它把模型、工具、系统提示、会话存储、扩展与交互界面组装成一个真正可运行的编码助手，核心能力包括：

- CLI 入口与 one-shot / interactive 两种模式
- 用户级与项目级配置加载
- 资源加载：skills、prompts、themes、`AGENTS.md`、extensions
- `AgentSession` 会话编排与事件映射
- 工具注册与安全策略
- session 持久化、恢复与分支摘要压缩
- 基于 `prompt_toolkit` 的交互界面

更详细说明见 [coding_agent/coding-agent模块.md](/Users/admin/PyCharmProject/LiuClaw/coding_agent/coding-agent模块.md)。

## 快速开始

### 安装依赖

```bash
uv sync
```

### 运行测试

```bash
uv run pytest
```

### 运行交互式编码助手

```bash
uv run python -m coding_agent
```

常用参数：

- `--model`：指定模型 ID
- `--cwd`：指定工作目录
- `--thinking`：`low`、`medium`、`high`
- `--session`：恢复历史会话
- `--new`：强制新建会话
- `--compact`：压缩当前会话后退出
- `--theme`：指定主题

### 单次提示模式

```bash
uv run python -m coding_agent "帮我总结当前项目结构"
```

## 作为库使用

### 直接使用 `ai`

```python
import asyncio

from ai import Context, UserMessage, complete


async def main() -> None:
    message = await complete(
        model="openai:gpt-5",
        context=Context(
            systemPrompt="你是一个简洁的中文助手。",
            messages=[UserMessage(content="请一句话介绍 LiuClaw。")],
        ),
    )
    print(message.content)


asyncio.run(main())
```

### 使用 `agent_core`

```python
import asyncio

from ai import UserMessage
from agent_core import Agent, AgentLoopConfig


async def main() -> None:
    agent = Agent(
        AgentLoopConfig(
            model="openai:gpt-5",
            systemPrompt="你是一个代码助手。",
        )
    )
    await agent.send(UserMessage(content="请解释这个仓库的三层结构。"))

    async for event in await agent.run():
        if event.type == "message_update":
            print(event.messageDelta, end="")


asyncio.run(main())
```

更多示例见 [examples/openai_simple.py](/Users/admin/PyCharmProject/LiuClaw/examples/openai_simple.py) 和 [examples/anthropic_simple.py](/Users/admin/PyCharmProject/LiuClaw/examples/anthropic_simple.py)。

## 关键设计

- 非流式调用不是单独实现的另一套路径，`complete()` 本质上是对 `stream()` 的统一聚合。
- `agent_core` 不直接依赖具体厂商协议，而是通过 `ai` 的统一流式接口驱动 Agent 循环。
- `coding_agent` 不把逻辑塞进入口函数，而是把配置、资源、工具、会话和交互拆到独立组件中装配。
- 会话摘要压缩与恢复是产品层能力，不污染 `ai` 与 `agent_core` 的纯运行时边界。
- 扩展机制已经预留工具、provider、监听器和系统提示扩展点，便于后续演进。

## 文档导航

- [ai/ai模块.md](/Users/admin/PyCharmProject/LiuClaw/ai/ai模块.md)
- [agent_core/agent-core模块.md](/Users/admin/PyCharmProject/LiuClaw/agent_core/agent-core模块.md)
- [coding_agent/coding-agent模块.md](/Users/admin/PyCharmProject/LiuClaw/coding_agent/coding-agent模块.md)
- [tests/test_client.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_client.py)
- [tests/test_agent_loop.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_agent_loop.py)
- [tests/test_coding_agent.py](/Users/admin/PyCharmProject/LiuClaw/tests/test_coding_agent.py)

## 适合从哪里读起

- 想看底层模型协议：先读 `ai`
- 想看 Agent 多轮循环：先读 `agent_core`
- 想看终端应用如何装配：先读 `coding_agent`
- 想看行为边界和当前契约：直接读 `tests/`
