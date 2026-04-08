# agent_core 模块文档

## 模块定位

`agent_core` 是本项目的 Agent 运行时内核，位于 [`agent_core/__init__.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/__init__.py)。它建立在 `ai` 模块提供的模型流式能力之上，但自己不关心 OpenAI、Anthropic 或其他 provider 的协议细节，而是专注解决下面几件事：

- 用统一的状态对象描述一次 Agent 运行过程
- 把用户消息、模型输出、工具调用和后续追问组织成可持续推进的循环
- 用事件流把运行中的关键节点暴露给上层
- 提供工具前置检查、工具结果后处理、provider 错误重试等扩展点
- 在低层 loop 之上再包一层高层 `Agent`，方便应用代码管理会话与队列

模块主体由三个文件组成：

- [`agent_core/types.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/types.py)：类型系统、配置对象、状态对象、事件对象和钩子协议
- [`agent_core/agent_loop.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/agent_loop.py)：真正执行循环的低层运行时
- [`agent_core/agent.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/agent.py)：对低层 loop 的高层封装，负责状态同步、消息队列和监听器管理

从代码分层上看，可以把它理解成：

- `types.py` 负责“定义问题”
- `agent_loop.py` 负责“执行问题”
- `agent.py` 负责“给业务侧一个更好用的控制面”

## 对外导出

[`agent_core/__init__.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/__init__.py) 暴露了这个模块的公共 API，主要包括：

- 高层入口：`Agent`、`AgentOptions`
- 低层入口：`agentLoop()`、`agentLoopContinue()`
- 配置与状态：`AgentLoopConfig`、`AgentState`、`AgentContext`、`AgentRuntimeFlags`
- 事件与错误：`AgentEvent`、`AgentError`
- 工具相关：`AgentTool`、`BeforeToolCall*`、`AfterToolCall*`
- 运行控制：`AbortSignal`、`RetryContext`、`RetryDecision`

测试 [`tests/test_agent.py`](/Users/admin/PyCharmProject/LiuClaw/tests/test_agent.py) 和 [`tests/test_agent_loop.py`](/Users/admin/PyCharmProject/LiuClaw/tests/test_agent_loop.py) 也明确验证了这些对外形状是当前模块的主接口。

## 一、核心数据模型

### 1. `AbortSignal`

定义位置：[`agent_core/types.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/types.py)

这是模块内部统一使用的中断信号对象，核心能力有三件：

- `abort(reason)`：标记中断并记录原因
- `wait()`：异步等待中断发生
- `throw_if_aborted()`：在已中断时抛出 `asyncio.CancelledError`

`agent_loop.py` 中几乎所有关键步骤都会调用 `signal.throw_if_aborted()`，所以取消不是只在最外层发生，而是会穿透：

- 打开模型流之前
- 获取 steering 或 follow-up 消息时
- 消费 provider 事件时
- 执行工具前后

这使得 `cancel()` 行为可以在较细粒度上生效。

### 2. `AgentError`

定义位置：[`agent_core/types.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/types.py)

统一错误结构如下：

- `kind`：错误分类
- `message`：面向运行时和事件流的主错误文本
- `details`：可选细节
- `retriable`：是否可重试

当前错误种类由 `AgentErrorKind` 约束为四类：

- `provider_error`
- `tool_error`
- `runtime_error`
- `aborted`

对应关系在实现中很清楚：

- provider 流打开失败或返回 error event 时，生成 `provider_error`
- 工具不存在、参数校验失败、工具执行异常时，生成 `tool_error`
- hook 返回了不支持的结果，或监听器抛错，通常记为 `runtime_error`
- loop 或桥接任务被取消时，记为 `aborted`

### 3. `AgentRuntimeFlags`

定义位置：[`agent_core/types.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/types.py)

这是纯运行期字段，不是业务消息历史的一部分，包含：

- `isStreaming`
- `isRunning`
- `isCancelled`
- `turnIndex`
- `retryCount`

它的职责是把“当前 loop 正在做什么”从历史消息中分离出来。实现里这些值主要在 [`agent_core/agent_loop.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/agent_loop.py) 中被驱动更新。

### 4. `AgentContext`

定义位置：[`agent_core/types.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/types.py)

这是一次模型调用或工具 hook 使用的上下文快照，字段很少：

- `systemPrompt`
- `messages`
- `tools`

它和 `ai.types.Context` 很像，但不是直接复用，而是作为 `agent_core` 自己的中间表示。原因是这个模块允许通过 `convert_to_llm` 和 `transform_context` 先改写上下文，再交给 `ai`。

### 5. `AgentTool`

定义位置：[`agent_core/types.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/types.py)

`AgentTool` 继承自 `ai.types.Tool` 的字段结构，并额外增加可执行器：

- `name`
- `description`
- `inputSchema`
- `metadata`
- `renderMetadata`
- `executor`

为了兼容旧代码，它也支持 `execute` 这个别名属性。真正执行工具时，`agent_loop.py` 会优先走新签名：

```python
executor(tool_call_id, params, signal, on_update)
```

如果因为签名不匹配抛出 `TypeError`，则回退到兼容旧形式：

```python
executor(arguments_text, agentContext)
```

这说明当前实现同时兼容新旧两套工具执行约定。

### 6. `AgentState`

定义位置：[`agent_core/types.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/types.py)

这是整个模块最重要的状态对象，字段包括：

- `systemPrompt`
- `model`
- `thinking`
- `tools`
- `messages`
- `stream_message`
- `pending_tool_calls`
- `error`
- `runtime_flags`

此外它保留了一组兼容旧命名的属性：

- `history` 对应 `messages`
- `currentMessage` 对应 `stream_message`
- `runningToolCall` 对应 `pending_tool_calls[0]`
- `isStreaming` 对应 `runtime_flags.isStreaming`

当前状态设计体现了一个很明确的分层：

- `messages` 保存已经进入会话历史的消息
- `stream_message` 保存当前尚在流式生成中的 assistant 消息
- `pending_tool_calls` 保存当前正在执行的工具调用
- `error` 保存本轮最后一次显式错误
- `runtime_flags` 保存纯运行标记

这组字段被高层 `Agent` 和低层 loop 同时共享，是整个系统的状态中枢。

## 二、配置和扩展点

### 1. `AgentLoopConfig`

定义位置：[`agent_core/types.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/types.py)

`AgentLoopConfig` 是低层运行时的总配置对象，字段如下：

- `systemPrompt`
- `model`
- `thinking`
- `tools`
- `stream`
- `convert_to_llm`
- `transform_context`
- `get_steering_messages`
- `get_follow_up_messages`
- `toolExecutionMode`
- `beforeToolCall`
- `afterToolCall`
- `retryPolicy`
- `registry`

同时保留兼容别名：

- `steer` 等价于 `get_steering_messages`
- `followUp` 等价于 `get_follow_up_messages`

这些字段大致可以分成四组：

#### 模型调用相关

- `model`
- `thinking`
- `stream`
- `registry`

如果 `stream` 为空，低层会使用默认的 `_default_stream()`，内部调用 `ai.streamSimple(...)`。

#### 上下文转换相关

- `convert_to_llm`
- `transform_context`

执行顺序是：

1. 从 `AgentState.messages` 中选出要送给模型的消息
2. 组装成 `AgentContext`
3. 再做二次变换
4. 最终转成 `ai.Context`

默认行为分别是：

- `default_convert_to_llm()`：只保留 user、assistant、tool result 这几类标准消息
- `default_transform_context()`：不做改写

#### 工具控制相关

- `tools`
- `toolExecutionMode`
- `beforeToolCall`
- `afterToolCall`

这组字段决定工具是否可执行、怎样执行、是否允许执行前短路，以及执行后是否替换结果。

#### 循环推进相关

- `get_steering_messages`
- `get_follow_up_messages`
- `retryPolicy`

其中：

- steering 消息用于“在当前运行中追加新的用户输入或控制输入”
- follow-up 消息用于“当一轮内层循环结束后，再决定是否继续下一轮”
- retry policy 只处理 provider 级错误，不处理工具错误

### 2. 工具前后置 hook

定义位置：[`agent_core/types.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/types.py)

#### `BeforeToolCallContext`

包含：

- `state`
- `tool`
- `toolCall`
- `params`
- `assistantMessage`
- `agentContext`
- `signal`

这个上下文已经完成了：

- 工具查找
- 参数 JSON 解析
- 参数 schema 校验
- 发送给模型的上下文构造

也就是说，`beforeToolCall` 拿到的是“已准备完毕但尚未执行”的工具调用。

#### `BeforeToolCallResult`

支持三种结果：

- `BeforeToolCallAllow`：允许继续执行
- `BeforeToolCallSkip`：跳过真实执行，直接返回替代结果
- `BeforeToolCallError`：阻止执行，并产出错误结果

测试覆盖见：

- [`tests/test_agent_loop.py`](/Users/admin/PyCharmProject/LiuClaw/tests/test_agent_loop.py)

其中验证了：

- allow 会正常执行工具
- skip 不会触发工具执行器
- error 会生成 `metadata["error"] = True` 的工具结果

#### `AfterToolCallContext`

相比前置 hook，它多了：

- `result`

表示工具已经执行完且结果已被标准化为 `ToolResultMessage`。

#### `AfterToolCallResult`

支持两种结果：

- `AfterToolCallPass`：保留原结果
- `AfterToolCallReplace`：用新结果替换原结果

如果 hook 返回其他不支持的对象，当前实现会把它记为 `runtime_error`。

### 3. 重试策略

定义位置：[`agent_core/types.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/types.py)

由两部分组成：

- `RetryContext`
- `RetryDecision`

`RetryContext` 包含：

- `error`
- `state`
- `attempt`
- `signal`

`RetryDecision` 包含：

- `shouldRetry`
- `delaySeconds`

默认策略 `default_retry_policy()` 永远不重试。真正生效的地方在 [`agent_core/agent_loop.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/agent_loop.py) 的 `_handle_provider_error()`。

需要注意的是：

- 这里只处理 provider 错误
- 工具执行异常不会走这套重试策略
- 如果决定重试，会累加 `state.runtime_flags.retryCount`

## 三、低层运行时：`agent_loop.py`

### 1. 文件职责

[`agent_core/agent_loop.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/agent_loop.py) 是真正驱动 Agent 运行的地方，负责：

- 初始化和快照状态
- 把消息转换为模型上下文
- 调用 `ai.streamSimple` 或自定义 `stream`
- 把 provider 的流式事件映射成 `AgentEvent`
- 解析并执行工具调用
- 处理 steering 和 follow-up 消息
- 管理取消、异常与结束事件

这个文件的设计不是“一次请求一次响应”，而是“一个会话里可能跑多轮 turn，并在每轮之间插入控制消息”。

### 2. 状态快照与消息归一化

几个基础辅助函数非常重要：

- `_copy_tools()`
- `_copy_messages()`
- `_snapshot_state()`
- `_normalize_messages()`

它们共同保证两件事：

- 事件流中发出的 `state` 是快照，不直接暴露内部可变引用
- 输入消息既可以是 dataclass 消息对象，也可以是带 `role` 的 dict

这也是高层 `Agent` 和低层 loop 都在频繁做 copy 的原因，目的是减少共享可变对象导致的状态串改。

### 3. 从 `AgentState` 到模型上下文

相关函数：

- `_to_agent_context()`
- `_context_to_llm_context()`
- `to_llm_context()`

其中：

- `_to_agent_context()` 是异步路径，给运行中的 loop 用
- `to_llm_context()` 是同步路径，更像一个辅助函数或调试入口

异步路径的执行顺序如下：

1. 读取 `loop.convert_to_llm`，没有则用 `default_convert_to_llm`
2. 基于 `state.messages` 得到要送给 LLM 的消息
3. 把 `state.tools` 转成纯 `ai.types.Tool`
4. 组装成 `AgentContext`
5. 读取 `loop.transform_context`，没有则用默认实现
6. 返回变换后的 `AgentContext`
7. 在真正调用 `ai` 时，再转成 `ai.types.Context`

这意味着：

- `AgentState` 是内部真实状态
- `AgentContext` 是模型调用前的中间投影
- `ai.Context` 是最终 provider 调用输入

### 4. 打开模型流

相关函数：

- `_default_stream()`
- `_open_stream()`

默认流函数 `_default_stream()` 最终会调用：

```python
ai.streamSimple(model, context, reasoning=thinking, registry=registry)
```

如果配置了自定义 `stream`，则 `_open_stream()` 会改为调用 `loop.stream(...)`。实现里还做了一个兼容处理：

- 先尝试带 `signal=...` 调用
- 如果自定义函数不接受这个参数，则退回旧签名

因此文档上应把 `stream` 理解为“可覆盖默认 LLM 流入口的适配器”。

### 5. assistant 消息流转

相关函数：

- `_append_text_delta()`
- `_append_thinking_delta()`
- `streamAssistantResponse()`

`streamAssistantResponse()` 是单轮模型响应的核心函数。它会：

1. 打开底层 provider 流
2. 把 provider 的 `StreamEvent` 转换成 `AgentEvent`
3. 维护 `state.stream_message`
4. 在收到 `done` 时把最终 `AssistantMessage` 追加进 `state.messages`

它处理的 provider 事件包括：

- 文本增量
- thinking 增量
- tool call 开始、更新、结束
- provider error
- done

特别要注意两点：

#### 第一，`message_start` / `message_update` / `message_end` 是 Agent 级事件

它们不是 provider 原始事件，而是 `agent_core` 自己映射出来的语义事件。

#### 第二，即使 provider 返回的是纯工具调用消息，也仍然会有消息生命周期事件

这一点有测试显式验证：

- [`tests/test_agent_loop.py`](/Users/admin/PyCharmProject/LiuClaw/tests/test_agent_loop.py)

也就是说，外部消费者可以统一依赖消息生命周期，而不必区分“这是文本回复还是工具调用回复”。

### 6. 工具调用流水线

相关函数：

- `prepareToolCall()`
- `executePreparedToolCall()`
- `emitToolCallOutcome()`
- `finalizeExecutedToolCall()`
- `executeToolCallsSequential()`
- `executeToolCallsParallel()`
- `executeToolCalls()`

可以把它拆成四个阶段看。

#### 阶段 A：准备

`prepareToolCall()` 会依次完成：

1. 根据 `tool_call.name` 查找 `AgentTool`
2. 检查工具是否存在且具备 `executor`
3. 解析参数
4. 用 `validate_tool_arguments()` 做 schema 校验
5. 构造 `BeforeToolCallContext`
6. 执行 `beforeToolCall`

这个阶段的产出有两类：

- `PreparedToolCall`
- `PreparedToolCallError`

如果是 `PreparedToolCallError`，可能代表：

- 参数错误
- 工具不存在
- hook 直接返回错误
- hook 直接返回替代结果

#### 阶段 B：执行

`executePreparedToolCall()` 会：

- 把当前 tool call 记入 `state.pending_tool_calls`
- 先发一条 `tool_execution_update`，内容是 `{"status": "running"}`
- 调用工具执行器
- 提供 `on_update` 回调，允许工具中途发进度更新
- 在结束后清空 `state.pending_tool_calls`

这里事件顺序有一个小特点：

- `tool_execution_start` 在更外层发出
- `tool_execution_update` 表示进入运行态
- `tool_execution_end` 才携带最终结果

#### 阶段 C：结果后处理

`finalizeExecutedToolCall()` 会先把任意执行结果标准化为 `ToolResultMessage`，再执行 `afterToolCall`。如果 hook 返回 `AfterToolCallReplace`，结果会被替换。

#### 阶段 D：落盘到历史

`emitToolCallOutcome()` 负责：

- 发出 `tool_execution_end`
- 把工具结果追加到 `state.messages`
- 再补发一组 `message_start` / `message_end`

这里很关键，因为它说明在 `agent_core` 里，工具结果既是“工具执行结果”，也是“会话消息历史的一部分”。

### 7. 串行与并行工具执行

当前支持两种模式：

- `serial`
- `parallel`

串行模式由 `executeToolCallsSequential()` 实现，逻辑最直观：一个个执行、一个个入历史。

并行模式由 `executeToolCallsParallel()` 实现，特点有两个：

- 先完成所有 tool call 的准备阶段
- 用 `asyncio.gather(...)` 并发执行真正的工具逻辑

但最终结果不会按完成先后顺序写回，而是会按 assistant 原始 `toolCalls` 的顺序重排。测试里有明确验证：

- 慢工具先定义、快工具后定义时，最终历史里仍保持原始顺序

因此这里的设计目标是：

- 运行上并行
- 语义上稳定
- 历史顺序可预期

### 8. Steering 与 Follow-up

相关函数：

- `_resolve_control_messages()`
- `runLoop()`

这是 `agent_core` 区别于简单“单次问答封装”的核心设计之一。

#### Steering 消息

`get_steering_messages` 会在每轮 turn 结束后检查。若返回新消息，这些消息会先进入历史，然后立即推动下一轮模型调用。

适合用于：

- 中途插入用户补充说明
- 自动纠偏
- 规划式多步推进

#### Follow-up 消息

`get_follow_up_messages` 会在内层循环退出之后再检查。若有消息，则重新进入下一轮外层推进。

适合用于：

- 本轮收尾后追加复查任务
- 在某个阶段完成后发起下一阶段
- 收束式多轮任务编排

测试显示两者的先后顺序是：

- steering 优先于 follow-up
- follow-up 只会在内层 turn 流程结束后运行

### 9. 主循环结构

相关函数：

- `runLoop()`
- `runAgentLoop()`

可以把运行过程概括为下面这条主线：

1. `runAgentLoop()` 发出 `agent_start`
2. 初始化 `turnIndex = 1` 并发出第一条 `turn_start`
3. 把本次新输入消息写入历史并发消息生命周期事件
4. 进入 `runLoop()`
5. `runLoop()` 先处理 steering 消息
6. 调用 `streamAssistantResponse()` 获取 assistant 回复
7. 若回复中含工具调用，则执行工具
8. 发出 `turn_end`
9. 再检查 steering，必要时继续下一轮 turn
10. 内层无更多 steering 后检查 follow-up
11. 如无更多 follow-up，发出 `agent_end`

从测试观察到的典型事件序列如下：

```text
agent_start
turn_start
message_start
message_end
message_start
message_update
message_update
message_end
turn_end
agent_end
```

如果中途有工具执行，还会穿插：

- `tool_execution_start`
- `tool_execution_update`
- `tool_execution_end`

### 10. 会话创建与继续运行

相关函数：

- `_createAgentLoopSession()`
- `agentLoop()`
- `agentLoopContinue()`

#### `agentLoop()`

用于启动一个新对话，会：

- 校验 `loop.model` 是否存在
- 用 `AgentLoopConfig` 构造一个新的 `AgentState`
- 把 `initialMessages` 规范化
- 创建 `StreamSession[AgentEvent]`
- 后台启动 `runAgentLoop(...)`

#### `agentLoopContinue()`

用于在已有历史上继续运行，有两个明确前置条件：

- `state.messages` 不能为空
- 历史最后一条消息不能是 `AssistantMessage`

第二条限制很关键，意味着继续运行时，历史尾部必须是用户消息或工具结果，这样模型才有新的上下文可回应。

它还会用传入的 `loop` 或从当前 `state` 推导出的配置，重新同步：

- `systemPrompt`
- `model`
- `thinking`
- `tools`

然后清空上一次的错误和取消标记，再启动新会话。

## 四、高层封装：`agent.py`

### 1. 文件职责

[`agent_core/agent.py`](/Users/admin/PyCharmProject/LiuClaw/agent_core/agent.py) 并不重写 loop 逻辑，而是在低层 `agentLoop` / `agentLoopContinue` 之上提供一个更适合业务代码直接使用的对象接口。

它主要解决四类问题：

- 状态持有和复制
- 高层消息队列管理
- 底层事件桥接
- 监听器订阅

### 2. `AgentOptions`

`AgentOptions` 包含：

- `loop`
- `initialState`
- `listeners`
- `pendingMessages`
- `steeringMessages`
- `followUpMessages`
- `autoCopyState`

它体现出高层 `Agent` 的设计重点不是“把配置塞满”，而是“围绕一次长生命周期对象保存运行上下文”。

### 3. 初始化与状态复制

`Agent.__init__()` 支持两种输入：

- 直接传 `AgentLoopConfig`
- 传 `AgentOptions`

如果提供 `initialState`，默认会先做深一些的复制；否则会根据 loop 构造一个初始状态。对应辅助函数是：

- `_copy_message()`
- `_copy_state()`
- `_build_initial_state()`

这里的目标非常明确：避免调用方和 `Agent` 内部共享同一批可变状态对象。

### 4. 高层持有的三类消息队列

与低层直接传消息不同，高层 `Agent` 额外维护三组队列：

- `_pendingMessages`
- `_steeringMessages`
- `_followUpMessages`

对应公开接口包括：

- `enqueue()` / `dequeueAll()` / `clearQueue()` / `queueSize()`
- `enqueueSteering()` / `dequeueSteeringAll()` / `clearSteeringQueue()` / `steeringQueueSize()`
- `enqueueFollowUp()` / `dequeueFollowUpAll()` / `clearFollowUpQueue()` / `followUpQueueSize()`

测试里也专门验证了三者是相互独立的。

这层设计的意义在于：

- 业务侧可以先积攒普通消息，再统一 `run()`
- 也可以在运行前预放 steering 或 follow-up 消息
- 高层对象比低层函数更适合做长期会话控制

### 5. 对底层 loop 的桥接

核心方法是 `_runLoopSession()`。

它会：

1. 检查是否重复运行
2. 根据是否有历史和是否有新消息决定走 `agentLoop()` 还是 `agentLoopContinue()`
3. 调用 `_buildLoopConfig()` 生成一次新的 loop 配置
4. 创建新的 `AbortSignal`
5. 启动底层 session
6. 起一个桥接任务消费底层事件
7. 每消费一条事件，就调用 `_handleLoopEvent()` 同步高层 `self.state`
8. 再把事件分发给监听器和对外 session

这里最重要的不是“简单转发”，而是“高层自己的状态始终跟着底层事件走”。因此 `Agent.state` 会随着事件流逐步更新，而不是只在结束时才替换。

### 6. `_buildLoopConfig()` 的作用

这个方法会把高层内部状态和原始 loop 配置合并成一次新的低层配置。它尤其做了一个关键动作：

- 把高层的 steering 队列和 follow-up 队列包成新的 `get_steering_messages` / `get_follow_up_messages` 函数

也就是说，高层队列不是让业务代码自己手工塞进低层，而是由高层在启动时动态注入到 loop 配置中。

如果原始 loop 自己也定义了 hook，那么高层会把：

- 本地队列中的消息
- 原始 hook 返回的消息

合并起来一起交给低层执行。

### 7. 事件处理与监听器

相关方法：

- `_handleLoopEvent()`
- `_emit_to_listeners()`
- `emit()`
- `subscribe()`
- `unsubscribe()`
- `clearListeners()`

处理逻辑大致如下：

- 若事件里带 `state`，先复制后同步到 `self.state`
- 再根据事件类型修正高层特定字段
- 再把事件分发给所有监听器
- 最后写入高层自己的输出队列

其中 `_handleLoopEvent()` 重点维护：

- `stream_message`
- `pending_tool_calls`
- `error`
- `runtime_flags.isStreaming`
- `runtime_flags.isRunning`

监听器出错不会打断主流程，测试显式验证了这一点。当前行为是：

- 把监听器异常记到 `agent.state.error`
- 不中止整体运行

### 8. 用户态 API

高层最常用的方法有：

- `prompt(message)`：先 `enqueue()` 再立即运行
- `run()`：如果普通队列非空，则把队列作为新消息启动；否则按继续对话模式运行
- `continueConversation()`：从已有上下文继续，不追加新消息
- `resume()`：`continueConversation()` 的别名
- `cancel()`：取消当前运行
- `wait()`：等待当前运行结束
- `reset()`：重置状态和全部队列

这里有几个边界值得特别说明：

#### `run()` 的分支逻辑

- 队列里有普通消息时，走“新消息驱动的一轮运行”
- 队列为空时，走“基于已有历史继续运行”

#### `continueConversation()` 的前提

和低层一致，必须已经有历史，而且不能从 assistant 尾消息继续。

#### `reset()` 的限制

运行中不能 reset，否则会抛 `RuntimeError`。

### 9. 取消与清理

相关方法：

- `cancel()`
- `wait()`
- `_cleanup_after_run()`

取消时会同时处理三件事：

- 标记 `_cancelRequested`
- 触发 `AbortSignal.abort(...)`
- 取消当前 session 的生产任务与桥接任务

结束清理时会重置：

- `_isRunning`
- `_currentSession`
- `_currentTask`
- `_abortSignal`
- `state.runtime_flags.isStreaming`
- `state.runtime_flags.isRunning`
- `state.stream_message`
- `state.pending_tool_calls`

因此高层 `Agent` 是显式区分“历史保留”和“运行时控制状态清空”的。

## 五、典型运行流程

把高层和低层串起来看，一次最常见的运行路径如下：

1. 业务代码创建 `AgentLoopConfig`
2. 用它构造 `Agent` 或直接调用 `agentLoop()`
3. 新消息进入 `state.messages`
4. loop 发出 `agent_start`、`turn_start`
5. 当前轮的输入消息发出 `message_start`、`message_end`
6. 组装 `AgentContext` 并打开模型流
7. assistant 文本或工具调用增量转成 `message_update`
8. assistant 完成后写入历史并发 `message_end`
9. 若有工具调用，执行工具并把 `ToolResultMessage` 写入历史
10. 发出 `turn_end`
11. 检查 steering，再检查 follow-up
12. 无后续工作后发出 `agent_end`

如果使用高层 `Agent`，这个过程中还会多一层：

- 底层事件先进入桥接器
- 高层同步自身状态
- 监听器收到事件
- 最后外部消费者从高层 session 里读取事件

## 六、模块边界

### `agent_core` 负责什么

- Agent 运行状态模型
- 多轮 turn 控制
- 工具执行与工具 hook
- 事件流封装
- 会话继续、取消和高层对象管理

### `agent_core` 不负责什么

- 不直接实现 provider 协议
- 不负责模型注册细节
- 不负责 CLI、TUI 或交互界面
- 不负责工作区资源装载、技能系统、主题或扩展

这些能力分别落在其他模块，尤其是 `ai` 和 `coding_agent`。

## 七、测试透露出的设计意图

从 [`tests/test_agent_loop.py`](/Users/admin/PyCharmProject/LiuClaw/tests/test_agent_loop.py) 和 [`tests/test_agent.py`](/Users/admin/PyCharmProject/LiuClaw/tests/test_agent.py) 可以总结出当前实现刻意保证的行为：

- `agentLoop()` 会立即返回 `StreamSession`，不会阻塞到整轮结束
- 默认底层流函数是 `ai.streamSimple`
- 允许完全自定义 `stream`，并优先于默认实现
- `agentLoopContinue()` 必须建立在有效历史上
- 工具前后置 hook 可以改变执行路径和最终结果
- 并行工具执行时，结果顺序仍和原始 tool call 顺序一致
- steering 的优先级高于 follow-up
- 纯工具调用消息也必须有完整消息生命周期事件
- 高层 `Agent` 的三类消息队列是分离的
- 监听器报错不会破坏主流程
- 公共类与关键方法带有中文 docstring

这些测试基本定义了 `agent_core` 当前版本的行为契约。

## 八、适合怎样使用

如果你只需要：

- 给定消息
- 执行一轮或多轮 Agent 循环
- 自己消费事件流

那么直接使用 `agentLoop()` / `agentLoopContinue()` 就够了。

如果你还需要：

- 长生命周期对象
- 累积消息队列
- 监听器订阅
- 随时取消和等待
- 高层状态读写接口

那么更适合使用 `Agent`。

## 九、小结

`agent_core` 的实现重点不是“再包一层模型调用”，而是把模型回复、工具执行、控制消息和运行状态整合成一个可持续推进的 Agent 循环。

它的核心价值体现在三点：

- 用 `AgentState` 和 `AgentEvent` 把运行过程变成显式状态机
- 用 `AgentLoopConfig` 提供清晰的扩展点，而不是把逻辑写死
- 用高层 `Agent` 把底层异步循环包装成更可控的会话对象

如果把整个项目看成一套分层架构，那么 `agent_core` 就是连接 `ai` 能力层和 `coding_agent` 产品层的中间运行时。
