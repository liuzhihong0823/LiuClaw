# coding_agent 模块说明

`coding_agent` 是 LiuClaw 面向终端产品层的执行模块。它不直接实现底层大模型流式循环，而是在 `ai` 与 `agent_core` 之上完成以下职责：

- 提供命令行入口与启动参数解析。
- 组装模型、配置、资源、工具、会话存储与运行时依赖。
- 维护可恢复的会话状态，并把底层 Agent 事件转换为产品层事件。
- 提供终端交互界面，包括输入、滚动、命令、状态栏与主题。
- 管理上下文压缩、扩展能力、技能与系统提示拼装。

从代码分层上看，`coding_agent` 更像“产品编排层”，负责把一套可运行的 Coding Agent 应用装起来，而不是单一算法模块。

## 目录结构

`coding_agent/` 当前主要分为四层：

- 入口层
  - `__main__.py`：模块入口，直接调用 `main()`。
  - `main.py`：程序启动主流程。
  - `cli/parser.py`：命令行参数定义与解析。
- 配置路径层
  - `config/paths.py`：约定 `~/.LiuClaw/agent/` 下的设置、模型、会话、技能、提示、主题、扩展目录，并定义项目级 `.LiuClaw/settings.json` 路径。
- 核心编排层 `core/`
  - 会话对象：`agent_session.py`
  - 运行时装配：`runtime_assembly.py`
  - 会话持久化：`session_manager.py`
  - 设置与模型：`settings_manager.py`、`model_registry.py`
  - 资源加载：`resource_loader.py` 及 `skills_loader.py`、`prompts_loader.py`、`themes_loader.py`、`agents_context_loader.py`、`extensions_runtime.py`
  - 系统提示：`system_prompt.py`、`system_prompt_builder.py`
  - 工具体系：`tools/`
  - 压缩体系：`compaction/`
  - 共享类型：`types.py`
- 交互层 `modes/interactive/`
  - `app.py`：交互应用入口
  - `controller.py`：输入命令与会话执行控制
  - `state.py`：界面状态树
  - `renderer.py`：基于 `prompt_toolkit` 的终端渲染

## 启动入口与主流程

### 1. 入口

`coding_agent/__main__.py` 只有一件事：

```python
from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
```

所以真正启动逻辑都在 `coding_agent/main.py`。

### 2. `main.py` 的职责

`main()` 会按顺序完成：

1. 用 `parse_args()` 解析 CLI 参数。
2. 根据 `--cwd` 解析工作区根目录。
3. 通过 `build_agent_paths()` 得到用户级目录结构，并调用 `ensure_exists()` 自动创建缺失目录与默认空配置文件。
4. 使用 `SettingsManager` 合并全局设置和项目级设置。
5. 用 `ModelRegistry` 解析内置模型与用户自定义模型。
6. 创建 `SessionManager` 与 `ResourceLoader`。
7. 构造 `AgentSession`，将模型、思考等级、配置、资源加载器、会话存储统一注入。
8. 若指定 `--session`，调用 `resume_session()` 恢复会话历史。
9. 若指定 `--compact`，直接压缩当前会话后退出。
10. 若命令行提供了单次 `prompt`，则直接执行一轮会话并把事件流打印到终端。
11. 否则进入 `InteractiveApp.run()`，启动完整的交互式 TUI。

这意味着 `main.py` 是整个模块的“装配入口”，但不直接处理业务细节。真正的执行控制被下沉到 `AgentSession` 和 interactive 模式里。

## CLI 能力

`cli/parser.py` 定义的参数比较克制，说明这个模块当前以单会话终端使用为核心：

- 位置参数 `prompt`：一条可选的 one-shot 提示词。
- `--model`：指定模型 ID。
- `--session`：恢复某个历史会话。
- `--cwd`：指定工作目录。
- `--thinking`：限定为 `low | medium | high`。
- `--new`：创建新会话。
- `--compact`：压缩活动会话并退出。
- `--theme`：切换主题。

这里没有把复杂逻辑塞进 CLI，而是把大部分行为留给进入交互模式后的斜杠命令处理。

## 配置与路径约定

### 1. 用户级目录

`config/paths.py` 规定默认根目录为：

`~/.LiuClaw/agent/`

其下包括：

- `settings.json`
- `models.json`
- `sessions/`
- `skills/`
- `prompts/`
- `themes/`
- `extensions/`

这说明 `coding_agent` 支持“用户级运行时环境”，而不是把所有东西都绑死在仓库内部。

### 2. 项目级设置

项目级配置固定为：

`<workspace_root>/.LiuClaw/settings.json`

`SettingsManager.load()` 会先读全局设置，再读项目设置，通过 `_deep_merge()` 做递归覆盖。最终生成 `CodingAgentSettings`，其中核心字段包括：

- `default_model`
- `default_thinking`
- `auto_compact`
- `compact_threshold`
- `compact_keep_turns`
- `compact_model`：压缩摘要时优先使用的模型
- `theme`
- `system_prompt_override`
- `tool_policy`

`ToolPolicy` 则定义工具侧的限制，例如：

- `max_read_chars`
- `max_command_chars`
- `max_ls_entries`
- `max_find_entries`
- `allow_bash`

所以配置体系的重点不是界面外观，而是“模型默认值 + 压缩策略 + 工具安全边界”。

## 运行时装配：`runtime_assembly.py`

这是 `coding_agent` 很关键的一层。`assemble_session_runtime()` 负责把一个会话运行所需的稳定依赖组装成 `SessionRuntimeAssembly`，其中包含：

- `resources`：加载好的资源包
- `tool_registry`：工具注册表
- `tools`：当前激活工具列表
- `provider_registry`：大模型 provider 注册表
- `prompt_builder`：系统提示构建器
- `compaction`：压缩协调器
- `listeners`：扩展注册的事件监听器

装配过程大致是：

1. `resource_loader.load()` 读取技能、提示、主题、项目上下文和扩展。
2. `build_tool_registry()` 创建内置工具注册表并激活工具。
3. 将扩展提供的工具追加注册到 `ToolRegistry`。
4. 创建或复用 `ProviderRegistry`，并注册扩展提供的 provider factory。
5. 构造 `SystemPromptBuilder` 与 `CompactionCoordinator`。
6. 收集扩展提供的事件监听器。

这层的意义是：`AgentSession` 不直接依赖一堆散乱组件，而是通过一次装配拿到一套可运行环境。

## 资源加载体系

### 1. `ResourceLoader`

`core/resource_loader.py` 统一编排资源加载，输出 `ResourceBundle`。内容包括：

- `skills`
- `prompts`
- `themes`
- `agents_context`
- `extensions`
- `extension_runtime`

并且会调用 `_ensure_no_conflicts()` 检查技能、提示、主题、扩展之间是否重名，避免运行时歧义。

### 2. 技能、提示、主题、项目上下文

- `skills_loader.py`
  - 递归扫描技能目录中的 `SKILL.md`。
  - 技能名默认取目录名。
- `prompts_loader.py`
  - 加载 `prompts/*.md`。
  - 若不存在 `SYSTEM.md`，自动提供默认系统提示。
- `themes_loader.py`
  - 加载 `themes/*.json`。
  - 若缺省，自动提供 `default` 主题。
- `agents_context_loader.py`
  - 从当前工作区一路向上查找最近的 `AGENTS.md`，将其作为项目上下文注入系统提示。

### 3. 扩展运行时

`extensions_runtime.py` 负责两步：

1. `scan_extensions()` 只扫描扩展目录，收集 `extension.json` 与模块路径，不执行代码。
2. `load_extension_runtime()` 再真正加载扩展模块，并调用其 `register(api)` 钩子。

扩展可通过 `ExtensionApi` 贡献：

- 新工具 `register_tool`
- 新命令 `register_command`
- 新 provider `register_provider`
- 事件监听器 `subscribe`
- 系统提示附加片段 `extend_system_prompt`

当前代码里，扩展命令会被收集到 `ExtensionRuntime.commands`，但 interactive 控制器暂时没有消费这些命令，这意味着“扩展命令注册能力已预留，但产品接入尚未闭环”。

## 系统提示拼装

系统提示由两层完成：

- `system_prompt.py`：构造基础提示
- `system_prompt_builder.py`：把扩展附加片段拼接进去

`build_system_prompt(context)` 的拼装顺序是固定的：

1. 默认 `SYSTEM` 提示，或 `settings.system_prompt_override`
2. 可用工具说明
3. 内建行为准则 `BEHAVIOR_RULES`
4. `AGENTS.md` 项目上下文
5. 技能摘要
6. 环境信息
   - 当前日期时间
   - 当前目录
   - 工作区根目录
   - 模型 ID
   - 思考等级
   - 平台信息

然后 `SystemPromptBuilder.build()` 再把扩展追加的 prompt fragment 拼接到末尾。

因此，这个模块的系统提示不是静态常量，而是“基础提示 + 当前工具 + 当前环境 + 项目上下文 + 扩展补丁”的动态结果。

## 会话核心：`AgentSession`

`core/agent_session.py` 是整个 `coding_agent` 的中心对象。它对外表现为“产品层会话”，对内则负责：

- 组装 runtime
- 创建底层 `agent_core.Agent`
- 管理当前会话 ID、分支 ID、当前 turn ID、最近节点 ID
- 在持久化层和底层 Agent 之间做消息同步
- 控制压缩、恢复、工具前后 steering、tool result 跟进总结
- 把 `agent_core.AgentEvent` 转换成 UI 可消费的 `SessionEvent`

### 1. 初始化阶段

构造 `AgentSession` 时会：

1. 调用 `_assemble_runtime()` 完成资源、工具、provider、压缩器装配。
2. 调用 `_build_agent()` 构造 `agent_core.Agent`。
3. 如果未传入 `session_id`，用 `SessionManager.create_session()` 新建会话。

### 2. 底层 Agent 的组装

`_build_agent()` 会创建 `AgentLoopConfig`，把以下能力挂给底层循环：

- `model`
- `thinking`
- `systemPrompt`
- `tools`
- `stream`
- `convert_to_llm`
- `steer`
- `followUp`
- `beforeToolCall`
- `afterToolCall`
- `registry`

这说明 `coding_agent` 并不是简单调用一个“问答接口”，而是深度参与了底层 Agent Loop 的多个钩子。

### 3. 用户消息与会话恢复

- `send_user_message(content)`
  - 创建新 turn ID
  - 将 `UserMessage` 持久化到会话文件
  - 更新 `last_node_id`
  - 把消息入队到底层 Agent
- `resume_session()`
  - 从 `SessionManager.build_context_messages()` 恢复上下文
  - 同步到底层 Agent state
  - 恢复 `last_node_id`

### 4. 模型与思考等级热切换

- `switch_model(model)`：
  - 重新装配 runtime
  - 更新底层 Agent 的模型、工具、系统提示
  - 回写 session meta 中的 `model_id`
- `set_thinking(thinking)`：
  - 更新当前 thinking
  - 刷新底层 Agent 的 thinking 与 system prompt

这两个操作都不需要重建完整应用，只刷新会话内部状态。

## 事件流与产品层映射

`AgentSession.run_turn()` 的职责不是简单 await 一次结果，而是输出 `AsyncIterator[SessionEvent]`。这是交互层能实时渲染的基础。

### 1. 执行过程

`run_turn()` 支持上下文溢出恢复：

- 先调用 `_run_turn_once()`
- 如果发现 overflow 错误
  - 先触发 `compaction.recover_from_overflow()`
  - 再 `resume_session()`
  - 然后重试一次

`_run_turn_once()` 则负责：

- 若当前无 pending message 且已有 history，则继续对话
- 否则刷新会话、尝试自动压缩、再运行 Agent
- 消费底层事件流
- 映射为产品层事件
- 在消息结束时把 assistant/control/tool result 持久化

### 2. 事件映射规则

`_map_event()` 把 `agent_core` 事件映射成 `SessionEvent`，主要包括：

- `message_start` -> assistant 输出块开始
- `message_update` -> `message_delta`
- `message_end(assistant)` -> assistant 最终消息
- `message_end(control)` -> status 事件
- `tool_execution_start` -> `tool_start`
- `tool_execution_update` -> `tool_update`
- `tool_execution_end` -> `tool_end`
- `agent_end(error)` -> `error`

其中有两个很关键的产品化设计：

1. `ControlMessage` 不直接渲染成普通对话消息，而是被转换成状态类事件。
2. 用户消息不会被错误映射成空 assistant 卡片，测试中对这一点专门做了保护。

## Steering 与 Follow-up 机制

这部分是 `coding_agent` 相比纯聊天壳子更有产品意识的地方。

### 1. 工具前后 steering

在 `_before_tool_call()` 中，系统会排入一条 `ControlMessage(kind="steering")`，内容说明“即将执行哪个工具及参数”。

在 `_after_tool_call()` 中，会再排入一条 steering，提示“工具已执行完成，请结合结果继续工作”。

这些控制消息不会作为普通聊天历史直接展示给用户，但会：

- 在 LLM 上下文里被转成带 `metadata` 的 `UserMessage`
- 在 UI 层以 `status` 形式显示
- 在会话存储中以 `control` 事件单独落盘

### 2. Follow-up

`_follow_up()` 的逻辑是：

- 如果已有待发的 follow-up 控制消息，直接发送。
- 如果本轮发生过工具调用、且还没发过 follow-up、且最后一条 assistant 消息不再要求继续调用工具，就自动注入一条 follow-up：

`请基于当前工具执行结果给出最终答复；如果任务尚未完成，请继续推进并明确下一步。`

这会强制模型从“工具执行完成”推进到“形成最终答复”，减少只调用工具不收尾的情况。

## 会话持久化：`SessionManager`

`core/session_manager.py` 采用追加式 JSONL + 内存索引树的方式管理会话。

### 1. 目录结构

每个 session 对应一个 `.jsonl` 文件，首行是 session header，后续每行都是树状 entry。

### 2. 事件类型

当前持久化支持：

- `message`
- `thinking_level_change`
- `model_change`
- `compaction`
- `branch_summary`
- `custom`
- `custom_message`
- `label`
- `session_info`

### 3. 存储与恢复逻辑

- `create_session()`：创建新会话文件并切到该会话。
- `append_message()`：写入普通消息节点。
- `set_session_file()`：加载整个 session file，重建 `by_id`、`labels_by_id`、`leaf_id` 等内存索引。
- `build_session_context()`：从当前 `leaf_id` 沿 `parent_id` 回溯 active path，再解释成真正发给模型的上下文消息。
- `build_context_messages()`：
  - 默认读取当前 leaf 对应分支
  - 若存在 `compaction`，先插入一条标记了 `summary=True` 的 `UserMessage`
  - 跳过已经被摘要覆盖的旧节点
  - 继续解释 `branch_summary` / `custom_message`

这套设计的重点是：session file 是持久化来源，`SessionManager` 是当前会话状态；恢复时不是“回放成快照”，而是“加载 entries + 建索引 + 按 leaf 还原上下文”。

## 工具体系

### 1. 工具注册表

`core/tools/registry.py` 的 `ToolRegistry` 负责：

- 注册工具定义 `register_definition()`
- 直接注册工具实例 `register_tool()`
- 激活全部定义 `activate_all()`
- 输出工具 Markdown 描述 `render_markdown()`

`activate_all()` 在真正生成工具实例时，会通过 `_wrap_tool()` 包一层统一执行包装，用于：

- 解析 arguments
- 组装 `ToolExecutionContext`
- 调用安全策略
- 透传或后处理输出
- 注入 group / source / mode 等元数据

### 2. 内置工具集合

`core/tools/__init__.py` 当前内置了 7 个工具：

- `read`
- `write`
- `edit`
- `bash`
- `grep`
- `find`
- `ls`

工具分组主要为：

- `filesystem`
- `shell`
- `search`

### 3. 各工具职责

- `read.py`
  - 读取 UTF-8 文本文件
  - 按 `max_read_chars` 截断
- `write.py`
  - 在工作区内创建或覆盖写入文件
- `edit.py`
  - 支持“精确替换一次”或“按行范围替换”
- `bash.py`
  - 在工作区内执行非交互 shell 命令
  - 返回命令、退出码、stdout、stderr
- `grep.py`
  - 调用 `rg -n --no-heading`
- `find.py`
  - 用 `rglob("*")` 按名称片段匹配路径
- `ls.py`
  - 列出目录直接子项

### 4. 安全边界

`common.ensure_within_workspace()` 会确保路径不能逃出工作区。

`security.build_default_tool_security_policy()` 会额外约束：

- 若 `allow_bash=False`，禁止 `bash`
- 若工具模式是 `read-only`，则阻止写工具和 `bash`

所以安全不是只靠提示词，而是落实到了工具执行前的代码校验。

## 交互模式：`modes/interactive`

交互层由 `InteractiveApp`、`InteractiveController`、`InteractiveState`、`InteractiveRenderer` 四部分组成。

### 1. `InteractiveApp`

`app.py` 是交互入口：

- 优先使用 `prompt_toolkit`
- 若依赖不存在，则退回 `_fallback_loop()` 标准输入模式

这说明该模块既支持完整 TUI，也支持简化降级运行。

### 2. `InteractiveController`

控制器负责把用户输入转为会话动作。

普通文本路径：

1. `handle_text()` 调用 `session.send_user_message()`
2. 初始化 turn
3. 遍历 `session.run_turn()` 的事件流
4. 将事件交给 `InteractiveState.apply_event()`
5. 触发 renderer 刷新与滚动跟随

斜杠命令路径则由 `handle_command()` 统一处理，当前支持：

- `/new`
- `/resume [id]`
- `/model <id>`
- `/thinking <level>`
- `/compact`
- `/theme <name>`
- `/pwd`
- `/sessions`
- `/help`
- `/clear`
- `/retry`
- `/exit`
- `/bottom`
- `/top`

还提供：

- `CommandCompleter` 自动补全
- `cancel_current()` 取消当前运行
- `toggle_focus()` 切换主输出区与输入区焦点

### 3. `InteractiveState`

`state.py` 不是简单缓存文本，而是维护一棵“按 turn 聚合”的可视状态树。

关键数据结构包括：

- `TranscriptBlock`
  - 单个可渲染块，可能是 user / assistant / thinking / tool / status / error
- `TranscriptTurn`
  - 一轮用户问题及其完整处理链
- `InteractiveState`
  - 保存 transcript、状态栏、最近会话、滚动模式、未读输出等全部 UI 状态

`apply_event()` 会按事件类型更新：

- assistant 文本
- thinking 内容
- tool 卡片
- status 状态
- error 输出

然后调用 `rebuild_transcript()` 把 turn 结构重新铺平成稳定文本，这也是 renderer 能高效工作的基础。

### 4. `InteractiveRenderer`

`renderer.py` 用 `prompt_toolkit` 构建 TUI，主要包含：

- 主输出区
- 侧边栏
- 输入区
- 状态栏

它还实现了比较完整的滚动与跟随语义：

- 自动跟随最新输出
- 历史浏览模式
- 跳到底部 / 跳到顶部
- 鼠标滚轮滚动
- 窗口尺寸变化后的视口修正
- 未读输出计数

快捷键支持也比较完整，例如：

- `Enter` 发送
- `Esc+Enter` 插入换行
- `Ctrl-C` 取消
- `Ctrl-L` 清屏
- `Tab` 补全
- `PgUp/PgDn` 翻页
- `Home/End` 跳转最早/最新
- `F6` 切换焦点

## 上下文压缩体系

### 1. 压缩触发

`compaction/coordinator.py` 中的 `CompactionCoordinator` 统一管理三类行为：

- `compact_manual()`：手动压缩
- `maybe_compact_for_threshold()`：发送前按阈值自动压缩
- `recover_from_overflow()`：上下文溢出后恢复

它会调用 `ai.utils.context_window.detect_context_overflow()` 估算上下文使用量，然后由 `triggers.should_compact()` 根据：

- `compact_threshold`
- 当前估算 token 占比
- 模型上下文上限

来决定是否压缩。

### 2. 实际压缩策略

`compaction/compactor.py` 中的 `SessionCompactor` 采用的是“保留最近 N 轮，其余消息转摘要”的策略：

- `keep_turns * 2` 作为保留节点数
- 更早的节点进入摘要
- 摘要写入 `summary` 事件
- 被摘要覆盖的旧节点在 `build_context_messages()` 中会被跳过

摘要内容改为通过 `ai.completeSimple()` 调用大模型生成，而不是本地字符串拼接。实现上会：

- 先把较早历史整理成稳定输入
- 使用专用 system prompt 约束摘要结构
- 只输出固定 5 段：
  - `任务目标`
  - `关键上下文`
  - `已完成事项`
  - `未完成事项`
  - `风险与注意点`

摘要模型优先读取 `compact_model`，未配置时回退当前会话模型。这样压缩后的 summary 更适合后续继续任务，而不只是机械压缩聊天记录。

## 共享类型设计

`core/types.py` 定义了这个模块的公共契约，重点类型包括：

- `CodingAgentSettings`
- `ToolPolicy`
- `ResourceBundle`
- `ExtensionRuntime`
- `ToolDefinition`
- `SessionContext`
- `SessionEvent`
- `CompactResult`
- `ContextStats`

这些类型把“配置、资源、工具、事件、持久化、压缩”几个子系统串联起来，是本模块稳定接口的核心。

## 与其他模块的关系

`coding_agent` 与其他模块的关系可以概括为：

- 依赖 `ai`
  - 使用 `Model`、`Context`、消息类型、provider registry、上下文溢出检测等能力。
- 依赖 `agent_core`
  - 使用 `Agent`、`AgentLoopConfig`、`AgentTool`、tool hook、event 流。
- 面向产品层输出
  - 通过 `SessionEvent` 和 interactive TUI 提供终端使用体验。

也就是说：

- `ai` 负责模型抽象与 provider 能力。
- `agent_core` 负责对话循环与工具调用框架。
- `coding_agent` 负责把它们装成一个真正能用的终端编码助手。

## 关键运行链路总结

一次典型交互的真实执行链路如下：

1. 用户启动 `coding-agent`。
2. `main.py` 加载路径、配置、模型、资源和会话管理器。
3. 构造 `AgentSession`，并通过 `runtime_assembly.py` 装配资源、工具、provider、压缩器。
4. `InteractiveApp` 启动 TUI。
5. 用户输入文本或命令。
6. `InteractiveController` 调用 `AgentSession.send_user_message()`。
7. `AgentSession.run_turn()` 启动底层 `agent_core.Agent`。
8. 底层事件经过 `_map_event()` 转为 `SessionEvent`。
9. `InteractiveState.apply_event()` 更新 transcript 与状态树。
10. `InteractiveRenderer` 将状态渲染到终端，并处理滚动、焦点和状态栏。
11. assistant/tool/control 结果同时通过 `SessionManager` 落盘，支持恢复、重试与压缩。

## 测试反映出的模块行为重点

`tests/test_coding_agent.py` 覆盖了本模块最关键的行为，能帮助理解源码设计重点：

- 设置支持全局与项目级合并。
- 资源加载支持技能、提示、项目上下文，并会检测命名冲突。
- 会话支持创建、持久化、列出最近会话和压缩。
- 内置工具的读写改查与 bash 执行可直接工作。
- `AgentSession` 能把流式输出恢复回持久化历史。
- 工具执行前后的 steering 与最终 follow-up 会被正确注入并落盘。
- interactive controller 支持模型切换、thinking 切换、主题切换、重试、列会话和清屏。
- renderer 支持完整 transcript 历史、滚动 API、自动跟随与历史浏览状态切换。
- 用户消息不会错误地产生空 assistant 卡片。

这些测试说明 `coding_agent` 的核心价值不只是“能跑”，而是已经具备了比较明确的产品行为约束。

## 当前实现特点与边界

基于当前代码，可以总结出这个模块的几个特点：

- 它已经具备完整的终端产品骨架：启动、会话、持久化、交互、工具、主题、扩展、压缩都已成形。
- 它的很多设计是“可扩展优先”的，例如资源包、扩展运行时、provider 注入、prompt fragment 注入。
- 它把控制消息、普通消息、工具结果、摘要事件做了清晰分层，没有把所有东西混成单一 transcript。
- 它当前仍有一些预留接口尚未完全接入，例如扩展命令已经能注册，但 interactive 命令系统还没有消费这部分能力。

总体来说，`coding_agent` 不是单一的 UI 层，也不是单纯的 Agent 包装器，而是 LiuClaw 项目里把底层能力产品化、终端化、会话化的核心模块。
