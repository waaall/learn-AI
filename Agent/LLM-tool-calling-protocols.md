# LLM 工具调用协议：技术深度剖析

**工具调用（Tool Calling）是把 LLM 从文本生成器转变为能够与真实世界交互的自治代理的关键机制。** 几乎所有主流 AI 编码助手、企业聊天机器人和 Agent 工作流，都依赖一个看似简单的协议：模型输出描述函数调用的结构化 JSON，应用执行该调用，然后把结果回传到对话上下文。本文会逐层拆解该协议的工作方式，从 API 线格式和模型训练，到 Vercel AI SDK 这类 SDK 抽象，再到 OpenCode、Claude Code、Aider 等开源编码工具中的生产实践。对于在 2025 年及以后构建 AI 应用的开发者，理解这些层次至关重要。

-----

## API 层面的工具调用如何工作

从本质上看，工具调用是一个五步循环，各家提供商仅在细节上略有差异。开发者先定义工具（名称、描述、JSON Schema 参数）并随用户提示一起发送。模型决定是否调用工具，并返回包含函数名和 JSON 参数的结构化对象。应用执行函数，把结果追加到对话历史，再把完整上下文发回模型。模型随后生成最终回答，或者继续发起下一次工具调用，循环往复。

**模型本身从不执行任何操作。** 它只是在输出 token，这些 token 刚好组成与函数签名匹配的有效 JSON。API 层会解析该输出，将其封装为结构化响应，并把执行责任交给应用程序。

### OpenAI 的格式

OpenAI 把每个工具放在 `tools[]` 数组里，使用 `type: "function"`，并在嵌套 `function` 对象中包含 `name`、`description` 和 `parameters`（标准 JSON Schema）。当模型决定调用工具时，响应消息会包含 `tool_calls` 数组；每次调用都有唯一 `id`、函数 `name`，以及字符串化的 `arguments`：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\":\"Boston\",\"unit\":\"celsius\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

工具结果通过专用 `role: "tool"` 消息返回，并引用 `tool_call_id`。OpenAI 默认支持**并行工具调用**，即模型可同时在 `tool_calls` 数组中返回多个调用。设置 `strict: true` 会启用受约束解码，在推理时用 token 级语法约束保证**100% 符合 JSON Schema**。

### Anthropic 的内容块架构

Anthropic 采用结构上不同的方式。工具定义使用 `input_schema`（不是 `parameters`），响应采用**内容块数组**，`text` 和 `tool_use` 可以自然地出现在同一条 assistant 消息中：

```json
{
  "role": "assistant",
  "stop_reason": "tool_use",
  "content": [
    {"type": "text", "text": "I'll check the weather for you."},
    {"type": "tool_use", "id": "toolu_01A09q90qw90lq917835lq9",
     "name": "get_weather", "input": {"location": "San Francisco, CA"}}
  ]
}
```

关键差异在于：工具结果是放在 **`user` 消息**中的 `tool_result` 内容块，而不是单独角色。这意味着 `tool_result` 必须紧跟包含对应 `tool_use` 的 assistant 消息。Anthropic 还提供独特能力：**编程式工具调用**（Claude 编写 Python 编排代码，而不是逐轮串行往返，可将 token 降低 **37%**）、用于从大规模工具库动态发现工具的**tool search tool**（上下文从约 55K 降至约 500 token），以及 `web_search`、`text_editor`、`bash` 等服务端工具。

### Google Gemini 的函数声明

Gemini 在工具中使用 `functionDeclarations`，schema 与 OpenAPI 3.0 兼容。工具调用以 `functionCall` 对象出现在 `content.parts` 中，结果以 `functionResponse` part 返回。Gemini 每次请求最多支持 **128 个函数声明**，支持组合/顺序调用（在一次推理中链式执行 `get_location()` -> `get_weather(location)`），并支持通过带 JSONPath 引用的 `partialArgs` 流式传输函数参数。

### 关键差异

|方面|OpenAI|Anthropic|Google Gemini|
|---|---|---|---|
|工具结果机制|`role: "tool"` message|`tool_result` in user message|`functionResponse` in user parts|
|调用 ID 匹配|`tool_call_id`|`tool_use_id`|通过函数 `name` 匹配|
|强制调用任意工具|`tool_choice: "required"`|`tool_choice: {"type": "any"}`|`functionCallingConfig: {mode: "ANY"}`|
|Schema 格式|JSON Schema|JSON Schema|OpenAPI 3.0 子集|
|并行调用|默认开启|默认开启（可关闭）|支持|

尽管表层格式不同，**在模型层面所有提供商都会把工具定义“展开”进 token 流。** 结构化 API 只是开发者抽象；在底层，工具 schema 会被注入系统提示（通常是带类 XML 标签的 JSON），模型生成文本 token，再由推理系统解析成结构化响应。

-----

## 模型如何学会调用工具

工具调用能力建立在三项已有能力之上：来自海量代码语料预训练的**代码能力**（理解函数签名、JSON 结构、API 模式）、来自 RLHF 后训练的**指令跟随能力**、以及因训练数据中 JSON/代码占比较高形成的**结构化输出能力**。

训练路径经历了几个阶段。早期方法（2019-2022），如 WebGPT，需要人工标注语料中的搜索调用，再用这些标注做微调，成本高且仅适用于固定工具集。**Toolformer**（Meta，2023）提出了自监督突破：先用 LLM 在文本中建议工具调用插入点，执行调用后用交叉熵过滤衡量是否改进下一 token 预测，再只用“有帮助”的插入样本微调。模型由此学会输出类似 `[Calculator(400/1400)] -> 0.29]` 的特殊 token。

当前 SOTA 方案（xLAM 2、NexusRaven 等）采用合成数据流水线：从代码语料提取函数定义；让 LLM 生成会触发这些函数的自然语言查询；加入无关“干扰函数”做判别训练；通过多阶段 LLM 校验和有用性指标筛选；加入推理步骤（CoT、ReAct）；再在增强语料上微调。开源模型通常使用**特殊 token**标记工具调用片段：Mistral 用 `[TOOL_CALLS]` 和 `[AVAILABLE_TOOLS]`，Hermes 用 `<tool_call>` 和 `<tool_response>`。

在推理阶段保证 schema 一致性时，**受约束解码**（如 OpenAI Structured Outputs）会在每一步生成时跟踪 JSON Schema 允许的 token，并在采样前屏蔽无效 token。相比无约束生成约 86% 的匹配率，该方法可实现完美 schema 匹配。

-----

## Vercel AI SDK：抽象层细节

Vercel AI SDK 提供了目前最易用的 TypeScript 工具调用抽象，通过统一 API 封装各提供商差异。理解其设计，有助于理解现代框架如何处理多步 Agent 工作流的复杂性。

### 使用 Zod schema 定义工具

工具通过 `tool()` 辅助函数创建，使用 **Zod schema** 做类型安全参数定义，并自动转换为供 LLM 使用的 JSON Schema：

```typescript
import { z } from 'zod';
import { tool, generateText, stepCountIs } from 'ai';

const weatherTool = tool({
  description: 'Get the weather in a location',
  inputSchema: z.object({
    location: z.string().describe('The location to get the weather for'),
    unit: z.enum(['celsius', 'fahrenheit']).optional(),
  }),
  execute: async ({ location, unit = 'celsius' }) => ({
    location,
    temperature: 72 + Math.floor(Math.random() * 21) - 10,
  }),
});
```

`execute` 函数会接收第二个上下文参数，包括 `toolCallId`、`messages`（到该步为止的对话历史）、用于取消的 `abortSignal`，以及可携带任意数据的 `experimental_context`。`execute` 是**可选的**：省略它表示该工具应在客户端执行或由外部队列处理。AI SDK 5 还把 `parameters` 更名为 `inputSchema`，把 `result` 更名为 `output`，以对齐 MCP 规范。

### 多步 Agent 循环

SDK 用可组合的 `stopWhen` + `stepCountIs()` 取代了简单的 `maxSteps` 整数，支持更丰富的终止逻辑：

```typescript
const { text, steps } = await generateText({
  model: 'anthropic/claude-sonnet-4.5',
  tools: { weather: weatherTool },
  stopWhen: stepCountIs(5),
  prompt: 'What is the weather in San Francisco?',
});
```

每一步遵循相同循环：提示 -> 模型生成工具调用 -> 工具执行 -> 结果回传 -> 模型生成文本或继续调用工具。达到步数上限，或模型返回文本且不再调用工具时，循环结束。**`stopWhen` 条件只会在最后一步包含工具结果时才评估**，从而避免过早终止。

**`prepareStep` 回调**支持逐步动态配置，例如切换模型、调整 `toolChoice`、限制 `activeTools`，或在循环中压缩消息：

```typescript
prepareStep: async ({ stepNumber, messages }) => {
  if (stepNumber === 0) return { toolChoice: { type: 'tool', toolName: 'search' } };
  if (messages.length > 20) return { messages: messages.slice(-10) };
  return {};
},
```

### 四种 toolChoice 策略

`toolChoice` 控制模型行为：`'auto'`（默认）让模型自行决定；`'required'` 强制调用工具但不指定具体哪个；`'none'` 禁止所有工具调用；`{ type: 'tool', toolName: 'weather' }` 强制调用特定工具。SDK 会在底层把它映射到各提供商原生格式。

### 服务端执行与客户端执行

SDK 支持三种执行模式。**服务端自动执行工具**：有 `execute` 函数，在 `generateText`/`streamText` 中自动运行。**客户端自动执行工具**：省略 `execute`，由 `useChat` 的 `onToolCall` 回调处理。**用户交互型工具**：同样省略 `execute`，但会渲染 UI（如确认弹窗），用户交互后通过 `addToolOutput` 提供结果。工具部件会经历状态流转：`input-streaming` -> `input-available` -> `output-available`（也可能是 `output-error` 或 `approval-requested`）。

### AI SDK 6：Agent、审批与编程式调用

AI SDK 6 引入了多项关键特性。`needsApproval` 属性（静态布尔值或异步函数）可在需要人工确认的工具调用处暂停执行。触发后，`generateText` 返回 `tool-approval-request` 部件；应用收集用户决策，再发送 `tool-approval-response` 消息恢复执行：

```typescript
const dangerousTool = tool({
  description: 'Delete a file',
  inputSchema: z.object({ path: z.string() }),
  needsApproval: async ({ path }) => path.includes('/production/'),
  execute: async ({ path }) => { /* delete logic */ },
});
```

`ToolLoopAgent` 类把完整 Agent 模式（模型、指令、工具、循环配置）封装为可复用、类型安全对象，提供 `generate()` 和 `stream()` 方法。默认 `stopWhen: stepCountIs(20)`，并支持 `callOptionsSchema` 做运行时配置。

**编程式工具调用**（Anthropic）允许 Claude 在代码执行环境中调用工具，把中间结果移出上下文。`@ai-sdk/mcp` 中的 MCP 支持已稳定，涵盖 OAuth 认证、resources、prompts 与 elicitation。

### 流式传输线协议

工具调用通过 **Server-Sent Events** 以类型化数据片段流式传输。一次工具调用会产生四个顺序事件：`tool-input-start`（含 `toolCallId` 与 `toolName`）、`tool-input-delta`（增量参数 token）、`tool-input-available`（完整且已校验输入）、`tool-output-available`（执行结果）。步骤边界由 `start-step` 与 `finish-step` 标记。流以 `finish` 和 `[DONE]` 结束。这让 UI 可以实时更新，例如边生成边展示工具参数、完成后立即展示结果。

-----

## 开源实现揭示了分化策略

观察生产级编码工具中的工具调用实现会发现，没有单一“正确”方案，最佳策略高度依赖模型训练目标。

### OpenCode 采用插件驱动的工具架构

OpenCode（anomalyco/opencode）是基于 Go 的 Agent 编码工具，内置 **15+ 工具**：`bash`、`edit`（搜索/替换）、`write`、`read`、`grep`（基于 ripgrep）、`glob`、`list`、`patch`、LSP 操作、用于任务跟踪的 `todowrite`/`todoread`、`webfetch`/`websearch`，以及用于澄清需求的 `question`。工具遵循**权限模型**：每个工具可设为 `allow`、`deny` 或 `ask`（需审批），并支持 MCP 工具通配符。

自定义工具通过 TypeScript 文件定义，借助 `@opencode-ai/plugin` 的 `tool()` 和 Zod schema。文件名即工具名，`execute` 可获得丰富上下文，如 `agent`、`sessionID`、`directory`、`worktree`。OpenCode 把 MCP server 作为一等扩展机制，也支持 ACP（Agent Communication Protocol）。其**技能系统**（可加载的 `SKILL.md`）用于注入领域上下文，本质上更像基于提示词的元工具，而非可执行函数。

### Claude Code 借助 Anthropic 原生能力

Claude Code 的架构与 Anthropic API 的内容块模型高度一致。Claude Agent SDK 通过 MCP server 暴露工具，定义使用 Zod schema，执行返回 `{content: [{type: "text", text: "..."}]}`。内置工具包括 `Read`、`Write`、`Edit`、`Bash`、`Glob`、`Grep`、`WebFetch`、`WebSearch` 和 `Task`（用于派生子代理）。

SDK 管理完整循环：LLM 调用 -> `tool_use` stop reason -> 执行 -> 回传结果 -> 重复，直到 `end_turn`。五类 Agent 分工明确：通用（全权限）、Explore（只读检索代码库）、Plan（架构规划）、claude-code-guide（文档）、statusline-setup（专项）。**Hooks**（`PreToolUse`/`PostToolUse`）支持确定性处理，例如拦截危险 bash 命令、改写输出或记录调用日志用于审计。

### Aider 有意拒绝结构化工具调用

Aider 的做法最反直觉：它发现对于代码编辑，**纯文本编辑格式持续优于结构化函数调用**。其基准结论是：即使 OpenAI 对 JSON 和函数调用提供了大量支持，GPT 在这类格式下编辑代码反而更差。把源码塞进 JSON 会引入转义复杂度和易错格式，降低模型表现。

因此 Aider 使用可插拔 `Coder` 类体系和文本编辑格式，如 Search/Replace 块、unified diff、整文件替换，以及 OpenAI patch 格式。其 **Architect/Editor** 模式把推理和编辑拆分为两次 LLM 调用：架构模型（如 o1-preview）先给方案，编辑模型再把方案转成文件修改。在其代码编辑基准上该模式达到 **85% 准确率**。应用修改时的模糊匹配链路（精确 -> 忽略空白 -> 保持缩进 -> difflib）提高了鲁棒性。

### Cursor 使用神经网络 Apply 模型

Cursor 采用另一种两阶段方案：主 LLM 先生成变更“草图”，再由**定制训练的 Apply 模型**将草图合并进代码库。Apply 模型是专门面向代码合并的神经网络，比纯文本匹配更强，能处理上下文、结构以及输入中的不完美之处。这样主 LLM 可专注“改什么”，Apply 模型负责“怎么改”。

### LangChain 与 LlamaIndex 提供框架抽象

**LangChain** 提供三种灵活度递增的工具定义方式：`@tool` 装饰器（从函数签名与 docstring 自动生成 schema）、`StructuredTool.from_function`（中等控制粒度）和 `BaseTool` 子类（完全控制 Pydantic schema）。工具通过 `llm.bind_tools(tools)` 绑定模型，由 `AgentExecutor` 编排循环。`Runnable` 接口使工具可与 `with_retry`、`bind` 等操作符组合。

**LlamaIndex** 区分 `FunctionTool`（封装 Python 函数）、`QueryEngineTool`（把向量索引封装为工具）和 `ToolSpecs`（如 `GmailToolSpec` 这类工具集合）。其事件驱动 `AgentWorkflow` 通过 `ToolCallEvent` 与 `FunctionOutputEvent` 驱动基于 DAG 的 Agent 循环，并支持“代理作为其他代理的工具”的多代理工作流。

-----

## 生产环境工具调用的实战模式

### 经典天气 API 流程

天气示例已经成为工具调用的“Hello World”。以下是 OpenAI 格式示例：

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

messages = [{"role": "user", "content": "What's the weather in Paris?"}]
response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)

# 提取并执行
tool_call = response.choices[0].message.tool_calls[0]
result = get_current_weather(**json.loads(tool_call.function.arguments))

# 把结果回传给模型
messages.append(response.choices[0].message)
messages.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})

# 获取最终回答
final = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
```

### 通过自然语言执行数据库查询

SQL Agent 工具可把自然语言转成经过校验的查询。典型模式是定义 `ask_database` 工具，接收 `query` 字符串，并在系统提示中提供数据库 schema。模型生成 SQL，应用在数据库执行，再把结果回传用于自然语言总结。Anthropic 的 tool-use 包为这一模式提供了专用 `execute_sqlite3_query` 工具。

### 用并行工具调用提升吞吐

最有价值的生产模式之一是并行执行相互独立的工具调用。只读工具（数据库查询、API 拉取）可安全并行；会改写状态的工具应串行执行以避免竞态。结果在回传模型前按原请求顺序重排。对 n 个独立操作，通常可获得约 **n 倍加速**。

### 对敏感操作做人在环审批

处理破坏性操作、金融交易或配置变更的生产系统需要审批闸门。常见模式是识别关键动作、通过中断机制暂停执行、向审核者展示摘要上下文（而非原始 JSON），并记录每次决策用于审计。LangGraph 用 `interrupt()` 实现，Vercel AI SDK 用 `needsApproval`，OpenCode 用权限策略，三者都收敛到同一架构原则。

-----

## 工业控制系统是前沿应用场景

制造和流程工业正成为工具调用的新前沿。LLM 位于 PLC、SCADA、DCS、ERP、MES 系统**上层**，负责编排、规划、解释，而安全关键的确定性控制环仍由 PLC 负责。LLM 不应直接控制执行器。

当前工业场景大致沿成熟度演进：**阶段 1**（只读）由 LLM 回答历史传感器数据问题；**阶段 2**（分析）加入 CMMS 与维护系统的自然语言接口；**阶段 3**（草拟动作）由 LLM 生成需人工审批的工单；**阶段 4**（主动）在校验层保护下允许自治动作。研究显示，在 CNC 机床编程场景，GPT-4 在 2 步流程上达到 **100% 成功率**，在 4 步流程上达到 **86%**。

工业自动化中的工具调用实现可定义如 `read_sensor_data(sensor_id, time_range)`、`query_alarm_history(equipment_id)`、`generate_work_order(description, priority, equipment)`、`lookup_maintenance_manual(equipment_model, section)` 等工具。关键安全约束是：**所有写操作都必须人工审批**，且生成的控制代码执行前必须经过确定性验证。在任务关键环境中，幻觉风险决定了强健校验层不可妥协。

-----

## MCP 正在成为通用连接层

**Model Context Protocol（MCP）** 由 Anthropic 于 2024 年 11 月发布，已迅速成为工具集成的事实标准，常被称为“AI 领域的 USB-C”。它通过 JSON-RPC 2.0（支持 stdio、Server-Sent Events、Streamable HTTP 等传输）标准化了 LLM 应用发现和调用外部工具的方式。

MCP 工具由 `name`、`description`、`inputSchema`（JSON Schema）定义。客户端通过 `tools/list` 发现工具，通过 `tools/call` 调用工具。协议定义了三类原语：**tools**（模型控制）、**resources**（应用控制）、**prompts**（用户控制）。目前生态已覆盖 Claude Code、OpenCode、Continue.dev、Cursor、Windsurf、OpenAI Agents SDK、Vercel AI SDK、LangChain，以及数千个社区 MCP server。

其意义在于架构解耦：MCP 让 Agent 不再绑定固定工具集。Agent 可在运行时查询 MCP server 动态发现可用工具，无需改代码即可实现即插即用扩展。再结合编程式工具调用（模型通过写代码编排 MCP 工具），可形成可组合、可扩展到数百集成的工具生态。

-----

## 结论：工具调用的收敛与分化

工具调用生态已经在两点上达成收敛：JSON Schema 成为参数定义的通用语言，请求-执行-回传循环成为通用架构。但在执行策略上，实践仍显著分化。Aider 关于“代码编辑中纯文本优于结构化 JSON”的结论、Cursor 的神经 Apply 模型、Anthropic 的编程式工具调用，都在挑战“标准工具调用 API 始终最优”的假设。

有三个关键洞见。第一，**最佳工具格式取决于模型训练目标**，而不取决于对开发者来说看起来是否“更结构化”。第二，**审批模式已经成为架构组成部分**，几乎所有严肃框架都提供人在环闸门，因为自治执行必须有信任边界。第三，**MCP 正在赢得集成层**，它在主要工具中的普及说明标准化的工具发现与调用会像 HTTP 之于 Web 服务一样基础。对实践者而言，建议很明确：理解协议层而不只停留在 SDK 抽象；从第一天就实现审批闸门；将工具构建为 MCP server 以获得最大复用；在最终确定架构前，针对你的具体场景同时基准测试结构化与非结构化方案。
