
## coding agent 系统的不同之处

下文探讨 coding agent 系统 和 传统的 LangChain / AutoGen / CrewAI 等 agent 编排框架的本质不同。这也是为什么 coding agent 的应用不基于框架而是自己实现的原因。

### 工作单元不同

框架通常把一次 agent 运行看成一个“job / workflow / graph execution”。
OpenClaw、OpenCode、nanobot 这类项目把 agent 看成一个“持续会话中的操作员”。

### 状态形态不同

框架更擅长结构化状态：节点状态、checkpoint、resume、flow state。
coding/local agent 的真实状态往往是非结构化的：工作区文件、git diff、终端输出、用户偏好、bootstrap files、会话压缩结果、权限上下文。

### 难点不同

框架的难点是 orchestration：图、路由、handoff、durable execution、tracing。
这类 agent 产品的难点是 runtime control：文件系统、shell、patch、沙箱、提权、打断、恢复、上下文压缩、工具权限、流式 UX。

### 安全模型不同

框架更多处理“逻辑正确性”和“流程可控性”。
本地 agent 产品首先处理“危险工具的副作用”：能不能写文件、能不能跑命令、什么时候请求授权、如何隔离工作区。这是完全不同的工程重心。

### 产品边界不同

框架注重的是“通用性”。
OpenClaw/OpenCode 这类产品注重的是“强默认、低摩擦、像一个能直接工作的 agent OS/IDE 伙伴”。

### 抽象成本不同

如果你的核心 loop 本质上就是“读上下文 -> 调模型 -> 执行工具 -> 回灌结果”，自己写一层往往比接入通用框架更便宜、更透明。
一旦你需要复杂审批流、长任务恢复、企业观测、多服务部署，框架的价值才明显上升。

## opencode-agent集成

### 1) 把 OpenCode 当成独立后端服务（推荐）

- 直接启动 headless HTTP server：`opencode serve` 会跑一个不带 TUI 的服务，并暴露 OpenAPI 3.1 接口（`/doc`）供客户端调用。

- 可以把它当“sidecar/microservice”，产品只需要实现：

    - 会话（session）创建/续聊/终止
    - 消息发送（prompt/command）
    - 文件/工具调用（OpenCode 内置工具体系）
    - （可选）事件流：服务端有全局事件流（SSE）接口用于 streaming/状态更新

- 认证：可用环境变量 `OPENCODE_SERVER_PASSWORD` 走 HTTP Basic Auth（用户名默认 `opencode`）。


适用：要把 OpenCode 变成“后端能力”，前端可以是 Web/桌面/ agent 平台统一 UI。

### 2) 直接用官方 JS/TS SDK 以“库”的方式嵌入

OpenCode 提供 type-safe JS/TS SDK：`@opencode-ai/sdk`，可以：

- 一行创建并启动 server + client（适合本地嵌入）
- 或只创建 client 连接到已经跑起来的 server（适合微服务化）

适用：后端/桌面端是 Node/TS 生态，希望少写 HTTP client、直接用类型化 API。

### 3) 把 OpenCode 当作“子进程 agent”，走标准协议或 JSON 流

这类更像“继承 CLI”，但仍能做到结构化集成：

ACP：Agent Client Protocol（更标准、更适合集成 IDE/自己的客户端）

- OpenCode 支持 ACP（Agent Client Protocol），通过 `opencode acp` 以子进程方式运行，用 stdio JSON-RPC 跟宿主通信。
- 软件只要实现 ACP 客户端，就能把 OpenCode 当“外部 agent server”。

适用：已经有编辑器/客户端形态，想对接一个“外部 agent”，并且希望协议标准化。