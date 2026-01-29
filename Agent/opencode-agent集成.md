

## 1) 把 OpenCode 当成独立后端服务（推荐）

- 直接启动 headless HTTP server：`opencode serve` 会跑一个不带 TUI 的服务，并暴露 OpenAPI 3.1 接口（`/doc`）供客户端调用。
    
- 可以把它当“sidecar/microservice”，产品只需要实现：
    
    - 会话（session）创建/续聊/终止
    - 消息发送（prompt/command）
    - 文件/工具调用（OpenCode 内置工具体系）
    - （可选）事件流：服务端有全局事件流（SSE）接口用于 streaming/状态更新

- 认证：可用环境变量 `OPENCODE_SERVER_PASSWORD` 走 HTTP Basic Auth（用户名默认 `opencode`）。
    

适用：要把 OpenCode 变成“后端能力”，前端可以是 Web/桌面/ agent 平台统一 UI。

## 2) 直接用官方 JS/TS SDK 以“库”的方式嵌入

OpenCode 提供 type-safe JS/TS SDK：`@opencode-ai/sdk`，可以：

- 一行创建并启动 server + client（适合本地嵌入）
- 或只创建 client 连接到已经跑起来的 server（适合微服务化）

适用：后端/桌面端是 Node/TS 生态，希望少写 HTTP client、直接用类型化 API。

## 3) 把 OpenCode 当作“子进程 agent”，走标准协议或 JSON 流

这类更像“继承 CLI”，但仍能做到结构化集成：

ACP：Agent Client Protocol（更标准、更适合集成 IDE/自己的客户端）

- OpenCode 支持 ACP（Agent Client Protocol），通过 `opencode acp` 以子进程方式运行，用 stdio JSON-RPC 跟宿主通信。
- 软件只要实现 ACP 客户端，就能把 OpenCode 当“外部 agent server”。

适用：已经有编辑器/客户端形态，想对接一个“外部 agent”，并且希望协议标准化。