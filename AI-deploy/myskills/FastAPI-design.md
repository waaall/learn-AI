# FastAPI 设计原则



[fastapi-best-practices](https://github.com/zhanymkanov/fastapi-best-practices.git)

## 分层设计

- router（API 层）：只做参数校验、鉴权依赖、返回模型、调用 service；不写业务规则。
- service（应用层）：业务规则、编排事务、调用 repo/外部客户端；可单测。
- repo（数据访问层）：只做持久化与查询，不掺业务判断。
- schema（DTO / contract）：Pydantic 模型作为边界契约，避免把 ORM 对象直接暴露到 API。

## API 设计与版本治理

### 路由约定

统一前缀：/api/v1/...（或 header 版本，但要统一策略）
统一响应模型（Response Model）与错误模型（Error Envelope）
router.py 聚合各子域路由，符合 FastAPI “bigger applications”组织方式。

### 版本与弃用（versioning & deprecation）

定义弃用策略：公告期、迁移期、移除期；并在 OpenAPI/文档中标注 deprecated。版本治理与弃用沟通的最佳实践可参考行业建议。
约定：破坏性变更必须升主版本（v1→v2），并保留旧版本一段时间。

## 统一错误处理与返回契约

### 错误返回 Envelope（建议标准化）

建议统一结构，便于前端与可观测系统解析：
```
{
  "error": {
    "code": "USER_NOT_FOUND",
    "message": "User not found",
    "request_id": "..."
  }
}
```

### 全局异常处理器（exception handlers）

统一捕获：业务异常、鉴权异常、数据异常、未知异常（500）

中间件/异常链路要放在正确层级；Starlette 文档强调通过应用包装方式保证异常处理生效。
后台任务（background tasks）里的异常不要指望返回给客户端，要自己捕获并记录/告警。

## 应用启动与资源管理

### 统一用 Lifespan 管理启动/关闭

把 DB 连接池、HTTP 客户端、模型加载等放到 lifespan（生命周期，lifespan events）里，避免散落 startup/shutdown。FastAPI/Starlette 都推荐这种方式。

### 外部依赖（DB/Redis/HTTP Client）一律依赖注入

用 Depends/Security（dependency injection）提供“当前用户、租户、权限、DB session”等。FastAPI 官方对 Depends()/Security()行为与 OpenAPI 集成有明确说明。

## 可观测性（Observability）

### 请求链路标识

X-Request-Id：网关生成优先；应用缺失则生成
日志必须结构化（JSON），至少包含：request_id、trace_id、user_id、tenant_id、latency_ms、status_code

## 性能与并发模型

明确 async 边界：I/O 用 async；CPU 密集型任务丢到 worker（Celery/RQ/自建队列）
数据库驱动与 ORM：async 方案要全链路一致（async engine / async session）
大文件上传/下载：流式（streaming），避免一次性读入内存
背景任务：对可靠性有要求就用队列系统，不要只靠进程内 background task（进程重启会丢）

## 安全基线（Security Baseline）

### 认证与授权

认证（authentication）：统一 Bearer/JWT 或 OIDC
授权（authorization）：RBAC/ABAC/Scopes（OAuth2 scopes）
对 scope 的声明用 Security(..., scopes=[...])，让 OpenAPI 自动生成安全需求与文档。

### 常见安全控制项

输入校验：严格使用 Pydantic；对 Query/Path/Body 加约束（长度、正则、枚举）
防注入：ORM/参数化查询；日志避免直接打印敏感字段
CORS、Trusted Host、HTTPS redirect（视部署而定）
速率限制（rate limiting）：建议放 API Gateway；或在应用层做兜底
多租户（multi-tenancy）：租户信息必须来自可信来源（token/网关注入），并贯穿到数据访问层过滤

