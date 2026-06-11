# Slidev Markdown PPT 设计原则

本文档定义一套使用 **Slidev + Markdown** 设计 HTML PPT 的通用原则。它关注的是：如何通过结构化内容、布局模板、组件和设计系统，稳定地产出高质量演示文稿。

核心理念：

> PPT 的主要输入应该是内容，而不是手写页面 DOM；Markdown 负责表达内容结构，Slidev 负责演示运行时，Layout 与 Component 负责视觉呈现。

---

## 1. 基本理念

### 1.1 内容优先

Slidev PPT 应首先被视为一份结构化内容文档，而不是一组孤立的页面截图。

内容源应清晰表达：

- 这页讲什么主题。
- 这页属于哪种页面类型。
- 这页有哪些核心论点、列表、数据、图片或备注。
- 这页需要什么布局参数。

不应让内容维护者反复处理：

- 复杂 HTML 嵌套。
- 大量 inline style。
- 手写页码。
- 复制粘贴布局结构。
- 与具体视觉实现强绑定的 DOM。

### 1.2 Markdown 是内容协议，不是排版容器

Markdown 应主要承担内容表达，而不是承担复杂视觉排版。

推荐用 Markdown 表达：

- 标题
- 段落
- 列表
- 表格
- 图片
- 引用
- 代码块
- 讲者备注
- 页面 frontmatter

不推荐在 Markdown 中堆叠：

- 大量 `<div>`
- 大量 `style="..."`
- 重复的卡片 HTML
- 与业务个例强绑定的复杂 DOM

原则：

> 普通内容用 Markdown，复杂页面用 Layout，重复零件用 Component。

---

## 2. 内容源原则

### 2.1 使用单一主内容源

每套 PPT 应优先使用一个主 Markdown 文件，例如：

```text
slides.md
```

它负责：

- 页面顺序
- 页面标题
- 正文内容
- 页面级参数
- 讲者备注
- 资源引用

这样可以让人和 Agent 都能快速理解整套 PPT 的叙事结构。

### 2.2 每页使用 frontmatter 描述元信息

每页应通过 frontmatter 描述页面类型和必要参数。

示例：

```md
---
layout: demo
eyebrow: DEMO 01 · 实时问答
image: /assets/demo-chat-1.png
label: 主蒸汽压力趋势
---

# 7 天主蒸汽压力趋势

4 秒内识别出 3 处异常波动，并给出可追溯的数据来源。
```

frontmatter 适合放：

- `layout`
- `title`
- `label`
- `eyebrow`
- `image`
- `items`
- `metrics`
- `tags`
- `class`
- `background`

正文 Markdown 适合放实际叙事内容。

### 2.3 内容字段要结构化

当页面包含卡片、指标、时间线、流程步骤等结构时，应使用结构化字段，而不是直接写死 HTML。

推荐：

```md
---
layout: cards
items:
  - title: 产品定位
    desc: 解决什么问题，系统边界是什么
  - title: 对话能力
    desc: 多轮问答，多数据源融合，现场实时使用
  - title: 报告能力
    desc: 趋势、诊断、事故复盘、综合分析
---

# 今天聊三件事
```

不推荐：

```md
<div class="card">...</div>
<div class="card">...</div>
<div class="card">...</div>
```

结构化内容更利于：

- 自动排版
- 批量修改
- Agent 生成
- 多主题复用
- 视觉一致性

---

## 3. Layout 原则

### 3.1 Layout 是页面类型

Layout 应表达一种可复用页面类型，而不是某一页的专属实现。

常见 Layout：

| Layout | 用途 |
| --- | --- |
| `cover` | 封面 |
| `agenda` | 议程 |
| `section` | 章节分隔 |
| `quote` | 大观点 / 核心判断 |
| `cards` | 多卡片信息组织 |
| `demo` | 截图或界面演示 |
| `compare` | 对比说明 |
| `flow` | 流程 / 链路 |
| `architecture` | 分层架构 |
| `metrics` | 数据指标 |
| `timeline` | 时间线 |
| `thanks` | 结束页 |

### 3.2 Layout 不硬编码业务内容

Layout 应接收内容参数，而不是写死某个具体项目的数据。

不推荐：

```vue
<h3>QA Agent · 对话</h3>
<p>多轮自然语言问答</p>
```

推荐：

```vue
<template>
  <section class="cards-layout">
    <h1>{{ $slidev.nav.currentPage?.title }}</h1>
    <GlassCard v-for="item in items" :key="item.title">
      <h3>{{ item.title }}</h3>
      <p>{{ item.desc }}</p>
    </GlassCard>
  </section>
</template>
```

业务内容应来自 `slides.md`。

### 3.3 Layout 管结构，Component 管细节

不要让一个 layout 变成巨型组件。

推荐分层：

```text
Layout       决定页面整体结构
Component    决定可复用视觉零件
Token/CSS     决定视觉变量
Markdown      决定页面内容
```

例如：

- `demo` layout 管左右两栏。
- `GlassCard` component 管玻璃卡片。
- `PageChrome` component 管页脚和页码。
- `SourceTag` component 管来源标签。

---

## 4. Component 原则

### 4.1 组件应语义化

组件名称应表达业务或展示语义，而不是仅表达视觉形状。

推荐：

- `PageChrome`
- `MetricCard`
- `SourceTag`
- `ArchitectureLayer`
- `FlowStep`
- `Callout`

谨慎使用：

- `BlueBox`
- `BigText1`
- `CardStyleA`

### 4.2 组件应可组合

组件应该小而稳定，可以在不同 layout 中组合使用。

例如：

```text
MetricCard 可用于 metrics、summary、outcomes 页面
GlassCard 可用于 agenda、cards、demo、architecture 页面
Callout 可用于 demo、quote、analysis 页面
```

### 4.3 组件不应依赖页面顺序

组件不应假设自己一定出现在第几页，也不应依赖全局硬编码页码或固定内容。

页面顺序属于 `slides.md`，组件只根据 props 渲染。

---

## 5. 视觉系统原则

### 5.1 Token 优先

颜色、字体、圆角、阴影、间距、动效等都应从 token 获取。

推荐：

```css
.glass-card {
  background: var(--surface-glass);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-glass);
}
```

不推荐：

```css
.glass-card {
  background: rgba(255,255,255,0.9);
  border-radius: 20px;
  box-shadow: 0 8px 24px rgba(91,141,239,0.08);
}
```

### 5.2 区分品牌 token 与幻灯片 token

品牌 token 描述品牌视觉：

- 主色
- 文本色
- 背景色
- 字体
- 阴影
- 圆角

幻灯片 token 描述 PPT 版式：

- 画布比例
- 页边距
- 标题字号
- 副标题字号
- 页脚尺寸
- 卡片间距
- 图文比例

示例：

```css
:root {
  --slide-width: 1920px;
  --slide-height: 1080px;
  --slide-margin-x: 110px;
  --slide-margin-y: 90px;
  --slide-title-size: 76px;
  --slide-body-size: 24px;
}
```

### 5.3 单页不临时发明视觉语言

新增样式前先判断：

1. 是否已有 token 可以表达？
2. 是否已有组件可以复用？
3. 是否应该抽象成新的 layout？
4. 是否会破坏整套 PPT 的一致性？

只有确实是一次性特殊页面时，才允许写局部样式。

---

## 6. 讲者备注原则

### 6.1 备注必须与页面同源

讲者备注应和对应页面写在一起，避免用独立数组按顺序匹配。

推荐使用 Slidev 支持的备注写法：

```md
# 页面标题

页面正文。

<!--
这一页的讲者备注。说明讲述重点、转场方式和需要强调的信息。
-->
```

这样插入、删除、移动页面时，备注不会错位。

### 6.2 备注写讲述逻辑，不重复页面正文

备注应描述：

- 这一页怎么讲。
- 哪些点需要强调。
- 如何和上一页、下一页衔接。
- 演示时需要注意什么。

备注不应只是逐字重复页面上的文字。

---

## 7. 页码与页面标识原则

### 7.1 页码自动生成

页码不应手写在每一页。

不推荐：

```html
<span>07 / 20</span>
```

推荐：

```vue
<PageChrome />
```

由组件或 Slidev runtime 自动读取当前页和总页数。

### 7.2 页面 label 用于识别，不用于页码

`label` 可以用于目录、截图命名、调试和审阅，例如：

```yaml
label: 演示 · 主蒸汽压力趋势
```

但页面编号应自动计算，不应写入 label 中作为真实页码来源。

---

## 8. 图片与资产原则

### 8.1 资产路径稳定

图片、logo、截图、视频等应统一放在公共资源目录中，例如：

```text
public/assets/
```

Markdown 中使用稳定路径：

```md
![界面演示](/assets/demo-chat-1.png)
```

### 8.2 图片服务于内容结构

图片不应只是装饰。每张图应对应明确的信息角色：

- 产品截图
- 架构图
- 流程图
- 数据图表
- Logo / 品牌元素
- 背景视觉

### 8.3 复杂图形优先组件化或数据化

如果一张图经常需要更新，优先考虑用组件、SVG、Mermaid 或结构化数据生成，而不是每次手改位图。

---

## 9. 导出原则

### 9.1 HTML 演示是主产物

Slidev 的核心产物是可演示的 HTML deck。PDF、PNG、PPTX 是分发格式。

推荐产物层级：

```text
HTML deck    主演示产物
PDF          正式分享产物
PNG          审阅和归档产物
PPTX         兼容交付产物
```

### 9.2 默认接受截图式 PPTX

复杂 HTML PPT 导出为 PPTX 时，默认优先保证视觉保真。多数情况下，PPTX 可以是逐页截图式。

如果必须生成可编辑的原生 PPTX，应单独定义 PptxGenJS 或其他原生 PPTX 生成路线，不应强迫 HTML deck 承担这个目标。

### 9.3 导出必须命令化

导出流程应通过命令复现，避免依赖手工点击。

推荐脚本：

```json
{
  "scripts": {
    "dev": "slidev slides.md",
    "build": "slidev build slides.md",
    "export:pdf": "slidev export slides.md --format pdf",
    "export:pptx": "slidev export slides.md --format pptx",
    "export:png": "slidev export slides.md --format png"
  }
}
```

---

## 10. 质量检查原则

### 10.1 检查内容完整性

每次交付前应检查：

- 是否缺页。
- 标题是否完整。
- 备注是否存在且未错位。
- 图片是否加载成功。
- 链接是否可用。
- 页面顺序是否符合叙事逻辑。

### 10.2 检查视觉稳定性

至少检查：

- 文本是否溢出。
- 元素是否重叠。
- 页脚是否位置一致。
- 背景是否正确。
- 字体是否符合预期。
- 导出 PDF/PPTX 是否缺失元素。

### 10.3 复杂 deck 建议截图回归

建议为关键 deck 保存截图基准：

```text
screenshots/baseline/
screenshots/current/
screenshots/diff/
```

当 token、layout 或 component 改动时，用截图对比确认没有误伤其他页面。

---

## 11. Agent 友好原则

### 11.1 内容应容易被 Agent 理解

为了方便 Agent 修改和生成 PPT：

- 内容集中在 `slides.md`。
- 每页 frontmatter 字段稳定。
- layout 名称语义清晰。
- 结构化字段优先于 raw HTML。
- 复杂组件有清晰注释和示例。
- 不把个例数据硬编码在通用组件中。

### 11.2 修改边界要清楚

Agent 修改时应优先判断修改对象：

| 修改目标 | 优先修改位置 |
| --- | --- |
| 改文案 | `slides.md` |
| 改页面顺序 | `slides.md` |
| 改某类页面版式 | `layouts/` |
| 改通用视觉零件 | `components/` |
| 改品牌视觉 | `styles/tokens.css` |
| 改导出流程 | `package.json` / `scripts/` |

---

## 12. 推荐工作流

### 12.1 新建 PPT

推荐流程：

1. 先写大纲。
2. 按大纲拆分页面。
3. 为每页选择 layout。
4. 补充每页 Markdown 正文。
5. 补充结构化字段和图片。
6. 补充讲者备注。
7. 浏览器预览。
8. 调整内容密度。
9. 导出 PDF/PNG/PPTX。
10. 做视觉和内容检查。

### 12.2 修改 PPT

推荐优先级：

1. 能改 `slides.md` 就不改 layout。
2. 能改 layout 参数就不写 raw HTML。
3. 能复用 component 就不复制 DOM。
4. 能改 token 就不逐页改样式。
5. 能自动生成就不手动维护。

---

## 13. 判断标准

一套合格的 Slidev Markdown PPT 应满足：

- 内容主要集中在 Markdown 中。
- 每页 layout 选择清晰。
- 复杂结构通过 frontmatter 或组件参数表达。
- 页码自动生成。
- 备注与页面绑定。
- 视觉由 token、layout、component 统一管理。
- 导出流程可命令化复现。
- 人和 Agent 都能安全修改内容。

最终目标：

> 用 Markdown 保持内容可编辑，用 Slidev 保持演示能力，用 Layout 和 Component 保持复杂视觉可复用，用 Tokens 和脚本保证一致性与可复现性。
