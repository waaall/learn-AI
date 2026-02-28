AI agents are here, and already they’re being deployed across contexts that vary widely in consequence, from email triage to [cyber espionage](https://www.anthropic.com/news/disrupting-AI-espionage). Understanding this spectrum is critical for deploying AI safely, yet we know surprisingly little about how people actually use agents in the real world.
> AI 智能体已经到来，而且已经被部署到后果差异极大的各种场景中，从电子邮件分流到网络间谍活动。理解这一光谱对于安全部署 AI 至关重要，但我们对人们在现实世界中究竟如何使用智能体仍知之甚少。

We analyzed millions of human-agent interactions across both Claude Code and our public API using our [privacy-preserving tool](https://www.anthropic.com/research/clio), to ask: How much autonomy do people grant agents? How does that change as people gain experience? Which domains are agents operating in? And are the actions taken by agents risky?
> 我们借助注重隐私的工具，分析了 Claude Code 和我们的公共 API 中数百万条人与智能体的交互，以回答以下问题：人们会赋予智能体多大程度的自主性？这种情况会如何随着人们经验的增加而变化？智能体正在哪些领域中运行？以及，智能体采取的行动是否具有风险？

We found that:
> 我们发现：

-   [**Claude Code is working autonomously for longer.**](https://anthropic.com/research/measuring-agent-autonomy#claude-code-is-working-autonomously-for-longer) Among the longest-running sessions, the length of time Claude Code works before stopping has nearly doubled in three months, from under 25 minutes to over 45 minutes. This increase is smooth across model releases, which suggests it isn’t purely a result of increased capabilities, and that existing models are capable of more autonomy than they exercise in practice.
    > **Claude Code 的自主工作时间更长了。** 在持续时间最长的会话中，Claude Code 在停止前连续工作的时长在三个月内几乎翻了一倍，从不到 25 分钟增长到 45 分钟以上。这个增长跨模型发布呈现平滑趋势，这说明它并不完全是能力提升的结果，也说明现有模型实际具备的自主性高于它们在实践中表现出来的程度。

-   [**Experienced users in Claude Code auto-approve more frequently, but interrupt more often.**](https://www.anthropic.com/research/measuring-agent-autonomy#experienced-users-in-claude-code-auto-approve-more-frequently-but-interrupt-more-often) As users gain experience with Claude Code, they tend to stop reviewing each action and instead let Claude run autonomously, intervening only when needed. Among new users, roughly 20% of sessions use full auto-approve, which increases to over 40% as users gain experience.
    > **Claude Code 中的资深用户更频繁地使用自动批准，但也更常打断它。** 随着用户使用 Claude Code 的经验增加，他们往往不再审查每一个动作，而是让 Claude 自主运行，只在需要时介入。在新用户中，大约 20% 的会话使用完全自动批准，而随着用户经验增加，这一比例会上升到 40% 以上。

-   [**Claude Code pauses for clarification more often than humans interrupt it.**](https://www.anthropic.com/research/measuring-agent-autonomy#claude-code-pauses-for-clarification-more-often-than-humans-interrupt-it) In addition to human-initiated stops, _agent_\-initiated stops are also an important form of oversight in deployed systems. On the most complex tasks, Claude Code stops to ask for clarification more than twice as often as humans interrupt it.
    > **Claude Code 为了澄清而暂停的次数，比人类打断它的次数更多。** 除了人类主动触发的停止之外，由智能体主动触发的停止也是已部署系统中一种重要的监督形式。在最复杂的任务上，Claude Code 停下来请求澄清的频率是人类打断它的两倍以上。

-   **[Agents are used in risky domains, but not yet at scale.](https://anthropic.com/research/measuring-agent-autonomy#agents-are-used-in-risky-domains-but-not-yet-at-scale)** Most agent actions on our public API are low-risk and reversible. Software engineering accounted for nearly 50% of agentic activity, but we saw emerging usage in healthcare, finance, and cybersecurity.
    > **智能体已经被用于高风险领域，但规模尚不大。** 我们公共 API 上的大多数智能体行为都是低风险且可逆的。软件工程占据了近 50% 的智能体活动，但我们也看到了它们在医疗、金融和网络安全领域的初步应用。

Below, we present our methodology and findings in more detail, and end with recommendations for model developers, product developers, and policymakers. Our central conclusion is that effective oversight of agents will require new forms of post-deployment monitoring infrastructure _and_ new human-AI interaction paradigms that help both the human and the AI manage autonomy and risk together.
> 下面，我们将更详细地介绍我们的方法论和研究发现，并以面向模型开发者、产品开发者和政策制定者的建议作结。我们的核心结论是，要想有效监督智能体，就需要新的部署后监测基础设施形式，以及新的 human-AI 交互范式，帮助人类和 AI 一起管理自主性与风险。

We view our research as a small but important first step towards empirically understanding how people deploy and use agents. We will continue to iterate on our methods and communicate our findings as agents are adopted more widely.
> 我们将这项研究视为迈向以实证方式理解人们如何部署和使用智能体的一小步，但却是重要的一步。随着智能体被更广泛地采用，我们将继续迭代方法并传达我们的发现。

## Studying agents in the wild
> 在真实世界中研究智能体

Agents are difficult to study empirically. First, there is no agreed-upon definition of what an agent _is._ Second, agents are evolving quickly. Last year, many of the most sophisticated agents—including Claude Code—involved a single conversational thread, but today there are multi-agent systems that operate autonomously for hours. Finally, model providers have limited visibility into the architecture of their customers’ agents. For example, we have no reliable way to associate independent requests to our API into “sessions” of agentic activity. (We discuss this challenge in more detail at the end of this post.)
> 智能体很难被实证研究。首先，对于什么是智能体，并没有被广泛认可的一致定义。其次，智能体正在快速演化。去年，许多最先进的智能体还包括 Claude Code 在内，都只涉及单一对话线程，但如今已经出现了能够连续自主运行数小时的多智能体系统。最后，模型提供方对客户智能体的架构可见性有限。例如，我们没有可靠的方法把发往我们 API 的独立请求关联成智能体活动的“会话”。（我们会在本文末尾更详细地讨论这一挑战。）

In light of these challenges, how can we study agents empirically?
> 面对这些挑战，我们该如何以实证方式研究智能体？

To start, for this study we adopted a definition of agents that is conceptually grounded and operationalizable: _an agent is an AI system equipped with tools that allow it to take actions_, like running code, calling external APIs, and sending messages to other agents.<sup>1</sup> Studying the tools that agents use tells us a great deal about what they are doing in the world.
> 首先，在这项研究中，我们采用了一个在概念上有根基且可操作的智能体定义：智能体是配备了工具、能够采取行动的 AI 系统，例如运行代码、调用外部 API，以及向其他智能体发送消息。研究智能体所使用的工具，可以让我们了解它们在现实世界中究竟在做什么。

Next, we developed a collection of metrics that draw on data from both agentic uses of our [public API](https://platform.claude.com/docs/en/api/overview) and [Claude Code](https://code.claude.com/docs/en/overview), our own coding agent. These offer a tradeoff between breadth and depth:
> 接下来，我们设计了一组指标，同时利用我们公共 API 中的智能体使用数据，以及我们自有编码智能体 Claude Code 的数据。这两类来源在广度与深度之间提供了一种权衡：

-   Our **public API** gives us broad visibility into agentic deployments across thousands of different customers. Rather than attempting to infer our customers’ agent architectures, we instead perform our analysis at the level of _individual tool calls_.<sup>2</sup> This simplifying assumption allows us to make grounded, consistent observations about real-world agents, even as the contexts in which those agents are deployed vary significantly. The limitation of this approach is that we must analyze actions in isolation, and cannot reconstruct how individual actions compose into longer sequences of behavior over time.
    > 我们的 **公共 API** 让我们能够广泛观察数千个不同客户中的智能体部署情况。我们并不试图推断客户的智能体架构，而是把分析层级放在单个工具调用上。这一简化假设使我们即便面对部署情境差异很大的真实世界智能体，也能做出有根据且一致的观察。这种方法的局限在于，我们必须孤立地分析动作，无法重建单个动作如何随着时间组合成更长的行为序列。

-   **Claude Code** offers the opposite tradeoff. Because Claude Code is our own product, we can link requests across sessions and understand entire agent workflows from start to finish. This makes Claude Code especially useful for studying autonomy—for example, how long agents run without human intervention, what triggers interruptions, and how users maintain oversight over Claude as they develop experience. However, because Claude Code is only one product, it does not provide the same diversity of insight into agentic use as API traffic.
    > **Claude Code** 提供了相反的权衡。由于 Claude Code 是我们自己的产品，我们可以跨会话关联请求，并从头到尾理解完整的智能体工作流。这使 Claude Code 特别适合用于研究自主性，例如智能体在无人干预下能运行多久、什么会触发打断，以及用户随着经验增长会如何维持对 Claude 的监督。然而，由于 Claude Code 只是一个产品，它无法像 API 流量那样在智能体使用上提供同样多样的洞见。

By drawing from both sources using our [privacy-preserving infrastructure](https://www.anthropic.com/research/clio), we can answer questions that neither could address alone.
> 通过使用我们注重隐私的基础设施同时利用这两个来源，我们能够回答任何单一来源都无法单独回答的问题。

## Claude Code is working autonomously for longer
> Claude Code 的自主工作时间更长了

How long do agents actually run without human involvement? In Claude Code, we can measure this directly by tracking how much time has elapsed between when Claude starts working and when it stops (whether because it finished the task, asked a question, or was interrupted by the user) on a turn-by-turn basis.<sup>3</sup>
> 智能体在没有人类参与的情况下，实际会运行多久？在 Claude Code 中，我们可以直接逐轮追踪 Claude 从开始工作到停止之间经过了多少时间来衡量这一点，而停止的原因可能是它完成了任务、提出了问题，或者被用户打断。

Turn duration is an imperfect proxy for autonomy.<sup>4</sup> For example, more capable models could accomplish the same work faster, and subagents allow more work to happen at once, both of which push towards shorter turns.<sup>5</sup> At the same time, users may be attempting more ambitious tasks over time, which would push towards longer turns. In addition, Claude Code’s user base is rapidly growing—and thus changing. We can’t measure these changes in isolation; what we measure is the net result of this interplay, including how long users let Claude work independently, the difficulty of the tasks they give it, and the efficiency of the product itself (which improves [daily](https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md)).
> 单轮持续时间并不是自主性的完美代理指标。例如，能力更强的模型可能更快完成相同工作，而子智能体允许更多工作同时发生，这两者都会推动单轮变短。同时，用户也可能随着时间推移尝试更有雄心的任务，这又会推动单轮变长。此外，Claude Code 的用户群正在快速增长，也因此在发生变化。我们无法把这些变化彼此隔离开来测量；我们所测到的是这些因素相互作用后的净结果，包括用户允许 Claude 独立工作的时长、他们交给 Claude 的任务难度，以及产品本身的效率，而后者还在每天持续改进。

Most Claude Code turns are short. The median turn lasts around 45 seconds, and this duration has fluctuated only slightly over the past few months (between 40 and 55 seconds). In fact, nearly every percentile below the 99th has remained relatively stable.<sup>6</sup> That stability is what we’d expect for a product experiencing rapid growth: when new users adopt Claude Code, they are comparatively inexperienced, and—as we show in the next section—less likely to grant Claude full latitude.
> Claude Code 的大多数单轮都很短。中位数单轮大约持续 45 秒，而过去几个月里这一时长只出现了轻微波动，大致在 40 到 55 秒之间。事实上，99 分位以下几乎所有分位数都相对稳定。对于一款正在快速增长的产品来说，这种稳定性正是我们预期会看到的：当新用户开始使用 Claude Code 时，他们相对缺乏经验，而且正如下一节所示，他们不太可能给予 Claude 完全的自主空间。

The more revealing signal is in the tail. The longest turns tell us the most about the most ambitious uses of Claude Code, and point to where autonomy is heading. Between October 2025 and January 2026, the 99.9th percentile turn duration nearly doubled, from under 25 minutes to over 45 minutes (Figure 1).
> 更能揭示趋势的信号出现在尾部。最长的单轮最能说明 Claude Code 最具雄心的使用方式，也指向自主性将走向何方。在 2025 年 10 月到 2026 年 1 月之间，99.9 分位的单轮持续时间几乎翻了一倍，从不到 25 分钟增长到 45 分钟以上（图 1）。

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fa86a9559ec987a33340c265265463843846ad8c7-3840x2160.png&w=3840&q=75)

**Figure 1.** 99.9th percentile turn duration (how long Claude works on a per-turn basis) in interactive Claude Code sessions, 7-day rolling average. The 99.9th percentile has grown steadily from under 25 minutes in late September to over 45 minutes in early January. This analysis reflects all interactive Claude Code usage.
> **图 1。** 交互式 Claude Code 会话中 99.9 分位的单轮持续时间（即 Claude 每一轮工作多久），采用 7 天滚动平均。99.9 分位从 9 月下旬的不到 25 分钟稳步增长到 1 月上旬的 45 分钟以上。这一分析反映了全部交互式 Claude Code 使用情况。

  
Notably, this increase is smooth across model releases. If autonomy were purely a function of model capability, we would expect sharp jumps with each new launch. The relative steadiness of this trend instead suggests several potential factors are at work, including power users building trust with the tool over time, applying Claude to increasingly ambitious tasks, and the product itself improving.
> 值得注意的是，这一增长跨模型发布呈现出平滑走势。如果自主性纯粹是模型能力的函数，我们会预期每次新版本发布时都出现明显跃升。相反，这一趋势的相对平稳说明可能有多个因素同时在起作用，包括重度用户随着时间推移建立起对该工具的信任、把 Claude 用于越来越有雄心的任务，以及产品本身在持续改进。

The extreme turn duration has declined somewhat since mid-January. We hypothesize a few reasons why. First, the Claude Code user base [doubled](https://www.anthropic.com/news/anthropic-raises-30-billion-series-g-funding-380-billion-post-money-valuation) between January and mid-February, and a larger and more diverse population of sessions could reshape the distribution. Second, as users returned from the holiday break, the projects they brought to Claude Code may have shifted from hobby projects to more tightly circumscribed work tasks. Most likely, it’s a combination of these factors and others we haven’t identified.
> 自 1 月中旬以来，极端单轮时长有所下降。我们推测有几个原因。首先，Claude Code 的用户群在 1 月到 2 月中旬之间翻了一倍，更大且更多样化的会话群体可能重塑了分布。其次，随着用户结束假期回归，他们带到 Claude Code 上的项目可能从兴趣项目转向了边界更清晰的工作任务。最有可能的情况是，这是这些因素以及其他尚未识别因素共同作用的结果。

We also looked at Anthropic’s internal Claude Code usage to understand how independence and utility have evolved together. From August to December, Claude Code’s success rate on internal users’ most challenging tasks doubled, at the same time that the average number of human interventions per session decreased from 5.4 to 3.3.<sup>7</sup> Users are granting Claude more autonomy and, at least internally, achieving better outcomes while needing to intervene less often.
> 我们还考察了 Anthropic 内部对 Claude Code 的使用情况，以了解独立性与实用性如何共同演化。从 8 月到 12 月，Claude Code 在内部用户最具挑战性任务上的成功率翻了一倍，与此同时，每个会话中人类干预的平均次数从 5.4 次下降到 3.3 次。用户正在赋予 Claude 更多自主性，而且至少在内部场景中，他们在更少干预的同时取得了更好的结果。

Both measurements point to a significant deployment overhang, where the autonomy models are capable of handling exceeds what they exercise in practice.
> 这两项测量都指向一个显著的部署悬差，即模型实际能够处理的自主性水平超过了它们在实践中被允许发挥出来的水平。

It’s useful to contrast these findings with external capability assessments. One of the most widely cited capability assessments is METR’s “Measuring AI Ability to Complete Long Tasks,” which [estimates](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) that Claude Opus 4.5 can complete tasks with a 50% success rate that would take a human nearly 5 _hours_. The 99.9th percentile turn duration in Claude Code, in contrast, is ~42 minutes, and the median is much shorter. However, the two metrics are not directly comparable. The METR evaluation captures what a model is capable of in an idealized setting with no human interaction and no real-world consequences. Our measurements capture what happens in practice, where Claude pauses to ask for feedback and users interrupt.<sup>8</sup> And METR’s five-hour figure measures task difficulty—how long the task would take a human—not how long the model actually runs.
> 将这些发现与外部能力评估做对照是有帮助的。被引用最广的能力评估之一是 METR 的“Measuring AI Ability to Complete Long Tasks”，其估计 Claude Opus 4.5 能以 50% 的成功率完成那些需要人类接近 5 小时才能完成的任务。相比之下，Claude Code 中 99.9 分位的单轮持续时间约为 42 分钟，中位数则短得多。不过，这两个指标并不能直接比较。METR 的评估衡量的是模型在没有人类交互、没有现实后果的理想化环境中能做什么。我们的测量捕捉的是实践中实际发生了什么，其中 Claude 会暂停以请求反馈，而用户也会打断它。而且，METR 所说的五小时衡量的是任务难度，也就是人类完成该任务要花多久，而不是模型实际运行了多久。

Neither capability evaluations nor our measurements alone give a complete picture of agent autonomy, but together they suggest that the latitude granted to models in practice lags behind what they can handle.
> 无论是能力评估还是我们的测量，单独来看都无法给出智能体自主性的完整图景，但把两者放在一起，它们表明实践中赋予模型的自主空间落后于模型实际能够承受的程度。

##  Experienced users in Claude Code auto-approve more frequently, but interrupt more often
> Claude Code 中的资深用户更频繁地自动批准，但也更常打断它

How do humans adapt how they work with agents over time? We found that people grant Claude Code more autonomy as they gain experience using it (Figure 2). Newer users (<50 sessions) employ full auto-approve roughly 20% of the time; by 750 sessions, this increases to over 40% of sessions.
> 随着时间推移，人类会如何调整自己与智能体协作的方式？我们发现，随着使用经验增长，人们会给予 Claude Code 更多自主性（图 2）。新用户（少于 50 次会话）大约有 20% 的时间会启用完全自动批准；到了 750 次会话时，这一比例会上升到 40% 以上。

This shift is gradual, suggesting a steady accumulation of trust. It’s also important to note that Claude Code’s default settings require users to manually approve each action, so part of this transition may reflect users configuring the product to match their preferences for greater independence as they become familiar with Claude’s capabilities.
> 这种转变是渐进发生的，说明信任是在稳定积累的。还需要注意的是，Claude Code 的默认设置要求用户手动批准每一个动作，因此这种转变的一部分也可能反映了用户在逐渐熟悉 Claude 的能力后，会把产品配置调整为更符合自己对更高独立性的偏好。

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Ff1e687f19e4e87590b73dd9fa1cf5726184d7209-3840x2160.png&w=3840&q=75)

**Figure 2.** Auto-approve rate by account tenure. Experienced users increasingly let Claude run without any manual approval. Data reflects all interactive Claude Code usage for users who signed up after September 19, 2025. Line and CI bounds are LOWESS-smoothed (0.15 bandwidth). The x-axis is a log scale.
> **图 2。** 按账户使用资历划分的自动批准率。资深用户越来越多地让 Claude 在完全没有人工批准的情况下运行。数据反映的是 2025 年 9 月 19 日之后注册用户的全部交互式 Claude Code 使用情况。曲线及置信区间边界采用 LOWESS 平滑（带宽 0.15）。x 轴为对数尺度。

Approving actions is only one method of supervising Claude Code. Users can also interrupt Claude while it is working to provide feedback. We find that interrupt rates increase with experience. New users (those with around 10 sessions) interrupt Claude in 5% of turns, while more experienced users interrupt in around 9% of turns (Figure 3).
> 批准动作只是监督 Claude Code 的一种方式。用户也可以在 Claude 工作时打断它并提供反馈。我们发现，打断率会随着经验增加而上升。新用户（大约有 10 次会话的用户）会在 5% 的单轮中打断 Claude，而更有经验的用户会在大约 9% 的单轮中打断它（图 3）。

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fabcddadbbdc63751449a6a7fc5731a693c9654bb-3840x2160.png&w=3840&q=75)

**Figure 3.** Interrupt rates by account tenure on a turn-by-turn basis. Experienced users interrupt Claude more often, not less. Data reflects all interactive Claude Code usage for users who signed up after September 19, 2025. Shaded region shows 95% Wilson score confidence interval. Line and CI bounds are LOWESS-smoothed (0.15 bandwidth). The x-axis is a log scale.
> **图 3。** 按账户使用资历划分的逐轮打断率。资深用户打断 Claude 的频率更高，而不是更低。数据反映的是 2025 年 9 月 19 日之后注册用户的全部交互式 Claude Code 使用情况。阴影区域表示 95% Wilson 评分置信区间。曲线及置信区间边界采用 LOWESS 平滑（带宽 0.15）。x 轴为对数尺度。

Both interruptions _and_ auto-approvals increase with experience. This apparent contradiction reflects a shift in users’ oversight strategy. New users are more likely to approve each action before it’s taken, and therefore rarely need to interrupt Claude mid-execution. Experienced users are more likely to let Claude work autonomously, stepping in when something goes wrong or needs redirection. The higher interrupt rate may also reflect active monitoring by users who have more honed instincts for when their intervention is needed. We expect the per-turn interrupt rate to eventually plateau as users settle into a stable oversight style, and indeed the curve may already be flattening among the most experienced users (though widening confidence intervals at higher session counts make this difficult to confirm).<sup>9</sup>
> 随着经验增加，打断和自动批准这两者都会上升。这个看似矛盾的现象反映的是用户监督策略的转变。新用户更可能在每个动作执行前先批准，因此很少需要在 Claude 执行过程中途打断它。资深用户则更可能让 Claude 自主工作，并在出现问题或需要重新引导时介入。更高的打断率也可能反映出，这些用户对何时需要自己介入已经形成了更敏锐的直觉，因此在主动监控。我们预计，随着用户形成稳定的监督风格，逐轮打断率最终会进入平台期，而事实上在经验最丰富的用户中，这条曲线可能已经开始趋于平缓，尽管更高会话数下不断扩大的置信区间让这一点难以确认。

We saw a similar pattern on our public API: 87% of tool calls on minimal-complexity tasks (like editing a line of code) have some form of human involvement, compared to only 67% of tool calls for high-complexity tasks (like [autonomously finding zero-day exploits](https://red.anthropic.com/2026/zero-days/) or [writing a compiler](https://www.anthropic.com/engineering/building-c-compiler)).<sup>10</sup> This may seem counterintuitive, but there are two likely explanations. First, step-by-step approval becomes less practical as the number of steps grows, so it is structurally harder to supervise each action on complex tasks. Second, our Claude Code data suggests that experienced users tend to grant the tool more independence, and complex tasks may disproportionately come from experienced users. While we cannot directly measure user tenure on our public API, the overall pattern is consistent with what we observe in Claude Code.
> 我们在公共 API 上也看到了类似模式：在最低复杂度任务上，例如修改一行代码，87% 的工具调用都带有某种形式的人类参与；相比之下，在高复杂度任务上，例如自主寻找零日漏洞或编写编译器，只有 67% 的工具调用有人类参与。这看起来也许违反直觉，但可能有两个解释。第一，随着步骤数量增加，逐步批准会变得不再那么现实，因此从结构上看，复杂任务中的每一个动作都更难被逐一监督。第二，我们的 Claude Code 数据表明，资深用户往往会赋予该工具更多独立性，而复杂任务可能更集中来自这些资深用户。虽然我们无法在公共 API 上直接测量用户资历，但整体模式与我们在 Claude Code 中观察到的现象是一致的。

Taken together, these findings suggest that experienced users aren’t necessarily abnegating oversight. The fact that interrupt rates increase with experience alongside auto-approvals indicates some form of active monitoring. This reinforces a point we have made [previously](https://www.anthropic.com/news/our-framework-for-developing-safe-and-trustworthy-agents): effective oversight doesn’t require approving every action but being in a position to intervene when it matters.
> 综合来看，这些发现表明资深用户并不一定是在放弃监督。打断率会随着经验和自动批准率一同上升，这说明存在某种形式的主动监控。这也强化了我们此前提出的一点：有效监督并不要求批准每一个动作，而是要在人类能够在关键时刻介入的位置上。

## Claude Code pauses for clarification more often than humans interrupt it
> Claude Code 为了澄清而暂停的次数，比人类打断它的次数更多

Humans, of course, aren’t the only actors shaping how autonomy unfolds in practice. Claude is an active participant too, stopping to ask for clarification when it’s unsure how to proceed. We found that as task complexity increases, Claude Code asks for clarification more often—and more frequently than humans choose to interrupt it (Figure 4).
> 当然，塑造自主性在实践中如何展开的，并不只有人类。Claude 本身也是积极参与者，当它不确定该如何继续时，会停下来请求澄清。我们发现，随着任务复杂性上升，Claude Code 请求澄清的次数会变得更多，而且比人类选择打断它的频率还要高（图 4）。

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fff8122208d6b9aa0739994f75cab63b476203904-1920x1080.png&w=3840&q=75)

_**Figure 4.** Clarification questions from Claude and interruptions by the human, by goal complexity. As tasks get more complex, Claude is more likely to ask for clarification and humans are more likely to interrupt. Claude-initiated stops increase faster than human-initiated stops. 95% CI < 0.9% for all categories, n = 500k interactive Claude Code sessions._
> _**图 4。** 按目标复杂度划分的 Claude 澄清提问与人类打断情况。随着任务变得更复杂，Claude 更可能请求澄清，人类也更可能打断。由 Claude 主动发起的停止增长速度快于由人类发起的停止。所有类别的 95% 置信区间均小于 0.9%，n = 50 万个交互式 Claude Code 会话。_

  
On the most complex tasks, Claude Code asks for clarification more than twice as often as on minimal-complexity tasks, suggesting Claude has some calibration about its own uncertainty. However, it’s important not to overstate this finding: Claude may not be stopping at the right moments, it may ask unnecessary questions, and its behavior might be affected by product features such as [Plan Mode](https://code.claude.com/docs/en/common-workflows#use-plan-mode-for-safe-code-analysis). Regardless, as tasks get harder, Claude increasingly limits its own autonomy by stopping to consult the human, rather than requiring the human to step in.<sup>11</sup>
> 在最复杂的任务上，Claude Code 请求澄清的频率是最低复杂度任务的两倍以上，这说明 Claude 对自身不确定性可能具备某种校准能力。不过，不应过度解读这一发现：Claude 可能并没有在正确的时刻停下来，它也可能提出没有必要的问题，而且它的行为还可能受到诸如 Plan Mode 之类产品功能的影响。无论如何，随着任务变得更难，Claude 越来越多地通过停下来向人类咨询来限制自身自主性，而不是要求人类主动介入。

Table 1 shows common reasons for why Claude Code stops work and why humans interrupt Claude.
> 表 1 展示了 Claude Code 停止工作以及人类打断 Claude 的常见原因。

What causes Claude Code to stop?
> 是什么导致 Claude Code 停下来？

| **Why does Claude stop itself?** | **Why do humans interrupt Claude?** |
| --- | --- |
| To present the user with a choice between proposed approaches (35%) | To provide missing technical context or corrections (32%) |
| To gather diagnostic information or test results (21%) | Claude was slow, hanging, or excessive (17%) |
| To clarify vague or incomplete requests (13%) | They received enough help to proceed independently (7%) |
| To request missing credentials, tokens, or access (12%) | They want to take the next step themselves (e.g., manual testing, deployment, committing, etc.) (7%) |
| To get approval or confirmation before taking action (11%) | To change requirements mid-task (5%) |

> 表 1 译文：
> Claude 为什么会自己停下来？ | 人类为什么会打断 Claude？
> 向用户呈现几种拟议方案供其选择（35%） | 提供缺失的技术上下文或纠正信息（32%）
> 收集诊断信息或测试结果（21%） | Claude 太慢、卡住了，或执行过度（17%）
> 澄清模糊或不完整的请求（13%） | 他们已经获得足够帮助，可以自行继续（7%）
> 请求缺失的凭证、令牌或访问权限（12%） | 他们想自己执行下一步，例如手动测试、部署、提交等（7%）
> 在采取行动前获取批准或确认（11%） | 在任务中途更改需求（5%）

Table 1. Common reasons why Claude stops itself and why humans interrupt Claude, as determined by Claude, based on a sample of 500k human interruptions and 500k completed turns in interactive Claude Code sessions. Some clusters have been lightly edited for clarity.¹²
> 表 1。基于 50 万次人类打断和 50 万次已完成单轮的交互式 Claude Code 会话样本，由 Claude 判断 Claude 自己停下来以及人类打断 Claude 的常见原因。为了更清晰起见，其中一些聚类名称经过了轻微编辑。¹²

  
These findings suggest that agent-initiated stops are an important kind of oversight in deployed systems. Training models to recognize and act on their own uncertainty is an important safety property that complements external safeguards like permission systems and human oversight. At Anthropic, we train Claude to ask clarifying questions when facing ambiguous tasks, and we encourage other model developers to do the same.
> 这些发现表明，在已部署系统中，由智能体主动发起的停止是一种重要的监督形式。训练模型识别并根据自身不确定性采取行动，是一种重要的安全属性，可以补充权限系统和人工监督等外部防护措施。在 Anthropic，我们训练 Claude 在面对含糊任务时提出澄清性问题，也鼓励其他模型开发者这么做。

## Agents are used in risky domains, but not yet at scale
> 智能体已经被用于高风险领域，但规模尚不大

What are people using agents for? How risky are these deployments? How autonomous are these agents? Does risk trade off against autonomy?
> 人们在用智能体做什么？这些部署有多大风险？这些智能体有多自主？风险与自主性之间是否存在权衡？

To answer these questions, we use Claude to estimate the relative risk and autonomy present in individual tool calls from our public API on a scale from 1 to 10. Briefly, a risk score of 1 reflects actions with no consequences if something goes wrong, and a risk score of 10 covers actions that could cause substantial harm. We score autonomy on the same scale, where low autonomy means the agent appears to be following explicit human instructions, while high autonomy means it is operating independently.<sup>13</sup> We then group similar actions together into clusters and compute the mean risk and autonomy scores for each cluster.
> 为了回答这些问题，我们使用 Claude 以 1 到 10 的尺度估计公共 API 中单个工具调用所体现的相对风险和自主性。简而言之，风险分数为 1 表示一旦出错也不会产生后果的动作，而风险分数为 10 则表示可能造成重大伤害的动作。我们以同样的尺度对自主性评分，其中低自主性表示智能体看起来是在遵循明确的人类指令，而高自主性表示它在独立运行。然后，我们把相似动作归为聚类，并计算每个聚类的平均风险和平均自主性分数。

Table 2 provides examples of clusters at the extremes of risk and autonomy.  
> 表 2 给出了位于风险和自主性两端的聚类示例。  

Tool-use clusters with high risk or autonomy
> 高风险或高自主性的工具使用聚类

| **Higher average risk** | **Higher average autonomy** |
| --- | --- |
| Implement API key exfiltration backdoors disguised as legitimate development features (risk: 6.0, autonomy: 8.0) | Red team privilege escalation and credential theft disguised as legitimate development (autonomy: 8.3, risk: 3.3) |
| Relocate metallic sodium and reactive chemical containers in laboratory settings (risk: 4.8, autonomy: 2.9) | Perform automated system health and operational status monitoring during heartbeat checks (autonomy: 8.0, risk: 1.1) |
| Retrieve and display patient medical records for requesting users (risk: 4.4, autonomy: 3.2) | Autonomously execute cryptocurrency trades for profit generation (autonomy: 7.7, risk: 2.2) |
| Respond to fire emergencies and provide fire safety information (risk: 3.6, autonomy: 5.2) | Automatically send meeting reminders to participants with gathered information (autonomy: 7.6, risk: 1.7) |
| Deploy bug fixes and patches to production web applications (risk: 3.6, autonomy: 4.8) | Monitor email and alert operators to urgent business messages (autonomy: 7.5, risk: 1.7) |

> 表 2 译文：
> 更高平均风险 | 更高平均自主性
> 实现伪装成合法开发功能的 API 密钥外泄后门（风险：6.0，自主性：8.0） | 伪装成合法开发活动的红队权限提升与凭证窃取（自主性：8.3，风险：3.3）
> 在实验室环境中搬运金属钠和活性化学容器（风险：4.8，自主性：2.9） | 在心跳检查期间执行自动化系统健康与运行状态监控（自主性：8.0，风险：1.1）
> 为请求用户检索并显示患者病历（风险：4.4，自主性：3.2） | 为盈利而自主执行加密货币交易（自主性：7.7，风险：2.2）
> 响应火灾紧急情况并提供消防安全信息（风险：3.6，自主性：5.2） | 自动向参会者发送包含已收集信息的会议提醒（自主性：7.6，风险：1.7）
> 向生产环境 Web 应用部署缺陷修复和补丁（风险：3.6，自主性：4.8） | 监控电子邮件并向操作人员提醒紧急业务消息（自主性：7.5，风险：1.7）

Table 2. Clusters with high average risk or high average autonomy. Based on a sample of 998,481 random tool calls from our public API.
> 表 2。平均风险高或平均自主性高的聚类。基于我们公共 API 上 998,481 次随机工具调用样本。

Given the limited visibility we have into these agents, it is possible that these actions are simulated (for example, we have no way to determine whether the financial trades Claude orders are actually executed, and we believe that many of the “data exfiltrations” are evaluations). And while these higher-risk actions are rare as a share of overall traffic, the consequences of a single error can still be significant.
> 由于我们对这些智能体的可见性有限，这些动作有可能是模拟出来的，例如，我们无法判断 Claude 下达的金融交易指令是否真的被执行，而且我们相信许多“数据外泄”实际上是评估。尽管这些高风险动作在整体流量中的占比很低，但单次错误造成的后果仍然可能相当严重。

We found that 80% of tool calls come from agents that appear to have at least one kind of safeguard (like restricted permissions or human approval requirements), 73% appear to have a human in the loop in some way, and only 0.8% of actions appear to be irreversible (such as sending an email to a customer).<sup>14</sup>
> 我们发现，80% 的工具调用来自看起来至少具有一种防护措施的智能体，例如受限权限或人工批准要求；73% 的调用看起来在某种程度上有人类参与；只有 0.8% 的动作看起来是不可逆的，例如向客户发送电子邮件。

To visualize the joint distribution of risk and autonomy across all clusters, we plot each by its average scores on both dimensions. Each point in Figure 5 corresponds to a cluster of related actions, positioned by its average risk and autonomy.
> 为了可视化所有聚类在风险与自主性上的联合分布，我们按照这两个维度的平均分绘制每个聚类。图 5 中的每一个点都对应一组相关动作的聚类，其位置由平均风险和平均自主性决定。

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F6f659cded3fb01e70d7eea0130d0c121dc74ea23-3840x2160.png&w=3840&q=75)

**Figure 5.** Mean Claude-estimated risk and autonomy by task cluster. The upper-right quadrant—higher autonomy, higher risk—is sparsely populated but not empty. Data reflects tool calls made via our public API. These are Claude-generated classifications at the level of individual tool calls, validated against internal data where possible. We describe the full methodology in the Appendix. Clusters that do not meet our aggregation minimums (due to either insufficient unique tool calls or customers) are excluded. Based on a sample of 998,481 tool calls on our public API.
> **图 5。** 按任务聚类划分的 Claude 估计平均风险与平均自主性。右上象限，也就是高自主性、高风险区域，虽然点很少，但并非空白。数据反映的是通过我们公共 API 发起的工具调用。这些分类由 Claude 在单个工具调用层面生成，并在可能情况下使用内部数据进行了验证。完整方法见附录。不满足聚合最小门槛的聚类会被排除，原因可能是唯一工具调用数不足，或客户数不足。基于我们公共 API 上 998,481 次工具调用样本。

The vast majority of actions on our public API are low-risk. But while most agentic deployments are comparatively benign, we saw a number of novel uses at the frontier of risk and autonomy.<sup>15</sup> The riskiest clusters—again, many of which we expect to be evaluations—tended to involve sensitive security-related actions, financial transactions, and medical information. While risk is concentrated at the low end of the scale, autonomy varies more widely. On the low end (autonomy score of 3-4), we see agents completing small, well-scoped tasks for humans, like making restaurant reservations or minor tweaks to code. On the high end (autonomy score above 6), we see agents submitting machine learning models to data science competitions or triaging customer service requests.
> 我们公共 API 上的绝大多数动作都是低风险的。不过，尽管大多数智能体部署相对无害，我们仍然看到了一些位于风险与自主性前沿的新型用途。风险最高的那些聚类，再次强调，其中许多我们认为是评估，往往涉及敏感的安全相关操作、金融交易以及医疗信息。虽然风险大多集中在评分尺度的低端，但自主性的分布要广得多。在低端，也就是自主性分数为 3 到 4 时，我们看到智能体在为人类完成范围小且界定清晰的任务，例如预订餐厅或对代码做细微调整。在高端，也就是自主性分数高于 6 时，我们看到智能体会把机器学习模型提交到数据科学竞赛中，或者分流处理客服请求。

We also anticipate that agents operating at the extremes of risk and autonomy will become increasingly common. Today, agents are concentrated in a single industry: software engineering accounts for nearly 50% of tool calls on our public API (Figure 6). Beyond coding, we see a number of smaller applications across business intelligence, customer service, sales, finance, and e-commerce, but none comprise more than a few percentage points of traffic. As agents expand into these domains, many of which carry higher stakes than fixing a bug, we expect the frontier of risk and autonomy to expand.
> 我们也预计，运行在风险和自主性两端的智能体会变得越来越常见。如今，智能体高度集中在单一行业中：软件工程占据了我们公共 API 上近 50% 的工具调用（图 6）。除编码之外，我们还看到了商业智能、客户服务、销售、金融和电子商务等领域的一些较小规模应用，但没有任何一个领域占流量的几个百分点以上。随着智能体扩展到这些领域，而其中许多领域的利害关系都高于修复一个 bug，我们预计风险与自主性的前沿也会随之扩张。

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F32a4492bec971b895b67a1736661635b2c412922-3840x2160.png&w=3840&q=75)

**Figure 6.** Distribution of tool calls by domain. Software engineering accounts for nearly 50% of tool calls. Data reflects tool calls made via our public API. 95% CI < 0.5% for all categories, n = 998,481.
> **图 6。** 按领域划分的工具调用分布。软件工程占据了近 50% 的工具调用。数据反映的是通过我们公共 API 发起的工具调用。所有类别的 95% 置信区间均小于 0.5%，n = 998,481。

These patterns suggest we are in the early days of agent adoption. Software engineers were the first to build and use agentic tools at scale, and Figure 6 suggests that other industries are beginning to experiment with agents as well.<sup>16</sup> Our methodology allows us to monitor how these patterns evolve over time. Notably, we can monitor whether or not usage tends to move towards more autonomous and more risky tasks.
> 这些模式表明，我们仍处于智能体采用的早期阶段。软件工程师是第一批大规模构建和使用智能体工具的人，而图 6 表明其他行业也开始尝试智能体。我们的方法使我们能够监测这些模式如何随时间演化。值得注意的是，我们可以持续监测使用情况是否趋向于更自主、也更高风险的任务。

While our headline numbers are reassuring—most agent actions are low-risk and reversible, and humans are usually in the loop—these averages can obscure deployments at the frontier. The concentration of adoption in software engineering, combined with growing experimentation in new domains, suggests that the frontier of risk and autonomy will expand. We discuss what this means for model developers, product developers, and policymakers in our recommendations at the end of this post.
> 虽然我们的 headline 数字令人安心，大多数智能体行为都低风险、可逆，而且人类通常在环路中，但这些平均值可能掩盖了前沿部署情况。采用高度集中在软件工程领域，再加上新领域中不断增长的实验，说明风险与自主性的前沿将会扩张。我们会在本文结尾的建议部分讨论这对模型开发者、产品开发者和政策制定者意味着什么。

##  Limitations
> 局限性

This research is just a start. We provide only a partial view into agentic activity, and we want to be upfront about what our data can and cannot tell us:
> 这项研究只是一个开始。我们只能提供对智能体活动的局部视图，因此希望坦率说明我们的数据能告诉我们什么，以及不能告诉我们什么：

-   We can only analyze traffic from a single model provider: Anthropic. Agents built on other models may show different adoption patterns, risk profiles, and interaction dynamics.
    > 我们只能分析单一模型提供方 Anthropic 的流量。基于其他模型构建的智能体可能会呈现不同的采用模式、风险画像和交互动态。
-   Our two data sources offer complementary but incomplete views. Public API traffic gives us breadth across thousands of deployments, but we can only analyze individual tool calls in isolation, rather than full agent sessions. Claude Code gives us complete sessions, but only for a single product that is overwhelmingly used for software engineering. Many of our strongest findings are grounded in data from Claude Code, and may not generalize to other domains or products.
    > 我们的两类数据来源提供了互补但并不完整的视角。公共 API 流量让我们获得了跨数千个部署的广度，但我们只能孤立地分析单个工具调用，而无法分析完整的智能体会话。Claude Code 则为我们提供了完整会话，但仅限于一个几乎主要用于软件工程的单一产品。我们许多最有力的发现都建立在 Claude Code 数据之上，因此未必能够推广到其他领域或产品。
-   Our classifications are generated by Claude. We provide an opt-out category (e.g., “not inferable,” “other”) for each dimension and validate against internal data where possible (see our [Appendix](https://cdn.sanity.io/files/4zrzovbb/website/55e4d2de6eb39b3a9259c3f74843f86b1a12e265.pdf) for more details), but we cannot manually inspect the underlying data due to privacy constraints. Some safeguards or oversight mechanisms may also exist outside the context we can observe.
    > 我们的分类由 Claude 生成。我们为每个维度都提供了可退出的类别，例如“无法推断”或“其他”，并在可能的情况下用内部数据进行验证，详见附录，但由于隐私限制，我们无法手动检查底层数据。某些防护措施或监督机制也可能存在于我们无法观察到的上下文之外。
-   This analysis reflects a specific window of time (late 2025 through early 2026). The landscape of agents is changing quickly, and patterns may shift as capabilities grow and adoption evolves. We plan to extend this analysis over time.
    > 这项分析反映的是一个特定时间窗口，也就是 2025 年末到 2026 年初。智能体版图变化很快，而随着能力增长和采用演进，这些模式也可能发生转移。我们计划随着时间推移扩展这项分析。
-   Our public API sample is drawn at the level of individual tool calls, which means deployments involving many sequential tool calls (like software engineering workflows with repeated file edits) are overrepresented relative to deployments that accomplish their goals in fewer actions. This sampling approach reflects the volume of agent activity but not necessarily the distribution of agent deployments or uses.
    > 我们的公共 API 样本是在单个工具调用层面抽取的，这意味着那些涉及大量连续工具调用的部署，例如带有反复文件编辑的软件工程工作流，相比于用更少动作就能实现目标的部署会被过度代表。这种采样方式反映的是智能体活动量，而不一定反映智能体部署或用途的分布。
-   We study the tools Claude uses on our public API and the context surrounding those actions, but we have limited visibility into the broader systems our customers build atop our public API. An agent that appears to operate autonomously at the API level may have human review downstream that we cannot observe. In particular, our risk, autonomy, and human involvement classifications reflect what Claude can infer from the context of individual tool calls, and do not distinguish between actions taken in production and actions taken as part of evaluations or red-teaming exercises. Several of the highest-risk clusters appear to be security evaluations, which highlights the limits of our visibility into the broader context surrounding each action.
    > 我们研究的是 Claude 在公共 API 上使用的工具以及这些动作周围的上下文，但我们对客户建立在公共 API 之上的更广泛系统可见性有限。一个在 API 层面看起来是自主运行的智能体，在下游可能存在我们无法观察到的人类审查。特别是，我们对风险、自主性和人类参与的分类反映的是 Claude 能从单个工具调用的上下文中推断出的内容，而不会区分这些动作究竟发生在生产环境中，还是作为评估或红队演练的一部分。若干最高风险聚类看起来就是安全评估，这凸显了我们对每个动作所处更广泛上下文的可见性边界。

## Looking ahead
> 展望未来

We are in the early days of agent adoption, but autonomy is increasing and higher-stakes deployments are emerging, especially as products like [Cowork](https://support.claude.com/en/articles/13345190-getting-started-with-cowork) make agents more accessible. Below, we offer recommendations for model developers, product developers, and policymakers. Given that we have only just begun measuring agent behavior in the wild, we avoid making strong prescriptions and instead highlight areas for future work.
> 我们仍处于智能体采用的早期阶段，但自主性正在提升，更高利害关系的部署也正在出现，尤其是在 Cowork 之类产品让智能体更易获得之后。下面，我们将为模型开发者、产品开发者和政策制定者提出建议。鉴于我们才刚刚开始在真实世界中测量智能体行为，我们避免提出过强的规定性主张，而是强调未来值得继续开展工作的领域。

**Model and product developers should invest in post-deployment monitoring.** Post-deployment monitoring is essential for understanding how agents are actually used. Pre-deployment evaluations test what agents are capable of in controlled settings, but many of our findings cannot be observed through pre-deployment testing alone. Beyond understanding a model’s capabilities, we must also understand how people interact with agents in practice. The data we report here exists because we _chose_ to build the infrastructure to collect it. But there’s more to do. We have no reliable way to link independent requests to our public API into coherent agent sessions, which limits what we can learn about agent behavior beyond first-party products like Claude Code. Developing these methods in a privacy-preserving way is an important area for cross-industry research and collaboration.
> **模型和产品开发者应当投资于部署后监测。** 部署后监测对于理解智能体究竟如何被使用至关重要。部署前评估测试的是智能体在受控环境中的能力，但我们的许多发现无法仅通过部署前测试观察到。除了理解模型的能力之外，我们还必须理解人们在实践中如何与智能体互动。我们在这里报告的数据之所以存在，是因为我们选择去建设收集这些数据所需的基础设施。但仍有更多工作要做。我们目前还没有可靠的方法把发往公共 API 的独立请求串联成一致的智能体会话，这限制了我们对 Claude Code 这类第一方产品之外的智能体行为的理解。以保护隐私的方式发展这些方法，是一个值得跨行业研究与合作的重要方向。

**Model developers should consider training models to recognize their own uncertainty.** Training models to recognize their own uncertainty and surface issues to humans proactively is an important safety property that complements external safeguards like human approval flows and access restrictions. We train Claude to do this (and our analysis shows that Claude Code asks questions more often than humans interrupt it), and we encourage other model developers to do the same.
> **模型开发者应考虑训练模型识别自身不确定性。** 训练模型识别自己的不确定性，并主动把问题暴露给人类，是一种重要的安全属性，可以补充人工批准流程和访问限制等外部防护措施。我们会这样训练 Claude，而我们的分析也表明，Claude Code 提问的频率高于人类打断它的频率，我们也鼓励其他模型开发者这样做。

**Product developers should design for user oversight.** Effective oversight of agents requires more than putting a human in the approval chain. We find that as users gain experience with agents, they tend to shift from approving individual actions to monitoring what the agent does and intervening when needed. In Claude Code, for example, experienced users auto-approve more but also interrupt more. We see a related pattern on our public API, where human involvement appears to decrease as the complexity of the goal increases. Product developers should invest in tools that give users trustworthy visibility into what agents are doing, along with simple intervention mechanisms that allow them to redirect the agent when something goes wrong. This is something we continue to invest in for Claude Code (for example, through [real-time steering](https://github.com/anthropics/claude-code/issues/535) and [OpenTelemetry](https://code.claude.com/docs/en/monitoring-usage)), and we encourage other product developers to do the same.
> **产品开发者应当围绕用户监督来设计。** 对智能体的有效监督，不只是把人放进批准链条里那么简单。我们发现，随着用户对智能体的使用经验增长，他们往往会从批准单个动作，转向监控智能体在做什么，并在需要时介入。以 Claude Code 为例，资深用户自动批准更多，但也打断得更多。我们在公共 API 上也看到了相关模式，即随着目标复杂性增加，人类参与似乎在减少。产品开发者应当投资于能够让用户可信地看见智能体在做什么的工具，并提供简单的干预机制，让用户在出问题时能够重新引导智能体。对于 Claude Code，我们也在持续投入这方面能力，例如实时引导和 OpenTelemetry，并鼓励其他产品开发者采取同样做法。

**It's too early to mandate specific interaction patterns.** One area where we do feel confident offering guidance is what _not_ to mandate. Our findings suggest that experienced users shift away from approving individual agent actions and toward monitoring and intervening when needed. Oversight requirements that prescribe specific interaction patterns, such as requiring humans to approve every action, will create friction without necessarily producing safety benefits. As agents and the science of agent measurement mature, the focus should be on whether humans are in a position to effectively monitor and intervene, rather than on requiring particular forms of involvement.
> **现在要求特定交互模式还为时过早。** 有一个方面我们确实有把握给出建议，那就是什么不应该被强制规定。我们的发现表明，资深用户会逐渐远离逐个批准智能体动作，转向在需要时进行监控和干预。那些规定特定交互模式的监督要求，例如要求人类批准每一个动作，会增加摩擦，却未必带来相应的安全收益。随着智能体以及智能体测量科学逐渐成熟，重点应当放在确保人类处于能够有效监控和干预的位置，而不是要求某种特定形式的参与。

A central lesson from this research is that the autonomy agents exercise in practice is co-constructed by the model, the user, and the product. Claude limits its own independence by pausing to ask questions when it’s uncertain. Users develop trust as they work with the model, and shift their oversight strategy accordingly. What we observe in any deployment emerges from all three of these forces, which is why it cannot be fully characterized by pre-deployment evaluations alone. Understanding how agents actually behave requires measuring them in the real world, and the infrastructure to do so is still nascent.
> 这项研究带来的一个核心启示是，智能体在实践中表现出的自主性，是由模型、用户和产品共同建构出来的。Claude 会在不确定时停下来提问，以此限制自身独立性。用户则会随着与模型协作而建立信任，并据此调整自己的监督策略。我们在任何部署中观察到的现象，都是这三种力量共同作用的结果，这也是为什么单靠部署前评估无法完整描述它。要理解智能体实际上如何行为，就必须在真实世界中测量它们，而支撑这件事的基础设施仍然处于早期阶段。

Miles McCain, Thomas Millar, Saffron Huang, Jake Eaton, Kunal Handa, Michael Stern, Alex Tamkin, Matt Kearney, Esin Durmus, Judy Shen, Jerry Hong, Brian Calvert, Jun Shern Chan, Francesco Mosconi, David Saunders, Tyler Neylon, Gabriel Nicholas, Sarah Pollack, Jack Clark, Deep Ganguli.
> 作者：Miles McCain、Thomas Millar、Saffron Huang、Jake Eaton、Kunal Handa、Michael Stern、Alex Tamkin、Matt Kearney、Esin Durmus、Judy Shen、Jerry Hong、Brian Calvert、Jun Shern Chan、Francesco Mosconi、David Saunders、Tyler Neylon、Gabriel Nicholas、Sarah Pollack、Jack Clark、Deep Ganguli。

#### Bibtex
> Bibtex 引用

If you’d like to cite this post, you can use the following Bibtex key:
> 如果你想引用这篇文章，可以使用下面这个 Bibtex 条目：

```
@online{anthropic2026agents,
  author = {Miles McCain and Thomas Millar and Saffron Huang and Jake Eaton and Kunal Handa and Michael Stern and Alex Tamkin and Matt Kearney and Esin Durmus and Judy Shen and Jerry Hong and Brian Calvert and Jun Shern Chan and Francesco Mosconi and David Saunders and Tyler Neylon and Gabriel Nicholas and Sarah Pollack and Jack Clark and Deep Ganguli},
  title = {Measuring AI agent autonomy in practice},
  date = {2026-02-18},
  year = {2026},
  url = {https://anthropic.com/research/measuring-agent-autonomy},
}
```

## Appendix
> 附录

We provide more details in the [PDF Appendix](https://cdn.sanity.io/files/4zrzovbb/website/55e4d2de6eb39b3a9259c3f74843f86b1a12e265.pdf) to this post.
> 我们在这篇文章对应的 PDF 附录中提供了更多细节。

1\. Our definition is compatible with [Russell and Norvig (1995)](https://dl.acm.org/doi/book/10.5555/773294), who define an agent as “anything that can be viewed as perceiving its environment through sensors and acting upon that environment through effectors.” Our definition is also compatible with Simon Willison’s, who [writes](https://simonwillison.net/2025/Sep/18/agents/) that an agent is a system that “runs tools in a loop to achieve a goal.”
> 1\. 我们的定义与 Russell 和 Norvig（1995）是一致的，他们将智能体定义为“任何能够被视为通过传感器感知其环境，并通过执行器作用于该环境的东西”。我们的定义也与 Simon Willison 的定义一致，他写道，智能体是一种“通过循环运行工具来实现目标”的系统。

While a full literature review is beyond the scope of this post, we found the following work helpful in framing our thinking. [Kasirzadeh and Gabriel (2025)](https://arxiv.org/pdf/2504.21848) propose a four-dimensional framework for characterizing AI agents along autonomy, efficacy, goal complexity, and generality, constructing “agentic profiles” that map governance challenges across different classes of systems. [Morris et al. (2024)](https://arxiv.org/abs/2311.02462) propose levels of AGI based on performance and generality, treating autonomy as a separable deployment choice. [Feng, McDonald, and Zhang (2025)](https://arxiv.org/abs/2506.12469) define five levels of autonomy based on user roles, from operator to observer. [Shavit et al. (2023)](https://openai.com/index/practices-for-governing-agentic-ai-systems/) propose practices for governing agentic systems, while [Mitchell et al. (2025)](https://arxiv.org/abs/2502.02649) argue that fully autonomous agents should not be developed given that risk scales with autonomy. [Chan et al. (2023)](https://arxiv.org/pdf/2302.10329) argue for anticipating harms from agentic systems before widespread deployment, highlighting risks like reward hacking, power concentration, and the erosion of collective decision-making. [Chan et al. (2024)](https://arxiv.org/pdf/2401.13138) assess how agent identifiers, real-time monitoring, and activity logging could increase visibility into AI agents.
> 虽然完整的文献综述超出了本文范围，但以下工作对我们形成思路很有帮助。Kasirzadeh 和 Gabriel（2025）提出了一个四维框架，从自主性、效能、目标复杂度和通用性四个维度刻画 AI 智能体，并构建出“智能体画像”，以映射不同系统类别中的治理挑战。Morris 等人（2024）基于性能与通用性提出 AGI 的层级，并把自主性视为一种可分离的部署选择。Feng、McDonald 和 Zhang（2025）基于用户角色定义了五个自主性层级，从操作者到观察者。Shavit 等人（2023）提出了治理智能体系统的实践，而 Mitchell 等人（2025）则认为，鉴于风险会随自主性上升，不应开发完全自主的智能体。Chan 等人（2023）主张在大规模部署前就预判智能体系统可能带来的伤害，重点指出了奖励黑客、权力集中以及集体决策侵蚀等风险。Chan 等人（2024）评估了智能体标识、实时监控和活动日志记录如何提升对 AI 智能体的可见性。

On the empirical side, [Kapoor et al. (2024)](https://arxiv.org/abs/2407.01502) critique agent benchmarks for neglecting cost and reproducibility; [Pan et al. (2025)](https://arxiv.org/abs/2512.04123) survey practitioners and find that production agents tend to be simple and human-supervised; [Yang et al. (2025)](https://arxiv.org/abs/2512.07828) analyze Perplexity usage data and find productivity and learning tasks dominate; and [Sarkar (2025)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5713646) finds that experienced developers are more likely to accept agent-generated code. At Anthropic, we’ve also studied how professionals incorporate AI into their work both [internally](https://www.anthropic.com/research/how-ai-is-transforming-work-at-anthropic) and [externally](https://www.anthropic.com/research/anthropic-interviewer). Our work complements these efforts by analyzing deployment patterns using first-party data across both our API and Claude Code, giving us visibility into autonomy, safeguards, and risk that is difficult to observe externally.  
> 在实证研究方面，Kapoor 等人（2024）批评智能体基准忽视了成本和可复现性；Pan 等人（2025）对从业者进行调查，发现生产环境中的智能体往往更简单且有人类监督；Yang 等人（2025）分析了 Perplexity 的使用数据，发现生产力和学习任务占主导；Sarkar（2025）则发现，经验更丰富的开发者更可能接受智能体生成的代码。在 Anthropic，我们也研究了专业人士如何在内部和外部将 AI 融入自己的工作。我们的工作通过分析 API 与 Claude Code 两个渠道的第一方数据中的部署模式，对这些研究形成补充，使我们能够看到那些从外部很难观测到的自主性、防护措施和风险。  

2\. Because we characterize agents as AI systems that use tools, we can analyze individual tool calls as the building blocks of agent behavior. To understand what agents are doing in the world, we study the tools they use and the context of those actions (such as the system prompt and conversation history at the time of the action).
> 2\. 因为我们将智能体刻画为使用工具的 AI 系统，所以可以把单个工具调用视为智能体行为的构建模块。为了理解智能体在现实世界中做什么，我们研究它们所使用的工具以及这些动作发生时的上下文，例如系统提示词和对话历史。

3\. These results reflect Claude’s performance on programming-related tasks, and do not necessarily translate to performance in other domains.
> 3\. 这些结果反映的是 Claude 在与编程相关任务上的表现，并不一定能够迁移到其他领域的表现上。

4\. Throughout this post, we use "autonomy" somewhat informally to refer to the degree to which an agent operates independently of human direction and oversight. An agent with minimal autonomy executes exactly what a human explicitly requests; an agent with high autonomy makes its own decisions about what to do and how to do it, with little or no human involvement. Autonomy is not a fixed property of a model or system but an emergent characteristic of a deployment, shaped by the model's behavior, the user's oversight strategy, and the product's design. We do not attempt a precise formal definition; for details on how we operationalize and measure autonomy in practice, see the [Appendix](https://cdn.sanity.io/files/4zrzovbb/website/55e4d2de6eb39b3a9259c3f74843f86b1a12e265.pdf).
> 4\. 在整篇文章中，我们以相对不那么严格的方式使用 “autonomy” 一词，用来指代智能体在多大程度上脱离人类指挥和监督而独立运行。自主性最低的智能体只会执行人类明确要求的事情；自主性高的智能体则会在很少甚至没有人类参与的情况下，自行决定做什么以及如何去做。自主性不是模型或系统的固定属性，而是部署过程中涌现出来的特征，受到模型行为、用户监督策略以及产品设计的共同塑造。我们不试图给出精确的形式化定义；关于我们如何在实践中操作化并测量自主性的细节，请见附录。

5\. Moreover, the same model deployed differently can generate output at different speeds. For example, we recently released [Fast Mode](https://code.claude.com/docs/en/fast-mode) for Opus 4.6, which generates output 2.5x faster than regular Opus.
> 5\. 此外，同一个模型在不同部署方式下也可能以不同速度生成输出。例如，我们最近为 Opus 4.6 发布了 Fast Mode，它的输出速度比常规 Opus 快 2.5 倍。

6\. For turn duration across other percentiles, see the [Appendix](https://cdn.sanity.io/files/4zrzovbb/website/55e4d2de6eb39b3a9259c3f74843f86b1a12e265.pdf).
> 6\. 关于其他分位数上的单轮持续时间，请参见附录。

7\. Specifically, we use Claude to classify each internal Claude Code session into four categories of complexity, and to determine whether the task was successful. Here, we report the success rate for the most difficult category of task.
> 7\. 具体来说，我们使用 Claude 将每个内部 Claude Code 会话划分到四个复杂度类别中，并判断任务是否成功。这里我们报告的是最困难那一类任务的成功率。

8\. METR’s five-hour figure is a measure of task difficulty (how long the task would take a human), whereas our measurements reflect actual elapsed time, which is affected by factors like model speed and the user’s computing environment. We do not attempt to reason across these metrics, and we include this comparison to explain to readers who may be familiar with the METR finding why the numbers we report here are substantially lower.
> 8\. METR 提出的五小时数字衡量的是任务难度，也就是人类完成该任务需要多久，而我们的测量反映的是实际经过时间，它会受到模型速度和用户计算环境等因素影响。我们并不试图在这些指标之间进行推理比较，我们加入这一对比，是为了向熟悉 METR 结果的读者解释为什么我们这里报告的数字会显著更低。

9\. These patterns come from interactive Claude Code sessions, which overwhelmingly reflect software engineering. Software is unusually amenable to supervisory oversight because the outputs can be tested, easily compared, and reviewed before they are released. In domains where verifying an agent’s output requires the same expertise as producing it, this shift may be slower or take a different form. The rising interrupt rate may also reflect experienced users completing more challenging tasks, which would naturally require more human input. Finally, Claude Code’s default settings push new users towards approval-based oversight (since actions are not auto-approved by default), so some of the shifts we observe may reflect Claude Code’s product design.
> 9\. 这些模式来自交互式 Claude Code 会话，而这类会话在很大程度上反映的是软件工程。软件之所以特别适合被监督，是因为其输出可以在发布前进行测试、轻松比较并审查。在那些验证智能体输出需要与生产输出同等专业知识的领域，这种转变可能会更慢，或者呈现不同形式。打断率上升也可能反映了资深用户在完成更具挑战性的任务，而这些任务天然需要更多人工输入。最后，Claude Code 的默认设置会把新用户推向基于批准的监督方式，因为默认并不会自动批准动作，因此我们观察到的一些变化也可能反映了 Claude Code 的产品设计。

10\. Both complexity and human involvement are estimated by having Claude analyze each tool call in its full context (including the system prompt and conversation history). The complete classification prompt is available in the [Appendix](https://cdn.sanity.io/files/4zrzovbb/website/55e4d2de6eb39b3a9259c3f74843f86b1a12e265.pdf). Defining human involvement is particularly difficult, as many transcripts include content from a human even when that human is not actively steering the conversation (for example, a user message being moderated or analyzed). In our manual validation, Claude was nearly always correct when it classified a tool call as having no human involved, but it sometimes identified human involvement where there was none. As a result, these estimates should be interpreted as an upper bound on human involvement.
> 10\. 复杂度和人类参与度都是通过让 Claude 在完整上下文中分析每一个工具调用来估计的，其中包括系统提示词和对话历史。完整的分类提示词可在附录中查看。定义人类参与尤其困难，因为许多对话记录即使在人类并未主动引导对话时，也会包含来自人类的内容，例如一条被审核或分析的用户消息。在我们的人工验证中，当 Claude 将某个工具调用判断为没有人类参与时，它几乎总是正确的，但有时也会在实际上没有人类参与的地方识别出人类参与。因此，这些估计应被理解为人类参与程度的上界。

11\. In a sense, stopping to ask the user a question is itself a form of agency. We use “limits its own autonomy” to mean that Claude chooses to seek guidance from the human when it could have continued operating independently.
> 11\. 从某种意义上说，停下来向用户提问本身就是一种能动性。我们所说的“限制自身自主性”，是指 Claude 在原本可以继续独立运行的情况下，选择去向人类寻求指导。

12\. These clusters were generated by having Claude analyze each interruption or pause, along with the surrounding session context, then grouping related reasons together. We manually combined some closely related clusters and edited their names for clarity. The clusters shown are not exhaustive.
> 12\. 这些聚类是通过让 Claude 分析每一次打断或暂停及其周围会话上下文后，再把相关原因归并在一起生成的。我们手动合并了一些关系很近的聚类，并为了清晰起见修改了它们的名称。这里展示的聚类并非穷尽全部情况。

13\. We treat these scores as comparative indicators rather than precise measurements. Rather than defining rigid criteria for each level, we rely on Claude’s general judgment about the context surrounding each tool call, which allows the classification to capture considerations we may not have anticipated. The tradeoff is that the scores are more meaningful for comparing actions against each other than for interpreting any single score in absolute terms. For the full prompts, see the [Appendix](https://cdn.sanity.io/files/4zrzovbb/website/55e4d2de6eb39b3a9259c3f74843f86b1a12e265.pdf).
> 13\. 我们把这些分数视为比较性指标，而不是精确测量。我们并没有为每个等级定义僵硬的标准，而是依赖 Claude 对每个工具调用周围上下文的总体判断，这让分类能够覆盖一些我们事先可能没有预料到的考量。这样的权衡是，这些分数在比较不同行动时比在绝对意义上解释任何单一分数时更有意义。完整提示词请见附录。

14\. For more information about how we validated these figures and our precise definitions, see the [Appendix.](https://cdn.sanity.io/files/4zrzovbb/website/55e4d2de6eb39b3a9259c3f74843f86b1a12e265.pdf) In particular, we found that Claude often overestimated human involvement, so we expect 80% to be an upper bound on the number of tool calls with direct human oversight.
> 14\. 关于我们如何验证这些数字以及我们精确定义的更多信息，请参见附录。特别是，我们发现 Claude 往往会高估人类参与程度，因此我们认为 80% 应当被视为存在直接人工监督的工具调用数量上界。

15\. Our systems also automatically exclude clusters that do not meet our aggregation minimums, which means that tasks that only a small number of customers are performing with Claude will not surface in this analysis.
> 15\. 我们的系统还会自动排除那些未达到聚合最小门槛的聚类，这意味着只有极少数客户使用 Claude 执行的任务不会出现在这项分析中。

16\. Whether the adoption curve in software engineering will repeat in other domains is an open question. Software is comparatively easy to test and review—you can run code and see if it works—which makes it easier to trust an agent and catch its mistakes. In domains like law, medicine, or finance, verifying an agent’s output may require significant effort, which could slow the development of trust.
> 16\. 软件工程中的采用曲线是否会在其他领域重现，仍然是一个开放问题。软件相对容易测试和审查，你可以运行代码并查看它是否工作，这使得人们更容易信任智能体，也更容易发现它的错误。而在法律、医疗或金融等领域，验证智能体输出可能需要付出相当大的努力，这可能会放缓信任的形成。
