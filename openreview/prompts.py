"""
所有 Prompt 模板集中管理

设计原则:
- 每个 prompt 都是一个函数，接收结构化参数，返回字符串
- 易于独立调优和测试
- 所有 prompt 明确要求模型以中文回复（可按需修改）
"""
import re


# ============================================================
# Issue 分类 Prompt
# ============================================================

def build_issue_classify_prompt(
    title: str,
    body: str,
    labels: list[str],
    repo_name: str = "",
    local_code_context: str = "",
    triage_context: str = "",
    discussion_context: str = "",
    author_association: str = "",
) -> str:
    """
    判断 Issue 是否属于 Agent 可以自动回答的类型。
    
    可回答类型: bug 报告、使用疑问、错误排查、配置问题
    不可回答类型: 非维护者决策已定的泛泛新特性请求、增强建议、讨论、项目管理相关
    """
    label_str = ", ".join(labels) if labels else "无"
    repo_section = f"\n**仓库**: {repo_name}" if repo_name else ""
    author_section = f"\n**作者关系**: {author_association}" if author_association else ""
    local_code_section = ""
    if local_code_context:
        local_code_section = f"""

## 当前本地仓库上下文

{local_code_context}
"""
    triage_section = ""
    if triage_context:
        triage_section = f"""

## GitHub 侧补充线索

{triage_context}
"""
    discussion_section = ""
    if discussion_context:
        discussion_section = f"""

## 当前讨论

{discussion_context}
"""
    return f"""\
你是一个 GitHub Issue 分类助手。你的任务是判断以下 Issue 是否属于 **可以自动回答** 的类型。

## 分类规则

**可以自动回答 (ANSWERABLE)**：
- Bug 报告、错误日志分析
- 使用方法提问、配置问题
- 已知行为的解释
- 代码报错排查
- 标题看似是 feature / enhancement，但从当前仓库实现来看，这个能力可能已经存在，此时可以给出说明、使用方法、配置方式或相关代码位置
- OWNER/MEMBER/COLLABORATOR 提出的具体功能规格，且内容已经包含目标、实施顺序、验收、重点文件或非目标；这类应输出实施分析，而不是回复等待维护者决策

**不应自动回答 (UNANSWERABLE)**：
- 泛泛的新功能请求 / Feature Request
- 增强建议 / Enhancement
- 架构讨论 / RFC
- 项目管理、发布计划相关
- 需要维护者决策的事项
- 内容不清楚或过于简短无法判断

## 额外判断要求

- 标签只能作为弱信号，**不能仅因 feature / enhancement / question 等标签就直接判定为 UNANSWERABLE**
- 如果结合本地仓库上下文，能够指出“项目里已有类似能力 / 已有配置项 / 已有实现入口”，则优先判为 ANSWERABLE
- 如果 GitHub 侧线索显示它可能是重复 issue、已经修复、已有明确处理结论，也优先判为 ANSWERABLE
- 如果问题本质上是在问“现有项目能不能做到、怎么做到、为什么没生效”，通常应判为 ANSWERABLE
- 如果它明确是在要求新增能力、改产品决策或做路线选择，即使上下文里出现历史 issue / PR，也仍应判为 UNANSWERABLE，除非当前仓库实现已经能直接满足该需求，或可信作者已经给出足够具体的实施规格
- 只有当它明确要求新增能力、路线决策、架构取舍，且无法依据现有代码或可信作者的具体规格回答时，才判为 UNANSWERABLE

## Issue 信息

**标题**: {title}
**标签**: {label_str}
{repo_section}
{author_section}
**内容**:
{body}
{triage_section}
{discussion_section}
{local_code_section}

## 输出要求

请严格按以下 JSON 格式输出，不要包含其他内容:
```json
{{"classification": "ANSWERABLE 或 UNANSWERABLE", "reason": "简短的分类理由", "confidence": 0.0到1.0之间的置信度}}
```

其中：
- `reason` 只写 4 到 12 个中文字符的类型标签，不要写整句解释
- `reason` 示例：`配置排查`、`新功能请求`、`架构讨论`、`信息不足`"""


# ============================================================
# Issue 分析 Prompt
# ============================================================

def build_issue_analysis_prompt(
    title: str,
    body: str,
    labels: list[str],
    repo_name: str,
    history_context: str = "",
    local_code_context: str = "",
    triage_context: str = "",
    discussion_context: str = "",
    is_followup: bool = False,
    author_association: str = "",
) -> str:
    """为可回答的 Issue 生成分析 prompt"""
    label_str = ", ".join(labels) if labels else "无"
    author_section = f"\n**作者关系**: {author_association}" if author_association else ""
    history_section = ""
    if history_context:
        history_section = f"""
## 历史参考（来自已关闭的相似 Issue）

{history_context}
"""
    local_code_section = ""
    if local_code_context:
        local_code_section = f"""
## 当前本地仓库上下文

{local_code_context}
"""
    triage_section = ""
    if triage_context:
        triage_section = f"""
## GitHub 侧补充线索

{triage_context}
"""
    discussion_section = ""
    if discussion_context:
        if is_followup:
            discussion_section = f"""
## ⚠️ 跟进讨论（你必须针对最新一条外部评论回应）

以下是该 Issue 的讨论记录。**你只需要回应最新一条非 bot 评论提出的问题或反馈**，不要重复之前已经回答过的内容。如果最新评论补充了新信息或追问了新问题，直接回应该信息。

{discussion_context}
"""
        else:
            discussion_section = f"""
## 当前讨论

{discussion_context}
"""
    return f"""\
你是项目 **{repo_name}** 的资深开发者助手。你现在要写的是一条 **直接发布到 GitHub Issue 下的最终回复**。

## 回复要求
1. 直接回答提问者的问题，只输出最终评论正文，不要写你的分析过程
2. **绝对不要**输出任何检索轨迹、工具调用、命令执行记录或文件查看过程，例如 `● Grep ...`、`Read ...`、`Explore ...`、`Now I have ...`
3. **绝对不要**输出“是否合理”“是否是 issue”“是否好解决”“建议动作”这类内部判定字段
4. 不要使用“问题理解”“结合仓库的结论”“最值得优先排查的点”“建议怎么处理这个 issue”之类的小节标题
5. 开头先用 1 到 2 句话直接给出结论或最关键答案
6. 然后给出最多 5 条可执行的分析 / 排查 / 配置 / 实施建议，使用简短 `-` 列表
7. 如果需要指出位置，只能写仓库内的文件名、环境变量名、命令名，例如 `config.py`、`PRIMARY_MODEL`、`.env.example`
8. **不要**生成 `/home/...` 这类本地绝对路径，不要生成 markdown 文件链接
9. 如需给命令示例，优先使用行内代码；除非没有代码块无法表达，否则不要输出 shell 代码块或命令回显
10. 语气友好专业，使用中文回复
11. **绝对不要**输出完整修复 PR、内部思考、维护者视角总结或流程性结论
12. 优先基于给出的本地仓库上下文分析，避免脱离当前代码实现空泛回答
13. 如果 GitHub 侧线索显示它很可能已经修复、与已有 issue 重复、或已经可以关闭，要直接说清楚，并简洁说明为什么可以考虑关闭
14. 如果存在后续评论，优先回应最新用户补充，不要重复首条回复已经讲过的大段内容
15. 如果最新外部评论已经明确说“成功了 / 已解决 / 可以了 / resolved / fixed”，回复应收敛为“建议关闭或标记已解决”，不要继续追加排查项
16. 对 bug、桌面 UI 黑屏/白屏、崩溃、无法保存、数据缺失、计算错误这类缺陷定位，不能只要求用户补日志；必须优先基于本地上下文给出候选文件/模块、最可能的 1 到 3 个根因假设，以及下一步最小验证动作
17. 如果本地上下文没有足够候选位置，要明确写“当前上下文还不能定位到具体文件”，但仍要说明需要哪一类日志或复现信息，不要泛泛要求“提供更多信息”
18. 优先按以下四种模式之一组织回复：
   - `重复 issue`：明确写出旧 issue 编号、标题、状态，并建议把当前 issue 关闭或并入已有讨论
   - `已修复`：明确写出相关 PR 编号；如果上下文给了 commit id，也一并写出短 SHA，说明问题大概率已修复
   - `使用/配置问题`：优先引用仓库文档、配置文件、环境变量、命令名和历史 issue / PR 线索
   - `新特性/无法直接回答`：不要假装已有结论，这类只做简短说明并保持克制
   - `功能实施分析`：如果 issue 是 OWNER/MEMBER/COLLABORATOR 提出的具体功能规格，且已经包含目标、实施顺序、验收、重点文件或非目标，不要回复“等待维护者决策”或“AutoCode 会继续”；应直接基于仓库上下文给出实施拆解、复用入口、主要风险和最小验证建议
19. 如果要建议关闭 issue，必须同时给出证据编号（旧 issue / PR / commit），不要只给模糊判断；但用户本人最新确认已解决时，可以直接建议关闭当前 issue

## 输出结构
- 第 1 段：直接结论，1 到 2 句话
- 第 2 段：如需要，再给“建议排查”的 `-` 列表
- 除这两部分外，不要追加其他总结、评分、流程判断或结尾说明

## Issue 信息

**标题**: {title}
**标签**: {label_str}
{author_section}
**内容**:
{body}
{history_section}
{triage_section}
{discussion_section}
{local_code_section}
## 请直接输出最终评论正文"""


# ============================================================
# PR Review Prompt
# ============================================================

def build_pr_review_prompt(
    title: str,
    body: str,
    repo_name: str,
    diff_excerpt: str,
    diff_stat: str = "",
    changed_files: list[str] | None = None,
    review_facts: list[str] | None = None,
    local_code_context: str = "",
    discussion_context: str = "",
) -> str:
    """为 PR 的本地 diff 摘要生成 review prompt。"""
    files_str = "\n".join(f"- {f}" for f in changed_files) if changed_files else "见 diff"
    facts_str = "\n".join(f"- {fact}" for fact in review_facts) if review_facts else "- 无"
    diff_stat_section = ""
    if diff_stat:
        diff_stat_section = f"""
**本地 diff 统计**:
```text
{diff_stat}
```
"""

    local_code_section = ""
    if local_code_context:
        local_code_section = f"""
## 当前本地仓库快照

{local_code_context}
"""
    discussion_section = ""
    if discussion_context:
        discussion_section = f"""
**当前讨论**:
{discussion_context}
"""

    return f"""\
你是项目 **{repo_name}** 的 Code Reviewer。你现在要写的是一条 **直接发布到 GitHub PR review 的最终评论**。

## Review 要求
1. 必须结合给出的仓库 `AGENTS.md` 规范审查
2. 顶层结论仍按：必要性、是否有关联 issue、PR 类型、description 完整性、是否可直接合入
3. 优先基于本地已检出的 PR 最新 head、完整改动文件列表和本地 merge-base diff 审查；不要只根据 PR 描述、零散讨论或单个 diff 片段下结论
4. 识别问题时，先看当前 diff 新引入的真实风险，再看模板、描述或 checklist 缺项
5. `主要问题` 默认按以下优先级排序：行为/正确性风险 > 兼容性/迁移风险 > 验证缺口 > 文档/描述/模板缺口
6. 判断“缺文档”前，先结合本地上下文中的 `README.md`、`docs/*`、`.env*`、配置文件快照；如果只是新增默认行为/新增语义未说明，要明确写成“已有通用文档，但本次新增语义未说明”，不要直接夸大成“完全无文档”
7. 如果当前仓库里已经存在相似模式或旧债，不要暗示这是第一次出现；应写成“本 PR 延续/扩大了已有模式”
8. `本地 git 重点 diff 片段` 只是摘要，不能替代完整改动文件列表和本地文件快照；如果片段不足以支持结论，要明确收窄判断
9. 对 `docs`、`README`、`CHANGELOG`、图片资源、示例注释、docstring-only 这类低风险文档/注释类 PR，缺少回滚方案、未更新 README 的落点说明、或未提供可复现命令，不能单独作为合入阻断；最多列为建议项。只有真实文档链接错误、错误示例、范围漂移、冲突、CI 失败或会误导用户的内容错误才可阻断
10. 如果仓库规范要求回滚方案，文档/注释类 PR 只需要“revert 本 PR”级别的最小回滚说明；不要要求复杂回滚预案，也不要仅因未写复杂回滚方案判为不可合入
11. 如果结构化事实提示“外部模型/API 兼容风险”或代码 diff 改动了模型名、供应商 provider、Base URL、SDK/依赖默认值、废弃日期，必须按兼容性/迁移风险审查：检查是否有官方来源链接、当前锁定依赖/运行时兼容验证、旧配置迁移/回退路径。缺少任一关键证据时，不能写“低风险”或“可直接合入”，至少列为待澄清/验证缺口
12. 如果结构化事实提示“文档型外部模型/API 提示”，先判断当前 docs-only diff 是否新增/改变用户可执行的 provider/model/Base URL/默认值/废弃日期声明；如果只是搬运、恢复或同步既有仓库文档，且 PR 描述明确无运行时变更、无新增外部兼容性语义并给出仓库内依据，不要仅因缺少外部官方来源判为不可合入，最多列为建议项
13. 如果改动会自动清理、过滤、迁移或重置用户运行时配置（例如主模型、fallback、Vision、API Base URL），必须检查是否有用户可见提示、恢复方式和回归测试；静默清空或静默迁移应列为主要问题
14. 如果有问题，补充“主要问题”列表，优先指出文件路径、证据摘要和风险；只有在确实需要分级时，才使用 `[Correctness blocker]`、`[Process blocker]` 或 `[Nice to have]`
15. 如果证据只来自局部 diff 或片段上下文，可以写“置信度：高/中/低”；不要虚构未看到的行号、历史事实或仓库现状
16. 如果当前 CI 未完成、CI 失败，或 PR 暂时有冲突，仍要基于当前代码给出 review；这些状态应体现在“是否可直接合入”里
17. 本仓库启用了 `Only allow users with bypass permission to update matching refs`；owner / 具备 bypass permission 的维护者仍可合入，所以如果结构化事实里的 merge 状态为 `mergeable_state=blocked`，不要仅凭这一点就把“是否可直接合入”判为不可
18. 只有明确存在冲突，或你能从其它结构化事实/代码风险给出阻断理由时，才写“不可”
19. 使用中文回复
20. `改动文件` 是完整列表；`结构化事实` 是程序检测结果；如果它们与自然语言描述冲突，优先相信结构化事实
21. **绝对不要**输出思考过程、内部分析大纲、绝对路径、markdown 本地文件链接
22. **绝对不要**生成新的 commit 或代码提交建议，只做 review 评论

## 输出格式
请严格按下面结构输出：

`必要性`：通过/不通过 + 一句话理由
`是否有对应 issue`：有/无（编号或说明）
`PR 类型`：fix/feat/refactor/docs/chore/test + 理由
`description 完整性`：完整/不完整 + 缺失项
`是否可直接合入`：可/不可 + 必改项或阻断点

如果存在需要指出的问题，再追加一个标题：
`主要问题`
- 每条一个独立问题，优先写清文件、证据摘要、风险，必要时再补建议动作

如果没有明显问题，再追加一行：
`主要问题`：未发现阻断性问题

## PR 信息

**标题**: {title}
**描述**: {body or "无"}
**改动文件（完整列表）**:
{files_str}
**结构化事实（以程序检测结果为准）**:
{facts_str}
{discussion_section}
{diff_stat_section}
{local_code_section}

## 本地 git 重点 diff 片段

```diff
{diff_excerpt or "(无可展示 diff 片段)"}
```

## 请开始 Review"""


def build_pr_followup_prompt(
    title: str,
    body: str,
    repo_name: str,
    discussion_context: str,
    diff_excerpt: str,
    diff_stat: str = "",
    changed_files: list[str] | None = None,
    review_facts: list[str] | None = None,
    local_code_context: str = "",
) -> str:
    """为 PR 后续讨论生成 follow-up review prompt。"""
    files_str = "\n".join(f"- {f}" for f in changed_files) if changed_files else "见 diff"
    facts_str = "\n".join(f"- {fact}" for fact in review_facts) if review_facts else "- 无"
    diff_stat_section = ""
    if diff_stat:
        diff_stat_section = f"""
**本地 diff 统计**:
```text
{diff_stat}
```
"""
    local_code_section = ""
    if local_code_context:
        local_code_section = f"""
## 当前本地仓库快照

{local_code_context}
"""

    return f"""\
你是项目 **{repo_name}** 的 Code Reviewer。你现在要写的是一条 **针对 PR 后续讨论的继续评审评论**。

## Follow-up Review 要求
1. 结合当前代码、完整改动文件、结构化事实和讨论上下文，判断最新用户回复是否成立
2. 如果用户说得对，要明确承认并收缩之前的结论，不要为了坚持原判断而硬拗
3. 继续优先基于本地已检出的 PR 最新 head、完整改动文件列表和本地文件快照判断，不要只围绕单条回复或局部片段争论
4. 如果问题仍然存在，要明确指出为什么仍未解决、风险落点在哪里
5. 如果最新讨论证明现有文档已覆盖通用能力，或问题其实是旧模式延续，要主动收窄表述，例如改成“新增语义未说明”或“延续已有模式”，不要维持过重结论
6. 不要重复一整份首轮 review，只聚焦“最新讨论引入的争议点/补充信息”
7. `说明` 里的每条结论优先给出文件路径、证据摘要和剩余风险；不要虚构未看到的行号或事实
8. 如果当前 CI 未完成、CI 失败，或 PR 暂时有冲突，也继续基于当前代码和讨论做 follow-up review，并在结论里反映这些状态
9. 本仓库启用了 `Only allow users with bypass permission to update matching refs`；owner / 具备 bypass permission 的维护者仍可合入，所以如果结构化事实里的 merge 状态为 `mergeable_state=blocked`，不要仅凭这一点就把“是否仍有阻断”判为有
10. 当最新讨论质疑“简单文档/注释修改是否需要复杂回滚方案”时，如果当前 diff 确实是 docs、截图、示例注释或 docstring-only 的低风险变更，应接受这个质疑：缺少回滚方案只能作为建议补充，不能单独判为仍有阻断
11. 如果结构化事实提示“外部模型/API 兼容风险”，或当前改动不是文档型且代码 diff/讨论涉及模型名、provider、Base URL、SDK/依赖默认值、废弃日期，必须复核官方来源、当前依赖/运行时兼容、旧配置迁移和用户可见提示；缺少关键证据时不要维持“无阻断/可合入”的过宽结论
12. 如果结构化事实提示“文档型外部模型/API 提示”，要先接受已补充的 docs-only 范围说明和仓库内依据；只有最新 diff 仍新增/改变外部兼容性事实、默认推荐、废弃时间或用户可执行配置语义且缺少来源时，才继续判定为阻断
13. 如果改动会自动清理、过滤、迁移或重置用户运行时配置，必须检查是否有用户提示、恢复方式和测试覆盖；静默清空或静默迁移可以构成仍有阻断
14. 只有明确存在冲突、CI 失败、真实文档错误、范围漂移、代码行为风险、外部模型/API 兼容验证缺口，或你能从其它结构化事实/代码风险给出阻断理由时，才写“有”
15. 使用中文回复
16. **绝对不要**输出思考过程、内部分析大纲、绝对路径、markdown 本地文件链接

## 输出格式
请严格按下面结构输出：

`结论`：接受/部分接受/不接受 + 一句话理由
`是否仍有阻断`：有/无 + 一句话理由

如果需要补充，再追加一个标题：
`说明`
- 每条一个独立说明，优先指出事实、代码位置、遗漏点或可关闭的判断依据

## PR 信息

**标题**: {title}
**描述**: {body or "无"}
**改动文件（完整列表）**:
{files_str}
**结构化事实（以程序检测结果为准）**:
{facts_str}
**当前讨论**:
{discussion_context or "无"}
{diff_stat_section}
{local_code_section}

## 本地 git 重点 diff 片段

```diff
{diff_excerpt or "(无可展示 diff 片段)"}
```

## 请直接输出继续评审评论正文"""


# ============================================================
# 不可回答 Issue 的礼貌回复模板（非 LLM 生成）
# ============================================================

UNANSWERABLE_REPLY_TEMPLATE = """\
👋 感谢提交此 Issue！

经过初步分析，这个 Issue 看起来属于 **{reason}** 类型，需要项目维护者进一步评估和决策。\
我是自动助手，暂时无法对此类问题给出合适的答复。

维护者会尽快查看，感谢你的耐心等待！🙏

---
*🤖 此回复由 OpenReview Bot 自动生成*\
"""

AUTOCODE_HANDOFF_REPLY_TEMPLATE = """\
👋 感谢提交此 Issue！

经过初步分析，这个 Issue 看起来属于 **{reason}** 类型。OpenReview 不会在这里替维护者拍板需求优先级，但这不会阻止后续自动实现流程。

如果仓库已启用 AutoCode 且该 issue 符合自动执行策略，AutoCode 会继续按队列生成实现计划、提交 PR 或在受限路径/风险过高时说明阻塞原因。OpenReview 后续会继续审查关联 PR。

---
*🤖 此回复由 OpenReview Bot 自动生成*\
"""


def build_unanswerable_reply(reason: str) -> str:
    """为不可自动回答的 Issue 生成礼貌的占位回复"""
    summarized = _summarize_unanswerable_reason(reason)
    if summarized == "新功能请求":
        return AUTOCODE_HANDOFF_REPLY_TEMPLATE.format(reason=summarized)
    return UNANSWERABLE_REPLY_TEMPLATE.format(reason=summarized)


# ============================================================
# 回复包装器（给所有 AI 回复添加 footer）
# ============================================================

REPLY_FOOTER = "\n\n---\n*🤖 此回复由 OpenReview Bot 自动生成，仅供参考。如有疑问请 @维护者。*"

_LOCAL_FILE_LINK_RE = re.compile(r"\[([^\]]+)\]\((/[^)]+)\)")
_TRACE_LINE_RE = re.compile(
    r"^\s*(?:●|└|\$|Now I have\b|Here's my reply\b|Let me compose\b|Explore:|Read\b|Grep\b|List\b|Check\b|head\b|tail\b)"
)
_SEPARATOR_LINE_RE = re.compile(r"^\s*(?:---+|\*\*\*+)\s*$")
_BOT_FOOTER_LINE_RE = re.compile(r"OpenReview Bot 自动生成")
_META_HEADINGS = {
    "问题理解",
    "结合仓库的结论",
    "最值得优先排查的点",
    "建议怎么处理这个 issue",
}
_ISSUE_TRIAGE_LABELS = (
    "是否合理",
    "是否是 issue",
    "是否好解决",
    "建议动作",
)
_ISSUE_META_LABELS = {
    "结论",
    "排查建议",
    "建议排查",
    "补充说明",
    "请按以下步骤排查",
}
_REVIEW_SUMMARY_LABELS = (
    "必要性",
    "是否有对应 issue",
    "PR 类型",
    "description 完整性",
    "是否可直接合入",
)
_FOLLOWUP_SUMMARY_LABELS = (
    "结论",
    "是否仍有阻断",
)


def _collapse_blank_lines(lines: list[str]) -> str:
    collapsed: list[str] = []
    for line in lines:
        if not line.strip():
            if collapsed and collapsed[-1] != "":
                collapsed.append("")
            continue
        collapsed.append(line.rstrip())
    while collapsed and collapsed[-1] == "":
        collapsed.pop()
    return "\n".join(collapsed)


def _summarize_unanswerable_reason(reason: str) -> str:
    text = re.sub(r"\s+", " ", reason).strip().strip("。；;")
    if not text:
        return "需要维护者决策的问题"
    if len(text) <= 12 and not re.search(r"[，。；;:：,.!?！？（）()]", text):
        return text

    lower = text.lower()
    if any(keyword in lower for keyword in ("feature", "enhancement", "新功能", "功能请求", "增强建议")):
        return "新功能请求"
    if any(keyword in lower for keyword in ("rfc", "架构", "路线", "方向", "设计讨论")):
        return "架构讨论"
    if any(keyword in lower for keyword in ("项目管理", "发布计划", "roadmap", "排期")):
        return "项目管理"
    if any(keyword in lower for keyword in ("无法判断", "信息不足", "过于简短", "不清楚")):
        return "信息不足"
    return "需要维护者决策的问题"


def sanitize_public_reply(content: str) -> str:
    """清理不适合直接发到 GitHub 的本地链接、工具轨迹和内部分析标题。"""
    sanitized = _LOCAL_FILE_LINK_RE.sub(lambda match: f"`{match.group(1)}`", content)
    lines: list[str] = []
    for line in sanitized.splitlines():
        stripped = line.strip()
        normalized = stripped.rstrip("：:")
        if _TRACE_LINE_RE.match(stripped) or _SEPARATOR_LINE_RE.match(stripped) or _BOT_FOOTER_LINE_RE.search(stripped):
            continue
        if normalized in _META_HEADINGS:
            continue
        lines.append(line)
    return _collapse_blank_lines(lines).strip()


def format_issue_reply(content: str) -> str:
    """将 issue 回复整理为适合公开发布的最终评论。"""
    sanitized = sanitize_public_reply(content)
    if not sanitized:
        return ""

    filtered_lines: list[str] = []
    for line in sanitized.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(f"{label}：") or stripped.startswith(f"{label}:") for label in _ISSUE_TRIAGE_LABELS):
            continue
        if stripped.rstrip("：:") in _ISSUE_META_LABELS:
            continue
        filtered_lines.append(line.rstrip())

    intro_lines: list[str] = []
    bullet_lines: list[str] = []
    tail_lines: list[str] = []
    section = "intro"
    in_code_block = False

    for line in filtered_lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            if section == "tail":
                tail_lines.append(line.rstrip())
            elif section == "intro":
                intro_lines.append(line.rstrip())
            else:
                section = "tail"
                tail_lines.append(line.rstrip())
            continue

        is_list_item = (
            not in_code_block and
            (bool(re.match(r"^[-*]\s+", stripped)) or bool(re.match(r"^\d+\.\s+", stripped)))
        )
        if is_list_item:
            section = "bullets"
            bullet_lines.append(_normalize_problem_line(stripped))
            continue

        if section == "bullets" and stripped:
            section = "tail"

        if section == "intro":
            intro_lines.append(line.rstrip())
        elif section == "tail":
            tail_lines.append(line.rstrip())

    intro_text = _collapse_blank_lines(intro_lines).strip()
    tail_text = _collapse_blank_lines(tail_lines).strip()
    sections: list[str] = []

    if intro_text:
        sections.append("**结论**")
        sections.append(intro_text)

    if bullet_lines:
        if sections:
            sections.append("")
        sections.append("**建议排查**")
        for item in bullet_lines:
            sections.append(f"- {item}")

    if tail_text:
        if sections:
            sections.append("")
        sections.append("**补充说明**")
        sections.append(tail_text)

    return "\n".join(sections).strip() if sections else _collapse_blank_lines(filtered_lines).strip()


def _normalize_review_label(text: str) -> str:
    return text.strip().strip("`").strip().rstrip("：:")


def _extract_review_summary_line(line: str, allowed_labels: tuple[str, ...] = _REVIEW_SUMMARY_LABELS) -> tuple[str, str] | None:
    for separator in ("：", ":"):
        if separator not in line:
            continue
        left, right = line.split(separator, 1)
        label = _normalize_review_label(left)
        if label in allowed_labels:
            return label, right.strip()
    return None


def _normalize_problem_line(line: str) -> str:
    normalized = re.sub(r"^[-*]\s+", "", line.strip())
    normalized = re.sub(r"^\d+\.\s+", "", normalized)
    return normalized.strip()


def format_pr_review_reply(content: str) -> str:
    """将模型输出整理为固定的 PR review Markdown 结构。"""
    sanitized = sanitize_public_reply(content)
    if not sanitized:
        return ""

    summary: dict[str, str] = {}
    problem_blocks: list[str] = []
    in_problems = False

    for raw_line in sanitized.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        summary_line = _extract_review_summary_line(line)
        if summary_line and not in_problems:
            label, value = summary_line
            summary[label] = value
            continue

        normalized = _normalize_review_label(line)
        if normalized == "主要问题":
            in_problems = True
            continue

        if line.startswith("主要问题：") or line.startswith("主要问题:"):
            in_problems = True
            inline_problem = line.split("：", 1)[1] if "：" in line else line.split(":", 1)[1]
            inline_problem = inline_problem.strip()
            if inline_problem:
                problem_blocks.append(inline_problem)
            continue

        if in_problems:
            normalized_problem = _normalize_problem_line(line)
            if normalized_problem:
                problem_blocks.append(normalized_problem)

    sections: list[str] = []
    if summary:
        sections.append("**评审结论**")
        for label in _REVIEW_SUMMARY_LABELS:
            value = summary.get(label)
            if value:
                sections.append(f"- **{label}**：{value}")

    if problem_blocks:
        sections.append("")
        sections.append("**主要问题**")
        if len(problem_blocks) == 1 and ("未发现阻断性问题" in problem_blocks[0] or problem_blocks[0] == "无"):
            sections.append(f"- {problem_blocks[0]}")
        else:
            for index, problem in enumerate(problem_blocks, start=1):
                sections.append(f"{index}. {problem}")

    if not sections:
        return sanitized

    return "\n".join(sections).strip()


def format_followup_reply(content: str) -> str:
    """将 follow-up review 整理为固定的 Markdown 结构。"""
    sanitized = sanitize_public_reply(content)
    if not sanitized:
        return ""

    summary: dict[str, str] = {}
    explanation_blocks: list[str] = []
    in_explanations = False

    for raw_line in sanitized.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        summary_line = _extract_review_summary_line(line, _FOLLOWUP_SUMMARY_LABELS)
        if summary_line and not in_explanations:
            label, value = summary_line
            summary[label] = value
            continue

        normalized = _normalize_review_label(line)
        if normalized == "说明":
            in_explanations = True
            continue

        if line.startswith("说明：") or line.startswith("说明:"):
            in_explanations = True
            inline_explanation = line.split("：", 1)[1] if "：" in line else line.split(":", 1)[1]
            inline_explanation = inline_explanation.strip()
            if inline_explanation:
                explanation_blocks.append(inline_explanation)
            continue

        if in_explanations:
            normalized_explanation = _normalize_problem_line(line)
            if normalized_explanation:
                explanation_blocks.append(normalized_explanation)

    sections: list[str] = []
    if summary:
        sections.append("**跟进结论**")
        for label in _FOLLOWUP_SUMMARY_LABELS:
            value = summary.get(label)
            if value:
                sections.append(f"- **{label}**：{value}")

    if explanation_blocks:
        if sections:
            sections.append("")
        sections.append("**说明**")
        for index, explanation in enumerate(explanation_blocks, start=1):
            sections.append(f"{index}. {explanation}")

    if not sections:
        return sanitized

    return "\n".join(sections).strip()


def wrap_reply(content: str) -> str:
    """给 AI 生成的回复添加统一的 footer 标识"""
    formatted = format_issue_reply(content)
    return formatted + REPLY_FOOTER if formatted else REPLY_FOOTER.strip()


def wrap_pr_review_reply(content: str) -> str:
    """给 PR review 添加格式化结构和统一 footer。"""
    formatted = format_pr_review_reply(content)
    return formatted + REPLY_FOOTER if formatted else REPLY_FOOTER.strip()


def wrap_followup_reply(content: str) -> str:
    """给跟进评论添加结构化格式和统一 footer。"""
    formatted = format_followup_reply(content)
    return formatted + REPLY_FOOTER if formatted else REPLY_FOOTER.strip()
