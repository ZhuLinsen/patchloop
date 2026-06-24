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

def build_issue_comment_intent_prompt(
    *,
    body: str,
    issue_title: str = "",
    repo_name: str = "",
) -> str:
    """判断 issue 评论是否是在要求 AutoCode 写代码并提交 PR。"""
    repo_section = f"\n**仓库**: {repo_name}" if repo_name else ""
    title_section = f"\n**Issue 标题**: {issue_title}" if issue_title else ""
    return f"""\
你是一个 GitHub Issue 评论意图分类器。你的任务是判断下面这条评论是否是在明确要求 AutoCode 进入写代码、修复、继续处理或提交 PR 的流程。

## 判定为 IMPLEMENT 的情况

- 明确要求实现、修复、改代码、处理问题、继续修、重新修、开 PR 或提交 PR
- 对已有分析、方案、方向表达执行批准，例如“按这个方案做”“可以直接修”“进行修复”“开始修复”
- 反馈当前问题仍未解决，并要求继续尝试，例如“还是不行，继续修”
- 英文同义表达，例如 “please fix it”“go ahead”“open a PR”“ship it”

## 判定为 IGNORE 的情况

- 只是感谢、确认收到、补充上下文、普通讨论或提问
- 明确否定执行，例如“不用实现”“先别修”“无需 PR”“已实现”“已经修复”
- 只是说明现状“还未实现”“需要实现”，但没有要求 AutoCode 现在执行
- 明确说不需要代码 PR、改配置即可、人工处理即可
- 自动机器人回复、模板性分析、审查意见摘要

## 重要要求

- 关注评论的真实语义，不要只按关键词判断。
- 如果评论同时包含直接执行命令和附加要求，例如“实现，记得补测试”，应判为 IMPLEMENT。
- 如果语义不确定，判为 IGNORE。

## 评论信息
{repo_section}{title_section}

**评论正文**:
{body or "无"}

## 输出要求

严格输出 JSON，不要包含其他文字:
```json
{{"intent": "IMPLEMENT 或 IGNORE", "reason": "简短中文理由", "confidence": 0.0到1.0之间的数字}}
```"""


def build_issue_classify_prompt(
    title: str,
    body: str,
    labels: list[str],
    repo_name: str = "",
    local_code_context: str = "",
    triage_context: str = "",
    discussion_context: str = "",
) -> str:
    """
    判断 Issue 是否属于 Agent 可以自动回答的类型。
    
    可回答类型: bug 报告、使用疑问、错误排查、配置问题
    不可回答类型: 新特性请求、增强建议、讨论、项目管理相关
    """
    label_str = ", ".join(labels) if labels else "无"
    repo_section = f"\n**仓库**: {repo_name}" if repo_name else ""
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

**不应自动回答 (UNANSWERABLE)**：
- 新功能请求 / Feature Request
- 增强建议 / Enhancement
- 架构讨论 / RFC
- 项目管理、发布计划相关
- 需要维护者决策的事项
- 内容不清楚或过于简短无法判断

## 额外判断要求

- 标签只能作为弱信号，**不能仅因 feature / enhancement / question 等标签就直接判定为 UNANSWERABLE**
- 如果结合本地仓库上下文，能够指出“项目里已有类似能力 / 已有配置项 / 已有实现入口”，则优先判为 ANSWERABLE
- 如果 GitHub 侧线索显示它更像配置/参数/工作流设置问题，或者讨论里已经给出更小修复路径，也优先判为 ANSWERABLE
- 如果问题本质上是在问“现有项目能不能做到、怎么做到、为什么没生效”，通常应判为 ANSWERABLE
- 如果当前讨论已经明确说明“无需改代码”“改配置即可”“调整 workflow / timeout / env 参数即可”，不要再把它判成需要新 PR 的实现型问题
- 只有当它明确要求新增能力、路线决策、架构取舍，且无法依据现有代码回答时，才判为 UNANSWERABLE

## Issue 信息

**标题**: {title}
**标签**: {label_str}
{repo_section}
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
) -> str:
    """为可回答的 Issue 生成分析 prompt"""
    label_str = ", ".join(labels) if labels else "无"
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
## ⚠️ 跟进讨论（优先回应最新一条有效评论）

以下是该 Issue 的讨论记录。**请优先回应最新一条有效评论中的新信息或明确结论**，不要重复前面已经说过的大段内容。

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
6. 然后给出最多 5 条可执行的排查 / 配置建议，使用简短 `-` 列表
7. 如果需要指出位置，只能写仓库内的文件名、环境变量名、命令名，例如 `config.py`、`PRIMARY_MODEL`、`.env.example`
8. **不要**生成 `/home/...` 这类本地绝对路径，不要生成 markdown 文件链接
9. 如需给命令示例，优先使用行内代码；除非没有代码块无法表达，否则不要输出 shell 代码块或命令回显
10. 语气友好专业，使用中文回复
11. **绝对不要**输出完整修复 PR、内部思考、维护者视角总结或流程性结论
12. 优先基于给出的本地仓库上下文分析，避免脱离当前代码实现空泛回答
13. 如果 GitHub 侧线索或讨论已经明确指出“改配置即可”“无需改代码”“当前 PR 路径不对”，要直接说明不建议继续生成代码 PR，并给出更小处理路径
14. 如果存在后续评论，优先回应最新评论里的有效结论，不要忽略维护者已经给出的明确处理方向
15. 对于配置/工作流/参数问题，优先给出最小处理路径，不要扩展成大范围代码改造建议

## 输出结构
- 第 1 段：直接结论，1 到 2 句话
- 第 2 段：如需要，再给“建议排查”的 `-` 列表
- 除这两部分外，不要追加其他总结、评分、流程判断或结尾说明

## Issue 信息

**标题**: {title}
**标签**: {label_str}
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

    return f"""\
你是项目 **{repo_name}** 的 Code Reviewer。你现在要写的是一条 **直接发布到 GitHub PR review 的最终评论**。

## Review 要求
1. 必须结合给出的仓库 `AGENTS.md` 规范审查
2. 优先判断：必要性、是否有关联 issue、PR 类型、description 完整性、是否可直接合入
3. 如果有问题，补充“主要问题”列表，尽量指出文件路径和风险
4. 使用中文回复
5. `改动文件` 是完整列表；`结构化事实` 是程序检测结果；如果它们与自然语言描述冲突，优先相信结构化事实
6. **绝对不要**输出思考过程、内部分析大纲、绝对路径、markdown 本地文件链接
7. **绝对不要**生成新的 commit 或代码提交建议，只做 review 评论

## 输出格式
请严格按下面结构输出：

`必要性`：通过/不通过 + 一句话理由
`是否有对应 issue`：有/无（编号或说明）
`PR 类型`：fix/feat/refactor/docs/chore/test + 理由
`description 完整性`：完整/不完整 + 缺失项
`是否可直接合入`：可/不可 + 必改项或阻断点

如果存在需要指出的问题，再追加一个标题：
`主要问题`
- 每条一个独立问题，尽量写清文件、风险、建议动作

如果没有明显问题，再追加一行：
`主要问题`：未发现阻断性问题

## PR 信息

**标题**: {title}
**描述**: {body or "无"}
**改动文件（完整列表）**:
{files_str}
**结构化事实（以程序检测结果为准）**:
{facts_str}
{diff_stat_section}
{local_code_section}

## 本地 git 重点 diff 片段

```diff
{diff_excerpt or "(无可展示 diff 片段)"}
```

## 请开始 Review"""


# ============================================================
# AutoCode Prompt
# ============================================================

def build_execution_triage_prompt(
    title: str,
    body: str,
    labels: list[str],
    repo_name: str,
    local_code_context: str = "",
) -> str:
    """判断一个 Issue 是否适合 AutoCode 执行。"""
    label_str = ", ".join(labels) if labels else "无"
    local_code_section = ""
    if local_code_context:
        local_code_section = f"""
## 当前本地仓库上下文

{local_code_context}
"""
    return f"""\
你是项目 **{repo_name}** 的 AutoCode 任务分诊器。请判断以下 Issue 是否适合进入自动编码流程。

## 任务类型
- `bug_fix`: 缺陷修复、回归修复、边界条件修复；即使涉及权限、安全、依赖、部署、CI，只要本质是纠正已有错误且范围清晰，仍优先归为 `bug_fix`
- `todo_refactor`: 小范围重构、补测试、补文档、清理 TODO
- `small_feature`: 范围清晰、验收标准明确的新功能
- `high_risk_feature`: 架构改造、迁移、权限模型调整、依赖升级、部署/CI 流程改造、密钥/安全机制重设计，或上述高敏感领域里的大范围变更

## 自动化策略
- 明显 bug 可判为 `auto_fix`
- 小范围 TODO/refactor 可判为 `auto_fix`
- 高敏感领域里的“明确缺陷修复”不必一律判成 `high_risk_feature`；可以判为 `bug_fix`，同时把 `risk_level` 提高为 `high`
- feature 默认更保守；只有边界清晰、风险低、验收标准明确时才允许 `implement`
- 如果内容不清楚、需要维护者做方案决策，输出 `plan_only`
- 如果任务不应该由机器人执行，输出 `reject`

## Issue 信息

**标题**: {title}
**标签**: {label_str}
**内容**:
{body or "无"}
{local_code_section}

## 输出要求

严格输出 JSON，不要包含其他文字:
```json
{{
  "task_type": "bug_fix|todo_refactor|small_feature|high_risk_feature",
  "action": "auto_fix|implement|plan_only|reject",
  "risk_level": "low|medium|high",
  "reason": "一句中文理由",
  "confidence": 0.0
}}
```"""


def build_execution_plan_prompt(
    title: str,
    body: str,
    labels: list[str],
    repo_name: str,
    task_type: str,
    local_code_context: str = "",
) -> str:
    """为 AutoCode 任务生成结构化计划。"""
    label_str = ", ".join(labels) if labels else "无"
    local_code_section = ""
    if local_code_context:
        local_code_section = f"""
## 当前本地仓库上下文

{local_code_context}
"""
    return f"""\
你是项目 **{repo_name}** 的资深实现规划助手。请针对以下任务输出结构化实现计划。

## 约束
- 计划必须保守，优先最小改动
- 不要修改 secrets、workflow、deploy、migration
- `needs_human_approval` 默认为 false；只有在缺少关键信息（如完全无法定位目标文件或模块）导致无法给出任何可执行方案时才设为 true
- `blocked_reasons` 只写真正的硬阻塞——即"如果不解决这个问题就物理上无法继续"的情况。以下都**不算**硬阻塞，请写进 `assumptions` 而非 `blocked_reasons`：
  - 实现方案有多种可选路径（选一个最合理的写进计划即可）
  - 不确定维护者偏好哪种方案（按常规最佳实践选择）
  - 风险偏高（通过 `risk_level` 体现，不要因此阻塞）
  - 改动范围可能超预期（给出保守范围即可）
  - 某些功能在仓库中未找到现成实现（这恰恰是本次任务要做的事）
- `estimated_files` 只列相对路径或路径前缀，不要写绝对路径

## 任务信息

**任务类型**: {task_type}
**标题**: {title}
**标签**: {label_str}
**内容**:
{body or "无"}
{local_code_section}

## 输出要求

严格输出 JSON，不要包含其他文字:
```json
{{
  "goal": "一句话目标",
  "assumptions": ["..."],
  "acceptance_criteria": ["..."],
  "risk_level": "low|medium|high",
  "estimated_files": ["src/x.py", "tests/"],
  "suggested_tests": ["pytest tests/test_x.py"],
  "needs_human_approval": true,
  "blocked_reasons": ["..."]
}}
```"""


def build_combined_triage_plan_prompt(
    title: str,
    body: str,
    labels: list[str],
    repo_name: str,
    local_code_context: str = "",
    discussion_context: str = "",
) -> str:
    """合并分诊和计划为单次 LLM 调用，减少重复上下文传输。"""
    label_str = ", ".join(labels) if labels else "无"
    local_code_section = ""
    if local_code_context:
        local_code_section = f"""
## 当前本地仓库上下文

{local_code_context}
"""
    discussion_section = ""
    if discussion_context:
        discussion_section = f"""
## 当前讨论

{discussion_context}
"""
    return f"""\
你是项目 **{repo_name}** 的 AutoCode 任务分析器。请对以下 Issue 同时完成分诊和实现计划。

## 第一步：任务分诊

判断任务类型和自动化策略：

- 任务类型：
  - `bug_fix`: 缺陷修复、回归修复、边界条件修复；即使涉及权限、安全、依赖、部署、CI，只要本质是纠正已有错误且范围清晰，仍优先归为 `bug_fix`
  - `todo_refactor`: 小范围重构、补测试、补文档、清理 TODO
  - `small_feature`: 范围清晰、验收标准明确的新功能
  - `high_risk_feature`: 架构改造、迁移、权限模型调整、依赖升级、部署/CI 流程改造、密钥/安全机制重设计，或上述高敏感领域里的大范围变更

- 自动化策略：
  - 明显 bug 可判为 `auto_fix`
  - 小范围 TODO/refactor 可判为 `auto_fix`
  - 高敏感领域里的"明确缺陷修复"不必一律判成 `high_risk_feature`；可以判为 `bug_fix`，同时把 `risk_level` 提高为 `high`
  - feature 默认更保守；只有边界清晰、风险低、验收标准明确时才允许 `implement`
  - 如果内容不清楚、需要维护者做方案决策，输出 `plan_only`
  - 如果任务不应该由机器人执行，输出 `reject`
  - 如果 issue 或当前讨论已经明确说明这是配置/参数/工作流调整、使用说明、已知行为解释、或“无需继续改代码/改配置即可”，不要硬凑代码方案；应保守输出 `plan_only` 或 `reject`
  - 优先判断“是否真的需要代码 PR”；如果更小修复路径只是调参数、改 workflow、补说明或按现有配置操作，不要把它扩展成大范围代码改造

## 第二步：实现计划

基于分诊结果，生成结构化实现计划：

- 计划必须保守，优先最小改动
- 不要修改 secrets、workflow、deploy、migration
- `needs_human_approval` 默认为 false；只有在缺少关键信息导致无法给出任何可执行方案时才设为 true
- `blocked_reasons` 只写真正的硬阻塞。以下都**不算**硬阻塞，请写进 `assumptions`：
  - 实现方案有多种可选路径
  - 不确定维护者偏好哪种方案
  - 风险偏高
  - 改动范围可能超预期
  - 某些功能在仓库中未找到现成实现
- `estimated_files` 只列相对路径或路径前缀
- 如果更小修复路径是工作流/参数/配置/操作步骤，而不是应用代码，就把它写进 `goal` / `assumptions`，不要伪造应用代码改造计划
- 如果当前讨论中已经存在维护者给出的明确处理方向，计划必须优先服从该方向，而不是另起一套更大的改法

## Issue 信息

**标题**: {title}
**标签**: {label_str}
**内容**:
{body or "无"}
{discussion_section}
{local_code_section}

## 输出要求

严格输出 JSON，不要包含其他文字:
```json
{{
  "triage": {{
    "task_type": "bug_fix|todo_refactor|small_feature|high_risk_feature",
    "action": "auto_fix|implement|plan_only|reject",
    "risk_level": "low|medium|high",
    "reason": "一句中文理由",
    "confidence": 0.0
  }},
  "plan": {{
    "goal": "一句话目标",
    "assumptions": ["..."],
    "acceptance_criteria": ["..."],
    "risk_level": "low|medium|high",
    "estimated_files": ["src/x.py", "tests/"],
    "suggested_tests": ["pytest tests/test_x.py"],
    "needs_human_approval": false,
    "blocked_reasons": ["..."]
  }}
}}
```"""


def build_execution_prompt(
    title: str,
    body: str,
    repo_name: str,
    task_type: str,
    plan_json: str,
    allowed_paths: list[str],
    validation_commands: list[str],
    local_code_context: str = "",
    review_feedback: list[str] | None = None,
    max_changed_files: int = 0,
    max_added_lines: int = 0,
    max_deleted_lines: int = 0,
    previous_failure_reasons: list[str] | None = None,
) -> str:
    """生成 AutoCode 执行 prompt。"""
    local_code_section = ""
    if local_code_context:
        local_code_section = f"""
## 当前本地仓库上下文

{local_code_context}
"""
    feedback_section = ""
    if review_feedback:
        feedback_section = "## 需要同时处理的 review 反馈\n" + "\n".join(f"- {item}" for item in review_feedback)
    normalized_allowed_paths = [str(path or "").strip() for path in allowed_paths if str(path or "").strip()]
    docs_only_task = bool(normalized_allowed_paths) and all(
        path.endswith((".md", ".mdx", ".rst", ".txt"))
        or path.startswith(("docs/", "doc/", "README", "readme", "deploy/", "examples/"))
        for path in normalized_allowed_paths
    )
    allowed_paths_title = "## 本任务只允许修改以下路径" if docs_only_task else "## 允许优先改动的路径"
    allowed_paths_section = "\n".join(f"- {path}" for path in normalized_allowed_paths) if normalized_allowed_paths else "- 仓库内相关文件"
    validation_section = "\n".join(f"- {command}" for command in validation_commands) if validation_commands else "- 无额外命令"
    task_guidance = ""
    if task_type == "todo_refactor":
        task_guidance = (
            "- 文档 / README / FAQ 整理优先做局部精简和内容下沉，避免整份大文档重写\n"
            "- 如果多个大文档都需要调整，优先最少文件集；超出必要范围时宁可保守收缩"
        )
        if docs_only_task:
            task_guidance += (
                "\n- 这是纯文档任务，禁止修改文档范围之外的代码、测试、配置和数据文件"
                "\n- 如果你判断必须改代码才能彻底解决，请不要扩散改动，而是在文档里保守说明限制与后续建议"
            )
    extra_constraints: list[str] = [
        "- 不要破坏现有公共接口、导入路径、CLI 参数或模块级导出；如必须调整，需同步更新全部调用点并补回归测试",
        "- 不要新增更严格的运行前提；如果必须引入新的目录、文件或权限依赖，需提供 fallback 或明确错误处理并补验证",
    ]
    if max_changed_files > 0:
        extra_constraints.append(f"- 改动文件数不超过 {max_changed_files} 个")
    if max_added_lines > 0:
        extra_constraints.append(f"- 新增行数不超过 {max_added_lines} 行，删除行数不超过 {max_deleted_lines} 行")
    if task_guidance:
        extra_constraints.append(task_guidance)
    extra_constraints_section = "\n".join(item for item in extra_constraints if item)
    previous_failure_section = ""
    if previous_failure_reasons:
        items = "\n".join(f"- {r}" for r in previous_failure_reasons[-3:])
        previous_failure_section = (
            "## ⚠ 前几轮执行失败原因（必须规避）\n"
            f"{items}\n"
            "请仔细分析上述失败原因，这次执行中必须主动避开相同问题。"
            " 如果前几轮是因为改动范围过大或破坏了现有功能，请缩小改动范围。\n"
        )
    return f"""\
你正在为项目 **{repo_name}** 执行自动编码任务。请直接在当前工作区内完成改动。

## 硬性约束
- 禁止使用 git merge、git rebase、git push 到默认分支
- 禁止修改 `.github/workflows/`、`deploy/`、`infra/`、`migrations/`、`.env`
- 尽量控制改动范围，只修改必要文件
- **每次改动代码，必须同步更新或新增对应的测试**（单元测试或集成测试均可）
- **完成改动后，必须运行与本次改动直接相关的测试命令并确认通过**，不要在测试未通过时结束任务
- 如需自检，运行与本次改动直接相关的最小测试集（如 `pytest tests/test_xxx.py` 或单个测试函数）
- 完成后只输出 3 到 8 行中文摘要，不要输出思考过程
{extra_constraints_section}

{previous_failure_section}
## 任务信息

**任务类型**: {task_type}
**标题**: {title}
**内容**:
{body or "无"}

## 已批准计划（JSON）
```json
{plan_json}
```

{allowed_paths_title}
{allowed_paths_section}

## 返回后系统会统一执行的校验命令
{validation_section}

{feedback_section}
{local_code_section}
## 现在开始执行，并在完成后只输出实现摘要"""


def _truncate_pr_body_for_feedback(pr_body: str, max_chars: int = 600) -> str:
    """截取 PR body 中与修复相关的关键信息，丢弃模板化内容。"""
    if not pr_body or len(pr_body) <= max_chars:
        return pr_body or "无"
    # 优先保留 goal/summary 段落，跳过 rollback/risk/validation 等模板段
    lines = pr_body.split("\n")
    kept: list[str] = []
    total = 0
    skip_sections = {
        "回滚方案", "风险说明", "验证结果", "风险", "兼容性",
        "Rollback", "Validation", "Verification", "Risk", "Compatibility",
    }
    skipping = False
    for line in lines:
        if line.startswith("## "):
            heading = line.lstrip("# ").strip()
            skipping = any(kw in heading for kw in skip_sections)
        if skipping:
            continue
        if total + len(line) > max_chars:
            kept.append("... [截断]")
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept) or "无"


def build_review_feedback_prompt(
    pr_title: str,
    pr_body: str,
    repo_name: str,
    review_feedback: list[str],
    local_code_context: str = "",
    changed_files: list[str] | None = None,
    scope_guard: str = "",
    repair_round: int = 0,
    previous_repair_notes: list[str] | None = None,
) -> str:
    """将 review 反馈整理为 follow-up 执行 prompt。"""
    feedback_section = "\n".join(f"- {item}" for item in review_feedback) if review_feedback else "- 无"
    context_section = ""
    if local_code_context:
        context_section = f"""
## 当前本地仓库上下文

{local_code_context}
"""
    files_section = ""
    if changed_files:
        files_section = "## 当前 PR 已改动的文件\n" + "\n".join(f"- `{f}`" for f in changed_files[:15])
    scope_guard_section = ""
    if scope_guard:
        scope_guard_section = f"""
## 本轮范围硬约束

{scope_guard}
"""
    trimmed_body = _truncate_pr_body_for_feedback(pr_body)
    round_section = ""
    if repair_round > 0:
        round_section = f"\n**当前修复轮次**: 第 {repair_round} 轮"
    previous_section = ""
    if previous_repair_notes:
        items = "\n".join(f"- {n}" for n in previous_repair_notes[-3:])
        previous_section = (
            "\n## ⚠ 前几轮修复情况（必须避免重蹈覆辙）\n"
            f"{items}\n"
            "请先理解上述问题再开始修复，不要重复犯同样的错误。\n"
        )
    return f"""\
你正在为项目 **{repo_name}** 的已有 AutoCode PR 处理 review 反馈。

## 硬性约束
- 只修复反馈中明确要求的问题，不做额外改动
- 禁止修改 `.github/workflows/`、`deploy/`、`infra/`、`migrations/`、`.env`
- **如果反馈中有 CI 测试失败，必须修复失败的测试，不能忽略**
- **修复完成后，必须运行与本次改动直接相关的测试命令并确认通过**，不要在测试未通过时结束任务
- 完成后只输出 3 到 5 行中文摘要

**PR 标题**: {pr_title}{round_section}
**PR 概要**: {trimmed_body}

{files_section}
{scope_guard_section}
{previous_section}
## 需要处理的反馈
{feedback_section}
{context_section}
## 现在开始修复，并在完成后只输出实现摘要"""


def build_source_issue_prompt(
    *,
    source_name: str,
    raw_title: str,
    raw_body: str,
    raw_labels: list[str] | None = None,
) -> str:
    """将内部 source 条目整理为对外可见的 GitHub issue 草稿。"""
    labels = ", ".join(raw_labels or []) if raw_labels else "无"
    return f"""\
你是 AutoCode 的 GitHub issue 起草助手。请把下面的内部 source 条目整理成一个正常、可执行的 GitHub issue，或者判断它根本不应该创建 issue。

## 目标
- `create`：输出一个适合直接发布的 issue 标题和正文
- `skip`：如果条目只是内部备注、审计反证、未证实结论、否定句、排查记录或不具备明确动作价值，就不要创建 issue

## create 要求
- `title` 必须像正常 issue 标题，简洁、可执行、不要直接照抄原文
- `title` 需要包含受影响对象或行为，不要只写“优化一下”“补一下”“查看一下”这类空泛表述
- `title` 必须带标准前缀，并与类型标签对应：`[Bug]`、`[Feature]`、`[Docs]` 或 `[Question]`
- 不要保留 `[TODO]`、`[Idle]`、`[Backlog]` 这类前缀
- 凡是“缺少校验 / 验签 / 认证 / 权限控制 / 安全防护”这类已存在能力缺陷，一律按 `bug` / `[Bug]` 处理，不要写成 `feature` / `enhancement`
- `body` 只写问题现象、原因、影响或下一步，使用中文
- `body` 不要暴露内部同步信息，不要出现 source file、line number、item key、section、TODO 文件路径
- 不要输出本地绝对路径，也不要输出 `projects/...`、`src/...:12` 这类仓库内部定位信息
- 如无必要，不要直接暴露内部类名、函数名、工作流文件名或审计记录原文，优先改写成用户可理解的问题描述
- `labels` 必须包含 `autocode`，并补一个类型标签：`bug`、`enhancement`、`documentation` 或 `question`

## skip 要求
- 如果原条目本身是在说“这个问题不成立 / 未坐实 / 不应直接认定”，或者只是内部提醒、排查动作、审计备注，输出 `skip`
- `reason` 用一句简短中文说明为什么不该建单

## 输入
- source: {source_name}
- raw labels: {labels}
- raw title: {raw_title}
- raw body:
{raw_body or "无"}

## 输出格式
严格输出 JSON，不要包含其他文字：
```json
{{
  "action": "create|skip",
  "title": "[Bug|Feature|Docs|Question] issue 标题",
  "body": "issue 正文",
  "labels": ["autocode", "bug|enhancement|documentation|question"],
  "reason": "简短原因"
}}
```"""


def build_idle_candidate_prompt(
    *,
    repo_name: str,
    category: str,
    rule_id: str,
    source_path: str,
    line_number: int,
    scope_hint: str,
    evidence: str,
    suggestion_hint: str,
    local_code_context: str = "",
) -> str:
    """为 idle 巡检候选生成补充说明。"""
    local_code_section = ""
    if local_code_context:
        local_code_section = f"""
## 当前本地仓库上下文

{local_code_context}
"""
    return f"""\
你正在为项目 **{repo_name}** 的 idle 巡检候选生成一条稳定、可执行的 GitHub Issue 说明。

## 巡检候选
- 分类: `{category}`
- 规则: `{rule_id}`
- 位置: `{source_path}:{line_number}`
- 所在作用域: `{scope_hint or "未知"}`
- 触发证据: `{evidence}`
- 初步建议: `{suggestion_hint}`

## 输出要求
- 只根据给出的代码证据和仓库上下文判断，不要臆造不存在的文件或行为
- 不要把候选描述成“已经确认的缺陷”，保持“候选/建议进一步确认”的语气
- 如果想到类似工具或项目中常见的可借鉴做法，可以写在 `reference_hint`，但不要假装引用了真实外部仓库
- 严格输出 JSON，不要包含其他文字

{local_code_section}

```json
{{
  "summary": "一句中文概括",
  "impact": "1到2句说明为什么值得跟进",
  "suggestion": "1到3句最小改动建议",
  "reference_hint": "如有可借鉴实现思路写一句，否则留空",
  "confidence": 0.0
}}
```"""


def build_execution_failure_reply(stage: str, reason: str) -> str:
    """构造 AutoCode 失败说明。"""
    action = _failure_next_action(stage, reason)
    return (
        "## AutoCode 执行未完成\n\n"
        f"- 阶段: {stage}\n"
        f"- 原因: {reason}\n"
        f"- 建议下一步: {action}\n\n"
        "本次不会继续自动提交代码或发起合并；处理上述事项后，可由仓库 owner 评论 `继续修`、`按你说的做` 或 `开个 PR` 重新触发。"
    )


def _failure_next_action(stage: str, reason: str) -> str:
    text = f"{stage}\n{reason}".lower()
    if "环境未就绪" in reason or "command not found" in text or "no module named" in text or "缺少" in reason:
        return "先补齐本地验证依赖或调整验证命令；如果这是 CI 专属依赖，请在评论里说明可跳过本地环境限制后重试。"
    if "越界" in reason or "blocked" in text or "forbid" in text or "禁止" in reason:
        return "确认改动范围是否应扩大；如确需修改受限路径，请人工调整策略或把任务拆成更小范围。"
    if "验证失败" in reason or "失败:" in reason or "pytest" in text or "lint" in text:
        return "优先根据失败输出修复测试或 lint；如果失败与本任务无关，请在评论中说明并让 AutoCode 继续修。"
    if "未检测到代码改动" in reason or "未产生改动" in reason:
        return "补充更明确的期望改动、目标文件或复现步骤，然后让 AutoCode 继续修。"
    if "policy" in text or "批准" in reason or "approval" in text:
        return "由维护者确认风险和范围后，用明确评论触发实现。"
    if "planning" in text or "计划" in reason:
        return "补充目标、验收标准或相关文件路径后重新触发。"
    return "查看失败原因，补充缺失上下文或确认执行范围后重新触发。"


def format_execution_plan_reply(
    *,
    header: str,
    task_type: str,
    action: str,
    plan_summary: str,
    risk_level: str,
    estimated_files: list[str],
    acceptance_criteria: list[str],
    needs_human_approval: bool,
    blocked_reasons: list[str] | None = None,
) -> str:
    """把结构化计划格式化为可公开发布的评论。"""
    lines = [
        header,
        "",
        f"- 任务类型: `{task_type}`",
        f"- 建议动作: `{action}`",
        f"- 风险等级: `{risk_level}`",
        f"- 是否需人工批准: {'是' if needs_human_approval else '否'}",
        "",
        "**计划摘要**",
        plan_summary,
        "",
        "**预估改动范围**",
    ]
    if estimated_files:
        lines.extend(f"- {path}" for path in estimated_files[:8])
    else:
        lines.append("- 待执行时按实际上下文确定")
    lines.extend(["", "**验收标准**"])
    if acceptance_criteria:
        lines.extend(f"- {item}" for item in acceptance_criteria[:8])
    else:
        lines.append("- 以问题修复和现有测试通过为准")
    blockers = [str(item).strip() for item in (blocked_reasons or []) if str(item).strip()]
    if blockers:
        lines.extend(["", "**当前阻塞/注意事项**"])
        lines.extend(f"- {item}" for item in blockers[:8])
    return "\n".join(lines).strip()


# ============================================================
# 不可回答 Issue 的礼貌回复模板（非 LLM 生成）
# ============================================================

UNANSWERABLE_REPLY_TEMPLATE = """\
👋 感谢提交此 Issue！

经过初步分析，这个 Issue 看起来属于 **{reason}** 类型，需要项目维护者进一步评估和决策。\
我是自动助手，暂时无法对此类问题给出合适的答复。

维护者会尽快查看，感谢你的耐心等待！🙏

---
*🤖 此回复由 AutoCode Bot 自动生成*\
"""


def build_unanswerable_reply(reason: str) -> str:
    """为不可自动回答的 Issue 生成礼貌的占位回复"""
    return UNANSWERABLE_REPLY_TEMPLATE.format(reason=_summarize_unanswerable_reason(reason))


# ============================================================
# 回复包装器（给所有 AI 回复添加 footer）
# ============================================================

REPLY_FOOTER = "\n\n---\n*🤖 此回复由 AutoCode Bot 自动生成，仅供参考。如有疑问请 @维护者。*"

_LOCAL_FILE_LINK_RE = re.compile(r"\[([^\]]+)\]\((/[^)]+)\)")
# Strip raw absolute paths that leak local directory structure (e.g. /home/ubuntu/...)
_RAW_LOCAL_PATH_RE = re.compile(r"/home/\S+")
_TRACE_LINE_RE = re.compile(
    r"^\s*(?:●|└|├|│|┌|┐|┘|\$"
    r"|Now I have\b|Here's my reply\b|Let me compose\b"
    r"|Explore:|Read\b|Grep\b|List\b|Check\b|Search\b"
    r"|head\b|tail\b|grep\b|wc\b|cat\b|find\b"
    r"|\d+ lines? found\b|\d+ files? found\b)"
)
_SEPARATOR_LINE_RE = re.compile(r"^\s*(?:---+|\*\*\*+)\s*$")
_BOT_FOOTER_LINE_RE = re.compile(r"此回复由\s+.*?Bot\s+自动生成")
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
    # Strip raw local filesystem paths (e.g. /home/ubuntu/autocode-xxx/src/foo.py)
    sanitized = _RAW_LOCAL_PATH_RE.sub(
        lambda m: "`" + m.group(0).rsplit("/", 1)[-1] + "`" if "/" in m.group(0) else "",
        sanitized,
    )
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


def _extract_review_summary_line(line: str) -> tuple[str, str] | None:
    for separator in ("：", ":"):
        if separator not in line:
            continue
        left, right = line.split(separator, 1)
        label = _normalize_review_label(left)
        if label in _REVIEW_SUMMARY_LABELS:
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


def wrap_reply(content: str) -> str:
    """给 AI 生成的回复添加统一的 footer 标识"""
    formatted = format_issue_reply(content)
    return formatted + REPLY_FOOTER if formatted else REPLY_FOOTER.strip()


def wrap_pr_review_reply(content: str) -> str:
    """给 PR review 添加格式化结构和统一 footer。"""
    formatted = format_pr_review_reply(content)
    return formatted + REPLY_FOOTER if formatted else REPLY_FOOTER.strip()
