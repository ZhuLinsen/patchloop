# Changelog

本文件记录 OpenReview 仓库的重要变更，便于在提交、部署和排查时快速回顾行为变化。

## Unreleased - 2026-03-30

### Changed

- PR 重复评审保护策略从指数退避冷却（30min → 1h → 2h → 4h → 8h）改为 **head SHA 稳定窗口（debounce）**。
  - 新 commit 推送后，系统等待 head SHA 在连续 polling 周期内不再变化（默认 300 秒）后才触发重新评审。
  - 快速连续 push 时只 review 最后一个稳定版本，不再每个中间 commit 都评审。
  - 一旦停止 push，等待时间始终固定（不再随 review 次数递增）。
  - 有外部讨论信号时仍立即跳过 debounce 触发 follow-up review。
- 新增配置项 `PR_HEAD_STABILIZE_SECONDS`（默认 300），控制 head SHA 稳定等待时间。

### Removed

- 移除 `consecutive_review_count` 指数冷却机制及相关方法（`is_pr_in_review_cooldown`、`pr_review_cooldown_seconds`、`increment_pr_review_count`、`reset_pr_review_cooldown`）。

### Compatibility Notes

- 升级后旧 state 文件中的 `consecutive_review_count` 字段不再使用，会被自然忽略。
- 新字段 `head_sha_first_seen_at` 会在下一次 `mark_pr_seen` 时自动写入，无需手动迁移。

## Unreleased - 2026-03-26

### Added

- 新增 `agent/discussion.py`，统一整理 issue comment、PR review、review comment 的 discussion 上下文。
- Issue 分析新增相似 issue、相关 PR、疑似已修复线索和高置信重复 issue 检测。
- PR 新增 follow-up review，能够在首轮评审后继续回应新的讨论。
- 新增针对 discussion、poller、state migration、main context、webhook 过滤和 GitHub recent pagination 的测试覆盖。

### Changed

- polling 继续作为当前主运行模式，围绕 `updated_at`、discussion cursor 和 review submission 扫描增强了增量跟进能力。
- Issue 去重从单次处理扩展为基于 `title + body + discussion fingerprint` 的持续跟进。
- PR 去重从单纯按 `head_sha` 扩展为 `head + base + 描述 + discussion fingerprint`。
- `README.md` 与 `AGENTS.md` 重新整理为当前文档主入口；原 `design.md` 已移除。
- `.env` is no longer treated as a shareable artifact. Public deployments should keep
  real credentials outside version control and start from `.env.example`.

### Fixed

- 修复 Issue / PR 在异常路径下可能遗留处理锁的问题，避免长时间运行后锁表残留。
- 优化 GitHub recent comment/review 读取逻辑，优先利用尾页信息抓取最近几页，避免为取尾部数据而翻完整历史分页。
- polling 与 webhook 的 slash command 过滤语义保持一致。
- 升级兼容逻辑会优先复用既有 `.openreview-state.json`，避免重置后重新分析大量历史 Issue / PR。

### Compatibility Notes

- 升级后应继续沿用现有 `.openreview-state.json`，不要随意删除或切换 state 路径。
- 如果手动重置状态文件、cursor 或 baseline，系统会把这视为显式重跑，可能重新分析历史对象。
- 当前配置层面一次只启用一种事件入口；使用 `polling` 时，`/webhook` 路由会保持 disabled。
