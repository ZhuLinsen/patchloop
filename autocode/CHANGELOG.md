# Changelog

本文档记录 AutoCode 近期值得关注的版本变化，优先覆盖会影响自动化稳定性、执行体验和日常观测的更新。

## [v2026.03.27] - 2026-03-27

### What's New

- CLI 执行链路补齐可写工作区支持、超时控制和更细粒度进度日志，定位卡住或失败原因更直接。
- source sync / backlog / idle 的 issue 同步更稳，已同步条目会优先复用既有 issue，减少重复建单和漏执行。
- PR repair 只聚焦最新仍有效的 review feedback，避免回放过期评论或自动化样板意见。
- 状态持久化、轮询与任务调度日志更完整，队列恢复、重试和运行观测更直观。

### 自动化执行

- 放宽 AutoCode autorun 策略，让范围明确的自动执行场景更容易直接跑通。
- source item 已经写回 issue 链接时，可直接复用并按需重新入队 issue 执行链路。
- `codex` 执行适配器补齐可写运行支持，减少只读环境导致的执行失败。
- CLI 超时、进度摘要和失败上下文更丰富，排查执行卡顿时不再只剩黑盒结果。

### Source Sync 与去重

- backlog/source 同步补上去重、回写和节流，避免重复创建 `autocode` issue。
- backlog 已同步或已关联的条目会优先复用现有 issue，而不是因为本地状态变化重复建单。
- 安全相关的 source item 会按 `bug` / `[Bug]` 口径建单，减少类型误判。
- source sync 命中既有 issue 后，也能按需把 issue 执行任务重新排队，补上“已建单但未继续处理”的缺口。

### PR 修复与观测

- PR repair 改为优先看每位 reviewer 最新仍有效的 review 状态与反馈。
- 过滤过期 review comment 和自动化 review boilerplate，降低误触发修复的概率。
- 调度器、issue polling、PR queue、CLI 执行阶段补齐结构化日志与摘要输出，排查更直接。
- 状态快照、队列重试与恢复观测增强，后台运行状态更容易追踪。

### 关联提交

- `e9e9e0e` `feat(autocode): improve cli observability and timeout handling`
- `7db8c80` `fix(autocode): enable writable codex runs and richer progress logs`
- `5521e7b` `update pipeline`
- `a0cf652` `Enable broader AutoCode autorun policies`
- `fc8e81b` `Classify security source issues as fixes`
- `9d44f6a` `Prevent duplicate source issues and throttle backlog sync`
- `a646cd5` `Fix source sync dedupe and backlog writeback`
- `1ef5ec3` `Improve source issue workflow and runtime diagnostics`
- `ca78f72` `Improve state persistence and runtime observability`
- `8a1df4b` `Improve PR repair feedback filtering and diagnostics`
- `85e5675` `Improve backlog source and idle execution workflow`
- `0a618b2` `Improve PR review local analysis context`
