<div align="center">

# PatchLoop

**A battle-tested, local-first GitHub agent loop for reviews, patches, and review feedback repair**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

PatchLoop turns GitHub issue and pull request work into a durable local loop.
It was extracted from real automation used on
[ZhuLinsen/daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis),
a public AI-driven stock analysis project, where the loop handled issue triage,
pull request review, and review-feedback-driven patch repair in a live repository.

The core idea is simple: do not stop at "the agent opened a PR." Keep the PR
moving after review.

```text
Issue / TODO / idle finding
        |
        v
Plan and policy gate
        |
        v
Local patch in isolated worktree
        |
        v
Validation and patch inspection
        |
        v
Pull request
        |
        v
Review feedback
        |
        +----> repair queue ----> local patch ----> push update
```

It is designed for maintainers who want agent automation without handing the
entire repository workflow to a hosted service. PatchLoop runs on your machine
or server, talks to GitHub, drives your installed coding CLI, and persists enough
state to recover from restarts, retries, and missed webhook events.

## Highlights

- **Battle-tested on a real public repo**: extracted from the automation loop
  used on [daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis).
- **Local-first by design**: your clone, your tokens, your validation commands,
  your machine.
- **Multi-CLI execution**: use `codex`, `copilot`, `cursor`, `opencode`, or a
  custom command profile.
- **Not just PR creation**: PatchLoop tracks review feedback, queues repair work,
  and pushes follow-up patches to the same PR branch.
- **Durable long-running service**: polling cursors, idempotency keys, task
  queues, dead letters, state snapshots, and restore APIs.
- **Conservative by default**: dry-run examples, manual execution mode,
  no auto-merge, and no direct pushes to the default branch.

## What PatchLoop Does

PatchLoop contains two cooperating services.

| Service | Role | GitHub write surface |
| --- | --- | --- |
| `openreview` | Read-only issue analysis and PR review follow-up | Comments and PR reviews only |
| `autocode` | Issue-to-PR execution, backlog sync, idle scan, and PR feedback repair | Branch pushes and PR create/update |

### OpenReview

- Analyzes issues with local repository context, similar issues, related PRs, and discussion history.
- Reviews non-draft PRs after CI succeeds, with controlled follow-up when new discussion appears.
- Uses polling cursors and discussion fingerprints to reduce missed events and duplicate reviews.
- Keeps a strict read-only implementation boundary: it never edits files or pushes code.

### AutoCode

- Turns eligible issues into structured plans, applies policy gates, and creates scoped patches.
- Runs each coding task in an isolated worktree before validation and publication.
- Opens or updates PRs, but never merges and never pushes directly to the default branch.
- Tracks PR review feedback, queues repair tasks, and pushes follow-up fixes to the PR branch.
- Can sync Markdown backlog items and idle findings into GitHub issues before entering the same execution loop.

## Why It Is Different

- **Local-first control**: use your own clone, tokens, validation commands, and CLI tools.
- **CLI-agnostic adapters**: supports `codex`, `copilot`, `cursor`, and `opencode` style execution backends.
- **Durable queues**: issue execution, PR repair, and source sync tasks survive restarts.
- **Polling plus webhook**: webhook can reduce latency, polling provides recovery and long-running stability.
- **Feedback repair loop**: review feedback is not just summarized; it can be converted into follow-up patches.
- **Conservative safety model**: no auto-merge, no default-branch pushes, diff size limits, blocked paths, and dry-run defaults.

## Event Ingestion: Polling and Webhook

PatchLoop supports both GitHub event ingestion styles:

| Mode | Best for | Notes |
| --- | --- | --- |
| `polling` | Long-running local/server deployments | Recommended default. Uses cursors and overlap windows to recover from missed webhook delivery, restarts, and transient GitHub/API failures. |
| `webhook` | Low-latency event delivery | Requires a public or tunneled HTTP endpoint and `GITHUB_WEBHOOK_SECRET`. |

Recommended setup:

- Start with `EVENT_SOURCE=polling` and `ENABLE_WEBHOOK=false`.
- For faster response, keep polling as the safety net and enable webhook where supported.
- `openreview` currently treats `EVENT_SOURCE=webhook` as the webhook-serving mode; when it runs in polling mode, `/webhook` is disabled.
- `autocode` can run with `EVENT_SOURCE=polling` and `ENABLE_WEBHOOK=true`, using webhook as an accelerator while background polling and queues handle recovery.

Minimal polling config:

```env
EVENT_SOURCE=polling
ENABLE_WEBHOOK=false
POLL_INTERVAL_SECONDS=60
```

Webhook config:

```env
EVENT_SOURCE=webhook
ENABLE_WEBHOOK=true
GITHUB_WEBHOOK_SECRET=replace_me_webhook_secret
```

## Supported CLI Backends

PatchLoop does not require one hosted agent runtime. It shells out to a local CLI
adapter, so you can choose the model/tooling stack that fits your environment.

| Backend | Typical use |
| --- | --- |
| `codex` | OpenAI/Codex-style local coding sessions |
| `copilot` | GitHub Copilot CLI workflows |
| `cursor` | Cursor agent CLI workflows |
| `opencode` | OpenCode-style local execution |
| Custom command | Any compatible CLI configured through `*_CLI_COMMAND` |

## Quick Start

PatchLoop is currently source-run infrastructure. Start with dry-run mode, then
turn on write operations only after the queues and validation commands behave as
expected on your repository.

### 1. Clone and install

```bash
git clone https://github.com/your-org/patchloop.git
cd patchloop
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

You also need at least one supported local coding CLI installed and authenticated
on the machine running PatchLoop.

### 2. Run read-only review automation

```bash
cd openreview
cp .env.example .env
```

Edit `openreview/.env`:

```env
GITHUB_TOKEN=replace_me_read_token
GITHUB_REPO=owner/repo
LOCAL_REPO_PATH=/path/to/your/repo
PRIMARY_CLI=codex
EVENT_SOURCE=polling
ENABLE_WEBHOOK=false
DRY_RUN=true
```

Start the review service:

```bash
python3 review.py
# or, from the repository root after `pip install -e .`:
patchloop review
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 3. Run coding automation in dry-run mode

```bash
cd ../autocode
cp .env.example .env
```

Edit `autocode/.env`:

```env
GITHUB_TOKEN=replace_me_read_token
GITHUB_REPO=owner/repo
LOCAL_REPO_PATH=/path/to/your/repo
PRIMARY_CLI=codex
EVENT_SOURCE=polling
ENABLE_WEBHOOK=false

DRY_RUN=true
ENABLE_EXECUTION=true
EXECUTION_MODE=manual
EXECUTION_AUTO_IMPLEMENT_ON_ISSUE_OPEN=false
EXECUTION_ENABLE_PR_QUEUE=true
```

Start the coding service:

```bash
python3 autocode.py
# or, from the repository root after `pip install -e .`:
patchloop code
```

Useful endpoints:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/observability
curl "http://127.0.0.1:8001/tasks?limit=20"
```

Manually approve an issue implementation:

```bash
curl -X POST http://127.0.0.1:8001/issues/123/implement
```

## Production Checklist

Before disabling dry-run:

- Use a dedicated local clone or worktree root.
- Keep `GITHUB_TOKEN` read-only where possible.
- Set `GITHUB_WRITE_TOKEN` separately for branch pushes and PR updates.
- Configure `EXECUTION_FORMAT_COMMANDS`, `EXECUTION_LINT_COMMANDS`, and `EXECUTION_TEST_COMMANDS`.
- Keep `EXECUTION_MODE=manual` until you trust the policy gates on your repository.
- Review `EXECUTION_BLOCKED_PATHS`, diff limits, and maximum open PR limits.
- Keep `.env`, state files, logs, generated patches, and worktrees out of Git.

## Repository Layout

```text
patchloop/
  openreview/      # read-only issue and PR review service
  autocode/        # coding, PR creation, queueing, and feedback repair service
  README.md
  LICENSE
  CONTRIBUTING.md
  SECURITY.md
```

The two services are intentionally still separated. `openreview` can run with
comment-only permissions, while `autocode` requires write permissions only when
`DRY_RUN=false`.

## Current Status

PatchLoop is alpha software extracted from a working local automation setup. The
core loop is present, but the public packaging is intentionally conservative:

- Single-repository operation is the primary target.
- Multi-tenant SaaS deployment is not a goal of this repository.
- GitHub App packaging, dashboards, and richer policy configuration are future work.

## Roadmap

- Package both services behind a single `patchloop` CLI.
- Share common GitHub, CLI adapter, context, and state utilities between services.
- Add Docker Compose examples for local deployment.
- Add GitHub App installation docs for teams that prefer app-based permissions.
- Expand policy presets for docs-only, bug-fix, and high-risk repository areas.

## License

Apache-2.0 License. See [LICENSE](LICENSE) for details.
