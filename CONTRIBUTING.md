# Contributing to PatchLoop

PatchLoop is early-stage infrastructure for running GitHub agent loops locally.
Contributions are welcome, especially around safety gates, queue reliability,
adapter support, documentation, and tests.

## Development Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

Run the review service tests:

```bash
cd openreview
python3 -m pytest tests -q
```

Run the coding service tests:

```bash
cd autocode
python3 -m pytest tests -q
```

## Contribution Rules

- Do not commit real `.env` files, tokens, state snapshots, logs, patches, or local worktrees.
- Keep `DRY_RUN=true` in examples unless a section explicitly explains production setup.
- Preserve the safety model: PatchLoop must not merge PRs automatically or push directly to a default branch.
- Prefer small, reviewable changes with focused tests.
- Update `README.md` when behavior, configuration, state, polling, or safety boundaries change.

## Security-Sensitive Changes

Changes touching authentication, GitHub write permissions, webhook verification,
branch protection, command execution, or path filtering should include tests and
clear documentation of the risk model.
