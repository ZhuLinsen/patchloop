# Security Policy

PatchLoop runs local CLI agents, reads repository content, and can optionally
push branches or open pull requests. Treat it as automation with access to your
source code and GitHub permissions.

## Supported Versions

PatchLoop is currently pre-1.0. Security fixes are made on the default branch.

## Reporting a Vulnerability

Please report vulnerabilities privately instead of opening a public issue when
the report includes tokens, exploit details, repository contents, or command
execution paths.

If you publish your own fork, rotate any token that was ever committed to Git
history before making the repository public.

## Deployment Guidance

- Use fine-grained GitHub tokens whenever possible.
- Keep read and write tokens separate.
- Start with `DRY_RUN=true` and `EXECUTION_MODE=manual`.
- Do not expose PatchLoop HTTP endpoints to the public internet without your own
  authentication and network controls.
- Keep `.env`, state files, logs, and generated patch files out of version control.
- Use a dedicated local clone or worktree root for PatchLoop tasks.
