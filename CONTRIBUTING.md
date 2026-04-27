# Contributing to AstroStack

Thank you for taking the time to contribute. This document describes the
workflow, conventions and quality gates used in this project. The full
development guide — architecture, layout, principles, recipes — lives in
[DEVELOPMENT.md](./DEVELOPMENT.md). Read it before opening your first PR.

---

## Table of contents

- [Getting started](#getting-started)
- [Development workflow](#development-workflow)
- [Code standards](#code-standards)
- [Commit messages](#commit-messages)
- [Pull requests](#pull-requests)
- [Reporting issues](#reporting-issues)

---

## Getting started

```bash
git clone https://github.com/x42en/AstroStack.git
cd AstroStack
pip install -e ".[dev]"
cp .env.example .env
```

Spin up Postgres + Redis (locally or via `docker compose up postgres redis`)
and run the API + a worker:

```bash
alembic upgrade head
uvicorn app.main:app --reload --port 8080
python -m arq app.workers.settings.WorkerSettings   # in another shell
```

See [DEVELOPMENT.md](./DEVELOPMENT.md) for the full local-setup guide.

---

## Development workflow

1. **Fork** the repository and create a feature branch from `main`.
2. Implement your change. Run `ruff check`, `ruff format --check`, `mypy`
   (where configured) and `pytest` before committing — CI rejects branches
   that fail these gates.
3. Open a **pull request** against `main`. Keep the scope focused: one
   logical change per PR.
4. At least one approving review is required before merging.

---

## Code standards

The full set of principles (English only, strict typing, fail fast, no dead
code, file-size targets, etc.) is documented in
[DEVELOPMENT.md → Principles](./DEVELOPMENT.md#principles). Highlights:

- **English only** for code, comments, commits, PRs and user-facing strings.
- **Strict typing.** Annotate every public function. `mypy` clean.
- **Docstrings** on every public function and class (Google or reST style).
- **Fail fast and loud** — validate at boundaries, do not add defensive null
  checks inside the perimeter.
- **No `Any`** unless justified inline.
- **No dead code, no duplication, no backward-compat shims** during a
  migration.
- **File size** under 500 lines (hard ceiling 1 000).
- **Tests** ship alongside the change.

Pipeline-specific rules:

- A new pipeline step subclasses `BaseStep` (`app/pipeline/base_step.py`)
  and is wrapped in an adapter under `app/pipeline/adapters/` if it shells
  out to an external tool.
- A new endpoint goes through the `domain → infrastructure → service → api`
  layering documented in [DEVELOPMENT.md → Project layout](./DEVELOPMENT.md#project-layout).

---

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short imperative summary>

[optional body — explain WHY, not WHAT]

[optional footer — breaking changes, closes #issue]
```

| Type | When to use |
|---|---|
| `feat` | A new feature visible to API consumers. |
| `fix` | A bug fix. |
| `refactor` | Internal restructuring, no behaviour change. |
| `perf` | Performance improvement. |
| `test` | Adding or updating tests. |
| `docs` | Documentation only. |
| `build` | Build system, Dockerfile, dependency, CI changes. |
| `chore` | Tooling, repo housekeeping. |

Rules:

- Imperative mood: `add`, `fix`, `remove` — not `added`, `fixes`, `removed`.
- Summary ≤ 72 characters.
- Reference issues in the footer: `Closes #42`.
- Breaking changes include `BREAKING CHANGE:` in the footer.

---

## Pull requests

- **Title** follows the Conventional Commits format above.
- **Description** explains *what* changed and *why*. Link related issues.
- Keep PRs small and focused. A diff above ~400 lines warrants a discussion
  before you open it.
- Ensure `ruff`, `mypy` (where configured), and `pytest` pass locally.
- A migration is included with any schema change, in the **same** PR
  (no deprecation period).
- Squash-merge into `main`; the PR title becomes the squash commit message.

---

## Reporting issues

Use [GitHub Issues](https://github.com/x42en/AstroStack/issues). Please
include:

- **Environment:** OS, Python version, Docker version, GPU model + driver,
  CUDA version.
- **Steps to reproduce.**
- **Expected vs. actual behaviour.**
- **Relevant logs** from the API or worker container (`docker compose logs
  astro-api astro-worker-0`).
