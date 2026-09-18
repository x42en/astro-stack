# AstroStack

[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](./LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async%20API-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker Image](https://img.shields.io/badge/ghcr.io-bitsdiver%2Fastro--stack-2496ED.svg)](https://github.com/bitsdiver/astro-stack/pkgs/container/astro-stack)
[![Build & Publish](https://github.com/bitsdiver/astro-stack/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/bitsdiver/astro-stack/actions/workflows/docker-publish.yml)

> Automated astrophotography processing pipeline with an event-driven API backend.

AstroStack watches an inbox for incoming RAW/FITS sessions, runs them through a
configurable, GPU-accelerated processing pipeline (calibration → stacking →
plate solving → AI gradient removal → AI denoise → AI sharpening → super
resolution → star separation → satellite-trail removal → export), and streams
real-time progress over WebSocket. An optional vision-LLM **adaptive critic**
can iteratively refine a handful of steps for you — fully autonomous and off
by default, so novices get a hands-off pipeline while advanced users can opt
into tighter control. It ships with a community gallery and shareable
processing profiles so astronomers can publish results and reuse one
another's recipes.

The companion web interface lives in the [AstroUI](https://github.com/x42en/astro-ui)
repository.

---

## Overview

| Component | Stack |
|---|---|
| API & orchestration | Python 3.12, FastAPI 0.115+, Uvicorn, SQLModel |
| Job queue | Redis 7 + ARQ |
| Database | PostgreSQL 16 (asyncpg), Alembic migrations |
| Stacking & calibration | Siril 1.4.x (headless CLI + `sirilpy` scripting) |
| Plate solving | ASTAP (headless CLI) |
| Gradient removal / denoise | GraXpert 3.x (AI + polynomial, CUDA) |
| AI enhancement | SetiAstroSuitePro (SASpro) — denoise, sharpen, super-resolution, star separation, satellite-trail removal, aberration correction (CUDA) |
| Adaptive critic (optional) | LangGraph 1.x + any OpenAI-compatible vision-LLM endpoint (e.g. self-hosted vLLM) |
| RAW handling | rawpy / LibRaw, ExifRead |
| Real-time events | WebSocket + Redis pub/sub |
| Container runtime | NVIDIA CUDA 12.8.1 + cuDNN 9 base image |

---

## Features

- **End-to-end pipeline** — 11 verified steps from RAW conversion to export
  (`app/pipeline/steps/`), including satellite/aircraft trail removal.
- **Adaptive vision-critic loop (optional, off by default)** — a LangGraph
  state machine sends a step's preview + numeric stats to a vision-LLM
  critic, which proposes a small, catalogue-validated config patch; the step
  re-runs and the loop repeats until the critic is satisfied or an iteration
  budget is reached. Fully autonomous: no pause, no required review. An
  optional human-approval gate exists for advanced users but auto-approves
  with a logged warning if left unconfigured — the pipeline never stalls
  waiting on a reviewer that doesn't exist. Covers `gradient_removal`,
  `stretch_color`, `denoise`, `sharpen`, `super_resolution` and
  `satellite_removal` (`app/pipeline/adaptive/`).
- **Live stacking with on-rig coaching** — drop frames as they come off the
  sensor and watch the stack build in real time. The `app/livestack/` engine
  computes per-frame statistics (median R/G/B, clipping, FWHM) and a
  recommender (`app/livestack/recommender.py`) emits prioritised exposure,
  white-balance and focus advice over WebSocket while you are still at the
  telescope. The same adaptive critic can also auto-tune the live MTF
  autostretch after a warm-up period, then freeze the result and reuse it for
  the rest of the session (opt-in, off by default).
- **Structured tool capability catalogue** (`app/pipeline/adaptive/tool_catalog.py`)
  — every tunable parameter of every tool is documented with its role, visual
  effect, risk level and valid range, doubling as the critic's prompt context
  and as living documentation; validated against the real profile schema by
  its own test suite so it can never silently drift out of date.
- **SetiAstroSuitePro (SASpro)** — actively maintained successor to the
  archived Cosmic Clarity script bundle; adds satellite-trail removal and a
  built-in aberration corrector on top of the existing denoise/sharpen/
  super-resolution/star-separation engines.
- **Siril Python scripting** (`SirilPyAdapter`) — structured image statistics
  (mean/median/sigma/bgnoise, star PSF data) read directly from Siril via its
  `sirilpy` module, alongside the existing named-pipe adapter used for
  real-time stacking progress.
- **Session ownership & resume** — each user gets at most one active live
  session at a time, exposed via `GET /sessions/active-live` so the UI can
  surface a persistent "Resume" banner and a 409 redirect if a second one is
  attempted.
- **Calibration libraries per session** — dedicated upload endpoints for
  `darks/`, `flats/` and `dark_flats/` sub-folders with progress tracking and
  frame-count snapshots on the session row.
- **Observation planning** — followed-objects catalogue, observation-site
  registry (`app/api/v1/observation_sites.py`, `app/api/v1/followed_objects.py`)
  and weather-window scoring drive the session-prep flow with curated target
  recommendations matched to the user's location and forecast.
- **Profile presets & per-session overrides** — `quick`, `standard`, `quality`,
  `advanced` plus a fully custom profile editor. Every pipeline step is
  individually tunable.
- **Profile import / export** — round-trip JSON via `GET /profiles/{id}/export`
  and `POST /profiles/import`.
- **Profile sharing** — toggle a profile public with `PATCH /profiles/{id}/share`;
  shared profiles are discoverable across the community.
- **EXIF metadata aggregation** — capture metadata is parsed at ingestion and
  attached to every session (camera, ISO, exposure, focal length, integration).
- **Job profile snapshot** — the resolved profile is frozen alongside each job
  so historical runs remain reproducible even if the source profile changes.
- **Public gallery** — anonymous browsing with email-gated download requests;
  audit trail in `GalleryDownload`.
- **Multi-GPU workers** — distribute processing across multiple ARQ worker
  containers, one per GPU.
- **Real-time progress** — per-step WebSocket events (progress, log, status,
  error, completed, adaptive-critic iteration) bridged via a Redis pub/sub
  event bus.
- **File watcher** — auto-detects newly deposited sessions in the inbox and
  triggers ingestion after a configurable stability delay.
- **Published container image** — every push to `main`/tag builds and
  publishes `ghcr.io/bitsdiver/astro-stack` via GitHub Actions
  (`.github/workflows/docker-publish.yml`), so a deployment doesn't need a
  local build.
- **OAuth 2.1 / OIDC authentication** — three modes controlled by `AUTH_MODE`:
  `disabled` (dev default), `mock` (staging, HS256 + `X-Mock-User` header) and
  `oidc` (production). In `oidc` mode every request is validated against RS256
  JWTs issued by [AuthService](https://auth.astromote.com). The JWKS endpoint is
  cached with a configurable TTL and refreshed automatically on key rotation.
  WebSocket connections authenticate via a short-lived Redis ticket obtained from
  `POST /api/v1/auth/ws-ticket`.

---

## Architecture

```mermaid
flowchart TD
    UI(["AstroUI (React SPA)"])
    AUTH(["AuthService\nauth.astromote.com\nOIDC · OAuth 2.1 · PKCE"])
    TRAEFIK["Traefik (optional profile)\nTLS termination · HSTS · gzip\nports 80 / 443"]
    API["astro-api\nFastAPI · Uvicorn · Watchdog\nREST /api/v1/* · WS /ws/*"]
    REDIS[("Redis 7\nARQ job queue · pub/sub")]
    W0["astro-worker GPU 0\nARQ · pipeline orchestration"]
    W1["astro-worker GPU 1\nARQ · pipeline orchestration"]
    PG[("PostgreSQL\nsessions · jobs · steps · profiles")]
    VLLM["vLLM (optional, external)\nvision-LLM · adaptive critic"]

    UI -->|"HTTPS (direct or via Traefik)"| TRAEFIK
    TRAEFIK -->|reverse proxy| API
    UI -- "OIDC redirect + PKCE" --> AUTH
    AUTH -- "RS256 access token" --> UI
    API -- "JWKS fetch · RS256 validation" --> AUTH
    API -->|enqueue| REDIS
    REDIS -->|dequeue| W0
    REDIS -->|dequeue| W1
    W0 <-->|pub/sub events| REDIS
    W1 <-->|pub/sub events| REDIS
    API <-->|read / write| PG
    W0 <-->|read / write| PG
    W1 <-->|read / write| PG
    W0 -.->|critique opt-in| VLLM
    W1 -.->|critique opt-in| VLLM
```

### Pipeline steps

| #   | Step                          | Tool                            | Enabled by                    |
| --- | ----------------------------- | ------------------------------- | ----------------------------- |
| 0   | RAW → FITS conversion         | rawpy (LibRaw)                  | Auto if RAW files detected    |
| 1   | Calibration + Stacking        | Siril 1.4.x (headless)          | Always                        |
| 2   | Plate Solving                 | ASTAP                           | standard / quality / advanced |
| 3   | Background Gradient Removal   | GraXpert 3.x AI                 | standard / quality / advanced |
| 4   | Stretch + Colour Calibration  | Siril (PCC)                     | Always                        |
| 5   | AI Noise Reduction            | SASpro Denoise / GraXpert (CUDA)| All presets                   |
| 6   | AI Sharpening / Deconvolution | SASpro Sharpen (CUDA)           | standard / quality            |
| 7   | AI Super-Resolution 2x        | SASpro Super Resolution (CUDA) | quality / advanced            |
| 8   | Star Separation               | SASpro Dark Star (CUDA)         | quality / advanced            |
| 9   | Satellite Trail Removal       | SASpro Satellite (CUDA)         | Opt-in (all presets)          |
| 10  | Export                        | astropy + Pillow                | Always                        |

Steps 3, 4, 5, 6, 7 and 9 can each optionally be refined by the adaptive
vision critic (see below) instead of running once with static config.

**Outputs per job:** `final.fits`, `final.tiff`, `preview.jpg`, `thumbnail.png`.

---

## Adaptive AI Processing (optional, off by default)

AstroStack's core promise is a fully automated pipeline that needs no manual
tweaking. On top of that, an optional **adaptive vision-critic loop** can
iteratively refine a handful of steps using any OpenAI-compatible vision-LLM
endpoint (a self-hosted vLLM server is the verified target):

1. A step runs once with the profile's static config, producing a preview and
   numeric stats (from Siril, GraXpert or the export pipeline).
2. The critic receives the preview image, the stats, and the relevant slice
   of the tool capability catalogue (`app/pipeline/adaptive/tool_catalog.py`)
   describing exactly which parameters it may adjust, their role, effect and
   risk.
3. If unsatisfied, it proposes a small patch. The patch is validated and
   clamped against the catalogue (never trusted blindly), applied, and the
   step re-runs.
4. This repeats until the critic is satisfied or an iteration budget is
   reached (`adaptive_critic_max_iterations`, default 3).

**Always fully autonomous by default** — `adaptive_critic_enabled` and
`adaptive_critic_require_human_approval` both default to `False`. An optional
human-approval hook exists for advanced deployments, but if left unconfigured
it auto-approves with a logged warning rather than stalling a job. A critic
that is unreachable or returns a malformed response degrades to "accept the
current result" instead of failing the job.

The same mechanism also applies to **live stacking**
(`live_adaptive_critic_enabled`, also off by default): once enough frames
have accumulated (5 by default), the critic tunes the MTF autostretch
parameters, then freezes the result and reuses it for every subsequent frame
without calling the critic again — no GPU-bound tools are touched in live
mode, only the CPU-cheap stretch parameters.

Set `VLLM_BASE_URL` / `VLLM_MODEL` / `VLLM_API_KEY` to point at your
OpenAI-compatible endpoint (see `.env.example`). A self-hosted SearXNG
instance (`docker compose --profile searxng up -d`) is bundled in
anticipation of a planned reference-image search feature (see Roadmap) — not
consumed by the application yet.

---

## Quick Start

### With Docker Compose (recommended)

Standard deployment exposes `astro-api` and `astro-ui` directly — no reverse
proxy is started unless you ask for one. This is the recommended path when
you already run your own Traefik/nginx/Caddy in front of the host.

```bash
# 1. Clone and copy env file
cp .env.example .env
# Edit .env: DATABASE_URL, REDIS_URL, OLLAMA_URL, etc.

# 2. Create data directories
mkdir -p data/inbox data/sessions data/output data/models

# 3. Download AI models (one-time, ~2–4 GB)
docker compose --profile init up astro-init-models

# 4. Run database migrations
docker compose run --rm astro-api alembic upgrade head

# 5. Start all services
docker compose up -d
```

| Endpoint | URL |
|---|---|
| API | `http://localhost:8080` |
| OpenAPI docs | `http://localhost:8080/docs` |
| Health | `http://localhost:8080/health` |
| Web UI | `http://localhost:3000` |

#### Use the published image instead of building locally

Every push to `main`/tag publishes `ghcr.io/bitsdiver/astro-stack` via
GitHub Actions ([.github/workflows/docker-publish.yml](./.github/workflows/docker-publish.yml)).
Swap the `build:` block for `astro-api`/`astro-worker-*` in `docker-compose.yml`
for:

```yaml
image: ghcr.io/bitsdiver/astro-stack:latest
```

#### Optional: bundled Traefik reverse proxy

If you don't already run a reverse proxy, enable the bundled Traefik profile
for TLS termination (Let's Encrypt), HTTP→HTTPS redirects, security headers
(HSTS, X-Frame-Options, X-Content-Type-Options) and gzip compression:

```bash
export TRAEFIK_HOST=astro.mydomain.com
export TRAEFIK_ACME_EMAIL=admin@mydomain.com
docker compose --profile traefik up -d
```

#### Optional: SearXNG (anticipates the reference-image search feature)

```bash
docker compose --profile searxng up -d
```

---

## Configuration

See `.env.example` for the full list. Key variables:

| Variable                  | Default                    | Description                                 |
| ------------------------- | -------------------------- | ------------------------------------------- |
| `DATABASE_URL`            | `postgresql+asyncpg://...` | PostgreSQL DSN                              |
| `REDIS_URL`               | `redis://redis:6379/0`     | Redis DSN                                   |
| `OLLAMA_URL`              | `http://ollama:11434`      | Optional Ollama API base URL                |
| `INBOX_PATH`              | `/inbox`                   | Session inbox directory                     |
| `MODELS_PATH`             | `/models`                  | AI model weights directory                  |
| `PIPELINE_MAX_RETRIES`    | `3`                        | Default max retry count per step            |
| `SESSION_STABILITY_DELAY` | `30.0`                     | Seconds before a session is considered stable |
| `AUTH_MODE`               | `disabled`                 | Auth mode: `disabled`, `mock`, or `oidc`    |
| `OIDC_ISSUER`             | *(required in oidc mode)*  | OIDC issuer URL (e.g. `https://auth.astromote.com`) |
| `OIDC_AUDIENCE`           | `astrostack`               | Expected JWT `aud` claim                    |
| `OIDC_JWKS_CACHE_TTL_SECONDS` | `300`                  | JWKS public-key cache TTL in seconds        |
| `CORS_ALLOWED_ORIGINS`    | `http://localhost:5173`    | Comma-separated list of allowed CORS origins |
| `VLLM_BASE_URL`           | `http://vllm:8000/v1`      | OpenAI-compatible endpoint for the adaptive critic |
| `VLLM_MODEL`              | `lagarde-vllm`              | Model name requested from the vLLM server   |
| `VLLM_API_KEY`            | *(empty)*                  | Bearer token, if your endpoint requires one |

---

## Usage

### Deposit a session

Drop frames into `./data/inbox/` with this layout — the watcher picks them up
after `SESSION_STABILITY_DELAY` seconds of inactivity:

```
data/inbox/
  2024-03-15_M42/
    darks/        ← dark calibration frames (.fits or .cr2/.nef/...)
    flats/        ← flat calibration frames
    bias/         ← bias frames (optional)
    lights/       ← science frames
```

### Start processing

```bash
# List detected sessions
curl http://localhost:8080/api/v1/sessions

# Start the pipeline with a preset
curl -X POST "http://localhost:8080/api/v1/sessions/{session_id}/process?preset=standard"
```

### Save a custom profile

```bash
curl -X POST http://localhost:8080/api/v1/profiles \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Nebula Profile",
    "description": "Optimised for emission nebulae",
    "config": {
      "rejection_algorithm": "winsorized",
      "drizzle_enabled": true,
      "denoise_strength": 0.85,
      "sharpen_stellar_amount": 0.4,
      "sharpen_nonstellar_amount": 0.9,
      "star_separation_enabled": true
    }
  }'
```

### Share or import a profile

```bash
# Mark a profile public
curl -X PATCH http://localhost:8080/api/v1/profiles/{id}/share \
  -H "Content-Type: application/json" -d '{"is_shared": true}'

# Export to JSON
curl http://localhost:8080/api/v1/profiles/{id}/export -o my-profile.json

# Import from JSON
curl -X POST http://localhost:8080/api/v1/profiles/import \
  -F "file=@my-profile.json"
```

### Monitor progress over WebSocket

```javascript
const ws = new WebSocket("ws://localhost:8080/ws/jobs/{job_id}");
ws.onmessage = (e) => {
  const event = JSON.parse(e.data);
  // event.type: "progress" | "log" | "step_status" | "error" | "completed"
};
```

When `AUTH_MODE=oidc`, pass the access token as `Authorization: Bearer <token>`
on REST calls. For WebSocket connections, first obtain a short-lived ticket with
`POST /api/v1/auth/ws-ticket` (authenticated REST call), then append it as
`?ticket=<ticket>` to the WebSocket URL.

---

## Roadmap

The following items are planned but not yet implemented. They are listed in
priority order; the order may change based on feedback.

1. **Reference-image search** for the adaptive critic — opt-in, disabled by
   default, backed by the bundled SearXNG service; results are shown to the
   critic as stylistic/structural context only, never stored in exports.
2. **Planet-dedicated processing pipeline** with lucky-imaging support.
3. **Observation time-slot suggestions** after selecting a celestial object and
   a location.
4. **AI-driven session scheduling** and observation recommendations based on
   weather forecast, location, and target.
5. **Pipeline tools and steps exposed as MCP servers** so external agents can
   compose them.
6. **Observation alerts** (cancel reminders for cloudy nights, favourite-target
   visibility windows, etc.).

> Auto-selecting and auto-improving pipeline configuration via an agent
> workflow is no longer purely aspirational — see
> [Adaptive AI Processing](#adaptive-ai-processing-optional-off-by-default) above.

---

## Development

See [DEVELOPMENT.md](./DEVELOPMENT.md) for the full development guide
(architecture, local setup, testing, migrations, common workflows) and the
shared coding principles that apply to every change.

Quick reference:

```bash
pip install -e ".[dev]"

# Run the API locally (requires Postgres + Redis)
export DATABASE_URL="postgresql+asyncpg://astro:astro@localhost:5432/astrostack"
export REDIS_URL="redis://localhost:6379/0"
uvicorn app.main:app --reload --port 8080

# Run a worker
python -m arq app.workers.settings.WorkerSettings

# Quality gates
ruff check .
ruff format .
pytest
```

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the workflow, commit-message
conventions, and review process.

---

## License

| Component                     | License |
| ------------------------------ | ------- |
| AstroStack (this project)      | GPL-3.0-or-later |
| Siril                          | GPLv3   |
| GraXpert                       | GPLv3   |
| ASTAP                          | GPL     |
| SetiAstroSuitePro (SASpro)     | GPL-3.0 |
| LangGraph                      | MIT     |
| rawpy / LibRaw                 | LGPL    |
| astropy                        | BSD     |
| FastAPI                        | MIT     |
| vLLM (optional, external)      | Apache-2.0 |
