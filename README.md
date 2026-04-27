# AstroStack

> Automated astrophotography processing pipeline with an event-driven API backend.

AstroStack watches an inbox for incoming RAW/FITS sessions, runs them through a
configurable, GPU-accelerated processing pipeline (calibration → stacking →
plate solving → AI gradient removal → AI denoise → AI sharpening → super
resolution → star separation → export), and streams real-time progress over
WebSocket. It ships with a community gallery and shareable processing profiles
so astronomers can publish results and reuse one another's recipes.

The companion web interface lives in the [AstroUI](https://github.com/x42en/astro-stack-ui)
repository.

---

## Overview

| Component | Stack |
|---|---|
| API & orchestration | Python 3.12, FastAPI, Uvicorn, SQLModel |
| Job queue | Redis + ARQ |
| Database | PostgreSQL (asyncpg), Alembic migrations |
| Pipeline tooling | Siril 1.4 (headless), ASTAP, GraXpert, Cosmic Clarity (CUDA) |
| RAW handling | rawpy / LibRaw, ExifRead |
| Real-time events | WebSocket + Redis pub/sub |
| Container runtime | NVIDIA CUDA 12.6.3 + cuDNN base image |

---

## Features

- **End-to-end pipeline** — 10 verified steps from RAW conversion to export
  (`app/pipeline/steps/`).
- **Profile presets** — `quick`, `standard`, `quality`, `advanced` plus full
  custom profile support.
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
  error, completed) bridged via a Redis pub/sub event bus.
- **File watcher** — auto-detects newly deposited sessions in the inbox and
  triggers ingestion after a configurable stability delay.
- **Optional JWT authentication** — disabled by default; activate with
  `AUTH_ENABLED=true`.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Traefik Reverse Proxy (ports 80/443)                │
│    - Let's Encrypt TLS auto-configuration            │
│    - Security headers (HSTS, X-Frame-Options)        │
│    - Gzip compression                                │
└────────────────────────┬─────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────┐
│  astro-api (FastAPI + Uvicorn + Watchdog)            │
│    - REST API   /api/v1/*                            │
│    - WebSocket  /ws/jobs/{id}, /ws/sessions/{id}     │
│    - File watcher → session auto-detection           │
└────────────────────────┬─────────────────────────────┘
                         │ enqueue
                    ┌────▼──────┐
                    │  Redis 7  │ ← ARQ queue + pub/sub event bus
                    └────┬──────┘
              ┌──────────┴──────────┐
    ┌─────────▼──────┐    ┌─────────▼──────┐
    │ astro-worker   │    │ astro-worker   │
    │   GPU 0        │    │   GPU 1        │
    │  (ARQ + orch.) │    │  (ARQ + orch.) │
    └────────────────┘    └────────────────┘
                         │
              ┌──────────▼──────────┐
              │     PostgreSQL      │
              │  sessions / jobs /  │
              │  steps / profiles   │
              └─────────────────────┘
```

### Pipeline steps

| #   | Step                          | Tool                            | Enabled by                    |
| --- | ----------------------------- | ------------------------------- | ----------------------------- |
| 0   | RAW → FITS conversion         | rawpy (LibRaw)                  | Auto if RAW files detected    |
| 1   | Calibration + Stacking        | Siril 1.4 (headless)            | Always                        |
| 2   | Plate Solving                 | ASTAP                           | standard / quality / advanced |
| 3   | Background Gradient Removal   | GraXpert 3.x AI                 | standard / quality / advanced |
| 4   | Stretch + Colour Calibration  | Siril (PCC)                     | Always                        |
| 5   | AI Noise Reduction            | Cosmic Clarity Denoise (CUDA)   | All presets                   |
| 6   | AI Sharpening / Deconvolution | Cosmic Clarity Sharpen (CUDA)   | standard / quality            |
| 7   | AI Super-Resolution 2×        | Cosmic Clarity SuperRes (CUDA)  | quality / advanced            |
| 8   | Star Separation               | Cosmic Clarity Dark Star (CUDA) | quality / advanced            |
| 9   | Export                        | astropy + Pillow                | Always                        |

**Outputs per job:** `final.fits`, `final.tiff`, `preview.jpg`, `thumbnail.png`.

---

## Quick Start

### With Docker Compose (recommended)

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
| Traefik | `http://localhost/` |
| Traefik dashboard | `http://localhost:8081/` |

### Reverse proxy (Traefik)

The bundled Traefik handles TLS termination (Let's Encrypt), HTTP→HTTPS
redirects, security headers (HSTS, X-Frame-Options, X-Content-Type-Options),
and gzip compression. Configure a custom domain with:

```bash
export TRAEFIK_HOST=astro.mydomain.com
export TRAEFIK_ACME_EMAIL=admin@mydomain.com
docker compose up -d
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
| `AUTH_ENABLED`            | `false`                    | Enable JWT authentication                   |
| `JWT_SECRET`              | (required if auth enabled) | HMAC signing secret                         |

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

When `AUTH_ENABLED=true`, pass the JWT as `Authorization: Bearer <token>`
on REST calls and as `?token=<token>` on the WebSocket URL.

---

## Roadmap

The following items are planned but not yet implemented. They are listed in
priority order; the order may change based on feedback.

1. **Authentication** via [auth-service](https://github.com/circle-rd/auth-service).
2. **Planet-dedicated processing pipeline** with lucky-imaging support.
3. **Observation time-slot suggestions** after selecting a celestial object and
   a location.
4. **AI-driven session scheduling** and observation recommendations based on
   weather forecast, location, and target.
5. **Pipeline tools and steps exposed as MCP servers** so external agents can
   compose them.
6. **AI-driven pipeline auto-selection and auto-improve** through agent
   workflows.
7. **Observation alerts** (cancel reminders for cloudy nights, favourite-target
   visibility windows, etc.).

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

| Component                  | License |
| -------------------------- | ------- |
| AstroStack (this project)  | MIT     |
| Siril                      | GPLv3   |
| GraXpert                   | GPLv3   |
| ASTAP                      | GPL     |
| Cosmic Clarity (SetiAstro) | MIT     |
| rawpy / LibRaw             | LGPL    |
| astropy                    | BSD     |
| FastAPI                    | MIT     |
