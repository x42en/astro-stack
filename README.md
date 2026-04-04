# Astro-Stack

Automated astrophotography processing pipeline with an event-driven API backend.

Watches for incoming RAW/FITS astrophotography sessions, automatically runs a configurable processing pipeline (calibration → stacking → plate solving → AI gradient removal → AI denoise → AI sharpening → export), and delivers real-time progress updates via WebSocket.

## Requirements

- Docker + Docker Compose with NVIDIA Container Runtime
- 2× Nvidia RTX 4090 (or any CUDA-capable GPU)
- A running PostgreSQL instance
- Optionally: a running Ollama instance for AI metadata enrichment

## Quick Start

```bash
# 1. Clone and copy env file
cp .env.example .env
# Edit .env: set DATABASE_URL, REDIS_URL, OLLAMA_URL, etc.

# 2. Create data directories
mkdir -p data/inbox data/sessions data/output data/models

# 3. Download AI models (one-time setup, ~2–4 GB)
docker compose --profile init up astro-init-models

# 4. Run database migrations
docker compose run --rm astro-api alembic upgrade head

# 5. Start all services
docker compose up -d

# API available at:   http://localhost:8080 (direct) or http://localhost (via Traefik)
# OpenAPI docs:       http://localhost:8080/docs
# Health check:       http://localhost:8080/health
# Traefik dashboard:  http://localhost:8081 (if enabled)
```

## Reverse Proxy (Traefik)

A Traefik reverse proxy is included in the Docker stack to handle:

- **TLS termination** with automatic Let's Encrypt certificate provisioning
- **HTTP → HTTPS redirect** (enabled by default)
- **Security headers**: HSTS, X-Frame-Options, X-Content-Type-Options
- **Gzip compression** for API responses
- **Docker service discovery** via labels

### Configuration

The Traefik configuration is in:
- `traefik/traefik.yml` — main configuration (entrypoints, TLS, Let's Encrypt)
- `traefik/dynamic/middleware.yml` — security headers and compression

### Exposed endpoints

| Service          | Internal Port | External URL                    |
| ---------------- | ------------- | ------------------------------  |
| astro-api        | 8080          | `http://localhost/` (Traefik)  |
| Traefik dashboard| 8080          | `http://localhost:8081/`      |

### Using with a custom domain

Set the `TRAEFIK_HOST` environment variable:

```bash
export TRAEFIK_HOST=astro.mydomain.com
export TRAEFIK_ACME_EMAIL=admin@mydomain.com
docker compose up -d
```

Traefik will automatically provision a Let's Encrypt certificate for your domain.

## Usage

### Deposit a session

Drop your astrophotography session into `./data/inbox/` with this structure:

```
data/inbox/
  2024-03-15_M42/
    darks/        ← dark calibration frames (.fits or .cr2/.nef/etc.)
    flats/        ← flat calibration frames
    bias/         ← bias frames (optional)
    lights/       ← science frames (brutes)
```

The watcher detects the session after 30 seconds of inactivity and triggers automatic ingestion.

### Start processing via API

```bash
# List detected sessions
curl http://localhost:8080/api/v1/sessions

# Start pipeline with standard preset
curl -X POST "http://localhost:8080/api/v1/sessions/{session_id}/process?preset=standard"

# Start with quality preset (drizzle + full AI suite)
curl -X POST "http://localhost:8080/api/v1/sessions/{session_id}/process?preset=quality"
```

### Monitor progress via WebSocket

```javascript
const ws = new WebSocket("ws://localhost:8080/ws/jobs/{job_id}");
ws.onmessage = (e) => {
  const event = JSON.parse(e.data);
  // event.type: "progress" | "log" | "step_status" | "error" | "completed"
  console.log(event);
};
```

### Authentication (when enabled)

Set `AUTH_ENABLED=true` in `.env` and provide a valid JWT:

```
# REST:       Authorization: Bearer <token>
# WebSocket:  ws://host/ws/jobs/{id}?token=<token>
```

## Processing Profiles

| Profile    | Speed  | Drizzle  | AI Suite              | Use Case                |
| ---------- | ------ | -------- | --------------------- | ----------------------- |
| `quick`    | Fast   | No       | Denoise only          | Preview / initial check |
| `standard` | Medium | No       | Denoise + Sharpen     | Most sessions           |
| `quality`  | Slow   | Yes (2×) | Full suite + SuperRes | Exhibition / print      |
| `advanced` | Custom | Custom   | Custom                | Per-session fine-tuning |

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

## Pipeline Steps

| #   | Step                          | Tool                            | Enabled by                    |
| --- | ----------------------------- | ------------------------------- | ----------------------------- |
| 0   | RAW → FITS conversion         | rawpy (LibRaw)                  | Auto if RAW files detected    |
| 1   | Calibration + Stacking        | Siril 1.4.2 headless            | Always                        |
| 2   | Plate Solving                 | ASTAP                           | standard / quality / advanced |
| 3   | Background Gradient Removal   | GraXpert 3.x AI                 | standard / quality / advanced |
| 4   | Stretch + Colour Calibration  | Siril (PCC)                     | Always                        |
| 5   | AI Noise Reduction            | Cosmic Clarity Denoise (CUDA)   | All presets                   |
| 6   | AI Sharpening / Deconvolution | Cosmic Clarity Sharpen (CUDA)   | standard / quality            |
| 7   | AI Super-Resolution 2×        | Cosmic Clarity SuperRes (CUDA)  | quality / advanced            |
| 8   | Star Separation               | Cosmic Clarity Dark Star (CUDA) | quality / advanced            |
| 9   | Export                        | astropy + Pillow                | Always                        |

**Outputs per job:**

- `final.fits` — 32-bit float FITS (scientific archive)
- `final.tiff` — 16-bit TIFF (for manual editing in e.g. GIMP, Darktable)
- `preview.jpg` — JPEG high quality preview
- `thumbnail.png` — 800px PNG for the UI

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Traefik Reverse Proxy (ports 80/443)                 │
│    - Let's Encrypt TLS auto-configuration             │
│    - Security headers (HSTS, X-Frame-Options)         │
│    - Gzip compression                                 │
└────────────────────────┬─────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────┐
│  astro-api (FastAPI + Uvicorn + Watchdog)             │
│    - REST API  /api/v1/                               │
│    - WebSocket /ws/jobs/{id}, /ws/sessions/{id}       │
│    - File watcher → session auto-detection            │
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
              │    PostgreSQL        │
              │  (sessions/jobs/     │
              │   steps/profiles)    │
              └─────────────────────┘
```

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint + format
ruff check .
ruff format .

# Run locally (without Docker)
export DATABASE_URL="postgresql+asyncpg://astro:astro@localhost:5432/astrostack"
export REDIS_URL="redis://localhost:6379/0"
uvicorn app.main:app --reload --port 8080

# Run worker
python -m arq app.workers.settings.WorkerSettings
```

## Environment Variables

See `.env.example` for the full list of configurable variables.

Key variables:

| Variable                  | Default                    | Description                                 |
| ------------------------- | -------------------------- | ------------------------------------------- |
| `AUTH_ENABLED`            | `false`                    | Enable JWT authentication                   |
| `JWT_SECRET`              | (required if auth enabled) | HMAC signing secret                         |
| `DATABASE_URL`            | `postgresql+asyncpg://...` | PostgreSQL DSN                              |
| `REDIS_URL`               | `redis://redis:6379/0`     | Redis DSN                                   |
| `OLLAMA_URL`              | `http://ollama:11434`      | Ollama API base URL                         |
| `INBOX_PATH`              | `/inbox`                   | Session inbox directory                     |
| `MODELS_PATH`             | `/models`                  | AI model weights directory                  |
| `PIPELINE_MAX_RETRIES`    | `3`                        | Default max retry count per step            |
| `SESSION_STABILITY_DELAY` | `30.0`                     | Seconds before session is considered stable |

## Licences

| Component                  | Licence |
| -------------------------- | ------- |
| Astro-Stack (this project) | MIT     |
| Siril                      | GPLv3   |
| GraXpert                   | GPLv3   |
| ASTAP                      | GPL     |
| Cosmic Clarity (SetiAstro) | MIT     |
| rawpy / LibRaw             | LGPL    |
| astropy                    | BSD     |
| FastAPI                    | MIT     |
