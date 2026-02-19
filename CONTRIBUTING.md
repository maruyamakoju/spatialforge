# Contributing to SpatialForge

Thank you for your interest in contributing to SpatialForge. This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11+
- Redis (for auth, rate limiting, and job queue)
- Git

### Local Installation

```bash
# Clone the repository
git clone https://github.com/maruyamakoju/spatialforge.git
cd spatialforge

# Install in development mode
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env
```

### Running Locally

```bash
# Start the API server
uvicorn spatialforge.main:create_app --factory --reload --host 0.0.0.0 --port 8000

# Or use the CLI entry point
spatialforge
```

### SDK Development

```bash
cd sdk
pip install -e ".[dev]"
```

## Testing

We maintain 139+ tests across server and SDK. All tests must pass before merging.

```bash
# Server tests (62 tests)
pytest tests/ -v

# SDK tests (77 tests)
cd sdk && pytest tests/ -v

# Lint
ruff check --config pyproject.toml spatialforge/ tests/

# Run everything
pytest tests/ -v && cd sdk && pytest tests/ -v && cd .. && ruff check --config pyproject.toml spatialforge/ tests/
```

### Writing Tests

- Place server tests in `tests/`
- Place SDK tests in `sdk/tests/`
- Use pytest fixtures from `conftest.py` (e.g., `test_client`, `override_auth`)
- Test fixtures use `dependency_overrides` — no live Redis or GPU needed
- Aim for clear, descriptive test names: `test_depth_returns_valid_metadata`

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check
ruff check --config pyproject.toml spatialforge/ tests/

# Auto-fix
ruff check --fix --config pyproject.toml spatialforge/ tests/
```

Key conventions:
- **Type hints** on all public function signatures
- **Docstrings** on all public modules, classes, and functions
- **`from __future__ import annotations`** at the top of every module
- No unused imports (ruff F401)
- Use `str | Path` not `isinstance(x, (str, Path))` (ruff UP038)
- Use `TimeoutError` not `asyncio.TimeoutError` (ruff UP041)

## Project Structure

```
spatialforge/
  api/v1/          # FastAPI route handlers
  auth/            # API key management + rate limiter
  billing/         # Stripe integration
  inference/       # Depth, pose, measure engines
  middleware/      # Security headers + request timeout
  models/          # Pydantic request/response models
  storage/         # MinIO object store
  utils/           # Image and video utilities
  workers/         # Celery async tasks
  config.py        # Environment-based configuration
  main.py          # App entry point
  metrics.py       # Prometheus metrics
sdk/               # Python SDK (spatialforge-client)
tests/             # Server test suite
site/              # Landing page, docs, demo
```

## Pull Request Process

1. **Fork and branch** — Create a feature branch from `master`
2. **Make changes** — Keep PRs focused on a single concern
3. **Add tests** — All new features and bug fixes should include tests
4. **Run checks** — Ensure all tests pass and lint is clean
5. **Write a clear PR description** — Explain what changed and why
6. **CI must pass** — Tests, lint, and Docker build are required

### PR Title Convention

Use clear, descriptive titles:
- `Add batch depth estimation endpoint`
- `Fix timeout handling for large video uploads`
- `Update SDK retry logic for rate-limited responses`

## Reporting Issues

When filing an issue, please include:

- **Description** of the problem or feature request
- **Steps to reproduce** (for bugs)
- **Expected vs. actual behavior**
- **Environment** (Python version, OS, deployment type)
- **Error messages** or logs if applicable

## Architecture Notes

### Request Flow

```
Client → CORS → SecurityHeaders → Timeout → RateLimit → Metrics → Route Handler
```

### Middleware Order (in `main.py`)

Middleware is applied in reverse order (last added = first executed):
1. CORS (outermost)
2. SecurityHeadersMiddleware
3. RequestTimeoutMiddleware
4. Rate limiting (in auth dependency)
5. MetricsMiddleware
6. Tracing

### Async Endpoints

Endpoints like `/reconstruct`, `/floorplan`, and `/segment-3d` use Celery tasks:
1. Upload video to MinIO
2. Submit Celery task
3. Return `job_id` immediately
4. Client polls `GET /endpoint/{job_id}` for results

## Deployment

### Stripe Billing (Fly.io)

The app runs without Stripe by default (billing endpoints return 503). To enable:

```bash
# Set secrets on Fly.io (never commit these to the repo)
flyctl secrets set \
  STRIPE_SECRET_KEY=sk_live_... \
  STRIPE_WEBHOOK_SECRET=whsec_...

# Verify secrets are set
flyctl secrets list
```

To get the webhook secret:
1. Go to Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://spatialforge-demo.fly.dev/billing/webhooks`
3. Listen for: `checkout.session.completed`, `customer.subscription.*`, `invoice.*`
4. Copy the signing secret → set as `STRIPE_WEBHOOK_SECRET`

### SDK — Publishing to PyPI

The `publish-sdk.yml` workflow uses OIDC trusted publishers (no API keys needed).
One-time setup on PyPI.org:

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new pending publisher:
   - **PyPI project name**: `spatialforge-client`
   - **GitHub owner**: `maruyamakoju`
   - **GitHub repo**: `spatialforge`
   - **Workflow filename**: `publish-sdk.yml`
   - **Environment**: `pypi`
3. Do the same on https://test.pypi.org with environment `testpypi`
4. Create a GitHub release tagged `sdk-v0.1.0` — the workflow triggers automatically

### Fly.io — Removing the 5-Minute Sleep Limit

The free trial auto-stops machines after 5 minutes of inactivity.
To remove this limit: add a credit card at https://fly.io/dashboard/billing

After adding a card, optionally set `min_machines_running = 1` in `fly.toml`
to keep the machine always warm (eliminates cold-start delay for demos).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
