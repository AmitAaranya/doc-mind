# doc-mind

Intelligent Document QA API built with FastAPI.

## Prerequisites

- Python 3.12+
- `curl`
- Docker (optional)

## Installation (UV)

1. Install UV:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Ensure UV is on your PATH (current shell):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

3. Install dependencies from the project root:

```bash
uv sync
```

4. Create environment file:

```bash
cp .env.example .env
```

## Run Locally

Use UV to run the app with Uvicorn:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

App URLs:

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Readiness: `http://localhost:8000/ready`

## Run with Docker

Build image:

```bash
docker build -t doc-mind:local .
```

Run container:

```bash
docker run --rm -p 8000:8000 --env-file .env doc-mind:local
```

## Development Notes

- Project dependencies are managed via `pyproject.toml` + `uv.lock`.
- Route registration is done in `app/main.py` by including `common_route` from `app/routes`.
