# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — dependency resolver (UV)
# ─────────────────────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS deps

WORKDIR /app

# Copy only dependency files first to maximise layer cache hits
COPY pyproject.toml uv.lock* ./

# Install production dependencies into a dedicated virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy the pre-built virtual environment from the deps stage
COPY --from=deps /app/.venv /app/.venv

# Copy application source
COPY app/ ./app/

# Make the venv the default python environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
