FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy only dependency files first to maximise layer cache hits
COPY pyproject.toml uv.lock* ./

# Install production dependencies into a dedicated virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY --chown=appuser:appgroup app/ ./app/

# Make the venv the default python environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
