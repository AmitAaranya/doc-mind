from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging import get_logger

common_route = APIRouter(tags=["common"])
logger = get_logger(__name__)
settings = get_settings()


@common_route.get("/health")
async def health_check() -> dict:
    logger.debug("Health check requested")
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.ENV,
    }


@common_route.get("/ready")
async def readiness_check() -> dict:
    logger.debug("Readiness check requested")
    return {"status": "ready"}
