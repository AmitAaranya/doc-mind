from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm import llm_chat

common_route = APIRouter(tags=["common"])
logger = get_logger(__name__)
settings = get_settings()


@common_route.get("/health")
async def health_check() -> dict:
    logger.debug("Health check requested")
    llm_ok, llm_message = llm_chat.health_check()
    return {
        "status": "ok" if llm_ok else "degraded",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.ENV,
        "llm": {
            "status": "ok" if llm_ok else "error",
            "response": llm_message if llm_ok else "",
            "error": "" if llm_ok else llm_message,
        },
    }


@common_route.get("/ready")
async def readiness_check() -> dict:
    logger.debug("Readiness check requested")
    return {"status": "ready"}
