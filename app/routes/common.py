from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm import embeddings, llm_chat

common_route = APIRouter(tags=["common"])
logger = get_logger(__name__)
settings = get_settings()


@common_route.get("/health")
async def health_check() -> dict:
    logger.debug("Health check requested")
    llm_ok, llm_message = llm_chat.health_check()
    embeddings_ok, embeddings_message = embeddings.health_check()

    overall_ok = llm_ok and embeddings_ok
    return {
        "status": "ok" if overall_ok else "degraded",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.ENV,
        "llm": {
            "status": "ok" if llm_ok else "error",
            "response": llm_message if llm_ok else "",
            "error": "" if llm_ok else llm_message,
        },
        "embeddings": {
            "status": "ok" if embeddings_ok else "error",
            "response": embeddings_message if embeddings_ok else "",
            "error": "" if embeddings_ok else embeddings_message,
        },
    }
