"""FastAPI application for Jetson LLM API."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .config import settings
from .system_monitor import get_system_stats
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelList,
    Model,
    HealthResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from .clients import (
    call_llama_server,
    stream_llama_server,
    call_llama_server_vision,
    stream_llama_server_vision,
    check_backend_health,
    call_llama_server_embeddings,
)
from .deps import verify_api_key
from .routing import get_available_models, is_vision_model, detect_vision_content
from .activity_logger import ActivityLoggerMiddleware, activity_log


# Configure logging
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Jetson LLM API")
    logger.info(f"DeepSeek backend: {settings.deepseek_base_url}")
    logger.info(f"Qwen backend: {settings.qwen_base_url}")
    yield
    logger.info("Shutting down Jetson LLM API")


app = FastAPI(
    title="Jetson LLM API",
    description="OpenAI-compatible API for local LLM inference on NVIDIA Jetson",
    version="1.0.0",
    lifespan=lifespan,
)

# Add activity logging middleware
app.add_middleware(ActivityLoggerMiddleware)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard HTML page."""
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    else:
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>Jetson LLM API</h1>
                    <p>Dashboard not found. API is running at /v1/chat/completions</p>
                    <p>Visit /docs for API documentation</p>
                </body>
            </html>
            """,
            status_code=200,
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Checks connectivity to all backend llama-server instances.
    """
    models_health = {}
    available_models = get_available_models()

    for model_id, model_info in available_models.items():
        base_url = model_info["base_url"]
        health = await check_backend_health(base_url)
        models_health[model_id] = {
            "base_url": base_url,
            **health,
        }

    # Determine overall status
    all_healthy = all(
        m["status"] == "healthy" for m in models_health.values()
    )
    overall_status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        models=models_health,
    )


@app.get("/v1/models", response_model=ModelList)
async def list_models(_: None = Depends(verify_api_key)):
    """
    List available models.
    OpenAI-compatible endpoint.
    """
    available_models = get_available_models()
    models = [
        Model(
            id=model_id,
            object="model",
            created=0,
            owned_by="jetson-api",
        )
        for model_id in available_models.keys()
    ]

    return ModelList(
        object="list",
        data=models,
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _: None = Depends(verify_api_key),
):
    """
    Unified chat completion endpoint.
    Supports both text-only and vision models with auto-detection.
    """
    # Validate model exists
    available_models = get_available_models()
    if request.model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available models: {list(available_models.keys())}",
        )

    # Detect vision model or vision content
    model_supports_vision = is_vision_model(request.model)
    has_vision_content = detect_vision_content(request.messages)

    # Validate compatibility
    if has_vision_content and not model_supports_vision:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' does not support vision/image inputs. "
                   f"Please use a vision-capable model like 'minicpm-v-2.5'.",
        )

    try:
        # Route to appropriate handler based on model capabilities
        if model_supports_vision:
            # Use vision-specific handlers
            if request.stream:
                return EventSourceResponse(
                    stream_llama_server_vision(request),
                    media_type="text/event-stream",
                )
            else:
                response = await call_llama_server_vision(request)
                return response
        else:
            # Use text-only handlers
            if request.stream:
                return EventSourceResponse(
                    stream_llama_server(request),
                    media_type="text/event-stream",
                )
            else:
                response = await call_llama_server(request)
                return response

    except Exception as e:
        logger.error(f"Error processing completion: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error communicating with backend: {str(e)}",
        )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    _: None = Depends(verify_api_key),
):
    """
    Create embeddings for input text.
    OpenAI-compatible endpoint.
    """
    try:
        response = await call_llama_server_embeddings(request)
        return response
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embeddings: {str(e)}",
        )


@app.get("/system/stats")
async def system_stats():
    """
    Get system statistics (GPU, CPU, RAM, disk, temperatures).
    No authentication required for monitoring endpoint.
    """
    try:
        stats = get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting system stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving system stats: {str(e)}",
        )


@app.get("/api/activity-logs")
async def get_activity_logs(limit: int = 100):
    """
    Get recent API activity logs.
    No authentication required for monitoring endpoint.

    Args:
        limit: Maximum number of logs to return (default: 100, max: 200)
    """
    try:
        # Clamp limit to max 200
        limit = min(limit, 200)
        logs = activity_log.get_logs(limit=limit)
        return {
            "logs": logs,
            "count": len(logs),
        }
    except Exception as e:
        logger.error(f"Error retrieving activity logs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving activity logs: {str(e)}",
        )


@app.post("/api/activity-logs/clear")
async def clear_activity_logs(_: None = Depends(verify_api_key)):
    """
    Clear all activity logs.
    Requires authentication.
    """
    try:
        activity_log.clear_logs()
        return {"status": "success", "message": "Activity logs cleared"}
    except Exception as e:
        logger.error(f"Error clearing activity logs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing activity logs: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )
