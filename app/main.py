"""FastAPI application for Jetson LLM API."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
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

    except httpx.ReadTimeout:
        logger.error(f"Timeout waiting for completion from {request.model} (max_tokens={request.max_tokens})")
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out waiting for model response. The model may be taking longer than expected to generate {request.max_tokens or 'the requested'} tokens. Try reducing max_tokens or using streaming mode.",
        )
    except httpx.ConnectTimeout:
        logger.error(f"Connection timeout to {request.model} backend")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to model backend. The backend server may be down or overloaded.",
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"Backend returned error {e.response.status_code}: {e.response.text}")
        raise HTTPException(
            status_code=502,
            detail=f"Backend server error: {e.response.status_code} - {e.response.text[:200]}",
        )
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
    except httpx.ReadTimeout:
        input_count = len(request.input) if isinstance(request.input, list) else 1
        logger.error(f"Timeout generating embeddings for {input_count} input(s)")
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out generating embeddings for {input_count} input(s). The server may be at capacity. Check /api/queue-status to see current load.",
        )
    except httpx.ConnectTimeout:
        logger.error(f"Connection timeout to embeddings backend")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to embeddings backend. The backend server may be down.",
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"Embeddings backend returned error {e.response.status_code}: {e.response.text}")
        raise HTTPException(
            status_code=502,
            detail=f"Embeddings backend error: {e.response.status_code} - {e.response.text[:200]}",
        )
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


@app.get("/api/queue-status")
async def get_queue_status():
    """
    Get queue status for the embeddings backend.
    Shows slot utilization and processing status.
    No authentication required for monitoring endpoint.
    """
    try:
        from .config import settings

        base_url = str(settings.qwen3_embedding_base_url).rstrip('/')
        slots_url = f"{base_url}/slots"

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(slots_url)
            response.raise_for_status()
            slots_data = response.json()

        # Parse slot information
        total_slots = len(slots_data)
        busy_slots = sum(1 for slot in slots_data if slot.get("is_processing", False))
        idle_slots = total_slots - busy_slots

        # Get active task IDs
        active_tasks = [
            slot.get("id_task")
            for slot in slots_data
            if slot.get("is_processing", False)
        ]

        return {
            "backend": "qwen3-embedding-8b",
            "backend_url": base_url,
            "total_slots": total_slots,
            "busy_slots": busy_slots,
            "idle_slots": idle_slots,
            "utilization_percent": round((busy_slots / total_slots * 100) if total_slots > 0 else 0, 1),
            "at_capacity": busy_slots >= total_slots,
            "active_tasks": active_tasks,
            "slots": slots_data,
        }
    except Exception as e:
        logger.error(f"Error retrieving queue status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving queue status: {str(e)}",
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
