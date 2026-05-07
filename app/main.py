"""FastAPI application for Jetson LLM API."""

import asyncio
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
    logger.info(f"Qwen backend: {settings.qwen_base_url}")
    logger.info(f"Qwen Coder backend: {settings.qwen_coder_base_url}")
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

    # Embedding backend isn't in AVAILABLE_MODELS (it's not a chat model) but
    # the dashboard still needs its health to avoid showing a permanent
    # "Loading..." state for an active service.
    embedding_url = str(settings.qwen3_embedding_base_url)
    embedding_health = await check_backend_health(embedding_url)
    models_health["qwen3-embedding-8b"] = {
        "base_url": embedding_url,
        **embedding_health,
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
                   f"Please use a vision-capable model like 'qwen3-vl-8b'.",
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
        upstream = e.response.status_code
        logger.error(f"Backend returned error {upstream}: {e.response.text}")
        status = upstream if 400 <= upstream < 500 else 502
        raise HTTPException(
            status_code=status,
            detail=f"Backend server error: {upstream} - {e.response.text[:200]}",
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
        upstream = e.response.status_code
        logger.error(f"Embeddings backend returned error {upstream}: {e.response.text}")
        status = upstream if 400 <= upstream < 500 else 502
        raise HTTPException(
            status_code=status,
            detail=f"Embeddings backend error: {upstream} - {e.response.text[:200]}",
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


MANAGED_SERVICES = {
    "qwen2.5-7b-instruct": {
        "service": "llama-qwen.service",
        "label": "Qwen 2.5 7B Instruct",
        "approx_ram_gb": 5,
    },
    "qwen2.5-coder-14b-instruct": {
        "service": "llama-qwen-coder.service",
        "label": "Qwen 2.5 Coder 14B",
        "approx_ram_gb": 10,
    },
    "qwen3-vl-8b": {
        "service": "llama-qwen3-vl.service",
        "label": "Qwen3 VL 8B",
        "approx_ram_gb": 8,
    },
    "qwen3-embedding-8b": {
        "service": "llama-qwen3-embedding.service",
        "label": "Qwen3 Embedding 8B",
        "approx_ram_gb": 6,
    },
}


async def _systemctl(*args: str) -> tuple[int, str]:
    """Run systemctl and return (exit_code, stdout+stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    return proc.returncode, stdout.decode().strip()


async def _service_state(service: str) -> str:
    """Return systemd state: active, inactive, activating, deactivating, failed, unknown."""
    code, out = await _systemctl("/usr/bin/systemctl", "is-active", service)
    return out or ("active" if code == 0 else "unknown")


@app.get("/api/models/control")
async def models_control_status():
    """Get start/stop control state for all managed models. No auth (read-only)."""
    states = await asyncio.gather(
        *(_service_state(info["service"]) for info in MANAGED_SERVICES.values())
    )
    models = [
        {
            "id": model_id,
            "service": info["service"],
            "label": info["label"],
            "approx_ram_gb": info["approx_ram_gb"],
            "state": state,
        }
        for (model_id, info), state in zip(MANAGED_SERVICES.items(), states)
    ]
    return {"models": models}


async def _control_service(model_id: str, action: str) -> dict:
    if model_id not in MANAGED_SERVICES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown managed model '{model_id}'. Available: {list(MANAGED_SERVICES)}",
        )
    if action not in ("start", "stop", "restart"):
        raise HTTPException(status_code=400, detail=f"Invalid action: {action}")

    service = MANAGED_SERVICES[model_id]["service"]
    code, out = await _systemctl("/usr/bin/sudo", "-n", "/usr/bin/systemctl", action, service)
    if code != 0:
        logger.error(f"systemctl {action} {service} failed (exit {code}): {out}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to {action} {service}: {out[:300]}",
        )
    state = await _service_state(service)
    return {"id": model_id, "service": service, "action": action, "state": state}


@app.post("/api/models/{model_id}/start")
async def start_model(model_id: str, _: None = Depends(verify_api_key)):
    """Start the systemd service backing a model. Requires authentication."""
    return await _control_service(model_id, "start")


@app.post("/api/models/{model_id}/stop")
async def stop_model(model_id: str, _: None = Depends(verify_api_key)):
    """Stop the systemd service backing a model (frees RAM). Requires authentication."""
    return await _control_service(model_id, "stop")


@app.post("/api/models/{model_id}/restart")
async def restart_model(model_id: str, _: None = Depends(verify_api_key)):
    """Restart the systemd service backing a model. Requires authentication."""
    return await _control_service(model_id, "restart")


@app.get("/api/queue-status")
async def get_queue_status():
    """
    Get queue/slot status for every configured backend.
    Backends that don't respond are returned with status="offline" so the
    dashboard can hide cards for models that aren't currently running.
    """
    backends = {
        "qwen2.5-7b-instruct": str(settings.qwen_base_url).rstrip('/'),
        "qwen2.5-coder-14b-instruct": str(settings.qwen_coder_base_url).rstrip('/'),
        "qwen3-vl-8b": str(settings.qwen3_vl_base_url).rstrip('/'),
        "qwen3-embedding-8b": str(settings.qwen3_embedding_base_url).rstrip('/'),
    }

    async def probe(model_id: str, base_url: str) -> tuple[str, dict]:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{base_url}/slots")
                response.raise_for_status()
                slots_data = response.json()
        except Exception:
            return model_id, {"backend_url": base_url, "status": "offline"}

        total_slots = len(slots_data)
        busy_slots = sum(1 for slot in slots_data if slot.get("is_processing", False))
        return model_id, {
            "backend_url": base_url,
            "status": "online",
            "total_slots": total_slots,
            "busy_slots": busy_slots,
            "idle_slots": total_slots - busy_slots,
            "utilization_percent": round((busy_slots / total_slots * 100) if total_slots > 0 else 0, 1),
            "at_capacity": total_slots > 0 and busy_slots >= total_slots,
        }

    pairs = await asyncio.gather(*(probe(m, u) for m, u in backends.items()))
    return {"backends": dict(pairs)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )
