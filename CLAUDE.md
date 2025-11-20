# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenAI-compatible API for local LLM inference on NVIDIA Jetson devices. Provides a unified FastAPI interface that routes requests to multiple llama.cpp backend servers running different models:
- **Text Models**: DeepSeek R1 7B, Qwen 2.5 7B
- **Vision Models**: MiniCPM-V 2.5, Qwen2.5-VL-7B (multimodal image+text)
- **Embedding Model**: Qwen3 Embedding 8B (text embeddings for semantic search, RAG, clustering)
- **Web Dashboard**: Interactive UI at `http://localhost:9000` for testing models and monitoring system metrics

## Development Commands

### Setup and Running

```bash
# Setup virtual environment and dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env to set API_KEY and backend URLs

# Run development server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 9000

# Run with debug logging
LOG_LEVEL=debug uvicorn app.main:app --reload
```

### Testing

```bash
# Run tests (if tests exist in tests/ directory)
pytest tests/

# Manual health check
curl http://localhost:9000/health

# Test chat completion
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-7b-instruct", "messages": [{"role": "user", "content": "Hello"}]}'

# Test embeddings
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Authorization: Bearer your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-8b", "input": "Hello world"}'
```

## Architecture

### Request Flow

1. **Client Request** → FastAPI endpoint (`/v1/chat/completions`)
2. **Authentication** → `deps.verify_api_key()` validates Bearer token
3. **Model Routing** → `routing.resolve_model_base_url()` selects llama-server backend
4. **Prompt Formatting** → `routing.format_chat_prompt()` applies model-specific chat template
5. **Backend Call** → `clients.call_llama_server()` or `clients.stream_llama_server()`
6. **Response Translation** → llama.cpp response → OpenAI-compatible format

### Key Components

**`app/main.py`** - FastAPI application with endpoints:
- `/` - Serves the interactive dashboard (dashboard.html)
- `/v1/chat/completions` - OpenAI-compatible chat endpoint (streaming & non-streaming, text & vision)
- `/v1/embeddings` - OpenAI-compatible embeddings endpoint for vector generation
- `/v1/models` - List available models
- `/health` - Check backend availability and response times
- `/system/stats` - Real-time Jetson system metrics (GPU, CPU, RAM, disk, temperatures)
- `/api/activity-logs` - Get recent API activity logs
- `/api/activity-logs/clear` - Clear all activity logs (requires auth)

**`dashboard.html`** - Single-page web dashboard:
- Real-time system metrics (updates every 3s)
- Model status monitoring (updates every 10s)
- Interactive chat interface with model selection
- Image upload for vision models (auto-shows for MiniCPM-V and Qwen2.5-VL)
- Streaming and non-streaming modes
- Activity logs modal with real-time request monitoring (auto-refresh every 5s)
- API reference documentation with examples

**`app/system_monitor.py`** - Jetson device monitoring:
- `get_gpu_stats()` - GPU utilization, memory, temperature via nvidia-smi
- `get_cpu_stats()` - CPU utilization from /proc/stat
- `get_memory_stats()` - RAM usage from /proc/meminfo
- `get_disk_stats()` - Disk usage via os.statvfs
- `get_thermal_zones()` - Temperature readings from /sys/class/thermal
- `get_jetson_model()` - Device model detection from device tree

**`app/routing.py`** - Model routing and chat template logic:
- Maps model names to backend URLs via environment variables
- `format_chat_prompt()` applies model-specific formatting:
  - **DeepSeek R1**: Simple "User:/Assistant:" format
  - **Qwen 2.5**: Uses `<|im_start|>` and `<|im_end|>` tokens
- Each model requires a different chat template to function properly

**`app/clients.py`** - llama-server communication:
- `call_llama_server()` - Non-streaming completions via `/completion` endpoint
- `stream_llama_server()` - SSE streaming via llama.cpp's streaming API
- `call_llama_server_vision()` / `stream_llama_server_vision()` - Vision model handlers
- `call_llama_server_embeddings()` - Embeddings generation via `/v1/embeddings` endpoint
- `check_backend_health()` - Backend health checks using `/props` endpoint
- Translates llama.cpp response format to OpenAI-compatible format

**`app/config.py`** - Pydantic settings loaded from `.env`:
- `api_key` - Bearer token authentication
- `deepseek_base_url` / `qwen_base_url` - Backend llama-server URLs
- `host`, `port`, `log_level` - Server configuration

**`app/models.py`** - Pydantic models for OpenAI-compatible API:
- `ChatCompletionRequest` / `ChatCompletionResponse`
- `ChatCompletionChunk` / `ChatCompletionStreamChoice` (for streaming)
- `EmbeddingRequest` / `EmbeddingResponse` (for embeddings)
- `ModelList`, `HealthResponse`

**`app/deps.py`** - FastAPI dependency for Bearer token authentication

**`app/activity_logger.py`** - Request monitoring and logging:
- `ActivityLoggerMiddleware` - Captures all API requests/responses
- `ActivityLog` - In-memory storage for last 200 requests (configurable)
- Tracks: timestamp, method, endpoint, model, status code, response time, client IP, request preview
- Auto-excludes monitoring endpoints (/system/stats, /api/activity-logs, /health)
- Sanitizes sensitive data (API keys hidden from logs)

### Backend Architecture

The API acts as a **proxy/router** between OpenAI-compatible clients and llama.cpp backends:

```
Browser / Client (OpenAI SDK)
    ↓
Dashboard / API (port 9000) - with system monitoring
    ↓
Model Router (app/routing.py)
    ↓
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ llama-server    │ llama-server    │ llama-server    │ llama-server    │ llama-server    │
│ DeepSeek R1     │ Qwen 2.5        │ MiniCPM-V 2.5   │ Qwen2.5-VL 7B   │ Qwen3 Embed 8B  │
│ (text)          │ (text)          │ (vision)        │ (vision)        │ (embeddings)    │
│ :8081           │ :8082           │ :8083           │ :8084           │ :8085           │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

Each llama-server instance runs a single quantized GGUF model with CUDA acceleration. Vision models include multimodal projection (mmproj) files for image understanding. The embeddings model runs in embeddings mode (`--embeddings` flag) for vector generation. The API routes requests based on the `model` field and endpoint.

## Configuration

Environment variables (`.env`):
- **Required**: `API_KEY` - Must be set to a secure value for production
- **Backend URLs**: `DEEPSEEK_BASE_URL`, `QWEN_BASE_URL`, `MINICPM_V_BASE_URL`, `QWEN2_5_VL_BASE_URL`, `QWEN3_EMBEDDING_BASE_URL` - Point to llama-server instances
- **Server**: `HOST` (default: 0.0.0.0), `PORT` (default: 9000)

## Production Deployment

Systemd services in `systemd/`:
- `llama-deepseek.service` - DeepSeek R1 7B text backend (port 8081)
- `llama-qwen.service` - Qwen 2.5 7B text backend (port 8082)
- `llama-minicpm-v.service` - MiniCPM-V 2.5 vision backend (port 8083)
- `llama-qwen2.5-vl.service` - Qwen2.5-VL-7B vision backend (port 8084)
- `llama-qwen3-embedding.service` - Qwen3 Embedding 8B backend (port 8085)
- `jetson-api.service` - Main FastAPI server with dashboard (port 9000)

### Installation

```bash
# Install all services
cd systemd
./install-services.sh

# Enable and start all services (starts on boot)
sudo systemctl enable llama-deepseek llama-qwen llama-minicpm-v llama-qwen2.5-vl jetson-api
sudo systemctl start llama-deepseek llama-qwen llama-minicpm-v llama-qwen2.5-vl jetson-api

# Or start only text models + API (recommended for testing)
sudo systemctl enable llama-deepseek llama-qwen jetson-api
sudo systemctl start llama-deepseek llama-qwen jetson-api

# Check status
sudo systemctl status jetson-api
sudo systemctl status llama-qwen2.5-vl

# View logs
sudo journalctl -u jetson-api -f
```

After starting services, the dashboard will be available at `http://localhost:9000`

## Adding New Models

To add a new model:
1. Add model entry to `AVAILABLE_MODELS` in `app/routing.py`
2. Add base URL setting to `Settings` class in `app/config.py`
3. Update `resolve_model_base_url()` to handle new model
4. Add chat template formatting function (e.g., `_format_newmodel_prompt()`)
5. Update `format_chat_prompt()` to use the new template
6. Create systemd service for the new llama-server backend
7. Update `.env.example` with new backend URL variable

## Important Notes

- **Chat Templates**: Each model requires specific prompt formatting. DeepSeek uses simple role prefixes while Qwen uses special tokens. Using the wrong template will produce poor results.
- **Backend Dependency**: The API is stateless and requires llama-server backends to be running. All inference happens in llama.cpp.
- **Streaming**: Uses SSE (Server-Sent Events) via `sse-starlette` for streaming responses. The API translates llama.cpp's streaming format to OpenAI's chunk format.
- **Authentication**: Simple Bearer token auth via `Authorization` header. Consider adding rate limiting for production.
- **Timeouts**: HTTP clients use 120s timeout for inference requests, 5s for health checks.
