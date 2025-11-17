# Jetson LLM API

OpenAI-compatible API for local LLM inference on NVIDIA Jetson, providing a unified interface for DeepSeek R1 and Qwen 2.5 models running via llama.cpp.

## Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI API endpoints
- **Streaming support** - Real-time token streaming via Server-Sent Events (SSE)
- **Multiple models** - Route requests to DeepSeek R1 7B or Qwen 2.5 7B
- **Model-specific chat templates** - Proper formatting for each model
- **Bearer token authentication** - Simple API key security
- **Health checks** - Monitor backend availability
- **Systemd integration** - Production-ready service management

## Prerequisites

- NVIDIA Jetson AGX Orin with JetPack 6.x
- llama.cpp built with CUDA support
- Models downloaded:
  - DeepSeek R1 Distill Qwen 7B (Q4_K_M GGUF)
  - Qwen 2.5 7B Instruct (Q4_K_M GGUF)
- Python 3.10+

## Quick Start

### 1. Clone or copy this project to your Jetson

```bash
cd ~
# If you received this as a directory, you're already set
```

### 2. Set up Python environment

```bash
cd ~/jetson-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
nano .env  # Edit to set your API_KEY and verify backend URLs
```

Example `.env`:
```bash
API_KEY=your-super-secret-key-here
DEEPSEEK_BASE_URL=http://127.0.0.1:8081
QWEN_BASE_URL=http://127.0.0.1:8082
HOST=0.0.0.0
PORT=9000
LOG_LEVEL=info
```

### 4. Start llama-server backends

Make sure both llama-server instances are running:

```bash
# Terminal 1 - DeepSeek
cd ~/llama.cpp
./build/bin/llama-server \
  --model ~/models/deepseek/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
  --ctx-size 8192 \
  --port 8081 \
  --n-gpu-layers 99 \
  --threads $(nproc) \
  --host 127.0.0.1

# Terminal 2 - Qwen
cd ~/llama.cpp
./build/bin/llama-server \
  --model ~/models/qwen/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --ctx-size 8192 \
  --port 8082 \
  --n-gpu-layers 99 \
  --threads $(nproc) \
  --host 127.0.0.1
```

### 5. Run the API

```bash
cd ~/jetson-api
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

The API will be available at `http://YOUR_JETSON_IP:9000`

## Production Deployment with systemd

For production use, install the systemd services to auto-start everything on boot:

```bash
cd ~/jetson-api/systemd
./install-services.sh
```

Enable and start all services:

```bash
sudo systemctl enable llama-deepseek llama-qwen jetson-api
sudo systemctl start llama-deepseek llama-qwen jetson-api
```

Check status:

```bash
sudo systemctl status llama-deepseek
sudo systemctl status llama-qwen
sudo systemctl status jetson-api
```

View logs:

```bash
# Follow logs in real-time
sudo journalctl -u jetson-api -f

# View recent logs
sudo journalctl -u llama-deepseek --since "1 hour ago"
```

## API Usage

### Authentication

All requests require a Bearer token in the Authorization header:

```bash
Authorization: Bearer your-api-key-here
```

### Endpoints

#### List Models

```bash
curl http://YOUR_JETSON_IP:9000/v1/models \
  -H "Authorization: Bearer your-api-key-here"
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "deepseek-r1-7b",
      "object": "model",
      "created": 0,
      "owned_by": "jetson-api"
    },
    {
      "id": "qwen2.5-7b-instruct",
      "object": "model",
      "created": 0,
      "owned_by": "jetson-api"
    }
  ]
}
```

#### Chat Completion (Non-Streaming)

```bash
curl -X POST http://YOUR_JETSON_IP:9000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

#### Chat Completion (Streaming)

```bash
curl -X POST http://YOUR_JETSON_IP:9000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-7b",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "stream": true,
    "max_tokens": 256
  }'
```

#### Health Check

```bash
curl http://YOUR_JETSON_IP:9000/health
```

Response:
```json
{
  "status": "healthy",
  "models": {
    "deepseek-r1-7b": {
      "base_url": "http://127.0.0.1:8081",
      "status": "healthy",
      "details": {...}
    },
    "qwen2.5-7b-instruct": {
      "base_url": "http://127.0.0.1:8082",
      "status": "healthy",
      "details": {...}
    }
  }
}
```

## Using with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://YOUR_JETSON_IP:9000/v1",
    api_key="your-api-key-here"
)

# Non-streaming
response = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="deepseek-r1-7b",
    messages=[{"role": "user", "content": "Write a haiku about AI"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Project Structure

```
jetson-api/
├── app/
│   ├── __init__.py       # Package initialization
│   ├── main.py           # FastAPI application and endpoints
│   ├── config.py         # Configuration and settings
│   ├── models.py         # Pydantic models for requests/responses
│   ├── routing.py        # Model routing and chat templates
│   ├── clients.py        # llama-server HTTP client
│   └── deps.py           # Authentication dependencies
├── systemd/
│   ├── llama-deepseek.service
│   ├── llama-qwen.service
│   ├── jetson-api.service
│   └── install-services.sh
├── .env.example          # Example environment variables
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Available Models

| Model ID | Description | Port |
|----------|-------------|------|
| `deepseek-r1-7b` | DeepSeek R1 distilled on Qwen 7B (Q4_K_M) | 8081 |
| `qwen2.5-7b-instruct` | Qwen 2.5 7B Instruct (Q4_K_M) | 8082 |

## Troubleshooting

### API returns 503 or connection errors

Check if llama-server backends are running:

```bash
# Manual check
curl http://127.0.0.1:8081/props
curl http://127.0.0.1:8082/props

# If using systemd
sudo systemctl status llama-deepseek
sudo systemctl status llama-qwen
```

### Authentication fails

Verify your API key matches the one in `.env`:

```bash
grep API_KEY ~/jetson-api/.env
```

### Slow inference

- Check GPU utilization: `nvidia-smi`
- Verify `--n-gpu-layers 99` is set (offloads model to GPU)
- Monitor memory usage
- Consider reducing `--ctx-size` if memory is limited

### Port conflicts

If ports 8081, 8082, or 9000 are in use, update:
- llama-server start commands (or systemd service files)
- `.env` file to match new backend ports

## Development

### Running tests

```bash
source .venv/bin/activate
pytest tests/
```

### Viewing logs in development

```bash
# Run with debug logging
LOG_LEVEL=debug uvicorn app.main:app --reload
```

## License

This project is provided as-is for use with your Jetson setup.

## Support

For issues specific to:
- **llama.cpp**: https://github.com/ggml-org/llama.cpp
- **FastAPI**: https://fastapi.tiangolo.com/
- **Jetson/CUDA**: NVIDIA Developer Forums
