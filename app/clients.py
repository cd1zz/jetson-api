"""Client for interacting with llama-server backends."""

import json
import time
import uuid
from typing import Dict, Any, AsyncIterator
import httpx

from .routing import resolve_model_base_url, format_chat_prompt
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionStreamChoice,
    Usage,
    EmbeddingRequest,
    EmbeddingResponse,
    Embedding,
    EmbeddingUsage,
)


async def call_llama_server(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Call llama-server for non-streaming completion.

    Args:
        req: Chat completion request

    Returns:
        Chat completion response

    Raises:
        httpx.HTTPError: If the backend request fails
    """
    base_url = resolve_model_base_url(req.model).rstrip('/')
    url = f"{base_url}/completion"

    # Format messages into prompt using model-specific template
    prompt = format_chat_prompt(req.messages, req.model)

    # Build request payload for llama.cpp
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "n_predict": req.max_tokens or 512,
        "stream": False,
    }

    if req.stop:
        payload["stop"] = req.stop

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    # Extract completion text from llama.cpp response
    text = data.get("content", "").strip()
    tokens_predicted = data.get("tokens_predicted", 0)
    tokens_evaluated = data.get("tokens_evaluated", 0)

    # Build OpenAI-compatible response
    choice = ChatCompletionChoice(
        index=0,
        message=ChatMessage(role="assistant", content=text),
        finish_reason=_map_stop_reason(data.get("stop_reason", "stop")),
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=req.model,
        choices=[choice],
        usage=Usage(
            prompt_tokens=tokens_evaluated,
            completion_tokens=tokens_predicted,
            total_tokens=tokens_evaluated + tokens_predicted,
        ),
    )


async def stream_llama_server(req: ChatCompletionRequest) -> AsyncIterator[str]:
    """
    Stream completion from llama-server using SSE.

    Args:
        req: Chat completion request

    Yields:
        SSE-formatted chunks as strings

    Raises:
        httpx.HTTPError: If the backend request fails
    """
    base_url = resolve_model_base_url(req.model).rstrip('/')
    url = f"{base_url}/completion"

    # Format messages into prompt
    prompt = format_chat_prompt(req.messages, req.model)

    # Build streaming request payload
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "n_predict": req.max_tokens or 512,
        "stream": True,
    }

    if req.stop:
        payload["stop"] = req.stop

    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                # llama.cpp sends SSE format: "data: {...}"
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Check if stream is complete
                    if data.get("stop", False):
                        # Send final chunk with finish_reason
                        chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created_at,
                            model=req.model,
                            choices=[
                                ChatCompletionStreamChoice(
                                    index=0,
                                    delta={},
                                    finish_reason=_map_stop_reason(
                                        data.get("stop_reason", "stop")
                                    ),
                                )
                            ],
                        )
                        yield chunk.model_dump_json()
                        yield "[DONE]"
                        break

                    # Extract content delta
                    content = data.get("content", "")
                    if content:
                        # Send content chunk
                        chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created_at,
                            model=req.model,
                            choices=[
                                ChatCompletionStreamChoice(
                                    index=0,
                                    delta={"content": content},
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield chunk.model_dump_json()


async def check_backend_health(base_url: str) -> Dict[str, Any]:
    """
    Check if a llama-server backend is healthy.

    Args:
        base_url: Base URL of the llama-server instance

    Returns:
        Health status dictionary
    """
    base_url = base_url.rstrip('/')
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try to hit the /health endpoint (if available) or /props
            health_url = f"{base_url}/health"
            props_url = f"{base_url}/props"

            try:
                response = await client.get(health_url)
                response.raise_for_status()
                return {"status": "healthy", "details": response.json()}
            except httpx.HTTPError:
                # Fallback to /props endpoint
                response = await client.get(props_url)
                response.raise_for_status()
                return {"status": "healthy", "details": response.json()}

    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def _map_stop_reason(llama_reason: str) -> str:
    """
    Map llama.cpp stop reason to OpenAI finish_reason.

    Args:
        llama_reason: Stop reason from llama.cpp

    Returns:
        OpenAI-compatible finish reason
    """
    mapping = {
        "stop": "stop",
        "length": "length",
        "eos": "stop",
    }
    return mapping.get(llama_reason, "stop")


# ============================================================================
# Vision Model Client Functions
# ============================================================================

async def call_llama_server_vision(
    req: ChatCompletionRequest
) -> ChatCompletionResponse:
    """
    Call llama-server for vision model completion.
    Vision models use the /v1/chat/completions endpoint with structured messages.

    Args:
        req: Chat completion request with vision content

    Returns:
        Chat completion response

    Raises:
        httpx.HTTPError: If the backend request fails
    """
    base_url = resolve_model_base_url(req.model).rstrip('/')
    url = f"{base_url}/v1/chat/completions"

    # Convert messages to dict format for llama.cpp
    messages_payload = []
    for msg in req.messages:
        if isinstance(msg.content, str):
            # Simple text message
            messages_payload.append({
                "role": msg.role,
                "content": msg.content
            })
        else:
            # Structured content with images
            content_parts = []
            for part in msg.content:
                if part.type == "text":
                    content_parts.append({"type": "text", "text": part.text})
                elif part.type == "image_url" and part.image_url:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": part.image_url.url}
                    })
            messages_payload.append({
                "role": msg.role,
                "content": content_parts
            })

    # Build request payload
    payload: Dict[str, Any] = {
        "messages": messages_payload,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens or 512,
        "top_p": req.top_p,
        "stream": False,
    }

    if req.stop:
        payload["stop"] = req.stop

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    # Parse OpenAI-compatible response from llama.cpp
    choice = data["choices"][0]
    message_content = choice["message"]["content"]

    return ChatCompletionResponse(
        id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
        created=data.get("created", int(time.time())),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=message_content),
                finish_reason=choice.get("finish_reason", "stop"),
            )
        ],
        usage=Usage(
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            total_tokens=data.get("usage", {}).get("total_tokens", 0),
        ),
    )


async def stream_llama_server_vision(
    req: ChatCompletionRequest
) -> AsyncIterator[str]:
    """
    Stream completion from llama-server vision model using SSE.

    Args:
        req: Chat completion request with vision content

    Yields:
        SSE-formatted chunks as strings

    Raises:
        httpx.HTTPError: If the backend request fails
    """
    base_url = resolve_model_base_url(req.model).rstrip('/')
    url = f"{base_url}/v1/chat/completions"

    # Convert messages to dict format
    messages_payload = []
    for msg in req.messages:
        if isinstance(msg.content, str):
            messages_payload.append({"role": msg.role, "content": msg.content})
        else:
            content_parts = []
            for part in msg.content:
                if part.type == "text":
                    content_parts.append({"type": "text", "text": part.text})
                elif part.type == "image_url" and part.image_url:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": part.image_url.url}
                    })
            messages_payload.append({"role": msg.role, "content": content_parts})

    # Build streaming request payload
    payload: Dict[str, Any] = {
        "messages": messages_payload,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens or 512,
        "top_p": req.top_p,
        "stream": True,
    }

    if req.stop:
        payload["stop"] = req.stop

    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())

    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                # llama.cpp vision sends SSE format: "data: {...}"
                # EventSourceResponse will add "data: " prefix, so we strip it here
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    # Check for done signal
                    if data_str.strip() == "[DONE]":
                        yield "[DONE]"
                        break

                    # Yield just the JSON content (EventSourceResponse adds "data: ")
                    yield data_str


# ============================================================================
# Embeddings Client Functions
# ============================================================================

async def call_llama_server_embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    """
    Call llama-server for embeddings generation.

    Args:
        req: Embedding request

    Returns:
        Embedding response

    Raises:
        httpx.HTTPError: If the backend request fails
    """
    from .config import settings

    base_url = str(settings.qwen3_embedding_base_url).rstrip('/')
    url = f"{base_url}/v1/embeddings"

    # Normalize input to list format
    inputs = req.input if isinstance(req.input, list) else [req.input]

    # Build request payload for llama.cpp embeddings endpoint
    payload: Dict[str, Any] = {
        "input": inputs,
        "model": req.model,
    }

    # Add optional parameters
    if req.encoding_format:
        payload["encoding_format"] = req.encoding_format

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    # Parse response - llama.cpp should return OpenAI-compatible format
    embeddings = []
    for idx, embedding_data in enumerate(data.get("data", [])):
        embeddings.append(
            Embedding(
                object="embedding",
                embedding=embedding_data.get("embedding", []),
                index=idx,
            )
        )

    # Extract usage information
    usage_data = data.get("usage", {})
    usage = EmbeddingUsage(
        prompt_tokens=usage_data.get("prompt_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0),
    )

    return EmbeddingResponse(
        object="list",
        data=embeddings,
        model=req.model,
        usage=usage,
    )
