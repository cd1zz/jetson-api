"""Pydantic models for OpenAI-compatible API requests and responses."""

from typing import List, Literal, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


# ============================================================================
# Chat Completion Models
# ============================================================================

Role = Literal["system", "user", "assistant"]


# ============================================================================
# Vision Support Models
# ============================================================================

class ImageUrl(BaseModel):
    """Image URL for vision requests."""

    url: str = Field(..., description="URL or data URI of the image")
    detail: Optional[str] = Field("auto", description="Image detail level")


class ContentPart(BaseModel):
    """A part of message content (text or image)."""

    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class ChatMessage(BaseModel):
    """A single chat message (supports both text and vision)."""

    role: Role
    content: Union[str, List[ContentPart]] = Field(
        ...,
        description="Message content (string for text-only, list for vision)"
    )


class ChatCompletionRequest(BaseModel):
    """Request for chat completion endpoint."""

    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(512, gt=0)
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0)
    stream: bool = Field(False, description="Whether to stream the response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response for chat completion endpoint."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ChatCompletionStreamChoice(BaseModel):
    """A single streaming choice delta."""

    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """A single chunk in a streaming response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


# ============================================================================
# Models Endpoint
# ============================================================================

class Model(BaseModel):
    """Information about an available model."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "jetson-api"


class ModelList(BaseModel):
    """List of available models."""

    object: str = "list"
    data: List[Model]


# ============================================================================
# Health Check
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models: Dict[str, Dict[str, Any]]
