"""Model routing and prompt template logic."""

from typing import Literal, List
from .config import settings
from .models import ChatMessage


ModelName = Literal[
    "qwen2.5-coder-14b-instruct",
    "qwen3-vl-8b",
    "ui-tars-1.5-7b",
    "gemma-4-26b-a4b-it",
    "qwen3-embedding-8b",
]


AVAILABLE_MODELS = {
    "qwen2.5-coder-14b-instruct": {
        "name": "qwen2.5-coder-14b-instruct",
        "base_url": None,
        "description": "Qwen 2.5 Coder 14B Instruct - code generation specialist (Q4_K_M)",
        "modality": "text",
        "supports_vision": False,
    },
    "qwen3-vl-8b": {
        "name": "qwen3-vl-8b",
        "base_url": None,
        "description": "Qwen3-VL 8B Instruct vision-language model (Q4_K_M GGUF via llama.cpp)",
        "modality": "multimodal",
        "supports_vision": True,
    },
    "ui-tars-1.5-7b": {
        "name": "ui-tars-1.5-7b",
        "base_url": None,
        "description": "UI-TARS 1.5 7B - ByteDance GUI agent for desktop control (Qwen2.5-VL arch, bf16 via vLLM)",
        "modality": "multimodal",
        "supports_vision": True,
    },
    "gemma-4-26b-a4b-it": {
        "name": "gemma-4-26b-a4b-it",
        "base_url": None,
        "description": "Gemma 4 26B-A4B MoE - multimodal, 4B active params per token (Q4_K_M GGUF via llama.cpp)",
        "modality": "multimodal",
        "supports_vision": True,
    },
}


def get_available_models() -> dict:
    """Get list of available models with their configurations."""
    models = {k: v.copy() for k, v in AVAILABLE_MODELS.items()}
    models["qwen2.5-coder-14b-instruct"]["base_url"] = str(settings.qwen_coder_base_url)
    models["qwen3-vl-8b"]["base_url"] = str(settings.qwen3_vl_base_url)
    models["ui-tars-1.5-7b"]["base_url"] = str(settings.ui_tars_base_url)
    models["gemma-4-26b-a4b-it"]["base_url"] = str(settings.gemma_4_base_url)
    return models


def resolve_model_base_url(model: str) -> str:
    """Resolve the base URL for a given model name."""
    if model == "qwen2.5-coder-14b-instruct":
        return str(settings.qwen_coder_base_url)
    if model == "qwen3-vl-8b":
        return str(settings.qwen3_vl_base_url)
    if model == "ui-tars-1.5-7b":
        return str(settings.ui_tars_base_url)
    if model == "gemma-4-26b-a4b-it":
        return str(settings.gemma_4_base_url)
    raise ValueError(f"Unsupported model: {model}")


def is_vision_model(model: str) -> bool:
    """Check if a model supports vision/image inputs."""
    models = get_available_models()
    return models.get(model, {}).get("supports_vision", False)


def detect_vision_content(messages: List[ChatMessage]) -> bool:
    """Detect if messages contain image content."""
    from .models import ContentPart

    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, ContentPart) and part.type == "image_url":
                    return True
    return False


def format_chat_prompt(messages: List[ChatMessage], model: str) -> str:
    """Format chat messages into a prompt string based on the model's chat template."""
    if model in ("qwen2.5-coder-14b-instruct",):
        return _format_qwen_prompt(messages)
    return _format_generic_prompt(messages)


def _format_qwen_prompt(messages: List[ChatMessage]) -> str:
    """Format messages for Qwen 2.5 family models using their chat template."""
    prompt_parts = []

    for msg in messages:
        role = msg.role
        content = msg.content
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)


def _format_generic_prompt(messages: List[ChatMessage]) -> str:
    """Generic fallback prompt formatter."""
    prompt_parts = []
    system_parts = [m.content for m in messages if m.role == "system"]
    chat_parts = [
        f"{m.role.upper()}: {m.content}"
        for m in messages
        if m.role in ("user", "assistant")
    ]

    if system_parts:
        prompt_parts.append("System:\n" + "\n".join(system_parts))

    prompt_parts.extend(chat_parts)
    prompt_parts.append("ASSISTANT:")

    return "\n\n".join(prompt_parts)
