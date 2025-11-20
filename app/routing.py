"""Model routing and prompt template logic."""

from typing import Literal, List
from .config import settings
from .models import ChatMessage


ModelName = Literal["deepseek-r1-7b", "qwen2.5-7b-instruct", "minicpm-v-2.5", "qwen2.5-vl-7b", "qwen3-embedding-8b"]


AVAILABLE_MODELS = {
    "deepseek-r1-7b": {
        "name": "deepseek-r1-7b",
        "base_url": None,  # Set dynamically from settings
        "description": "DeepSeek R1 distilled on Qwen 7B (Q4_K_M quantization)",
        "modality": "text",
        "supports_vision": False,
    },
    "qwen2.5-7b-instruct": {
        "name": "qwen2.5-7b-instruct",
        "base_url": None,  # Set dynamically from settings
        "description": "Qwen 2.5 7B Instruct (Q4_K_M quantization)",
        "modality": "text",
        "supports_vision": False,
    },
    "minicpm-v-2.5": {
        "name": "minicpm-v-2.5",
        "base_url": None,  # Set dynamically from settings
        "description": "MiniCPM-V 2.5 vision-language model (Q4_K_M quantization)",
        "modality": "multimodal",
        "supports_vision": True,
    },
    "qwen2.5-vl-7b": {
        "name": "qwen2.5-vl-7b",
        "base_url": None,  # Set dynamically from settings
        "description": "Qwen 2.5 VL 7B vision-language model (Q4_K quantization)",
        "modality": "multimodal",
        "supports_vision": True,
    },
    "qwen3-embedding-8b": {
        "name": "qwen3-embedding-8b",
        "base_url": None,  # Set dynamically from settings
        "description": "Qwen3 Embedding 8B for vector embeddings (Q5_K_M quantization, 4096 dims)",
        "modality": "embedding",
        "supports_vision": False,
    },
}


def get_available_models() -> dict:
    """Get list of available models with their configurations."""
    models = AVAILABLE_MODELS.copy()
    models["deepseek-r1-7b"]["base_url"] = str(settings.deepseek_base_url)
    models["qwen2.5-7b-instruct"]["base_url"] = str(settings.qwen_base_url)
    models["minicpm-v-2.5"]["base_url"] = str(settings.minicpm_v_base_url)
    models["qwen2.5-vl-7b"]["base_url"] = str(settings.qwen2_5_vl_base_url)
    models["qwen3-embedding-8b"]["base_url"] = str(settings.qwen3_embedding_base_url)
    return models


def resolve_model_base_url(model: str) -> str:
    """
    Resolve the base URL for a given model name.

    Args:
        model: Model identifier

    Returns:
        Base URL string for the llama-server instance

    Raises:
        ValueError: If model is not supported
    """
    if model == "deepseek-r1-7b":
        return str(settings.deepseek_base_url)
    if model == "qwen2.5-7b-instruct":
        return str(settings.qwen_base_url)
    if model == "minicpm-v-2.5":
        return str(settings.minicpm_v_base_url)
    if model == "qwen2.5-vl-7b":
        return str(settings.qwen2_5_vl_base_url)
    raise ValueError(f"Unsupported model: {model}")


def is_vision_model(model: str) -> bool:
    """
    Check if a model supports vision/image inputs.

    Args:
        model: Model identifier

    Returns:
        True if model supports vision, False otherwise
    """
    models = get_available_models()
    return models.get(model, {}).get("supports_vision", False)


def detect_vision_content(messages: List[ChatMessage]) -> bool:
    """
    Detect if messages contain image content.

    Args:
        messages: List of chat messages

    Returns:
        True if any message contains images, False otherwise
    """
    from .models import ContentPart

    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, ContentPart) and part.type == "image_url":
                    return True
    return False


def format_chat_prompt(messages: List[ChatMessage], model: str) -> str:
    """
    Format chat messages into a prompt string based on the model's chat template.

    Args:
        messages: List of chat messages
        model: Model identifier

    Returns:
        Formatted prompt string
    """
    if model == "deepseek-r1-7b":
        return _format_deepseek_prompt(messages)
    elif model == "qwen2.5-7b-instruct":
        return _format_qwen_prompt(messages)
    else:
        # Fallback to generic format
        return _format_generic_prompt(messages)


def _format_deepseek_prompt(messages: List[ChatMessage]) -> str:
    """
    Format messages for DeepSeek R1 model.
    Uses a simple format that works well with the model.
    """
    prompt_parts = []

    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")

    # Add assistant prefix to prompt completion
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)


def _format_qwen_prompt(messages: List[ChatMessage]) -> str:
    """
    Format messages for Qwen 2.5 model using its chat template.
    Qwen uses <|im_start|> and <|im_end|> tokens.
    """
    prompt_parts = []

    for msg in messages:
        role = msg.role
        content = msg.content
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    # Add assistant prefix
    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)


def _format_generic_prompt(messages: List[ChatMessage]) -> str:
    """
    Generic fallback prompt formatter.
    """
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
