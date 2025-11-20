#!/usr/bin/env python3
"""
Vision API Client Examples
Demonstrates how to use the Jetson LLM API with vision-language models
"""

import base64
from pathlib import Path
from openai import OpenAI

# Configuration
API_BASE_URL = "http://localhost:9000/v1"
API_KEY = "change-me-to-a-secure-key"  # Match your .env file

# Initialize client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_image_data_uri(image_path: str) -> str:
    """Create a data URI for an image."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    mime_type = mime_types.get(ext, 'image/jpeg')
    base64_image = encode_image(image_path)
    return f"data:{mime_type};base64,{base64_image}"


def example_1_image_description(image_path: str):
    """Example 1: Basic image description"""
    print("\n" + "="*60)
    print("Example 1: Image Description")
    print("="*60)

    image_data_uri = create_image_data_uri(image_path)

    response = client.chat.completions.create(
        model="minicpm-v-2.5",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image? Concisely describe it."},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri}
                    }
                ]
            }
        ],
        max_tokens=512
    )

    print(f"\nResponse: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.total_tokens}")


def example_2_ocr(image_path: str):
    """Example 2: OCR / Text extraction from image"""
    print("\n" + "="*60)
    print("Example 2: OCR - Extract Text from Image")
    print("="*60)

    image_data_uri = create_image_data_uri(image_path)

    response = client.chat.completions.create(
        model="minicpm-v-2.5",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all visible text from this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri}
                    }
                ]
            }
        ],
        max_tokens=1024
    )

    print(f"\nExtracted Text:\n{response.choices[0].message.content}")


def example_3_streaming(image_path: str):
    """Example 3: Streaming vision response"""
    print("\n" + "="*60)
    print("Example 3: Streaming Vision Response")
    print("="*60)

    image_data_uri = create_image_data_uri(image_path)

    print("\nStreaming response: ", end="", flush=True)

    stream = client.chat.completions.create(
        model="minicpm-v-2.5",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image concisely."},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri}
                    }
                ]
            }
        ],
        stream=True,
        max_tokens=256
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")


def example_4_multi_image(image1_path: str, image2_path: str):
    """Example 4: Compare multiple images"""
    print("\n" + "="*60)
    print("Example 4: Multi-Image Comparison")
    print("="*60)

    img1_uri = create_image_data_uri(image1_path)
    img2_uri = create_image_data_uri(image2_path)

    response = client.chat.completions.create(
        model="minicpm-v-2.5",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images. What are the key differences?"},
                    {"type": "image_url", "image_url": {"url": img1_uri}},
                    {"type": "image_url", "image_url": {"url": img2_uri}}
                ]
            }
        ],
        max_tokens=512
    )

    print(f"\nComparison:\n{response.choices[0].message.content}")


def example_5_conversation_with_images(image_path: str):
    """Example 5: Multi-turn conversation about an image"""
    print("\n" + "="*60)
    print("Example 5: Multi-turn Vision Conversation")
    print("="*60)

    image_data_uri = create_image_data_uri(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What objects can you see in this image?"},
                {"type": "image_url", "image_url": {"url": image_data_uri}}
            ]
        }
    ]

    # First question
    response = client.chat.completions.create(
        model="minicpm-v-2.5",
        messages=messages,
        max_tokens=256
    )

    first_response = response.choices[0].message.content
    print(f"\nUser: What objects can you see in this image?")
    print(f"Assistant: {first_response}")

    # Continue conversation
    messages.append({"role": "assistant", "content": first_response})
    messages.append({"role": "user", "content": "Which object is the largest?"})

    response = client.chat.completions.create(
        model="minicpm-v-2.5",
        messages=messages,
        max_tokens=256
    )

    print(f"\nUser: Which object is the largest?")
    print(f"Assistant: {response.choices[0].message.content}")


def example_6_specific_questions(image_path: str):
    """Example 6: Specific analytical questions"""
    print("\n" + "="*60)
    print("Example 6: Specific Analytical Questions")
    print("="*60)

    image_data_uri = create_image_data_uri(image_path)

    questions = [
        "How many people are in this image?",
        "What colors are most prominent?",
        "What is the setting or location?",
        "What appears to be happening?",
    ]

    for question in questions:
        response = client.chat.completions.create(
            model="minicpm-v-2.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }
            ],
            max_tokens=128
        )

        print(f"\nQ: {question}")
        print(f"A: {response.choices[0].message.content}")


def interactive_vision_chat(image_path: str):
    """Interactive chat with an image"""
    print("\n" + "="*60)
    print("Interactive Vision Chat")
    print("="*60)
    print("Type 'quit' to exit\n")

    image_data_uri = create_image_data_uri(image_path)

    messages = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'quit':
            break
        elif not user_input:
            continue

        # Add user message with image reference
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_input},
                {"type": "image_url", "image_url": {"url": image_data_uri}}
            ]
        })

        print("Assistant: ", end="", flush=True)

        stream = client.chat.completions.create(
            model="minicpm-v-2.5",
            messages=messages,
            stream=True,
            #max_tokens=512,
            temperature=0.3,      # More deterministic
            max_tokens=100,       # Shorter responses
            top_p=0.9            # More focused
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print("\n")
        messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    import sys

    print("\nJetson Vision API Client Examples")
    print("===================================\n")

    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} <image_path>               # Run all examples")
        print(f"  {sys.argv[0]} <image_path> interactive   # Interactive mode")
        print(f"  {sys.argv[0]} <img1> <img2> compare      # Compare two images")
        print("\nExample:")
        print(f"  {sys.argv[0]} /path/to/image.jpg")
        print(f"  {sys.argv[0]} /path/to/image.jpg interactive")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    try:
        if len(sys.argv) > 2 and sys.argv[2] == "interactive":
            interactive_vision_chat(image_path)
        elif len(sys.argv) > 3 and sys.argv[3] == "compare":
            image2_path = sys.argv[2]
            if not Path(image2_path).exists():
                print(f"Error: Image file not found: {image2_path}")
                sys.exit(1)
            example_4_multi_image(image_path, image2_path)
        else:
            # Run all examples
            example_1_image_description(image_path)
            example_2_ocr(image_path)
            example_3_streaming(image_path)
            example_5_conversation_with_images(image_path)
            example_6_specific_questions(image_path)

            print("\n" + "="*60)
            print("All examples completed!")
            print("="*60)
            print("\nTry interactive mode:")
            print(f"  python3 {sys.argv[0]} {image_path} interactive")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. The vision model is running (sudo systemctl status llama-minicpm-v)")
        print("  2. The API is running (sudo systemctl status jetson-api)")
        print("  3. API_KEY matches your .env file")
