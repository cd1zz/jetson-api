#!/usr/bin/env python3
"""
Example client scripts for Jetson LLM API
Demonstrates various ways to interact with the OpenAI-compatible API
"""

import os
from openai import OpenAI

# Configuration
API_BASE_URL = "http://localhost:9000/v1"
API_KEY = "change-me-to-a-secure-key"  # Match your .env file

# Initialize client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


def example_1_simple_chat():
    """Basic chat completion with Qwen model"""
    print("\n" + "="*60)
    print("Example 1: Simple Chat with Qwen")
    print("="*60)

    response = client.chat.completions.create(
        model="qwen2.5-7b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the Jetson AGX Orin?"}
        ],
        temperature=0.7,
        max_tokens=256
    )

    print(f"\nResponse: {response.choices[0].message.content}")
    print(f"\nTokens used: {response.usage.total_tokens}")


def example_2_streaming_chat():
    """Streaming chat completion with DeepSeek"""
    print("\n" + "="*60)
    print("Example 2: Streaming Chat with DeepSeek")
    print("="*60)

    print("\nStreaming response: ", end="", flush=True)

    stream = client.chat.completions.create(
        model="deepseek-r1-7b",
        messages=[
            {"role": "user", "content": "Write a haiku about machine learning."}
        ],
        stream=True,
        max_tokens=128,
        temperature=0.8
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")


def example_3_conversation():
    """Multi-turn conversation"""
    print("\n" + "="*60)
    print("Example 3: Multi-turn Conversation")
    print("="*60)

    messages = [
        {"role": "system", "content": "You are a Python programming expert."},
        {"role": "user", "content": "How do I read a file in Python?"}
    ]

    # First response
    response = client.chat.completions.create(
        model="qwen2.5-7b-instruct",
        messages=messages,
        max_tokens=256
    )

    assistant_msg = response.choices[0].message.content
    print(f"\nUser: {messages[-1]['content']}")
    print(f"Assistant: {assistant_msg}")

    # Continue conversation
    messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": "Can you show me an example with error handling?"})

    response = client.chat.completions.create(
        model="qwen2.5-7b-instruct",
        messages=messages,
        max_tokens=256
    )

    print(f"\nUser: {messages[-1]['content']}")
    print(f"Assistant: {response.choices[0].message.content}")


def example_4_list_models():
    """List available models"""
    print("\n" + "="*60)
    print("Example 4: List Available Models")
    print("="*60)

    models = client.models.list()

    print("\nAvailable models:")
    for model in models.data:
        print(f"  - {model.id} (owned by: {model.owned_by})")


def example_5_temperature_comparison():
    """Compare responses with different temperatures"""
    print("\n" + "="*60)
    print("Example 5: Temperature Comparison")
    print("="*60)

    prompt = "Complete this sentence: The future of AI is"

    for temp in [0.3, 0.7, 1.0]:
        response = client.chat.completions.create(
            model="qwen2.5-7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=50
        )

        print(f"\nTemperature {temp}: {response.choices[0].message.content}")


def example_6_code_generation():
    """Use for code generation"""
    print("\n" + "="*60)
    print("Example 6: Code Generation")
    print("="*60)

    response = client.chat.completions.create(
        model="deepseek-r1-7b",
        messages=[
            {"role": "system", "content": "You are an expert Python programmer."},
            {"role": "user", "content": "Write a function to calculate fibonacci numbers using memoization."}
        ],
        temperature=0.3,  # Lower temperature for more deterministic code
        max_tokens=512
    )

    print(f"\n{response.choices[0].message.content}")


def example_7_raw_requests():
    """Using raw HTTP requests with curl (shown as command)"""
    print("\n" + "="*60)
    print("Example 7: Raw HTTP Request (curl command)")
    print("="*60)

    curl_command = f"""
curl -X POST {API_BASE_URL.replace('/v1', '')}/v1/chat/completions \\
  -H "Authorization: Bearer {API_KEY}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "qwen2.5-7b-instruct",
    "messages": [
      {{"role": "user", "content": "Hello, how are you?"}}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }}'
"""
    print(curl_command)


def interactive_chat():
    """Interactive chat session"""
    print("\n" + "="*60)
    print("Interactive Chat Mode")
    print("="*60)
    print("Type 'quit' to exit, 'clear' to start new conversation")
    print("Choose model: 1) qwen2.5-7b-instruct  2) deepseek-r1-7b")

    choice = input("Select model (1/2): ").strip()
    model = "qwen2.5-7b-instruct" if choice == "1" else "deepseek-r1-7b"
    print(f"\nUsing model: {model}\n")

    messages = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            messages = []
            print("Conversation cleared.\n")
            continue
        elif not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=512
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

    print("\nJetson LLM API Client Examples")
    print("================================\n")

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_chat()
    else:
        # Run all examples
        try:
            example_1_simple_chat()
            example_2_streaming_chat()
            example_3_conversation()
            example_4_list_models()
            example_5_temperature_comparison()
            example_6_code_generation()
            example_7_raw_requests()

            print("\n" + "="*60)
            print("All examples completed!")
            print("="*60)
            print("\nTry interactive mode: python3 client_examples.py interactive")

        except Exception as e:
            print(f"\nError: {e}")
            print("\nMake sure:")
            print("  1. The API is running (systemctl status jetson-api)")
            print("  2. API_KEY matches your .env file")
            print("  3. Backends are healthy (curl http://localhost:9000/health)")
