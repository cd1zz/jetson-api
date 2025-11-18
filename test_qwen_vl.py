#!/usr/bin/env python3
"""Test script for Qwen2.5-VL-7B vision model."""

import requests
import json

API_URL = "http://localhost:9000/v1/chat/completions"
API_KEY = "change-me-to-a-secure-key"

# Use a publicly available test image
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

def test_qwen_vl():
    """Test Qwen2.5-VL with a simple image description task."""

    payload = {
        "model": "qwen2.5-vl-7b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in detail."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": IMAGE_URL
                        }
                    }
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    print("Testing Qwen2.5-VL-7B vision model...")
    print(f"Image URL: {IMAGE_URL}")
    print("\nSending request to API...")

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()

        result = response.json()

        print("\n" + "="*60)
        print("RESPONSE:")
        print("="*60)
        print(json.dumps(result, indent=2))

        # Extract and display the description
        if "choices" in result and len(result["choices"]) > 0:
            description = result["choices"][0]["message"]["content"]
            print("\n" + "="*60)
            print("IMAGE DESCRIPTION:")
            print("="*60)
            print(description)
            print("="*60)

            print("\n✅ Test successful!")
            return True
        else:
            print("\n❌ Unexpected response format")
            return False

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_qwen_vl()
