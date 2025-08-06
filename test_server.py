"""Test suite for the SitEmb-v1.5-Qwen3-note Server.

This module provides tests for verifying the functionality of the embedding server,
including health checks and encoding endpoints.
"""

import sys

import requests

# Server configuration
SERVER_URL = "http://localhost:8000"


def test_server_health():
    """Test if the server is running and healthy"""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✓ Server health check passed")
            return True
        print(f"✗ Server health check failed with status code: {response.status_code}")
        return False
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running or not accessible")
        return False


def test_encode_texts():
    """Test encoding texts"""
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
    ]

    payload = {
        "texts": texts,
        "batch_size": 2,
        "max_length": 512,
        "pooling_type": "eos",
        "normalize": True,
    }

    try:
        response = requests.post(f"{SERVER_URL}/encode_texts", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("✓ Text encoding test passed")
            print(f"  Number of embeddings: {len(result['embeddings'])}")
            print(f"  Embedding dimension: {len(result['embeddings'][0])}")
            print(f"  Sample embedding (first 5 values): {result['embeddings'][0][:5]}")
            return True
        print(f"✗ Text encoding test failed with status code: {response.status_code}")
        print(f"  Response: {response.text}")
        return False
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running or not accessible")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Text encoding test failed with error: {str(e)}")
        return False


def test_encode_queries():
    """Test encoding queries"""
    queries = [
        "What is machine learning?",
        "How to program in Python?",
        "Explain artificial intelligence",
    ]

    payload = {
        "queries": queries,
        "batch_size": 2,
        "max_length": 512,
        "pooling_type": "eos",
        "normalize": True,
    }

    try:
        response = requests.post(
            f"{SERVER_URL}/encode_queries", json=payload, timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print("✓ Query encoding test passed")
            print(f"  Number of embeddings: {len(result['embeddings'])}")
            print(f"  Embedding dimension: {len(result['embeddings'][0])}")
            print(f"  Sample embedding (first 5 values): {result['embeddings'][0][:5]}")
            return True
        print(f"✗ Query encoding test failed with status code: {response.status_code}")
        print(f"  Response: {response.text}")
        return False
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running or not accessible")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Query encoding test failed with error: {str(e)}")
        return False


def main():
    """Main function to run all tests for the SitEmb-v1.5-Qwen3-note Server."""
    print("Testing SitEmb-v1.5-Qwen3-note Server...\n")

    # Test server health
    if not test_server_health():
        print("\nPlease make sure the server is running before testing.")
        sys.exit(1)

    print()

    # Test text encoding
    test_encode_texts()

    print()

    # Test query encoding
    test_encode_queries()

    print("\nTesting completed!")


if __name__ == "__main__":
    main()
