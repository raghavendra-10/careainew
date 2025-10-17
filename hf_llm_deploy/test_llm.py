#!/usr/bin/env python3
"""
Test script for Qwen LLM deployment
"""
import requests
import json

def test_llm_local():
    """Test LLM running locally"""
    base_url = "http://localhost:7860"
    
    print("🧪 Testing LLM deployment locally...")
    
    # Test health check
    print("\n1. Health check:")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test simple generation
    print("\n2. Simple generation:")
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "prompt": "Explain artificial intelligence in simple terms.",
                "max_length": 200,
                "temperature": 0.7
            }
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {result.get('response', 'No response')[:200]}...")
    except Exception as e:
        print(f"Generation test failed: {e}")
    
    # Test RAG chat
    print("\n3. RAG chat:")
    try:
        context = """
        Artificial Intelligence (AI) is a branch of computer science that focuses on creating 
        systems capable of performing tasks that typically require human intelligence. 
        These tasks include learning, reasoning, perception, and decision-making.
        """
        
        response = requests.post(
            f"{base_url}/chat",
            json={
                "query": "What is AI?",
                "context": context,
                "max_length": 150,
                "temperature": 0.7
            }
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Answer: {result.get('answer', 'No answer')}")
        print(f"Context used: {result.get('context_used', False)}")
    except Exception as e:
        print(f"RAG chat test failed: {e}")

def test_llm_remote(space_url):
    """Test LLM on HuggingFace Spaces"""
    print(f"🧪 Testing LLM on HF Spaces: {space_url}")
    
    # Test health check
    print("\n1. Health check:")
    try:
        response = requests.get(f"{space_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test chat endpoint
    print("\n2. Chat test:")
    try:
        response = requests.post(
            f"{space_url}/chat",
            json={
                "query": "What is machine learning?",
                "max_length": 100
            }
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Answer: {result.get('answer', 'No answer')}")
    except Exception as e:
        print(f"Chat test failed: {e}")

if __name__ == "__main__":
    # Test locally first
    test_llm_local()
    
    # Uncomment to test on HF Spaces
    # test_llm_remote("https://your-username-space-name.hf.space")