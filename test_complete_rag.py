#!/usr/bin/env python3
"""
Test complete RAG pipeline: Upload → Retrieval → LLM Answer
"""
import requests
import json
import time

def test_complete_rag_pipeline():
    """Test the full RAG pipeline"""
    base_url = "http://localhost:8002"
    
    print("🧪 Testing Complete RAG Pipeline...")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. 📋 Health Check")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Main API Status: {response.status_code}")
    except:
        print("❌ Main API not responding")
        return
    
    # Test 2: Check LLM health
    print("\n2. 🤖 LLM Health Check")
    try:
        from llm_integration import Qwen3LLMClient
        client = Qwen3LLMClient()
        health = client.health_check()
        if health["healthy"]:
            print("✅ Qwen3 LLM is healthy")
        else:
            print(f"⚠️ Qwen3 LLM issue: {health.get('error', 'Unknown')}")
    except Exception as e:
        print(f"❌ LLM health check failed: {e}")
    
    # Test 3: Simple search (without LLM)
    print("\n3. 🔍 Testing Search (Retrieval Only)")
    search_data = {
        "query": "test document processing"
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/v3/search",
            json=search_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Search Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Found {result['total_results']} results")
            if result['results']:
                print(f"📋 Top result: {result['results'][0]['text'][:100]}...")
                print(f"🎯 Similarity: {result['results'][0]['similarity_score']:.3f}")
        else:
            print(f"❌ Search failed: {response.text}")
    except Exception as e:
        print(f"❌ Search test failed: {e}")
    
    # Test 4: Complete RAG Chat
    print("\n4. 💬 Testing Complete RAG Chat")
    chat_data = {
        "query": "What is this document about?"
    }
    
    try:
        print("🚀 Sending chat request...")
        response = requests.post(
            f"{base_url}/api/v3/chat",
            json=chat_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Chat Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n📝 Query: {result['query']}")
            print(f"🤖 Answer: {result.get('answer', 'No answer generated')}")
            print(f"📊 Sources: {result.get('sources', [])}")
            print(f"🔧 Pipeline: {result.get('pipeline', 'unknown')}")
            print(f"🧠 Model: {result.get('llm_metadata', {}).get('model', 'unknown')}")
            print(f"📋 Search Results: {len(result.get('search_results', []))}")
            
            # Show context metadata
            ctx_meta = result.get('context_metadata', {})
            print(f"📈 Avg Similarity: {ctx_meta.get('avg_similarity', 0):.3f}")
            print(f"📄 Total Chunks: {ctx_meta.get('total_chunks', 0)}")
            
            if result.get('pipeline') == 'complete_rag':
                print("✅ Complete RAG pipeline successful!")
            elif result.get('pipeline') == 'retrieval_only':
                print("⚠️ RAG pipeline partially successful (retrieval only)")
                print(f"LLM Error: {result.get('llm_error', 'Unknown')}")
            
        else:
            print(f"❌ Chat failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
    
    # Test 5: Different questions
    print("\n5. 🎯 Testing Different Questions")
    test_questions = [
        "How does LlamaIndex process documents?",
        "What are the main sections in this document?",
        "Explain the chunking functionality"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n   Question {i}: {question}")
        try:
            response = requests.post(
                f"{base_url}/api/v3/chat",
                json={"query": question},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', 'No answer')
                pipeline = result.get('pipeline', 'unknown')
                print(f"   Answer ({pipeline}): {answer[:150]}...")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 RAG Pipeline Test Complete!")

if __name__ == "__main__":
    test_complete_rag_pipeline()