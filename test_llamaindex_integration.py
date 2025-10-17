#!/usr/bin/env python3
"""
Test script for LlamaIndex integration
"""
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# Test configuration
BASE_URL = "http://localhost:8083"  # Your Flask app URL
TEST_ORG_ID = "test_org_123"
TEST_USER_ID = "test_user_456"
TEST_FILE_ID = "test_file_789"

def test_setup():
    """Test NeonDB setup"""
    print("🔧 Testing NeonDB setup...")
    try:
        from setup_neondb import setup_neondb
        setup_neondb()
        print("✅ NeonDB setup successful")
        return True
    except Exception as e:
        print(f"❌ NeonDB setup failed: {e}")
        return False

def test_llamaindex_processor():
    """Test LlamaIndex processor initialization"""
    print("🤖 Testing LlamaIndex processor...")
    try:
        from llamaindex_integration import get_llamaindex_processor
        processor = get_llamaindex_processor()
        print("✅ LlamaIndex processor initialized successfully")
        print(f"📊 Embedding model: {processor.embed_model}")
        print(f"🤖 LLM model: {processor.llm}")
        return True
    except Exception as e:
        print(f"❌ LlamaIndex processor failed: {e}")
        return False

def test_file_upload():
    """Test file upload via API"""
    print("📤 Testing file upload...")
    
    # Create a simple test file
    test_content = """
    This is a test document for LlamaIndex integration.
    
    LlamaIndex is a powerful framework for building RAG applications.
    It provides advanced document processing, chunking, and retrieval capabilities.
    
    The system uses Qwen2.5 for answer generation and GTE-Large for embeddings.
    All data is stored in NeonDB PostgreSQL with pgvector for efficient vector search.
    """
    
    test_file_path = "/tmp/test_document.txt"
    with open(test_file_path, 'w') as f:
        f.write(test_content)
    
    try:
        url = f"{BASE_URL}/api/v2/upload"
        params = {
            "orgId": TEST_ORG_ID,
            "fileId": TEST_FILE_ID,
            "userId": TEST_USER_ID
        }
        
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_document.txt', f, 'text/plain')}
            response = requests.post(url, params=params, files=files)
        
        if response.status_code in [200, 202]:
            print("✅ File upload successful")
            print(f"📋 Response: {response.json()}")
            return True
        else:
            print(f"❌ File upload failed: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

def test_search():
    """Test search functionality"""
    print("🔍 Testing search...")
    try:
        url = f"{BASE_URL}/api/v2/search"
        data = {
            "query": "What is LlamaIndex?",
            "orgId": TEST_ORG_ID,
            "top_k": 5
        }
        
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Search successful")
            print(f"📊 Found {result.get('total_found', 0)} results")
            return True
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

def test_answer_generation():
    """Test answer generation"""
    print("🤖 Testing answer generation...")
    try:
        url = f"{BASE_URL}/api/v2/answer"
        data = {
            "query": "What models does the system use for embeddings and LLM?",
            "orgId": TEST_ORG_ID
        }
        
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Answer generation successful")
            print(f"🤖 Answer: {result.get('answer', 'No answer')}")
            print(f"📊 Confidence: {result.get('confidence', 'Unknown')}")
            print(f"📚 Sources: {len(result.get('sources', []))}")
            return True
        else:
            print(f"❌ Answer generation failed: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Answer generation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting LlamaIndex Integration Tests")
    print("=" * 50)
    
    tests = [
        ("NeonDB Setup", test_setup),
        ("LlamaIndex Processor", test_llamaindex_processor),
        ("File Upload", test_file_upload),
        ("Search", test_search),
        ("Answer Generation", test_answer_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
        print(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LlamaIndex integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()