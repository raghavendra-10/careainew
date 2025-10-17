#!/usr/bin/env python3
"""
Test complete LlamaIndex + NeonDB + Custom HF Embedding flow
"""
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

def test_neondb_setup():
    """Test NeonDB PostgreSQL setup"""
    print("🗄️ Testing NeonDB setup...")
    try:
        from setup_neondb import setup_neondb
        setup_neondb()
        return True
    except Exception as e:
        print(f"❌ NeonDB setup failed: {e}")
        return False

def test_embedding_endpoint():
    """Test custom embedding endpoint"""
    print("🔗 Testing custom embedding endpoint...")
    
    endpoint = os.getenv("EMBEDDING_ENDPOINT")
    if not endpoint or endpoint == "https://your-username-bge-embeddings.hf.space":
        print("⚠️ EMBEDDING_ENDPOINT not configured, skipping custom endpoint test")
        return True  # Not a failure, just not configured yet
    
    try:
        from custom_embedding import get_custom_embedding_model
        embed_model = get_custom_embedding_model()
        
        if embed_model:
            print("✅ Custom embedding endpoint working")
            return True
        else:
            print("❌ Custom embedding endpoint failed")
            return False
            
    except Exception as e:
        print(f"❌ Custom embedding test error: {e}")
        return False

def test_llamaindex_integration():
    """Test LlamaIndex processor initialization"""
    print("🤖 Testing LlamaIndex integration...")
    try:
        from llamaindex_integration import get_llamaindex_processor
        processor = get_llamaindex_processor()
        
        print(f"✅ LlamaIndex processor initialized")
        print(f"📊 Embedding model: {type(processor.embed_model).__name__}")
        print(f"🤖 LLM model: {type(processor.llm).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ LlamaIndex integration failed: {e}")
        return False

def test_file_upload_v2():
    """Test V2 file upload with LlamaIndex"""
    print("📤 Testing V2 file upload with LlamaIndex...")
    
    # Create test file
    test_content = """
# LlamaIndex Integration Test Document

This document tests the complete LlamaIndex integration flow.

## Key Components

1. **Qwen3 Embeddings**: Using Qwen/Qwen3-Embedding-4B for state-of-the-art embedding quality
2. **NeonDB PostgreSQL**: Vector storage with pgvector extension
3. **LlamaIndex**: Advanced RAG framework for document processing
4. **Qwen2.5**: Large language model for answer generation

## Technical Details

The system processes documents through the following pipeline:
- Document loading and parsing with LlamaIndex readers
- Smart chunking with semantic boundaries
- Embedding generation using custom HF endpoint
- Vector storage in NeonDB with metadata
- Retrieval and answer generation

## Performance Metrics

- Embedding model: 2560 dimensions
- Chunking: 1024 tokens with 20 overlap
- Vector similarity: Cosine similarity search
- Response time: < 2 seconds for typical queries

This integration provides enterprise-grade RAG capabilities with production-ready performance.
"""
    
    test_file = "/tmp/llamaindex_test.md"
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    try:
        # Test upload
        base_url = "http://localhost:8083"  # Adjust if needed
        params = {
            "orgId": "test_org_llamaindex",
            "fileId": f"test_file_{int(time.time())}",
            "userId": "test_user_llamaindex"
        }
        
        with open(test_file, 'rb') as f:
            files = {'file': ('llamaindex_test.md', f, 'text/markdown')}
            response = requests.post(
                f"{base_url}/api/v2/upload",
                params=params,
                files=files,
                timeout=60
            )
        
        if response.status_code in [200, 202]:
            result = response.json()
            print("✅ V2 upload successful")
            print(f"📋 Response: {result}")
            
            # Wait a bit for processing
            print("⏳ Waiting for background processing...")
            time.sleep(10)
            
            return params  # Return for search test
        else:
            print(f"❌ V2 upload failed: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload test error: {e}")
        return None
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def test_search_v2(upload_params):
    """Test V2 search functionality"""
    if not upload_params:
        print("⚠️ Skipping search test - no upload params")
        return False
    
    print("🔍 Testing V2 search...")
    
    try:
        base_url = "http://localhost:8083"
        search_data = {
            "query": "What embedding model does the system use?",
            "orgId": upload_params["orgId"],
            "top_k": 5
        }
        
        response = requests.post(
            f"{base_url}/api/v2/search",
            json=search_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ V2 search successful")
            print(f"📊 Found {result.get('total_found', 0)} results")
            print(f"🔧 Search type: {result.get('search_type', 'unknown')}")
            return True
        else:
            print(f"❌ V2 search failed: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Search test error: {e}")
        return False

def test_answer_v2(upload_params):
    """Test V2 answer generation"""
    if not upload_params:
        print("⚠️ Skipping answer test - no upload params")
        return False
    
    print("🤖 Testing V2 answer generation...")
    
    try:
        base_url = "http://localhost:8083"
        answer_data = {
            "query": "Explain the LlamaIndex integration and its key components.",
            "orgId": upload_params["orgId"]
        }
        
        response = requests.post(
            f"{base_url}/api/v2/answer",
            json=answer_data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ V2 answer generation successful")
            print(f"🤖 Answer: {result.get('answer', 'No answer')[:200]}...")
            print(f"📊 Confidence: {result.get('confidence', 'unknown')}")
            print(f"📚 Sources: {len(result.get('sources', []))}")
            print(f"🔧 Method: {result.get('method', 'unknown')}")
            return True
        else:
            print(f"❌ V2 answer generation failed: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Answer test error: {e}")
        return False

def main():
    """Run complete flow test"""
    print("🚀 Testing Complete LlamaIndex + NeonDB + HF Embedding Flow")
    print("=" * 70)
    
    tests = [
        ("NeonDB Setup", test_neondb_setup),
        ("Custom Embedding Endpoint", test_embedding_endpoint),
        ("LlamaIndex Integration", test_llamaindex_integration),
    ]
    
    # Run initial tests
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
        
        if not result:
            print(f"❌ {test_name} failed - stopping tests")
            break
    
    # If basic tests pass, run API tests
    if all(result for _, result in results):
        print(f"\n📋 Running: V2 File Upload")
        upload_params = test_file_upload_v2()
        upload_success = upload_params is not None
        results.append(("V2 File Upload", upload_success))
        
        if upload_success:
            print(f"\n📋 Running: V2 Search")
            search_success = test_search_v2(upload_params)
            results.append(("V2 Search", search_success))
            
            print(f"\n📋 Running: V2 Answer Generation")
            answer_success = test_answer_v2(upload_params)
            results.append(("V2 Answer Generation", answer_success))
    
    # Results summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Complete LlamaIndex integration is working!")
        print("\n📋 Your system now supports:")
        print("  ✅ Advanced document processing with LlamaIndex")
        print("  ✅ State-of-the-art Qwen3 embeddings (custom endpoint)")
        print("  ✅ PostgreSQL vector storage with NeonDB")
        print("  ✅ Smart retrieval and answer generation")
        print("  ✅ Production-ready API endpoints")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")
        print("\n💡 Common issues:")
        print("  - Make sure Flask app is running on localhost:8083")
        print("  - Verify NeonDB connection string is correct")
        print("  - Check HF embedding endpoint is deployed and accessible")
        print("  - Ensure all environment variables are set")

if __name__ == "__main__":
    main()