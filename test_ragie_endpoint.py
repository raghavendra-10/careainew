#!/usr/bin/env python3
"""
Test script for the new /chat-ragie endpoint
"""
import requests
import json

def test_chat_ragie_endpoint():
    """Test the new /chat-ragie endpoint"""
    
    # Test endpoint URL (assuming app is running on localhost:8000)
    url = "http://localhost:8000/chat-ragie"
    
    # Test payload
    test_payload = {
        "query": "How long is andor in the market?",
        "orgId": "test-org-123",
        "top_k": 5,
        "rerank": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("🧪 Testing /chat-ragie endpoint...")
        print(f"📡 Sending request to: {url}")
        print(f"📝 Payload: {json.dumps(test_payload, indent=2)}")
        
        response = requests.post(url, json=test_payload, headers=headers, timeout=60)
        
        print(f"\n✅ Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("📊 Response Structure:")
            print(f"  - Response length: {len(response_data.get('response', ''))}")
            print(f"  - Ragie chunks: {response_data.get('sources', {}).get('ragie_chunks', 0)}")
            print(f"  - Total context chunks: {response_data.get('sources', {}).get('total_context_chunks', 0)}")
            print(f"  - API version: {response_data.get('api_version', 'N/A')}")
            
            # Show first 200 chars of response
            ai_response = response_data.get('response', '')
            if ai_response:
                print(f"\n💬 AI Response Preview: {ai_response[:200]}...")
            
            # Show sources
            ragie_docs = response_data.get('ragie_documents', [])
            if ragie_docs:
                print(f"\n📚 Ragie Documents ({len(ragie_docs)}):")
                for i, doc in enumerate(ragie_docs[:3], 1):
                    print(f"  {i}. {doc.get('document_name', 'Unknown')} (Score: {doc.get('score', 0):.3f})")
            
            print(f"\n✅ Test completed successfully!")
            return True
            
        else:
            print(f"❌ Error Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the Flask app is running on localhost:8000")
        print("   Start the app with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Test Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_chat_ragie_endpoint()
    exit(0 if success else 1)