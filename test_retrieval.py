#!/usr/bin/env python3
"""
Test the retrieval system
"""
import sys
sys.path.append('/Users/raghavendra/careainew')

from retrieval_system import search_documents, get_context

def test_search():
    print("🧪 Testing retrieval system...")
    
    # Test search
    print("\n1. Testing document search:")
    results = search_documents(
        query="comprehensive test document",
        org_id="test-org",
        top_k=3
    )
    
    print(f"📊 Search results: {len(results)} found")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['similarity_score']:.3f}")
        print(f"     File: {result['file_id']}")
        print(f"     Text: {result['text'][:100]}...")
        print()
    
    # Test context
    print("\n2. Testing context generation:")
    context_result = get_context(
        query="test document processing",
        org_id="test-org",
        max_context_length=2000
    )
    
    print(f"📝 Context length: {len(context_result['context'])} chars")
    print(f"📋 Sources: {context_result['sources']}")
    print(f"🎯 Avg similarity: {context_result['avg_similarity']:.3f}")
    print(f"📄 Context preview: {context_result['context'][:300]}...")

if __name__ == "__main__":
    test_search()