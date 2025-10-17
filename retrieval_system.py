#!/usr/bin/env python3
"""
Retrieval system for NeonDB vector search
"""
import os
import asyncpg
import asyncio
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from simple_custom_embedding import get_embedding_from_hf_endpoint

load_dotenv()

class NeonDBRetrieval:
    def __init__(self):
        self.db_url = os.getenv("NEON_DATABASE_URL")
        if not self.db_url:
            raise ValueError("NEON_DATABASE_URL not found")
    
    async def search_similar_chunks(
        self, 
        query: str, 
        org_id: str = None,
        file_ids: List[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query: Search query text
            org_id: Filter by organization ID (optional)
            file_ids: Filter by specific file IDs (optional)
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar chunks with metadata and similarity scores
        """
        print(f"🔍 Searching for: '{query[:100]}...' (top_k={top_k})")
        
        # Get query embedding
        print("🧮 Generating query embedding...")
        query_embedding = get_embedding_from_hf_endpoint(query)
        query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Build SQL query with filters
        where_conditions = []
        params = [query_embedding_str, top_k]
        param_count = 2
        
        if org_id:
            param_count += 1
            where_conditions.append(f"org_id = ${param_count}")
            params.append(org_id)
        
        if file_ids:
            param_count += 1
            where_conditions.append(f"file_id = ANY(${param_count})")
            params.append(file_ids)
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # SQL query with cosine similarity
        sql_query = f"""
        SELECT 
            node_id,
            text,
            file_id,
            org_id,
            chunk_index,
            metadata,
            1 - (embedding <=> $1::vector) as similarity_score,
            created_at
        FROM llamaindex_embeddings
        {where_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """
        
        print(f"📊 Executing similarity search...")
        print(f"🔧 Filters: org_id={org_id}, file_ids={file_ids}")
        
        conn = await asyncpg.connect(self.db_url)
        try:
            rows = await conn.fetch(sql_query, *params)
            
            results = []
            for row in rows:
                similarity_score = float(row['similarity_score'])
                
                # Apply similarity threshold
                if similarity_score >= similarity_threshold:
                    result = {
                        "node_id": row['node_id'],
                        "text": row['text'],
                        "file_id": row['file_id'],
                        "org_id": row['org_id'],
                        "chunk_index": row['chunk_index'],
                        "metadata": row['metadata'],
                        "similarity_score": similarity_score,
                        "created_at": row['created_at'].isoformat() if row['created_at'] else None
                    }
                    results.append(result)
            
            print(f"✅ Found {len(results)} similar chunks (threshold: {similarity_threshold})")
            
            # Log top results
            for i, result in enumerate(results[:3]):
                print(f"📋 Result {i+1}: score={result['similarity_score']:.3f}, file={result['file_id']}, text='{result['text'][:100]}...'")
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            raise
        finally:
            await conn.close()
    
    async def get_context_for_query(
        self,
        query: str,
        org_id: str = None,
        file_ids: List[str] = None,
        max_context_length: int = 4000,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Get relevant context for a query, optimized for LLM consumption
        
        Args:
            query: User query
            org_id: Filter by organization
            file_ids: Filter by specific files
            max_context_length: Maximum context length in characters
            top_k: Maximum chunks to consider
            
        Returns:
            Dictionary with context and metadata
        """
        print(f"🎯 Getting context for query: '{query[:100]}...'")
        
        # Get similar chunks
        similar_chunks = await self.search_similar_chunks(
            query=query,
            org_id=org_id,
            file_ids=file_ids,
            top_k=top_k,
            similarity_threshold=0.6  # Lower threshold for context
        )
        
        if not similar_chunks:
            return {
                "context": "",
                "sources": [],
                "total_chunks": 0,
                "avg_similarity": 0.0
            }
        
        # Build context string
        context_parts = []
        current_length = 0
        used_chunks = []
        sources = set()
        
        for chunk in similar_chunks:
            chunk_text = chunk['text']
            chunk_length = len(chunk_text)
            
            # Check if adding this chunk would exceed max length
            if current_length + chunk_length > max_context_length:
                # Try to add a truncated version
                remaining_space = max_context_length - current_length
                if remaining_space > 100:  # Only if we have reasonable space
                    truncated_text = chunk_text[:remaining_space - 10] + "..."
                    context_parts.append(f"[Source: {chunk['file_id']}, Score: {chunk['similarity_score']:.3f}]\n{truncated_text}")
                    used_chunks.append(chunk)
                    sources.add(chunk['file_id'])
                break
            
            # Add full chunk
            context_parts.append(f"[Source: {chunk['file_id']}, Score: {chunk['similarity_score']:.3f}]\n{chunk_text}")
            current_length += chunk_length + 50  # Extra for formatting
            used_chunks.append(chunk)
            sources.add(chunk['file_id'])
        
        context = "\n\n---\n\n".join(context_parts)
        avg_similarity = sum(chunk['similarity_score'] for chunk in used_chunks) / len(used_chunks) if used_chunks else 0.0
        
        result = {
            "context": context,
            "sources": list(sources),
            "total_chunks": len(used_chunks),
            "avg_similarity": avg_similarity,
            "query": query,
            "chunks_metadata": [
                {
                    "file_id": chunk['file_id'],
                    "chunk_index": chunk['chunk_index'],
                    "similarity_score": chunk['similarity_score']
                }
                for chunk in used_chunks
            ]
        }
        
        print(f"📝 Context built: {len(context)} chars, {len(used_chunks)} chunks, avg_sim={avg_similarity:.3f}")
        return result

def search_documents(
    query: str,
    org_id: str = None,
    file_ids: List[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Synchronous wrapper for document search
    """
    retrieval = NeonDBRetrieval()
    return asyncio.run(retrieval.search_similar_chunks(query, org_id, file_ids, top_k))

def get_context(
    query: str,
    org_id: str = None,
    file_ids: List[str] = None,
    max_context_length: int = 4000
) -> Dict[str, Any]:
    """
    Synchronous wrapper for getting context
    """
    retrieval = NeonDBRetrieval()
    return asyncio.run(retrieval.get_context_for_query(query, org_id, file_ids, max_context_length))