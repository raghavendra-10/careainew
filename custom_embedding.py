#!/usr/bin/env python3
"""
Custom embedding class for LlamaIndex that uses our deployed HF endpoint
"""
import os
import requests
import numpy as np
from typing import List, Any
from llama_index.core.embeddings import BaseEmbedding
from dotenv import load_dotenv

load_dotenv()

class CustomHFEmbedding(BaseEmbedding):
    """Custom embedding class that uses deployed HuggingFace endpoint"""
    
    def __init__(
        self,
        endpoint_url: str = None,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        embed_batch_size: int = 32,
        timeout: int = 30,
        **kwargs
    ):
        # Initialize parent class first with all required fields
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            **kwargs
        )
        
        # Set instance variables after parent initialization
        endpoint_url = endpoint_url or os.getenv("EMBEDDING_ENDPOINT")
        
        if not endpoint_url:
            raise ValueError("EMBEDDING_ENDPOINT not found in environment")
        
        # Ensure endpoint URL is properly formatted
        if not endpoint_url.startswith("http"):
            endpoint_url = f"https://{endpoint_url}"
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"
        
        # Store as both private and public attributes for robustness
        self._endpoint_url = endpoint_url
        self.endpoint_url = endpoint_url
        self._timeout = timeout
        self.timeout = timeout
            
        print(f"🔗 Custom embedding endpoint: {endpoint_url}")
        print(f"🤖 Model: {model_name}")
        print(f"🔧 Timeout: {timeout}s")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query"""
        return self._get_text_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            # Ensure endpoint URL is available (try multiple attributes)
            endpoint_url = (getattr(self, '_endpoint_url', None) or 
                          getattr(self, 'endpoint_url', None) or 
                          os.getenv("EMBEDDING_ENDPOINT"))
            timeout = (getattr(self, '_timeout', None) or 
                      getattr(self, 'timeout', None) or 30)
            
            if not endpoint_url:
                raise ValueError("Embedding endpoint URL not available")
            
            # Ensure endpoint URL is properly formatted
            if not endpoint_url.startswith("http"):
                endpoint_url = f"https://{endpoint_url}"
            if not endpoint_url.endswith("/"):
                endpoint_url += "/"
            
            response = requests.post(
                f"{endpoint_url}embed_single",
                json={"text": text},
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["embedding"]
            else:
                print(f"❌ Embedding API error: {response.status_code}")
                print(f"📋 Response: {response.text}")
                raise Exception(f"Embedding API failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {str(e)}")
            raise Exception(f"Embedding request failed: {str(e)}")
        except Exception as e:
            print(f"❌ Embedding error: {str(e)}")
            raise
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        try:
            # Ensure endpoint URL is available (try multiple attributes)
            endpoint_url = (getattr(self, '_endpoint_url', None) or 
                          getattr(self, 'endpoint_url', None) or 
                          os.getenv("EMBEDDING_ENDPOINT"))
            timeout = (getattr(self, '_timeout', None) or 
                      getattr(self, 'timeout', None) or 30)
            
            if not endpoint_url:
                raise ValueError("Embedding endpoint URL not available")
            
            # Ensure endpoint URL is properly formatted
            if not endpoint_url.startswith("http"):
                endpoint_url = f"https://{endpoint_url}"
            if not endpoint_url.endswith("/"):
                endpoint_url += "/"
            
            # Split into batches if needed
            all_embeddings = []
            batch_size = getattr(self, 'embed_batch_size', 32)
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                response = requests.post(
                    f"{endpoint_url}embed",
                    json={"texts": batch_texts},
                    timeout=timeout,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    batch_embeddings = result["embeddings"]
                    all_embeddings.extend(batch_embeddings)
                    
                    print(f"✅ Processed batch {i//batch_size + 1}: {len(batch_texts)} texts")
                else:
                    print(f"❌ Batch embedding API error: {response.status_code}")
                    print(f"📋 Response: {response.text}")
                    raise Exception(f"Batch embedding API failed: {response.status_code}")
            
            print(f"🎯 Total embeddings generated: {len(all_embeddings)}")
            return all_embeddings
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Batch request error: {str(e)}")
            raise Exception(f"Batch embedding request failed: {str(e)}")
        except Exception as e:
            print(f"❌ Batch embedding error: {str(e)}")
            raise
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of get_query_embedding"""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of get_text_embedding"""
        return self._get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async version of get_text_embeddings"""
        return self._get_text_embeddings(texts)
    
    def test_connection(self) -> bool:
        """Test if the embedding endpoint is working"""
        try:
            # Ensure endpoint URL is available
            endpoint_url = getattr(self, '_endpoint_url', None) or os.getenv("EMBEDDING_ENDPOINT")
            timeout = getattr(self, '_timeout', 30)
            
            if not endpoint_url:
                print("❌ No endpoint URL available for testing")
                return False
            
            # Ensure endpoint URL is properly formatted
            if not endpoint_url.startswith("http"):
                endpoint_url = f"https://{endpoint_url}"
            if not endpoint_url.endswith("/"):
                endpoint_url += "/"
            
            print(f"🧪 Testing connection to: {endpoint_url}")
            response = requests.get(
                f"{endpoint_url}",
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Endpoint healthy: {result}")
                return True
            else:
                print(f"❌ Endpoint unhealthy: {response.status_code}")
                print(f"📋 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            import traceback
            print(f"🐛 Traceback: {traceback.format_exc()}")
            return False

def get_custom_embedding_model():
    """Get the custom embedding model instance"""
    try:
        embedding_model = CustomHFEmbedding()
        print("✅ Custom embedding model initialized (skipping connection test)")
        print("⚠️ Note: Server-side padding configuration needs to be fixed")
        return embedding_model
            
    except Exception as e:
        print(f"❌ Failed to initialize custom embedding model: {str(e)}")
        return None