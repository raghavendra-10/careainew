#!/usr/bin/env python3
"""
Simple custom embedding that works reliably with LlamaIndex
Using deployed HF Spaces Gradio API
"""
import os
import json
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()

def get_embedding_from_hf_endpoint(text: str) -> List[float]:
    """Get embedding for a single text using HF Spaces Gradio API"""
    try:
        from gradio_client import Client
    except ImportError:
        # Fallback to requests if gradio_client not available
        return get_embedding_from_hf_endpoint_fallback(text)
    
    space_name = os.getenv("EMBEDDING_SPACENAME")
    
    
    
    print(f"🔗 Calling HF Space: {space_name}")
    
    try:
        client = Client(space_name)
        result = client.predict(
            text,  # text_input
            1      # batch_size_input
        )
        
        # Parse JSON result (Gradio returns JSON string)
        if isinstance(result, str):
            result_data = json.loads(result)
        else:
            result_data = result
        embeddings = result_data.get("embeddings", [])
        
        if not embeddings:
            raise ValueError("No embeddings returned from HF endpoint")
        
        print(f"✅ Got embedding with dimension: {len(embeddings[0])}")
        return embeddings[0]
        
    except Exception as e:
        print(f"❌ HF Spaces API error: {str(e)}")
        # Fallback to direct HTTP requests
        return get_embedding_from_hf_endpoint_fallback(text)

def get_embedding_from_hf_endpoint_fallback(text: str) -> List[float]:
    """Fallback method using direct HTTP requests"""
    import requests
    
    endpoint_url = os.getenv("EMBEDDING_ENDPOINT")
    
    if not endpoint_url:
        raise ValueError("EMBEDDING_ENDPOINT not found in environment")
    
    # Ensure endpoint URL is properly formatted
    if not endpoint_url.startswith("http"):
        endpoint_url = f"https://{endpoint_url}"
    if not endpoint_url.endswith("/"):
        endpoint_url += "/"
    
    print(f"🔗 Fallback: Calling HF endpoint: {endpoint_url}embed_single")
    print(f"📝 Text preview: {text[:100]}...")
    
    try:
        response = requests.post(
            f"{endpoint_url}embed_single",
            json={"text": text},
            timeout=None,  # No timeout
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            embedding = result["embedding"]
            print(f"✅ Got embedding, dimension: {len(embedding)}")
            print(f"🔢 First 10 values: {embedding[:10]}")
            print(f"🔢 Last 10 values: {embedding[-10:]}")
            return embedding
        else:
            print(f"❌ HF API error: {response.status_code}")
            print(f"📋 Response: {response.text}")
            raise Exception(f"HF API failed: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {str(e)}")
        raise Exception(f"HF request failed: {str(e)}")
    except Exception as e:
        print(f"❌ Embedding error: {str(e)}")
        raise

def get_embeddings_from_hf_endpoint(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts using HF Spaces Gradio API with batch processing"""
    try:
        from gradio_client import Client
    except ImportError:
        # Fallback to requests if gradio_client not available
        return get_embeddings_from_hf_endpoint_fallback(texts)
    
    endpoint_url = os.getenv("EMBEDDING_ENDPOINT")
    
    if not endpoint_url:
        raise ValueError("EMBEDDING_ENDPOINT not found in environment")
    
    # Extract space name from URL
    if "hf.space" in endpoint_url:
        space_name = endpoint_url.replace("https://", "").replace(".hf.space", "").replace("-", "/", 1)
    else:
        space_name = endpoint_url
    
    print(f"🔗 Calling HF Space: {space_name}")
    print(f"📊 Processing {len(texts)} texts")
    
    try:
        client = Client(space_name)
        
        # Join texts with separator for batch processing
        texts_str = "|||".join(texts)
        
        result = client.predict(
            texts_str,  # text_input
            16          # batch_size_input  
        )
        
        # Parse JSON result (Gradio returns JSON string)
        if isinstance(result, str):
            result_data = json.loads(result)
        else:
            result_data = result
        embeddings = result_data.get("embeddings", [])
        
        if not embeddings:
            raise ValueError("No embeddings returned from HF endpoint")
        
        print(f"✅ Got {len(embeddings)} embeddings with dimension: {len(embeddings[0])}")
        return embeddings
        
    except Exception as e:
        print(f"❌ HF Spaces API error: {str(e)}")
        # Fallback to direct HTTP requests
        return get_embeddings_from_hf_endpoint_fallback(texts)

def get_embeddings_from_hf_endpoint_fallback(texts: List[str]) -> List[List[float]]:
    """Fallback method using direct HTTP requests"""
    endpoint_url = os.getenv("EMBEDDING_ENDPOINT")
    
    if not endpoint_url:
        raise ValueError("EMBEDDING_ENDPOINT not found in environment")
    
    # Ensure endpoint URL is properly formatted
    if not endpoint_url.startswith("http"):
        endpoint_url = f"https://{endpoint_url}"
    if not endpoint_url.endswith("/"):
        endpoint_url += "/"
    
    print(f"🔗 Fallback: Calling HF endpoint: {endpoint_url}embed")
    print(f"📊 Processing {len(texts)} texts")
    
    # For large batches, split into smaller chunks to avoid timeout
    batch_size = 5  # Process 5 chunks at a time
    all_embeddings = []
    
    if len(texts) > batch_size:
        print(f"📦 Large batch detected, splitting into batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch)} texts)")
            
            batch_embeddings = _get_single_batch_embeddings(endpoint_url, batch, batch_num=i//batch_size + 1)
            all_embeddings.extend(batch_embeddings)
            
            # Small delay between batches to avoid overwhelming the endpoint
            import time
            time.sleep(1)
            
        print(f"✅ Completed all batches: {len(all_embeddings)} total embeddings")
        return all_embeddings
    else:
        # Small batch, process directly
        return _get_single_batch_embeddings(endpoint_url, texts)

def _get_single_batch_embeddings(endpoint_url: str, texts: List[str], batch_num: int = 0) -> List[List[float]]:
    """Process a single batch of embeddings with no timeout and retry logic"""
    batch_label = f" (batch {batch_num})" if batch_num > 0 else ""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Attempt {attempt + 1}/{max_retries}: Processing {len(texts)} texts{batch_label} (no timeout)")
            
            # No timeout - let it run as long as needed
            response = requests.post(
                f"{endpoint_url}embed",
                json={"texts": texts},
                timeout=None,  # No timeout
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📊 Response status: {response.status_code}{batch_label}")
            
            if response.status_code == 200:
                result = response.json()
                embeddings = result["embeddings"]
                print(f"✅ Got {len(embeddings)} embeddings, dimension: {len(embeddings[0]) if embeddings else 0}{batch_label}")
                return embeddings
            else:
                print(f"❌ HF API error: {response.status_code}{batch_label}")
                print(f"📋 Response: {response.text}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying in 5 seconds...")
                    import time
                    time.sleep(5)
                else:
                    raise Exception(f"HF API failed after {max_retries} attempts: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error{batch_label} (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                raise Exception(f"HF request failed after {max_retries} attempts: {str(e)}")
        except Exception as e:
            print(f"❌ Embedding error{batch_label}: {str(e)}")
            raise