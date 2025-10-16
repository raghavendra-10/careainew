"""
Celery Background Tasks for Production-Ready Processing
Phase 1: Core Infrastructure
"""

import os
import traceback
import time
import hashlib
from celery import Task
from celery_config import celery_app
from redis_manager import redis_manager
from tenacity import retry, stop_after_attempt, wait_exponential

# Import existing functions from app.py - we'll reference them
# This preserves all existing functionality while adding queue support

class CallbackTask(Task):
    """Base task class with callback support"""
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        redis_manager.set_progress(
            task_id, "completed", 100, "Task completed successfully"
        )
        
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        redis_manager.set_progress(
            task_id, "error", 0, f"Task failed: {str(exc)}"
        )

@celery_app.task(bind=True, base=CallbackTask)
def process_file_upload(self, file_data, upload_params):
    """
    Background task for file upload processing
    Replaces the threading approach in app.py
    """
    task_id = self.request.id
    filename = upload_params.get('filename', 'unknown')
    
    try:
        # Update progress
        redis_manager.set_progress(
            task_id, "processing", 10, "Starting file processing", filename
        )
        
        # Import here to avoid circular imports
        from file_processor import process_file_upload_internal
        
        # Process the file using existing functionality
        result = process_file_upload_internal(file_data, upload_params, task_id)
        
        # Update final progress
        redis_manager.set_progress(
            task_id, "completed", 100, "File processing completed", filename
        )
        
        # Store result
        redis_manager.set_task_result(task_id, result)
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ File upload task failed: {error_msg}")
        traceback.print_exc()
        
        # Update error progress
        redis_manager.set_progress(
            task_id, "error", 0, f"Error: {error_msg}", filename
        )
        
        raise self.retry(
            exc=e, 
            countdown=int(os.environ.get('RETRY_DELAY', 5)), 
            max_retries=int(os.environ.get('RETRY_MAX_ATTEMPTS', 3))
        )

@celery_app.task(bind=True, base=CallbackTask)
def process_embedding_task(self, chunks, embedding_params):
    """
    Background task for embedding generation
    """
    task_id = self.request.id
    
    try:
        # Update progress
        redis_manager.set_progress(
            task_id, "embedding", 20, "Generating embeddings"
        )
        
        # Check for cached embeddings
        embedding_results = []
        cache_hits = 0
        
        for i, chunk in enumerate(chunks):
            # Create hash for caching
            content_hash = hashlib.md5(chunk.encode()).hexdigest()
            
            # Check cache first
            cached_embedding = redis_manager.get_cached_embedding(content_hash)
            
            if cached_embedding:
                embedding_results.append(cached_embedding)
                cache_hits += 1
            else:
                # Import here to avoid circular imports
                from utils import embed_chunks
                
                # Generate new embedding
                embedding = embed_chunks([chunk])
                if embedding and len(embedding) > 0:
                    embedding_results.append(embedding[0])
                    # Cache the result
                    redis_manager.cache_embedding(content_hash, embedding[0])
                else:
                    raise Exception(f"Failed to generate embedding for chunk {i}")
            
            # Update progress
            progress = 20 + int((i + 1) / len(chunks) * 60)
            redis_manager.set_progress(
                task_id, "embedding", progress, f"Processing chunk {i+1}/{len(chunks)}"
            )
        
        print(f"📊 Embedding task completed. Cache hits: {cache_hits}/{len(chunks)}")
        
        # Update completion progress
        redis_manager.set_progress(
            task_id, "completed", 100, "Embeddings generated successfully"
        )
        
        return {
            "embeddings": embedding_results,
            "cache_hits": cache_hits,
            "total_chunks": len(chunks)
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Embedding task failed: {error_msg}")
        traceback.print_exc()
        
        redis_manager.set_progress(
            task_id, "error", 0, f"Embedding error: {error_msg}"
        )
        
        raise self.retry(
            exc=e,
            countdown=int(os.environ.get('RETRY_DELAY', 5)),
            max_retries=int(os.environ.get('RETRY_MAX_ATTEMPTS', 3))
        )

@celery_app.task(bind=True, base=CallbackTask)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def send_webhook(self, webhook_url, payload, headers=None):
    """
    Background task for webhook delivery with retry logic
    """
    task_id = self.request.id
    
    try:
        import requests
        
        redis_manager.set_progress(
            task_id, "sending", 50, "Sending webhook"
        )
        
        if headers is None:
            headers = {"Content-Type": "application/json"}
        
        response = requests.post(
            webhook_url,
            json=payload,
            headers=headers,
            timeout=int(os.environ.get('WEBHOOK_TIMEOUT', 30))
        )
        
        if response.status_code == 200:
            redis_manager.set_progress(
                task_id, "completed", 100, "Webhook delivered successfully"
            )
            
            return {
                "status": "success",
                "status_code": response.status_code,
                "response": response.text[:500]  # Truncate response
            }
        else:
            raise Exception(f"Webhook failed with status {response.status_code}")
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Webhook task failed: {error_msg}")
        
        redis_manager.set_progress(
            task_id, "error", 0, f"Webhook error: {error_msg}"
        )
        
        raise self.retry(
            exc=e,
            countdown=int(os.environ.get('RETRY_DELAY', 5)),
            max_retries=int(os.environ.get('RETRY_MAX_ATTEMPTS', 3))
        )

@celery_app.task(bind=True, base=CallbackTask)
def process_agent_task(self, agent_params):
    """
    Background task for agent processing
    """
    task_id = self.request.id
    agent_type = agent_params.get('type', 'unknown')
    
    try:
        redis_manager.set_progress(
            task_id, "processing", 25, f"Starting {agent_type} agent"
        )
        
        # Import here to avoid circular imports
        # For now, return a placeholder - agent processing can be implemented later
        result = {
            "status": "completed",
            "agent_type": agent_type,
            "message": "Agent processing placeholder"
        }
        
        redis_manager.set_progress(
            task_id, "completed", 100, f"{agent_type} agent completed"
        )
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Agent task failed: {error_msg}")
        traceback.print_exc()
        
        redis_manager.set_progress(
            task_id, "error", 0, f"Agent error: {error_msg}"
        )
        
        raise self.retry(
            exc=e,
            countdown=int(os.environ.get('RETRY_DELAY', 5)),
            max_retries=int(os.environ.get('RETRY_MAX_ATTEMPTS', 3))
        )

# Helper function to get task status
def get_task_status(task_id: str):
    """Get comprehensive task status"""
    # Check Redis progress first
    progress_data = redis_manager.get_progress(task_id)
    
    # Check Celery task state
    task = celery_app.AsyncResult(task_id)
    
    return {
        "task_id": task_id,
        "celery_state": task.state,
        "celery_info": task.info,
        "progress_data": progress_data,
        "is_ready": task.ready(),
        "successful": task.successful() if task.ready() else False,
        "failed": task.failed() if task.ready() else False
    }

# Health check task
@celery_app.task
def health_check():
    """Simple health check task"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "redis_connected": redis_manager.is_connected()
    }