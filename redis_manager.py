"""
Redis-based Progress Tracking and Cache Management
Phase 1: Core Infrastructure
"""

import json
import time
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class RedisManager:
    """Manages Redis connections and provides progress tracking functionality"""
    
    def __init__(self):
        self.redis_client = None
        self.client_type = None
        self.connect()
    
    def connect(self):
        """Establish Redis connection with retry logic"""
        try:
            # Check for Upstash Redis first (HTTP REST API)
            upstash_url = os.environ.get('UPSTASH_REDIS_REST_URL', '')
            upstash_token = os.environ.get('UPSTASH_REDIS_REST_TOKEN', '')
            
            if upstash_url and upstash_token:
                # Use Upstash REST client
                from upstash_redis import Redis
                self.redis_client = Redis(url=upstash_url, token=upstash_token)
                self.client_type = 'upstash'
                
                # Test connection
                self.redis_client.set("test", "connection")
                test_result = self.redis_client.get("test")
                if test_result == "connection":
                    print("✅ Upstash Redis connection established")
                    return
                else:
                    raise Exception("Upstash test failed")
            
            # Try standard Redis URL connection
            redis_url = os.environ.get('REDIS_URL', '')
            
            if redis_url and redis_url.startswith(('redis://', 'rediss://')):
                import redis
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=10,
                    socket_timeout=10,
                    retry_on_timeout=True
                )
                self.client_type = 'standard'
                
                # Test connection
                self.redis_client.ping()
                print("✅ Standard Redis connection established")
                return
            
            # Fallback to individual parameters (for local Redis)
            import redis
            self.redis_client = redis.Redis(
                host=os.environ.get('REDIS_HOST', 'localhost'),
                port=int(os.environ.get('REDIS_PORT', 6379)),
                db=int(os.environ.get('REDIS_DB', 0)),
                password=os.environ.get('REDIS_PASSWORD', None) or None,
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10,
                retry_on_timeout=True
            )
            self.client_type = 'standard'
            
            # Test connection
            self.redis_client.ping()
            print("✅ Local Redis connection established")
            
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            self.redis_client = None
            self.client_type = None
    
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        if not self.redis_client:
            return False
        try:
            if self.client_type == 'upstash':
                # Upstash doesn't have ping, test with a simple operation
                self.redis_client.set("health_check", "ok", ex=10)
                return True
            else:
                # Standard Redis ping
                self.redis_client.ping()
                return True
        except:
            return False
    
    def set_progress(self, task_id: str, status: str, progress: int, 
                    stage: str, filename: str = "", extra_data: Dict = None):
        """Set task progress in Redis with expiration"""
        if not self.is_connected():
            print(f"⚠️ Redis unavailable, progress not saved for {task_id}")
            return False
            
        try:
            progress_data = {
                "task_id": task_id,
                "status": status,
                "progress": progress,
                "stage": stage,
                "filename": filename,
                "timestamp": time.time(),
                "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if extra_data:
                progress_data.update(extra_data)
            
            # Store with 24 hour expiration
            key = f"progress:{task_id}"
            self.redis_client.setex(
                key, 
                86400,  # 24 hours
                json.dumps(progress_data)
            )
            
            print(f"📊 Progress updated: {task_id} → {status} ({progress}%)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to set progress: {e}")
            return False
    
    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task progress from Redis"""
        if not self.is_connected():
            return None
            
        try:
            key = f"progress:{task_id}"
            data = self.redis_client.get(key)
            
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            print(f"❌ Failed to get progress: {e}")
            return None
    
    def delete_progress(self, task_id: str) -> bool:
        """Delete task progress from Redis"""
        if not self.is_connected():
            return False
            
        try:
            key = f"progress:{task_id}"
            self.redis_client.delete(key)
            print(f"🗑️ Progress cleaned up: {task_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to delete progress: {e}")
            return False
    
    def cache_embedding(self, content_hash: str, embedding: list, ttl: int = 3600):
        """Cache embedding with TTL"""
        if not self.is_connected():
            return False
            
        try:
            key = f"embedding:{content_hash}"
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(embedding)
            )
            return True
            
        except Exception as e:
            print(f"❌ Failed to cache embedding: {e}")
            return False
    
    def get_cached_embedding(self, content_hash: str) -> Optional[list]:
        """Get cached embedding"""
        if not self.is_connected():
            return None
            
        try:
            key = f"embedding:{content_hash}"
            data = self.redis_client.get(key)
            
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            print(f"❌ Failed to get cached embedding: {e}")
            return None
    
    def set_task_result(self, task_id: str, result: Dict, ttl: int = 3600):
        """Store task result with TTL"""
        if not self.is_connected():
            return False
            
        try:
            key = f"result:{task_id}"
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(result)
            )
            return True
            
        except Exception as e:
            print(f"❌ Failed to set task result: {e}")
            return False
    
    def get_task_result(self, task_id: str) -> Optional[Dict]:
        """Get task result"""
        if not self.is_connected():
            return None
            
        try:
            key = f"result:{task_id}"
            data = self.redis_client.get(key)
            
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            print(f"❌ Failed to get task result: {e}")
            return None

# Global Redis manager instance
redis_manager = RedisManager()