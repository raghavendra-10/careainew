"""
File processing logic separated from app.py to avoid circular imports
Phase 1: Core Infrastructure
"""

import os
import traceback
from datetime import datetime
from redis_manager import redis_manager

def process_file_upload_internal(file_data, upload_params, task_id=None):
    """
    Internal function to process file uploads
    This preserves existing functionality while being callable from tasks
    """
    try:
        file_path = file_data["file_path"]
        filename = file_data["filename"] 
        file_id = file_data["file_id"]
        
        org_id = upload_params["org_id"]
        user_id = upload_params["user_id"]
        
        # Update progress
        if task_id and redis_manager.is_connected():
            redis_manager.set_progress(
                task_id, "processing", 25, "Extracting content", filename
            )
        
        # Get file extension
        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        # Import functions from utils.py to avoid circular imports
        from utils import parse_and_chunk
        
        # Extract content using existing function
        chunks = parse_and_chunk(file_path, file_ext, chunk_size=50)
        
        if not chunks:
            raise Exception("No content extracted from file")
        
        # Limit chunks to prevent memory issues
        if len(chunks) > 500:
            chunks = chunks[:500]
            print(f"⚠️ Limited to 500 chunks for {filename}")
        
        # Update progress
        if task_id and redis_manager.is_connected():
            redis_manager.set_progress(
                task_id, "embedding", 50, "Generating embeddings", filename
            )
        
        # Import embedding function
        from utils import embed_chunks
        
        # Generate embeddings using existing function
        embeddings = embed_chunks(chunks)
        
        if not embeddings:
            raise Exception("Failed to generate embeddings")
        
        # Store in Firestore using existing functionality
        if task_id and redis_manager.is_connected():
            redis_manager.set_progress(
                task_id, "storing", 75, "Storing in database", filename
            )
        
        # Store embeddings
        store_embeddings_in_firestore(embeddings, chunks, file_id, org_id, filename)
        
        # Send webhook notification
        if user_id:
            send_webhook_notification(file_id, user_id, org_id)
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ Cleaned up: {file_path}")
        
        return {
            "success": True,
            "file_id": file_id,
            "chunks_processed": len(chunks),
            "embeddings_generated": len(embeddings),
            "filename": filename
        }
        
    except Exception as e:
        # Clean up on error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise e

def store_embeddings_in_firestore(embeddings, chunks, file_id, org_id, filename):
    """Store embeddings in Firestore using existing logic"""
    try:
        # Import Firebase here to avoid circular imports
        import firebase_admin
        from firebase_admin import credentials, firestore
        
        # Get existing Firebase app or initialize
        try:
            app = firebase_admin.get_app()
            db = firestore.client(app)
        except ValueError:
            # App doesn't exist, check if we can get it from app.py globals
            import sys
            if 'app' in sys.modules and hasattr(sys.modules['app'], 'db'):
                db = sys.modules['app'].db
            else:
                raise Exception("Firebase not initialized")
        
        collection_ref = db.collection("document_embeddings")
        
        # Use batch operations for better performance
        batch = db.batch()
        batch_count = 0
        
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            doc_data = {
                "file_id": file_id,
                "org_id": org_id,
                "filename": filename,
                "chunk_index": i,
                "chunk_text": chunk,
                "embedding": embedding,
                "timestamp": datetime.now(),
                "created_at": datetime.now().isoformat(),
                "source": "flask_ai_queue"
            }
            
            doc_ref = collection_ref.document()
            batch.set(doc_ref, doc_data)
            batch_count += 1
            
            # Commit in batches of 450 (Firestore limit is 500)
            if batch_count >= 450:
                batch.commit()
                print(f"📦 Committed batch of {batch_count} embeddings")
                batch = db.batch()
                batch_count = 0
        
        # Commit remaining items
        if batch_count > 0:
            batch.commit()
            print(f"📦 Committed final batch of {batch_count} embeddings")
        
        print(f"✅ Stored {len(embeddings)} embeddings for {filename}")
        
    except Exception as e:
        print(f"❌ Firestore storage error: {e}")
        raise e

def send_webhook_notification(file_id, user_id, org_id):
    """Send webhook notification"""
    try:
        import requests
        
        backend_api_url = os.environ.get("BACKEND_API_URL", "http://localhost:8080")
        webhook_url = f"{backend_api_url}/api/v2/files/webhook/ai-status"
        
        payload = {
            "fileId": file_id,
            "userId": user_id,
            "status": "completed",
            "embeddingComplete": True,
            "timestamp": datetime.now().isoformat(),
            "source": "flask_ai_queue"
        }
        
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✅ Webhook sent successfully for {file_id}")
        else:
            print(f"⚠️ Webhook failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Webhook error: {str(e)}")
        # Don't raise - webhook failure shouldn't fail the whole process