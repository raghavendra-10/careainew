#!/usr/bin/env python3
"""
Safe Collection Deletion Script
Based on Firebase best practices for avoiding "Transaction too big" errors
"""

import sys
import time
from firebase_admin import credentials, firestore, initialize_app
import firebase_admin

def initialize_firestore():
    """Initialize Firestore using fire.json credentials"""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("fire.json")
            initialize_app(cred)
        
        db = firestore.client()
        print("✅ Successfully connected to Firestore")
        return db
    except Exception as e:
        print(f"❌ Failed to initialize Firestore: {e}")
        sys.exit(1)

def delete_query_batch(query, batch_size=100):
    """
    Delete documents in batches using Firestore batch operations
    Following Firebase recommended pattern
    """
    docs = query.get()
    
    if len(docs) == 0:
        return 0
    
    # Use Firestore batch (max 500 operations, we'll use smaller batches)
    batch = firestore.client().batch()
    count = 0
    
    for doc in docs:
        batch.delete(doc.reference)
        count += 1
        
        # If we reach batch_size, commit and start new batch
        if count >= batch_size:
            break
    
    try:
        batch.commit()
        print(f"   ✓ Deleted batch of {count} documents")
        return count
    except Exception as e:
        print(f"   ❌ Error in batch delete: {e}")
        return 0

def delete_collection_safe(db, collection_path, batch_size=100):
    """
    Safely delete a collection in batches
    Implements the pattern recommended by Firebase documentation
    """
    print(f"🗑️  Safely deleting collection: {collection_path}")
    print(f"📦 Using batch size: {batch_size}")
    
    collection_ref = db.collection(collection_path)
    total_deleted = 0
    
    while True:
        # Create a query for the next batch
        query = collection_ref.limit(batch_size)
        
        # Delete this batch
        deleted_count = delete_query_batch(query, batch_size)
        
        if deleted_count == 0:
            break
        
        total_deleted += deleted_count
        print(f"📊 Total deleted so far: {total_deleted}")
        
        # Small delay to avoid overwhelming Firestore
        time.sleep(0.5)
    
    return total_deleted

def delete_subcollections_recursively(db, doc_ref):
    """Delete subcollections of a document recursively"""
    subcollections = doc_ref.collections()
    
    for subcoll in subcollections:
        print(f"   🔄 Processing subcollection: {subcoll.id}")
        delete_collection_safe(db, subcoll.path, batch_size=50)

def delete_collection_with_subcollections(db, collection_path):
    """
    Delete collection and handle subcollections
    This is the most comprehensive deletion approach
    """
    print(f"🗑️  Comprehensive deletion of: {collection_path}")
    
    collection_ref = db.collection(collection_path)
    total_deleted = 0
    batch_size = 50  # Smaller batches for safety
    
    while True:
        # Get a batch of documents
        docs = collection_ref.limit(batch_size).get()
        
        if len(docs) == 0:
            print("✅ No more documents to delete")
            break
        
        print(f"🔄 Processing batch of {len(docs)} documents...")
        
        # Process each document individually to handle subcollections
        for doc in docs:
            try:
                # First, delete subcollections
                delete_subcollections_recursively(db, doc.reference)
                
                # Then delete the document
                doc.reference.delete()
                total_deleted += 1
                
                print(f"   ✓ Deleted: {doc.id}")
                
                # Small delay between individual deletions
                time.sleep(0.1)
                
            except Exception as e:
                print(f"   ❌ Error deleting {doc.id}: {e}")
        
        print(f"📊 Batch complete. Total deleted: {total_deleted}")
        
        # Delay between batches
        time.sleep(1)
    
    return total_deleted

def main():
    """Main function"""
    collection_name = "document_embeddings"
    
    print(f"⚠️  SAFE DELETION OF: {collection_name}")
    print("This uses Firebase recommended patterns to avoid transaction limits.")
    print("The process will be slower but more reliable.")
    print("This operation is IRREVERSIBLE!")
    print()
    
    # Safety confirmation
    confirmation = input(f"Type 'SAFE_DELETE' to confirm deletion of {collection_name}: ")
    if confirmation != "SAFE_DELETE":
        print("❌ Operation cancelled")
        sys.exit(0)
    
    print()
    print("🚀 Starting safe deletion process...")
    
    # Initialize Firestore
    db = initialize_firestore()
    
    try:
        start_time = time.time()
        
        # Use the comprehensive deletion method
        deleted_count = delete_collection_with_subcollections(db, collection_name)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Safe deletion completed!")
        print(f"📊 Total documents deleted: {deleted_count}")
        print(f"⏱️  Time taken: {duration:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error during safe deletion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()