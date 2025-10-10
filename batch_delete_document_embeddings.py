#!/usr/bin/env python3
"""
Batch Delete document_embeddings Collection Script
Handles large collections by deleting in small batches
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

def delete_small_batch(collection_ref, batch_size=10):
    """Delete a small batch of documents"""
    docs = collection_ref.limit(batch_size).get()
    
    if not docs:
        return 0
    
    deleted_count = 0
    for doc in docs:
        try:
            # Delete subcollections first
            subcollections = doc.reference.collections()
            for subcoll in subcollections:
                # Delete subcollection documents one by one
                subdocs = subcoll.limit(10).get()
                for subdoc in subdocs:
                    subdoc.reference.delete()
                    time.sleep(0.1)  # Small delay to avoid rate limits
            
            # Delete the main document
            doc.reference.delete()
            deleted_count += 1
            print(f"   ✓ Deleted: {doc.id}")
            
            # Small delay between deletions
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   ❌ Error deleting {doc.id}: {e}")
    
    return deleted_count

def count_documents(collection_ref):
    """Count total documents in collection"""
    try:
        # Use a more efficient way to count
        docs = list(collection_ref.stream())
        return len(docs)
    except Exception:
        return "unknown"

def delete_collection_in_batches(db, collection_name, batch_size=10):
    """Delete collection in small batches to avoid transaction limits"""
    print(f"🗑️  Batch deleting collection: {collection_name}")
    
    collection_ref = db.collection(collection_name)
    
    # Count initial documents
    initial_count = count_documents(collection_ref)
    print(f"📊 Initial document count: {initial_count}")
    
    total_deleted = 0
    batch_number = 1
    
    while True:
        print(f"\n🔄 Processing batch {batch_number}...")
        
        # Delete a small batch
        deleted_in_batch = delete_small_batch(collection_ref, batch_size)
        
        if deleted_in_batch == 0:
            print("✅ No more documents to delete")
            break
        
        total_deleted += deleted_in_batch
        print(f"   Batch {batch_number}: Deleted {deleted_in_batch} documents")
        print(f"   Total deleted so far: {total_deleted}")
        
        batch_number += 1
        
        # Add a longer delay between batches to be safe
        time.sleep(1)
        
        # Safety check - avoid infinite loops
        if batch_number > 10000:  # Adjust based on expected size
            print("⚠️  Reached maximum batch limit. Stopping for safety.")
            break
    
    return total_deleted

def main():
    """Main function"""
    collection_name = "document_embeddings"
    
    print(f"⚠️  BATCH DELETING COLLECTION: {collection_name}")
    print("This will delete the collection in small batches to avoid transaction limits.")
    print("This operation is IRREVERSIBLE but will take some time!")
    print()
    
    # Safety confirmation
    confirmation = input(f"Type 'BATCH_DELETE' to confirm batch deletion of {collection_name}: ")
    if confirmation != "BATCH_DELETE":
        print("❌ Operation cancelled")
        sys.exit(0)
    
    print()
    print("🚀 Starting batch deletion (this may take a while)...")
    
    # Initialize Firestore
    db = initialize_firestore()
    
    try:
        # Delete the collection in batches
        start_time = time.time()
        deleted_count = delete_collection_in_batches(db, collection_name, batch_size=5)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\n✅ Batch deletion completed!")
        print(f"📊 Total documents deleted: {deleted_count}")
        print(f"⏱️  Time taken: {duration:.2f} seconds")
        
        # Final verification
        collection_ref = db.collection(collection_name)
        remaining = count_documents(collection_ref)
        print(f"📋 Remaining documents: {remaining}")
        
    except Exception as e:
        print(f"❌ Error during batch deletion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()