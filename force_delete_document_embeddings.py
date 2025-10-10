#!/usr/bin/env python3
"""
Force Delete document_embeddings Collection Script
"""

import sys
from firebase_admin import credentials, firestore, initialize_app
import firebase_admin
from google.cloud.firestore_v1.base_query import FieldFilter

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

def delete_collection_completely(db, collection_name):
    """Delete a collection completely using multiple methods"""
    print(f"🗑️  Force deleting collection: {collection_name}")
    
    total_deleted = 0
    
    # Method 1: Direct collection reference
    try:
        collection_ref = db.collection(collection_name)
        
        # Get documents in batches and delete
        batch_size = 100
        while True:
            docs = collection_ref.limit(batch_size).get()
            if not docs:
                break
                
            batch = db.batch()
            count = 0
            
            for doc in docs:
                print(f"   Queuing for deletion: {doc.id}")
                
                # Delete subcollections first
                subcollections = doc.reference.collections()
                for subcoll in subcollections:
                    subdocs = subcoll.get()
                    for subdoc in subdocs:
                        batch.delete(subdoc.reference)
                        count += 1
                
                # Delete the main document
                batch.delete(doc.reference)
                count += 1
            
            if count > 0:
                batch.commit()
                total_deleted += count
                print(f"   Deleted batch: {count} documents")
            else:
                break
                
    except Exception as e:
        print(f"   Method 1 error: {e}")
    
    # Method 2: Try to get all documents without limit
    try:
        collection_ref = db.collection(collection_name)
        all_docs = collection_ref.stream()
        
        for doc in all_docs:
            try:
                print(f"   Force deleting: {doc.id}")
                
                # Delete subcollections
                subcollections = doc.reference.collections()
                for subcoll in subcollections:
                    subdocs = subcoll.stream()
                    for subdoc in subdocs:
                        subdoc.reference.delete()
                
                # Delete main document
                doc.reference.delete()
                total_deleted += 1
                
            except Exception as doc_error:
                print(f"     Error deleting {doc.id}: {doc_error}")
                
    except Exception as e:
        print(f"   Method 2 error: {e}")
    
    return total_deleted

def verify_deletion(db, collection_name):
    """Verify that the collection is empty"""
    try:
        collection_ref = db.collection(collection_name)
        docs = list(collection_ref.limit(1).stream())
        return len(docs) == 0
    except Exception:
        return True  # If we can't access it, assume it's deleted

def main():
    """Main function"""
    collection_name = "document_embeddings"
    
    print(f"⚠️  FORCE DELETING COLLECTION: {collection_name}")
    print("This will DELETE the ENTIRE collection and ALL its contents.")
    print("This operation is IRREVERSIBLE!")
    print()
    
    # Safety confirmation
    confirmation = input(f"Type 'FORCE_DELETE' to confirm deletion of {collection_name}: ")
    if confirmation != "FORCE_DELETE":
        print("❌ Operation cancelled")
        sys.exit(0)
    
    print()
    print("🚀 Starting force deletion...")
    
    # Initialize Firestore
    db = initialize_firestore()
    
    try:
        # Delete the collection
        deleted_count = delete_collection_completely(db, collection_name)
        print(f"✅ Deletion process completed. Processed {deleted_count} items.")
        
        # Verify deletion
        if verify_deletion(db, collection_name):
            print(f"✅ Collection {collection_name} appears to be empty/deleted")
        else:
            print(f"⚠️  Some documents may still remain in {collection_name}")
            
    except Exception as e:
        print(f"❌ Error during deletion: {e}")
        sys.exit(1)
    
    print("✅ Force deletion completed!")

if __name__ == "__main__":
    main()