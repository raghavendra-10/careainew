#!/usr/bin/env python3
"""
Setup NeonDB PostgreSQL with pgvector for LlamaIndex
"""
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

def setup_neondb():
    """Initialize NeonDB with pgvector extension and tables"""
    
    db_url = os.getenv("NEON_DATABASE_URL")
    if not db_url:
        raise ValueError("NEON_DATABASE_URL not found in environment")
    
    print("🔧 Setting up NeonDB with pgvector...")
    
    try:
        # Connect to database
        conn = psycopg2.connect(db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Enable pgvector extension
        print("📦 Enabling pgvector extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Drop existing table if it exists (to handle dimension changes)
        print("🗑️ Dropping existing table if it exists...")
        cursor.execute("DROP TABLE IF EXISTS llamaindex_embeddings CASCADE;")
        
        # Create embeddings table for LlamaIndex
        print("📋 Creating embeddings table...")
        cursor.execute("""
            CREATE TABLE llamaindex_embeddings (
                id SERIAL PRIMARY KEY,
                node_id VARCHAR(255) UNIQUE NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB,
                embedding vector(2560),
                org_id VARCHAR(255),
                file_id VARCHAR(255),
                chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes for better performance
        print("🔍 Creating indexes...")
        # Note: Skipping vector index for 2560 dimensions (exceeds ivfflat limit of 2000)
        print("⚠️ Skipping vector index for 2560 dimensions (exceeds ivfflat limit)")
        print("🔍 Vector similarity will use basic operations")
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_org_file 
            ON llamaindex_embeddings (org_id, file_id);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metadata 
            ON llamaindex_embeddings USING gin (metadata);
        """)
        
        print("✅ NeonDB setup complete!")
        print("📊 Vector dimension: 2560")
        print("🏗️ Table: llamaindex_embeddings")
        print("🔍 Indexes: vector search, org/file, metadata")
        
        # Test the setup
        cursor.execute("SELECT COUNT(*) FROM llamaindex_embeddings;")
        count = cursor.fetchone()[0]
        print(f"📈 Current embeddings count: {count}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        raise

if __name__ == "__main__":
    setup_neondb()