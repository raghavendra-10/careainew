-- Recreate llamaindex_embeddings table for NeonDB
-- Run this SQL script to restore the table structure

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing table if it exists (to handle dimension changes)
DROP TABLE IF EXISTS llamaindex_embeddings CASCADE;

-- Create embeddings table for LlamaIndex
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

-- Create indexes for better performance
-- Note: Skipping vector index for 2560 dimensions (exceeds ivfflat limit of 2000)
CREATE INDEX IF NOT EXISTS idx_org_file 
ON llamaindex_embeddings (org_id, file_id);

CREATE INDEX IF NOT EXISTS idx_metadata 
ON llamaindex_embeddings USING gin (metadata);

-- Test the setup
SELECT COUNT(*) FROM llamaindex_embeddings;

-- Show table structure
\d llamaindex_embeddings;