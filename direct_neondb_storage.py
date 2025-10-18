#!/usr/bin/env python3
"""
Direct NeonDB storage - bypassing LlamaIndex vector store issues
"""
import os
import asyncpg
import asyncio
import uuid
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from simple_custom_embedding import get_embeddings_from_hf_endpoint

load_dotenv()

class DirectNeonDBStorage:
    def __init__(self):
        self.db_url = os.getenv("NEON_DATABASE_URL")
        if not self.db_url:
            raise ValueError("NEON_DATABASE_URL not found")
    
    async def store_embeddings_directly(self, chunks: List[str], file_id: str, org_id: str, user_id: str, filename: str) -> Dict[str, Any]:
        """Store embeddings directly in NeonDB without LlamaIndex"""
        print(f"🚀 Direct storage: {len(chunks)} chunks for file {file_id}")
        
        # Get embeddings for all chunks
        print("🔗 Getting embeddings from HF endpoint...")
        embeddings = get_embeddings_from_hf_endpoint(chunks)
        print(f"✅ Got {len(embeddings)} embeddings")
        
        # Connect to database
        conn = await asyncpg.connect(self.db_url)
        
        try:
            stored_count = 0
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                node_id = f"{file_id}_chunk_{i}_{uuid.uuid4().hex[:8]}"
                
                # Create metadata with additional fields
                metadata = {
                    "org_id": org_id,
                    "user_id": user_id,
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_id": file_id
                }
                
                # Convert embedding list to pgvector format
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                
                # Insert directly into database
                await conn.execute("""
                    INSERT INTO llamaindex_embeddings (
                        node_id, text, embedding, file_id, org_id, chunk_index, metadata, created_at
                    ) VALUES ($1, $2, $3::vector, $4, $5, $6, $7::jsonb, NOW())
                """, node_id, chunk, embedding_str, file_id, org_id, i, json.dumps(metadata))
                
                stored_count += 1
                print(f"✅ Stored chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            
            print(f"🎯 Successfully stored {stored_count} embeddings directly in NeonDB")
            
            # Verify storage
            count = await conn.fetchval("SELECT COUNT(*) FROM llamaindex_embeddings WHERE file_id = $1", file_id)
            print(f"🔍 Verification: {count} records found for file_id {file_id}")
            
            return {
                "success": True,
                "chunks_stored": stored_count,
                "file_id": file_id,
                "method": "direct_neondb_storage"
            }
            
        except Exception as e:
            print(f"❌ Direct storage error: {str(e)}")
            raise
        finally:
            await conn.close()

async def store_project_support_embeddings(chunks: List[str], file_id: str, org_id: str, user_id: str, project_id: str, filename: str) -> Dict[str, Any]:
    """Store embeddings in project_support_embeddings table with project_id"""
    print(f"🚀 Storing project support embeddings: {len(chunks)} chunks for project {project_id}")
    
    # Get embeddings for all chunks
    print("🔗 Getting embeddings from HF endpoint...")
    embeddings = get_embeddings_from_hf_endpoint(chunks)
    print(f"✅ Got {len(embeddings)} embeddings")
    
    # Connect to database
    db_url = os.getenv("NEON_DATABASE_URL")
    if not db_url:
        raise ValueError("NEON_DATABASE_URL not found")
    
    conn = await asyncpg.connect(db_url)
    
    try:
        # Create table if it doesn't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS project_support_embeddings (
                id SERIAL PRIMARY KEY,
                node_id TEXT UNIQUE NOT NULL,
                text TEXT NOT NULL,
                embedding vector(2560),
                file_id TEXT NOT NULL,
                org_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                user_id TEXT,
                chunk_index INTEGER,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Create indexes for better query performance
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_project_support_org_id ON project_support_embeddings(org_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_project_support_project_id ON project_support_embeddings(project_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_project_support_file_id ON project_support_embeddings(file_id)
        """)
        
        stored_count = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            node_id = f"project_{project_id}_file_{file_id}_chunk_{i}_{uuid.uuid4().hex[:8]}"
            
            # Create metadata with project info
            metadata = {
                "org_id": org_id,
                "user_id": user_id,
                "project_id": project_id,
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_id": file_id,
                "document_type": "project_support"
            }
            
            # Convert embedding list to pgvector format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            # Insert into project support table
            await conn.execute("""
                INSERT INTO project_support_embeddings (
                    node_id, text, embedding, file_id, org_id, project_id, user_id, chunk_index, metadata, created_at
                ) VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8, $9::jsonb, NOW())
            """, node_id, chunk, embedding_str, file_id, org_id, project_id, user_id, i, json.dumps(metadata))
            
            stored_count += 1
            print(f"✅ Stored project chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
        
        print(f"🎯 Successfully stored {stored_count} project support embeddings in NeonDB")
        
        # Verify storage
        count = await conn.fetchval("SELECT COUNT(*) FROM project_support_embeddings WHERE file_id = $1 AND project_id = $2", file_id, project_id)
        print(f"🔍 Verification: {count} records found for file_id {file_id} in project {project_id}")
        
        return {
            "success": True,
            "chunks_stored": stored_count,
            "file_id": file_id,
            "project_id": project_id,
            "method": "direct_project_support_storage"
        }
        
    except Exception as e:
        print(f"❌ Project support storage error: {str(e)}")
        raise
    finally:
        await conn.close()

def process_file_direct_storage_project_support(file_path: str, file_id: str, org_id: str, user_id: str, project_id: str, filename: str) -> Dict[str, Any]:
    """Process file for project support using direct storage approach"""
    print(f"🚀 Project Support Processing: {filename}")
    print(f"📋 Project: {project_id}, Org: {org_id}, File: {file_id}")
    
    try:
        # Get file extension to determine processing method
        file_ext = filename.split('.')[-1].lower()
        documents = []
        
        # Use LlamaParse for complex file types, fallback to LlamaIndex for simple ones
        llamaparse_supported = ['pdf', 'docx', 'pptx', 'xlsx', 'html', 'htm']
        
        if file_ext in llamaparse_supported:
            print(f"🚀 Using LlamaParse for {file_ext.upper()} file")
            try:
                from llama_parse import LlamaParse
                
                # Initialize LlamaParse
                api_key = os.getenv("LLAMA_CLOUD_API_KEY")
                print(f"🔑 API Key present: {bool(api_key)}")
                print(f"🔑 API Key length: {len(api_key) if api_key else 0}")
                
                parser = LlamaParse(
                    api_key=api_key,
                    result_type="markdown",
                    verbose=True,
                    language="en"
                )
                print(f"✅ LlamaParse parser initialized successfully")
                
                # Parse the document
                print(f"📊 Parsing {filename} with LlamaParse...")
                print(f"🔄 About to call parser.load_data({file_path})")
                documents = parser.load_data(file_path)
                print(f"✅ parser.load_data completed successfully")
                print(f"📄 LlamaParse loaded {len(documents)} document(s)")
                
            except Exception as e:
                print(f"⚠️ LlamaParse failed: {str(e)}")
                print(f"🔄 Falling back to LlamaIndex readers...")
                documents = []
        
        # Fallback to LlamaIndex readers for simple files or if LlamaParse fails
        if not documents:
            try:
                from llama_index.readers.file import UnstructuredReader
                from llama_index.core import Document
                
                print(f"📖 Using LlamaIndex UnstructuredReader for {filename}")
                reader = UnstructuredReader()
                
                # Load document
                documents = reader.load_data(file=file_path)
                print(f"📄 Loaded {len(documents)} document(s)")
                
            except Exception as e:
                print(f"⚠️ LlamaIndex parsing also failed: {str(e)}")
                print(f"🔄 Using simple text reading...")
                
                # Final fallback - simple text reading
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        content = f"Binary file: {filename} (size: {os.path.getsize(file_path)} bytes)"
                
                from llama_index.core import Document
                documents = [Document(text=content)]
                print(f"📄 Created document from simple text reading")
        
        if not documents:
            return {
                "success": False,
                "error": "No content extracted from document"
            }
        
        print(f"📚 Extracted {len(documents)} document(s)")
        
        # Create chunks using LlamaIndex
        print(f"✂️ Parsing documents into chunks...")
        try:
            from llama_index.core.node_parser import SentenceSplitter
            
            chunk_size = int(os.getenv("LLAMAINDEX_CHUNK_SIZE", "1024"))
            chunk_overlap = int(os.getenv("LLAMAINDEX_CHUNK_OVERLAP", "20"))
            
            node_parser = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            nodes = node_parser.get_nodes_from_documents(documents)
            chunks = [node.text for node in nodes]
            print(f"✂️ Created {len(chunks)} chunks")
            
        except Exception as e:
            print(f"⚠️ Chunking failed, using simple split: {str(e)}")
            
            # Simple chunking fallback
            all_text = " ".join([doc.text for doc in documents])
            chunk_size = int(os.getenv("LLAMAINDEX_CHUNK_SIZE", "1024"))
            chunks = []
            
            # Split into chunks
            words = all_text.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                current_chunk.append(word)
                current_length += len(word) + 1
                
                if current_length >= chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            print(f"✂️ Created {len(chunks)} chunks (simple split)")
        
        if not chunks:
            return {
                "success": False,
                "error": "No chunks created from document"
            }
        
        print(f"📦 Created {len(chunks)} chunks")
        
        # Store in project support table
        result = asyncio.run(store_project_support_embeddings(
            chunks=chunks,
            file_id=file_id,
            org_id=org_id,
            user_id=user_id,
            project_id=project_id,
            filename=filename
        ))
        
        return result
        
    except Exception as e:
        print(f"❌ Project Support processing error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def process_file_direct_storage(file_path: str, file_id: str, org_id: str, user_id: str, filename: str) -> Dict[str, Any]:
    """Process file using LlamaParse + direct storage method"""
    print(f"📄 Direct processing with LlamaParse: {filename}")
    
    # Get file extension to determine processing method
    file_ext = filename.split('.')[-1].lower()
    documents = []
    
    # Use LlamaParse for complex file types, fallback to LlamaIndex for simple ones
    llamaparse_supported = ['pdf', 'docx', 'pptx', 'xlsx', 'html', 'htm']
    
    if file_ext in llamaparse_supported:
        print(f"🚀 Using LlamaParse for {file_ext.upper()} file")
        try:
            from llama_parse import LlamaParse
            
            # Initialize LlamaParse
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                result_type="markdown",  # Get markdown for better structure
                verbose=True,
                language="en"
            )
            
            # Parse the document
            print(f"📊 Parsing {filename} with LlamaParse...")
            print(f"🔄 About to call parser.load_data({file_path})")
            documents = parser.load_data(file_path)
            print(f"✅ parser.load_data completed successfully")
            print(f"📄 LlamaParse loaded {len(documents)} document(s)")
            
            if documents:
                print(f"📝 First document preview: {documents[0].text[:200]}...")
                
        except Exception as e:
            print(f"⚠️ LlamaParse failed: {str(e)}")
            print(f"🔄 Falling back to LlamaIndex readers...")
            documents = []
    
    # Fallback to LlamaIndex readers for simple files or if LlamaParse fails
    if not documents:
        try:
            from llama_index.readers.file import UnstructuredReader
            from llama_index.core import Document
            
            print(f"📖 Using LlamaIndex UnstructuredReader for {filename}")
            reader = UnstructuredReader()
            
            # Load document
            documents = reader.load_data(file=file_path)
            print(f"📄 Loaded {len(documents)} document(s)")
            
            if documents:
                print(f"📝 First document preview: {documents[0].text[:200]}...")
                
        except Exception as e:
            print(f"⚠️ LlamaIndex parsing also failed: {str(e)}")
            print(f"🔄 Using simple text reading...")
            
            # Final fallback - simple text reading with proper encoding handling
            try:
                # Try UTF-8 first
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    # Try latin-1 if UTF-8 fails
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # For binary files, just extract basic info
                    content = f"Binary file: {filename} (size: {os.path.getsize(file_path)} bytes)"
            
            from llama_index.core import Document
            documents = [Document(text=content)]
            print(f"📄 Created document from simple text reading")
    
    # Add metadata to documents
    for doc in documents:
        doc.metadata.update({
            "file_id": file_id,
            "org_id": org_id,
            "user_id": user_id,
            "filename": filename,
            "file_extension": file_ext
        })
    
    # Parse into chunks using LlamaIndex
    print(f"✂️ Parsing documents into chunks with LlamaIndex...")
    try:
        from llama_index.core.node_parser import SentenceSplitter
        
        chunk_size = int(os.getenv("LLAMAINDEX_CHUNK_SIZE", "1024"))
        chunk_overlap = int(os.getenv("LLAMAINDEX_CHUNK_OVERLAP", "20"))
        
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        nodes = node_parser.get_nodes_from_documents(documents)
        print(f"✂️ Created {len(nodes)} chunks with LlamaIndex")
        
        # Extract text from nodes
        chunks = [node.text for node in nodes]
        
        # Add chunk metadata
        for i, node in enumerate(nodes):
            node.metadata.update({
                "chunk_index": i,
                "total_chunks": len(nodes)
            })
        
        if chunks:
            print(f"📝 First chunk preview: {chunks[0][:200]}...")
            
    except Exception as e:
        print(f"⚠️ Chunking failed, using simple split: {str(e)}")
        
        # Simple chunking fallback
        all_text = " ".join([doc.text for doc in documents])
        chunk_size = int(os.getenv("LLAMAINDEX_CHUNK_SIZE", "1024"))
        chunks = []
        
        # Split into chunks
        words = all_text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space
            
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        print(f"✂️ Created {len(chunks)} chunks (simple split method)")
    
    # Store directly
    storage = DirectNeonDBStorage()
    return asyncio.run(storage.store_embeddings_directly(chunks, file_id, org_id, user_id, filename))