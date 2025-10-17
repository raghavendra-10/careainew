#!/usr/bin/env python3
"""
LlamaIndex integration with NeonDB PostgreSQL
Handles document processing, embedding, and retrieval
"""
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.readers.file import PDFReader, DocxReader, UnstructuredReader

# Standard libraries
import tempfile
import shutil
import requests
import json

load_dotenv()

class LlamaIndexProcessor:
    """LlamaIndex processor for document handling and retrieval"""
    
    def __init__(self):
        self.setup_models()
        self.setup_vector_store()
        self.setup_parsers()
    
    def setup_models(self):
        """Initialize embedding and LLM models"""
        print("🤖 Initializing models...")
        print(f"🔧 Environment check - HF_TOKEN: {os.getenv('HF_TOKEN', 'NOT_SET')[-10:]}")
        print(f"🔧 Environment check - EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL', 'NOT_SET')}")
        print(f"🔧 Environment check - QWEN2_MODEL: {os.getenv('QWEN2_MODEL', 'NOT_SET')}")
        print(f"🔧 Environment check - EMBEDDING_ENDPOINT: {os.getenv('EMBEDDING_ENDPOINT', 'NOT_SET')}")
        
        # Check for LlamaIndex API key
        llamaindex_api_key = os.getenv("LLAMAINDEX_API_KEY")
        
        # Note: LlamaIndex hosted services packages don't exist yet
        # Using HuggingFace models directly
        if False:  # Disabled until packages are available
            pass
        else:
            print("🤗 Using Hugging Face models")
            
            # Use custom embedding endpoint as requested
            embedding_endpoint = os.getenv("EMBEDDING_ENDPOINT")
            if embedding_endpoint:
                print("🚀 Using simple custom HF embedding endpoint")
                try:
                    # Use a simple approach that works reliably with LlamaIndex
                    from llama_index.core.embeddings import BaseEmbedding
                    from simple_custom_embedding import get_embedding_from_hf_endpoint, get_embeddings_from_hf_endpoint
                    
                    class SimpleCustomEmbedding(BaseEmbedding):
                        def __init__(self):
                            super().__init__(
                                model_name="custom_hf_endpoint",
                                embed_batch_size=32
                            )
                        
                        def _get_query_embedding(self, query: str):
                            return get_embedding_from_hf_endpoint(query)
                        
                        def _get_text_embedding(self, text: str):
                            return get_embedding_from_hf_endpoint(text)
                        
                        def _get_text_embeddings(self, texts: list):
                            return get_embeddings_from_hf_endpoint(texts)
                        
                        async def _aget_query_embedding(self, query: str):
                            return self._get_query_embedding(query)
                        
                        async def _aget_text_embedding(self, text: str):
                            return self._get_text_embedding(text)
                        
                        async def _aget_text_embeddings(self, texts: list):
                            return self._get_text_embeddings(texts)
                    
                    self.embed_model = SimpleCustomEmbedding()
                    print("✅ Simple custom embedding endpoint initialized")
                    print("⚠️ Note: Server-side padding strategy issue needs to be fixed")
                    
                except Exception as e:
                    print(f"❌ Custom embedding error: {str(e)}")
                    raise ValueError(f"Failed to initialize custom embedding: {str(e)}")
            else:
                print("❌ EMBEDDING_ENDPOINT not configured")
                raise ValueError("Custom embedding endpoint required")
            
            # Initialize Qwen2 LLM using HuggingFace
            self.qwen2_model = os.getenv("QWEN2_MODEL", "Qwen/Qwen2.5-7B-Instruct")
            hf_token = os.getenv("HF_TOKEN")
            
            if not hf_token:
                raise ValueError("HF_TOKEN not found in environment")
                
            # For now, use a simple approach since HuggingFaceInferenceAPI is not available
            # We'll create a minimal LLM wrapper or use requests directly
            self.llm = None  # Will implement direct API calls for now
            print("⚠️ Using direct HF API calls instead of LlamaIndex LLM wrapper")
            
            print(f"✅ Embedding model: Custom HF endpoint")
            print(f"🤖 LLM model: {self.qwen2_model}")
            print(f"🔑 Using HF token: ...{hf_token[-10:]}")
        
        # Set global models
        Settings.embed_model = self.embed_model
        # Settings.llm = self.llm  # Skip for now since we're using direct API calls
        Settings.chunk_size = int(os.getenv("LLAMAINDEX_CHUNK_SIZE", "1024"))
        Settings.chunk_overlap = int(os.getenv("LLAMAINDEX_CHUNK_OVERLAP", "20"))
        
        print(f"📏 Vector dimension: {os.getenv('VECTOR_DIM', '1024')}")
        print(f"🔧 Global embedding model set to: {type(Settings.embed_model)}")
        print(f"🔧 Model name: {getattr(Settings.embed_model, 'model_name', 'unknown')}")
        
        # Test the embedding model to make sure it works
        print("🧪 Testing custom embedding model...")
        try:
            test_embedding = self.embed_model._get_text_embedding("test")
            print(f"✅ Custom embedding test successful, dimension: {len(test_embedding)}")
        except Exception as e:
            print(f"❌ Custom embedding test failed: {str(e)}")
            raise
    
    def setup_vector_store(self):
        """Initialize PostgreSQL vector store"""
        print("🗄️ Setting up NeonDB vector store...")
        
        db_url = os.getenv("NEON_DATABASE_URL")
        if not db_url:
            raise ValueError("NEON_DATABASE_URL not found")
        
        # Use connection string directly for better reliability
        print(f"🔗 Database URL: {db_url[:50]}...")
        
        try:
            self.vector_store = PGVectorStore.from_params(
                database=db_url.split('/')[-1].split('?')[0],  # Extract DB name
                host=db_url.split('@')[1].split(':')[0],       # Extract host
                password=db_url.split(':')[2].split('@')[0],   # Extract password
                port=5432,
                user=db_url.split('://')[1].split(':')[0],     # Extract user
                table_name="llamaindex_embeddings",
                embed_dim=int(os.getenv("VECTOR_DIM", "2560")),  # Fixed dimension
                schema_name="public"
            )
            print(f"✅ Vector store connection established")
            print(f"📊 Using embed_dim: {os.getenv('VECTOR_DIM', '2560')}")
            
        except Exception as e:
            print(f"❌ Vector store setup error: {str(e)}")
            # Try alternative connection method
            print("🔄 Trying alternative connection...")
            self.vector_store = PGVectorStore(
                connection_string=db_url,
                table_name="llamaindex_embeddings",
                embed_dim=int(os.getenv("VECTOR_DIM", "2560"))
            )
        
        print("✅ Vector store initialized")
    
    def setup_parsers(self):
        """Initialize document parsers"""
        self.parsers = {
            '.pdf': PDFReader(),
            '.docx': DocxReader(),
            '.txt': UnstructuredReader(),
            '.md': UnstructuredReader(),
            '.json': UnstructuredReader(),
        }
        
        # Node parser for chunking
        self.node_parser = SentenceSplitter(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap,
            separator=" "
        )
        
        print("✅ Document parsers initialized")
    
    def process_file(self, file_path: str, file_id: str, org_id: str, 
                    user_id: str, filename: str) -> Dict[str, Any]:
        """
        Process a file using LlamaIndex and store in NeonDB
        
        Args:
            file_path: Path to the file
            file_id: Unique file identifier
            org_id: Organization ID
            user_id: User ID
            filename: Original filename
            
        Returns:
            Processing results
        """
        try:
            print(f"🔄 Processing {filename} with LlamaIndex...")
            print(f"📁 File path: {file_path}")
            print(f"📋 File exists: {os.path.exists(file_path)}")
            print(f"📏 File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'} bytes")
            
            # Determine file type and parser
            file_ext = os.path.splitext(filename)[1].lower()
            print(f"🔍 File extension: {file_ext}")
            parser = self.parsers.get(file_ext, self.parsers['.txt'])
            print(f"📖 Using parser: {type(parser).__name__}")
            
            # Load document
            print(f"📥 Loading document with parser...")
            documents = parser.load_data(file_path)
            print(f"📄 Loaded {len(documents)} document(s)")
            
            if documents:
                print(f"📝 First document preview: {documents[0].text[:200]}...")
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "file_id": file_id,
                    "org_id": org_id,
                    "user_id": user_id,
                    "filename": filename,
                    "file_type": file_ext
                })
            
            # Parse into nodes (chunks)
            print(f"✂️ Parsing documents into chunks...")
            nodes = self.node_parser.get_nodes_from_documents(documents)
            print(f"✂️ Created {len(nodes)} chunks")
            
            if nodes:
                print(f"📝 First chunk preview: {nodes[0].text[:200]}...")
            
            # Add chunk metadata
            for i, node in enumerate(nodes):
                node.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(nodes)
                })
            
            # Create index and store in vector database
            print(f"🔗 Creating vector index and storing in NeonDB...")
            print(f"🧮 Generating embeddings for {len(nodes)} chunks...")
            print(f"📊 Vector store type: {type(self.vector_store)}")
            print(f"🔧 Vector store table: {getattr(self.vector_store, 'table_name', 'unknown')}")
            
            try:
                # Explicitly pass the embedding model to ensure it's used
                print(f"🔧 Creating index with embed_model: {type(self.embed_model)}")
                index = VectorStoreIndex(
                    nodes=nodes,
                    vector_store=self.vector_store,
                    embed_model=self.embed_model,  # Explicitly pass our custom model
                    show_progress=True
                )
                
                print(f"✅ Index created successfully")
                
                # Verify storage by checking the vector store directly
                print(f"🔍 Verifying storage...")
                
                # Force a refresh/commit if possible
                if hasattr(self.vector_store, '_client') and hasattr(self.vector_store._client, 'commit'):
                    self.vector_store._client.commit()
                    print(f"💾 Database committed")
                
                print(f"✅ Stored {len(nodes)} embeddings in NeonDB")
                print(f"🎯 Vector store table: llamaindex_embeddings")
                
            except Exception as e:
                print(f"❌ Error creating index: {str(e)}")
                import traceback
                print(f"🐛 Traceback: {traceback.format_exc()}")
                raise
            
            return {
                "success": True,
                "file_id": file_id,
                "filename": filename,
                "chunks_processed": len(nodes),
                "documents_loaded": len(documents),
                "vector_store": "neondb_postgresql"
            }
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_id": file_id,
                "filename": filename
            }
    
    def search_documents(self, query: str, org_id: str, 
                        top_k: int = 10, file_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Search documents using LlamaIndex
        
        Args:
            query: Search query
            org_id: Organization ID to filter by
            top_k: Number of results to return
            file_ids: Optional list of file IDs to search within
            
        Returns:
            Search results
        """
        try:
            print(f"🔍 Searching for: '{query}' in org: {org_id}")
            
            # Create index from existing vector store
            index = VectorStoreIndex.from_vector_store(self.vector_store)
            
            # Create retriever with filters
            filters = {"org_id": org_id}
            if file_ids:
                filters["file_id"] = {"$in": file_ids}
            
            retriever = index.as_retriever(
                similarity_top_k=top_k,
                filters=filters
            )
            
            # Retrieve relevant nodes
            nodes = retriever.retrieve(query)
            
            # Format results
            results = []
            for node in nodes:
                results.append({
                    "text": node.text,
                    "score": node.score,
                    "metadata": node.metadata,
                    "file_id": node.metadata.get("file_id"),
                    "filename": node.metadata.get("filename"),
                    "chunk_index": node.metadata.get("chunk_index")
                })
            
            print(f"📊 Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            return []
    
    def create_query_engine(self, org_id: str, file_ids: Optional[List[str]] = None):
        """
        Create a query engine for conversational Q&A
        
        Args:
            org_id: Organization ID
            file_ids: Optional file IDs to limit search
            
        Returns:
            Query engine
        """
        # Create index
        index = VectorStoreIndex.from_vector_store(self.vector_store)
        
        # Create filters
        filters = {"org_id": org_id}
        if file_ids:
            filters["file_id"] = {"$in": file_ids}
        
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=20,
            filters=filters,
            response_mode="compact"
        )
        
        return query_engine
    
    def query_qwen2(self, prompt: str) -> str:
        """
        Query Qwen2 model using direct HuggingFace API
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        try:
            print(f"🤖 Querying Qwen2 model via HF API...")
            
            hf_endpoint = os.getenv("HUGGING_FACE_ENDPOINT")
            hf_token = os.getenv("HF_TOKEN")
            
            if not hf_endpoint or not hf_token:
                return "Error: HF endpoint or token not configured"
            
            headers = {
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True
                }
            }
            
            response = requests.post(hf_endpoint, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "No response generated").strip()
                else:
                    return str(result)
            else:
                print(f"❌ HF API error: {response.status_code} - {response.text}")
                return f"Error: HF API returned {response.status_code}"
                
        except Exception as e:
            print(f"❌ Qwen2 query error: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_answer(self, query: str, org_id: str, file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive answer using retrieval + Qwen2
        
        Args:
            query: User question
            org_id: Organization ID
            file_ids: Optional file IDs to search
            
        Returns:
            Generated answer with sources
        """
        try:
            # Step 1: Retrieve relevant documents
            print(f"🔍 Retrieving relevant documents...")
            search_results = self.search_documents(query, org_id, top_k=5, file_ids=file_ids)
            
            if not search_results:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "sources": [],
                    "confidence": "low"
                }
            
            # Step 2: Build context from retrieved documents
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results[:3]):  # Use top 3 results
                context_parts.append(f"Source {i+1}: {result['text']}")
                sources.append({
                    "filename": result['metadata'].get('filename', 'Unknown'),
                    "chunk_index": result['metadata'].get('chunk_index', 0),
                    "score": result['score']
                })
            
            context = "\n\n".join(context_parts)
            
            # Step 3: Create prompt for Qwen2
            prompt = f"""Based on the following context, please answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
            
            # Step 4: Generate answer using Qwen2
            print(f"🤖 Generating answer with Qwen2...")
            answer = self.query_qwen2(prompt)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": "high" if len(search_results) >= 3 else "medium",
                "total_sources_found": len(search_results)
            }
            
        except Exception as e:
            print(f"❌ Answer generation error: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "confidence": "low"
            }

# Global instance
llamaindex_processor = None

def get_llamaindex_processor():
    """Get global LlamaIndex processor instance"""
    global llamaindex_processor
    if llamaindex_processor is None:
        llamaindex_processor = LlamaIndexProcessor()
    return llamaindex_processor

def process_file_with_llamaindex(file_path: str, file_id: str, org_id: str, 
                                user_id: str, filename: str) -> Dict[str, Any]:
    """Convenience function for file processing"""
    processor = get_llamaindex_processor()
    return processor.process_file(file_path, file_id, org_id, user_id, filename)

def search_with_llamaindex(query: str, org_id: str, top_k: int = 10, 
                          file_ids: Optional[List[str]] = None) -> List[Dict]:
    """Convenience function for searching"""
    processor = get_llamaindex_processor()
    return processor.search_documents(query, org_id, top_k, file_ids)