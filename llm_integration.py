#!/usr/bin/env python3
"""
LLM integration for Qwen3 model
"""
import os
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class Qwen3LLMClient:
    def __init__(self):
        self.endpoint_url = os.getenv("QWEN3_LLM_ENDPOINT")
        if not self.endpoint_url:
            raise ValueError("QWEN3_LLM_ENDPOINT not found in environment")
        
        # Ensure endpoint URL is properly formatted
        if not self.endpoint_url.startswith("http"):
            self.endpoint_url = f"https://{self.endpoint_url}"
        if not self.endpoint_url.endswith("/"):
            self.endpoint_url += "/"
    
    def generate_answer(
        self, 
        query: str, 
        context: str = "", 
        max_length: int = 1024,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate answer using Qwen3 LLM with optional context (RAG)
        
        Args:
            query: User question
            context: Retrieved context from documents
            max_length: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            Dictionary with answer and metadata
        """
        print(f"🤖 Generating answer for: '{query[:100]}...'")
        print(f"📝 Context length: {len(context)} chars")
        
        try:
            # Try using Gradio Client first
            try:
                from gradio_client import Client
                llm_space_name = os.getenv("LLM_SPACENAME")
                
                if llm_space_name:
                    print(f"🔗 Using HF Space: {llm_space_name}")
                    client = Client(llm_space_name)
                    
                    # Use exact format from your deployed app documentation
                    if context:
                        # RAG chat endpoint - Tab 2 (index 1)
                        result = client.predict(
                            query,          # query parameter
                            context,        # context parameter
                            max_length,     # max_length parameter
                            temperature,    # temperature parameter
                            fn_index=1      # RAG Chat tab function
                        )
                    else:
                        # Simple generation endpoint - Tab 1 (index 0)  
                        result = client.predict(
                            query,          # prompt parameter
                            max_length,     # max_length parameter
                            temperature,    # temperature parameter
                            0.9,            # top_p parameter
                            fn_index=0      # Simple Chat tab function
                        )
                    
                    # Clean up response - extract assistant's answer
                    clean_answer = result
                    
                    # Handle tuple response from Gradio (extract the first string element)
                    if isinstance(result, (tuple, list)):
                        # Find the first string in the tuple/list
                        for item in result:
                            if isinstance(item, str):
                                result = item
                                break
                        else:
                            # If no string found, convert first element to string
                            result = str(result[0]) if result else ""
                    
                    if isinstance(result, str):
                        # Remove chat template artifacts
                        if "assistant\n" in result:
                            parts = result.split("assistant\n")
                            if len(parts) > 1:
                                clean_answer = parts[-1].strip()
                        elif "assistant" in result:
                            parts = result.split("assistant")
                            if len(parts) > 1:
                                clean_answer = parts[-1].strip()
                        
                        # Remove system prompts if present
                        if "system\nYou are Qwen" in clean_answer:
                            lines = clean_answer.split('\n')
                            # Find where actual response starts
                            for i, line in enumerate(lines):
                                if isinstance(line, str) and line.strip() and not line.startswith("system") and not line.startswith("user") and not line.startswith("You are Qwen"):
                                    clean_answer = '\n'.join(lines[i:]).strip()
                                    break
                    
                    return {
                        "success": True,
                        "answer": clean_answer,
                        "model": "Qwen2.5-3B-Instruct",
                        "context_used": len(context) > 0,
                        "context_length": len(context),
                        "source": "hf_space_gradio"
                    }
                    
            except Exception as e:
                print(f"⚠️ Gradio client failed, falling back to HTTP: {str(e)}")
            
            # Fallback to direct HTTP requests to the configured endpoint
            if self.endpoint_url and "hf.space" not in self.endpoint_url:
                # Use the configured endpoint (non-HF Space)
                response = requests.post(
                    f"{self.endpoint_url}chat",
                    json={
                        "query": query,
                        "context": context,
                        "max_length": max_length,
                        "temperature": temperature
                    },
                    timeout=None,  # No timeout for LLM generation
                    headers={"Content-Type": "application/json"},
                    verify=False  # Skip SSL verification for potential SSL issues
                )
            else:
                # No fallback available for HF Spaces
                print("❌ No HTTP fallback available for HF Spaces")
                return {
                    "success": False,
                    "error": "LLM service unavailable",
                    "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                    "model": "fallback",
                    "context_used": len(context) > 0,
                    "context_length": len(context)
                }
            
            print(f"📊 LLM Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "")
                
                print(f"✅ Generated answer: {len(answer)} chars")
                print(f"📋 Answer preview: {answer[:150]}...")
                
                return {
                    "success": True,
                    "answer": answer,
                    "query": query,
                    "context_used": len(context) > 0,
                    "context_length": len(context),
                    "model": "Qwen3-7B-Instruct",
                    "parameters": {
                        "max_length": max_length,
                        "temperature": temperature
                    }
                }
            else:
                error_msg = f"LLM API error: {response.status_code}"
                try:
                    error_detail = response.json().get("error", response.text)
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text}"
                
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "query": query
                }
                
        except requests.exceptions.RequestException as e:
            error_msg = f"LLM request failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "query": query
            }
        except Exception as e:
            error_msg = f"LLM generation error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "query": query
            }
    
    def simple_generate(
        self, 
        prompt: str, 
        max_length: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Simple text generation without RAG context
        
        Args:
            prompt: Text prompt
            max_length: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            Dictionary with response and metadata
        """
        print(f"🔤 Simple generation for: '{prompt[:100]}...'")
        
        try:
            response = requests.post(
                f"{self.endpoint_url}generate",
                json={
                    "prompt": prompt,
                    "max_length": max_length,
                    "temperature": temperature
                },
                timeout=None,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📊 LLM Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                print(f"✅ Generated response: {len(response_text)} chars")
                
                return {
                    "success": True,
                    "response": response_text,
                    "prompt": prompt,
                    "model": "Qwen3-7B-Instruct",
                    "parameters": result.get("parameters", {})
                }
            else:
                error_msg = f"LLM API error: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "prompt": prompt
                }
                
        except Exception as e:
            error_msg = f"LLM generation error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "prompt": prompt
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check if LLM endpoint is healthy"""
        try:
            response = requests.get(f"{self.endpoint_url}", timeout=10)
            if response.status_code == 200:
                return {
                    "healthy": True,
                    "endpoint": self.endpoint_url,
                    "status": response.json()
                }
            else:
                return {
                    "healthy": False,
                    "endpoint": self.endpoint_url,
                    "error": f"Status {response.status_code}"
                }
        except Exception as e:
            return {
                "healthy": False,
                "endpoint": self.endpoint_url,
                "error": str(e)
            }

# Convenience functions
def generate_rag_answer(query: str, context: str = "", **kwargs) -> Dict[str, Any]:
    """Generate RAG answer using Qwen3"""
    client = Qwen3LLMClient()
    return client.generate_answer(query, context, **kwargs)

def generate_simple_response(prompt: str, **kwargs) -> Dict[str, Any]:
    """Generate simple response using Qwen3"""
    client = Qwen3LLMClient()
    return client.simple_generate(prompt, **kwargs)