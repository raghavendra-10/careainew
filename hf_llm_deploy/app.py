#!/usr/bin/env python3
"""
Qwen2.5-3B-Instruct LLM API for HuggingFace Spaces with ZeroGPU
"""

import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import gc
import spaces
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load Qwen2.5-3B-Instruct model"""
    global model, tokenizer
    
    if model is not None:
        return
    
    try:
        logger.info("🚀 Loading Qwen2.5-3B-Instruct model...")
        
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        
        # Load tokenizer
        logger.info("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        logger.info("🧠 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        model.eval()
        logger.info("✅ Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise

@spaces.GPU
def generate_response(prompt: str, max_length: int = 1024, temperature: float = 0.7, top_p: float = 0.9) -> str:
    """Generate response using Qwen model - MUST be decorated with @spaces.GPU"""
    
    # Load model inside GPU function
    if model is None or tokenizer is None:
        load_model()
    
    try:
        # Format prompt for chat
        messages = [{"role": "user", "content": prompt}]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to(device)
        
        generation_kwargs = {
            "max_new_tokens": min(max_length, 1024),
            "temperature": max(temperature, 0.1),
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
            "use_cache": True,
        }
        
        if 'attention_mask' in inputs:
            generation_kwargs['attention_mask'] = inputs['attention_mask']
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                **generation_kwargs
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "<|im_start|>assistant" in response:
            parts = response.split("<|im_start|>assistant")
            if len(parts) > 1:
                response = parts[-1].replace("<|im_end|>", "").strip()
        elif formatted_prompt in response:
            response = response.replace(formatted_prompt, "").strip()
        
        # Cleanup
        del inputs, outputs
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Generation error: {str(e)}")
        if device == "cuda":
            torch.cuda.empty_cache()
        raise

@spaces.GPU
def chat_with_context(query: str, context: str = "", max_length: int = 1024, temperature: float = 0.7) -> str:
    """Chat with RAG context support"""
    
    if model is None or tokenizer is None:
        load_model()
    
    # Build RAG prompt
    if context:
        prompt = f"""Based on the following context, please answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
    else:
        prompt = query
    
    logger.info(f"💬 Chat request: {query[:100]}...")
    
    return generate_response(prompt, max_length, temperature)

# Create Gradio Interface
with gr.Blocks(title="Qwen2.5-3B-Instruct API", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Qwen2.5-3B-Instruct LLM API
    
    Powered by ZeroGPU - Optimized for RAG applications
    """)
    
    with gr.Tabs():
        # Tab 1: Simple Generation
        with gr.Tab("💬 Simple Chat"):
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your question or prompt...",
                        lines=5
                    )
                    with gr.Row():
                        max_len_slider = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=1024,
                            step=128,
                            label="Max Length"
                        )
                        temp_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top P"
                        )
                    generate_btn = gr.Button("🚀 Generate", variant="primary")
                
                with gr.Column():
                    output_text = gr.Textbox(
                        label="Response",
                        lines=15,
                        interactive=False
                    )
            
            generate_btn.click(
                fn=generate_response,
                inputs=[prompt_input, max_len_slider, temp_slider, top_p_slider],
                outputs=output_text
            )
        
        # Tab 2: RAG Chat
        with gr.Tab("📚 RAG Chat"):
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="Question",
                        placeholder="Enter your question...",
                        lines=3
                    )
                    context_input = gr.Textbox(
                        label="Context (for RAG)",
                        placeholder="Paste your context here...",
                        lines=8
                    )
                    with gr.Row():
                        rag_max_len = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=1024,
                            step=128,
                            label="Max Length"
                        )
                        rag_temp = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                    chat_btn = gr.Button("💬 Ask Question", variant="primary")
                
                with gr.Column():
                    chat_output = gr.Textbox(
                        label="Answer",
                        lines=15,
                        interactive=False
                    )
            
            chat_btn.click(
                fn=chat_with_context,
                inputs=[query_input, context_input, rag_max_len, rag_temp],
                outputs=chat_output
            )
    
    gr.Markdown("""
    ### 🔌 API Usage
    
    You can call this Space programmatically:
    
    ```
    from gradio_client import Client
    
    client = Client("YOUR_USERNAME/YOUR_SPACE_NAME")
    
    # Simple generation
    result = client.predict(
        prompt="What is artificial intelligence?",
        max_length=1024,
        temperature=0.7,
        top_p=0.9,
        api_name="/predict"
    )
    print(result)
    
    # RAG chat
    result = client.predict(
        query="What is the main topic?",
        context="Your context here...",
        max_length=1024,
        temperature=0.7,
        api_name="/predict_1"
    )
    print(result)
    ```
    
    ### 📝 Features
    - ✅ ZeroGPU support for cost-effective inference
    - ✅ Optimized for RAG applications
    - ✅ Context-aware question answering
    - ✅ Adjustable generation parameters
    - ✅ API access via Gradio Client
    """)

if __name__ == "__main__":
    logger.info("🚀 Starting Qwen2.5-3B-Instruct Gradio App...")
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True
    )
