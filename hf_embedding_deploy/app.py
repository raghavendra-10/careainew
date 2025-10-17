import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import gradio as gr
import logging
import spaces

# Try to import flash attention (optional)
try:
    import flash_attn
    FLASH_ATTENTION_AVAILABLE = True
    print("✅ Flash attention available")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("⚠️ Flash attention not available, using standard attention")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qwen3-Embedding-4B model for retrieval
MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 2560

# Global model instances
tokenizer = None
model = None

def initialize_model():
    """Initialize model"""
    global tokenizer, model
    
    if tokenizer is None:
        logger.info(f"Loading tokenizer for {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
    
    if model is None:
        logger.info(f"Loading {MODEL_NAME} on {DEVICE}")
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32
        }
        
        if FLASH_ATTENTION_AVAILABLE and DEVICE == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("🚀 Using Flash Attention 2")
        
        model = AutoModel.from_pretrained(MODEL_NAME, **model_kwargs)
        model.to(DEVICE)
        model.eval()
        logger.info("✅ Model loaded successfully")

@spaces.GPU
def encode_texts_gpu(texts_str, batch_size=16):
    """
    Encode texts to embeddings using Qwen3-Embedding-4B
    Args:
        texts_str: Either a single text string or multiple texts separated by '|||'
        batch_size: Batch size for encoding
    Returns:
        JSON string with embeddings
    """
    global tokenizer, model
    
    # Ensure model is initialized
    if model is None or tokenizer is None:
        initialize_model()
    
    # Parse input - support both single text and multiple texts
    if '|||' in texts_str:
        texts = [t.strip() for t in texts_str.split('|||')]
    else:
        texts = [texts_str]
    
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Qwen3 instruction format for retrieval
        batch_texts = [f"Instruct: Retrieve semantically similar text.\nQuery: {text}" for text in batch_texts]
        
        inputs = tokenizer(
            batch_texts,
            padding="longest",
            truncation=True,
            max_length=32768,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            eos_token_id = tokenizer.eos_token_id
            sequence_lengths = (inputs['input_ids'] == eos_token_id).long().argmax(-1) - 1
            
            batch_embeddings = []
            for j, seq_len in enumerate(sequence_lengths):
                embedding = outputs.last_hidden_state[j, seq_len, :].cpu().numpy()
                batch_embeddings.append(embedding)
            
            batch_embeddings = np.array(batch_embeddings)
            batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            
            embeddings.extend(batch_embeddings)
    
    # Format output
    import json
    result = {
        "embeddings": [emb.tolist() for emb in embeddings],
        "model": MODEL_NAME,
        "dimension": len(embeddings[0]) if embeddings else 0,
        "count": len(embeddings)
    }
    
    return json.dumps(result, indent=2)

# Create Gradio interface
with gr.Blocks(title="Qwen3-Embedding-4B API") as demo:
    gr.Markdown("""
    # Qwen3-Embedding-4B Embedding Service
    
    This service generates embeddings using Qwen3-Embedding-4B (2560 dimensions).
    
    **Usage:**
    - Single text: Enter your text directly
    - Multiple texts: Separate texts with `|||` (e.g., `text1|||text2|||text3`)
    """)
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text Input",
                placeholder="Enter text or multiple texts separated by '|||'",
                lines=5
            )
            batch_size_input = gr.Slider(
                minimum=1,
                maximum=64,
                value=16,
                step=1,
                label="Batch Size"
            )
            submit_btn = gr.Button("Generate Embeddings", variant="primary")
        
        with gr.Column():
            output = gr.JSON(label="Embeddings Output")
    
    submit_btn.click(
        fn=encode_texts_gpu,
        inputs=[text_input, batch_size_input],
        outputs=output
    )
    
    gr.Markdown("""
    ### API Usage
    You can also call this Space via API:
    ```
    from gradio_client import Client
    
    client = Client("YOUR_USERNAME/YOUR_SPACE_NAME")
    result = client.predict(
        texts_str="Your text here",
        batch_size=16,
        api_name="/predict"
    )
    print(result)
    ```
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
