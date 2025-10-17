---
title: Qwen3-7B-Instruct LLM API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: apache-2.0
---

# Qwen3-7B-Instruct LLM API

This Space provides a REST API for text generation using the Qwen3-7B-Instruct model.

## Features

- 🤖 **Qwen3-7B-Instruct** - Latest high-quality instruction-following model
- 💬 **Chat Interface** - Optimized for conversational AI
- 🔗 **RAG Support** - Context-aware question answering
- ⚡ **Fast Inference** - Optimized for production use
- 🎯 **Simple API** - Easy integration

## API Endpoints

### 1. Health Check
```bash
GET /
```

### 2. Text Generation
```bash
POST /generate
Content-Type: application/json

{
  "prompt": "Your prompt here",
  "max_length": 1024,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### 3. Chat (RAG-ready)
```bash
POST /chat
Content-Type: application/json

{
  "query": "Your question",
  "context": "Optional context for RAG",
  "max_length": 1024,
  "temperature": 0.7
}
```

## Usage Examples

### Simple Generation
```python
import requests

response = requests.post(
    "https://your-space-url/generate",
    json={"prompt": "Explain quantum computing"}
)
print(response.json()["response"])
```

### RAG Chat
```python
import requests

response = requests.post(
    "https://your-space-url/chat",
    json={
        "query": "What is the main topic?",
        "context": "Your retrieved context here..."
    }
)
print(response.json()["answer"])
```

## Model Details

- **Model**: Qwen/Qwen3-7B-Instruct
- **Parameters**: 7 billion
- **Context Length**: 32K tokens
- **Languages**: English, Chinese, and more
- **Use Case**: Instruction following, chat, RAG

## Performance

- **GPU**: Optimized for T4/A10G
- **Memory**: ~14GB VRAM required
- **Speed**: ~20-50 tokens/second
- **Quantization**: FP16 for efficiency

Built for production RAG applications! 🚀