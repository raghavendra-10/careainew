---
title: BGE Large EN v1.5 Embedding Service
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

# BGE Large EN v1.5 Embedding Service

High-performance embedding service using BAAI/bge-large-en-v1.5 - currently the best English embedding model on MTEB leaderboard.

## Features

- **State-of-the-art performance**: Top ranking on MTEB benchmark
- **1024 dimensions**: Rich semantic representations
- **Optimized for retrieval**: Query prefixing for better search performance
- **Batch processing**: Efficient handling of multiple texts
- **Normalized embeddings**: Ready for cosine similarity

## API Endpoints

### Health Check
```bash
GET /
```

### Batch Embedding
```bash
POST /embed
Content-Type: application/json

{
  "texts": ["Your text here", "Another text"]
}
```

### Single Text Embedding
```bash
POST /embed_single
Content-Type: application/json

{
  "text": "Your text here"
}
```

## Usage Example

```python
import requests

# Your HF Space URL
url = "https://your-username-bge-embeddings.hf.space"

# Embed texts
response = requests.post(f"{url}/embed", json={
    "texts": ["What is machine learning?", "How does AI work?"]
})

embeddings = response.json()["embeddings"]
print(f"Got {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
```

## Performance

- **Model**: BAAI/bge-large-en-v1.5
- **Dimensions**: 1024
- **MTEB Score**: 67.2 (top English model)
- **Speed**: ~1000 texts/second on T4 GPU