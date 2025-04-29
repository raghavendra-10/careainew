import pandas as pd
import docx
import json
from PyPDF2 import PdfReader
from vertexai.preview.language_models import TextEmbeddingModel

def extract_text_from_file(file_path, file_type):
    text = ""
    if file_type == "pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif file_type == "docx":
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file_type == "txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif file_type == "json":
        with open(file_path, 'r', encoding='utf-8') as f:
            text = json.dumps(json.load(f), indent=2)
    return text

def parse_and_chunk(file_path, file_ext, chunk_size=100):
    text = extract_text_from_file(file_path, file_ext)
    words = text.split()
    return [' '.join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
    response = model.get_embeddings(chunks)
    return [embedding.values for embedding in response]

def embed_query(query):
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
    response = model.get_embeddings([query])
    return response[0].values
