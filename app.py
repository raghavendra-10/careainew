import os
import torch
import pandas as pd
import docx
import json
from PyPDF2 import PdfReader
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from firebase_admin import credentials, firestore, initialize_app
from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai import init as vertex_init
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from vertexai.preview.language_models import TextEmbeddingModel
from waitress import serve

#  Load .env  #
load_dotenv()

#  Flask Setup  #
app = Flask(__name__)
CORS(app)

#  Google Cloud Setup  #
cred_path = "fire.json"  # Ensure this file exists
gcp_credentials = service_account.Credentials.from_service_account_file(cred_path)

aiplatform.init(project="buildnblog-450618", location="us-central1", credentials=gcp_credentials)
vertex_init(project="buildnblog-450618", location="us-central1")

#  Firebase Setup  #
firebase_cred = credentials.Certificate(cred_path)
initialize_app(firebase_cred)
db = firestore.client()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#  Embedding Functions  #
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
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file_type == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            text = json.dumps(json.load(f), indent=2)
    elif file_type == "csv":
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)  # Convert CSV to a readable string
    elif file_type == "xlsx":
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
            text = df.to_string(index=False)
        except ImportError:
            return "❌ Error: Missing optional dependency 'openpyxl'. Please install it using `pip install openpyxl`."

    return text

def parse_and_chunk(file_path, file_ext, chunk_size=100):
    text = extract_text_from_file(file_path, file_ext)
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
    response = model.get_embeddings(chunks)
    return [embedding.values for embedding in response]

def embed_query(query):
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
    response = model.get_embeddings([query])
    return response[0].values

#  GCP AI Model - Text Generation  #
def generate_answer_with_gcp(query, context_chunks):
    context_text = "\n\n".join(context_chunks)
    
    prompt = f"""
You are an intelligent assistant. Below is a set of information retrieved from various documents.

Context:
{context_text}

Question: {query}

Answer (based ONLY on the above context):
"""

    model = GenerativeModel("gemini-1.5-pro-002")
    # gemini 2.0 flash latest 

    responses = model.generate_content(
        [Part.from_text(prompt)],
        generation_config={
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        },
        safety_settings=[
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
        ],
        stream=False  # Set to True if you want real-time streaming response
    )

    return responses.text.strip()

#  Upload Route  #
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        org_id = request.args.get("orgId")
        if not org_id:
            return jsonify({"error": "orgId is required"}), 400

        if "file" not in request.files:
            return jsonify({"error": "No file provided."}), 400

        file = request.files["file"]
        filename = file.filename
        file_ext = filename.split(".")[-1].lower()

        if file_ext not in ["csv", "txt", "pdf", "docx", "json", "xlsx"]:
            return jsonify({"error": "Unsupported file type"}), 400

        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)
        print(f"✅ File saved at {save_path}")

        # Extract text and create chunks
        chunks = parse_and_chunk(save_path, file_ext)
        if not chunks:
            return jsonify({"error": "No content extracted from file."}), 400

        # Get embeddings for chunks
        embeddings = embed_chunks(chunks)

        # Store in Firestore
        org_doc_ref = db.collection("document_embeddings").document(f"org-{org_id}")
        org_doc = org_doc_ref.get()
        files = org_doc.to_dict().get("files", []) if org_doc.exists else []

        new_file = {
            "filename": filename,
            "chunks": [{"index": idx, "content": chunk, "embedding": embed} for idx, (chunk, embed) in enumerate(zip(chunks, embeddings))]
        }
        files.append(new_file)
        org_doc_ref.set({"files": files}, merge=True)

        print("✅ Embeddings stored in Firestore.")
        return jsonify({"message": "File processed, embedded, and stored successfully."}), 200

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"🧹 File {save_path} deleted successfully.")
            
            
@app.route("/delete", methods=["DELETE"])
def delete_file():
    try:
        org_id = request.args.get("orgId")
        filename = request.args.get("filename")

        if not org_id or not filename:
            return jsonify({"error": "orgId and filename are required"}), 400

        org_doc_ref = db.collection("document_embeddings").document(f"org-{org_id}")
        org_doc = org_doc_ref.get()
        if not org_doc.exists:
            return jsonify({"message": "No documents found for this organization."}), 200

        files = org_doc.to_dict().get("files", [])
        updated_files = [file for file in files if file.get("filename") != filename]

        if len(files) == len(updated_files):
            return jsonify({"message": "No matching file found to delete."}), 404

        org_doc_ref.set({"files": updated_files}, merge=True)
        return jsonify({"message": f"File '{filename}' and its embeddings deleted successfully."}), 200

    except Exception as e:
        print("❌ Deletion Error:", str(e))
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


#  Chat Route  #
@app.route("/chat", methods=["POST"])
def chat_with_doc():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No valid JSON input found"}), 400

        query = data.get("query")
        org_id = data.get("orgId")
        if not query or not org_id:
            return jsonify({"error": "Query and orgId are required."}), 400

        query_embedding = embed_query(query)

        org_doc_ref = db.collection("document_embeddings").document(f"org-{org_id}")
        org_doc = org_doc_ref.get()
        if not org_doc.exists:
            return jsonify({"query": query, "retrieved_chunks": [], "answer": "No documents found for this organization."}), 200

        files_data = org_doc.to_dict().get("files", [])
        retrieved_docs = []

        for file in files_data:
            for chunk_info in file.get("chunks", []):
                chunk_embedding = torch.tensor(chunk_info["embedding"])
                query_tensor = torch.tensor(query_embedding)
                score = torch.nn.functional.cosine_similarity(query_tensor, chunk_embedding.unsqueeze(0))[0].item()
                retrieved_docs.append({"content": chunk_info["content"], "score": score})

        top_chunks = sorted([doc for doc in retrieved_docs if doc["score"] >= 0.2], key=lambda x: x["score"], reverse=True)[:3]

        if not top_chunks:
            return jsonify({"query": query, "retrieved_chunks": [], "answer": "No relevant information found."}), 200

        context_chunks = [doc["content"] for doc in top_chunks]
        answer = generate_answer_with_gcp(query, context_chunks)

        return jsonify({"query": query, "retrieved_chunks": context_chunks, "answer": answer}), 200

    except Exception as e:
        print("❌ Exception occurred:", e)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
