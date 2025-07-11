import os
import numpy as np
import pandas as pd
import json
import time
import uuid
import threading
import traceback
import requests
import re
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from threading import Thread, Lock
import asyncio
from waitress import serve
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import zipfile
import io

# Load environment variables
load_dotenv()

# Global variables
FIREBASE_AVAILABLE = False
OPENAI_AVAILABLE = False
VERTEX_AVAILABLE = False
db = None
openai_client = None

# Flask Setup
app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "DELETE", "OPTIONS", "PUT"], 
     allow_headers=["Content-Type", "Authorization"])

# Global progress tracking with thread safety
upload_progress = {}
progress_lock = Lock()

# Create uploads folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================================
# DEPENDENCY CHECKING
# ================================

def check_file_processing_dependencies():
    """Check what file types are currently supported and provide installation guidance"""
    supported = {"always": ["txt", "json"], "conditional": []}
    missing = []
    
    print("🔍 Checking file processing capabilities...")
    
    try:
        import openpyxl
        supported["conditional"].append("xlsx")
        print("✅ openpyxl available - Excel .xlsx files supported")
    except ImportError:
        missing.append("openpyxl")
        print("❌ openpyxl missing - Excel .xlsx files NOT supported")
    
    try:
        import xlrd
        supported["conditional"].append("xls") 
        print("✅ xlrd available - Excel .xls files supported")
    except ImportError:
        missing.append("xlrd")
        print("❌ xlrd missing - Excel .xls files NOT supported")
    
    try:
        from PyPDF2 import PdfReader
        supported["conditional"].append("pdf")
        print("✅ PyPDF2 available - PDF files supported")
    except ImportError:
        missing.append("PyPDF2")
        print("❌ PyPDF2 missing - PDF files NOT supported")
    
    try:
        import docx
        supported["conditional"].append("docx")
        print("✅ python-docx available - Word files supported")
    except ImportError:
        missing.append("python-docx")
        print("❌ python-docx missing - Word files NOT supported")
    
    try:
        from pptx import Presentation
        supported["conditional"].append("pptx")
        print("✅ python-pptx available - PowerPoint files supported")
    except ImportError:
        missing.append("python-pptx")
        print("⚠️ python-pptx missing - PowerPoint files NOT supported")
    
    # Always supported with pandas
    supported["conditional"].extend(["csv"])
    print("✅ CSV files always supported (pandas)")
    
    if missing:
        print(f"\n⚠️ Missing dependencies for: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        print("For all file support: pip install openpyxl xlrd PyPDF2 python-docx python-pptx")
    
    all_supported = supported["always"] + supported["conditional"]
    print(f"\n✅ Currently supported file types: {', '.join(sorted(all_supported))}")
    
    return supported, missing

# ================================
# ENHANCED FILE EXTRACTION
# ================================

def extract_pdf_text(file_path):
    """Extract text from PDF files"""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {i+1} ---\n{page_text}\n"
            except Exception as page_error:
                print(f"⚠️ Error extracting page {i+1}: {page_error}")
                continue
        
        return text
    except ImportError:
        return "ERROR: PyPDF2 not installed. Install with: pip install PyPDF2"
    except Exception as e:
        return f"PDF extraction error: {str(e)}"

def extract_docx_text(file_path):
    """Extract text from Word documents"""
    try:
        import docx
        doc = docx.Document(file_path)
        text = ""
        
        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
        
        # Extract tables
        for i, table in enumerate(doc.tables):
            text += f"\n--- Table {i+1} ---\n"
            for row in table.rows:
                row_text = " | ".join([cell.text.strip() for cell in row.cells])
                if row_text.strip():
                    text += row_text + "\n"
        
        return text
    except ImportError:
        return "ERROR: python-docx not installed. Install with: pip install python-docx"
    except Exception as e:
        return f"DOCX extraction error: {str(e)}"

def extract_excel_text(file_path, file_type):
    """Extract text from Excel files with comprehensive error handling"""
    try:
        engines = {"xlsx": "openpyxl", "xls": "xlrd"}
        engine = engines.get(file_type, "openpyxl")
        
        try:
            # Read the Excel file
            excel_file = pd.ExcelFile(file_path, engine=engine)
            sheet_names = excel_file.sheet_names
            
            text = f"Excel File: {len(sheet_names)} sheet(s) - {', '.join(sheet_names)}\n\n"
            
            # Process each sheet (limit to first 5 sheets)
            for i, sheet_name in enumerate(sheet_names[:5]):
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine, nrows=200)
                    
                    text += f"=== Sheet: {sheet_name} ===\n"
                    text += f"Dimensions: {len(df)} rows × {len(df.columns)} columns\n"
                    text += f"Columns: {', '.join(df.columns.astype(str))}\n\n"
                    
                    # Show sample data
                    text += "Sample Data:\n"
                    text += df.head(10).to_string(index=False) + "\n\n"
                    
                    # Add summary statistics for numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        text += f"Numeric Summary:\n"
                        text += df[numeric_cols].describe().to_string() + "\n\n"
                    
                except Exception as sheet_error:
                    text += f"Error reading sheet '{sheet_name}': {str(sheet_error)}\n\n"
            
            return text
            
        except ImportError as ie:
            error_msg = str(ie)
            if "openpyxl" in error_msg:
                return "ERROR: openpyxl not installed. Install with: pip install openpyxl"
            elif "xlrd" in error_msg:
                return "ERROR: xlrd not installed. Install with: pip install xlrd"
            else:
                return f"ERROR: Missing Excel dependency: {error_msg}"
                
    except Exception as e:
        error_msg = str(e)
        if "openpyxl" in error_msg or "xlrd" in error_msg:
            return "ERROR: Missing Excel dependencies. Install with: pip install openpyxl xlrd"
        return f"Excel processing error: {error_msg}"
def download_single_file_as_zip(backend_api_url, file_id, filename, headers):
    """Download a single file by requesting it as a ZIP and extracting it"""
    try:
        # Use the existing ZIP endpoint but with a single file ID
        zip_download_url = f"{backend_api_url}/api/files/org/files/download"
        
        payload = {"fileIds": [file_id]}
        
        print(f"📦 Downloading {filename} as ZIP from: {zip_download_url}")
        print(f"📋 Payload: {payload}")
        
        zip_response = requests.post(
            zip_download_url,
            json=payload,
            headers=headers,
            timeout=120  # Increased timeout for ZIP downloads
        )
        
        if zip_response.status_code != 200:
            error_text = zip_response.text[:500] if zip_response.text else "No response text"
            print(f"❌ ZIP download failed with status {zip_response.status_code}")
            print(f"Response: {error_text}")
            raise Exception(f"Failed to download ZIP: {zip_response.status_code} - {error_text}")
        
        # Validate ZIP content
        content_type = zip_response.headers.get('Content-Type', '').lower()
        content_length = zip_response.headers.get('Content-Length', 'Unknown')
        
        print(f"📊 ZIP Download Info:")
        print(f"  Content-Type: {content_type}")
        print(f"  Content-Length: {content_length}")
        print(f"  Actual Size: {len(zip_response.content)} bytes")
        
        # Check if we got a ZIP file
        if 'application/zip' not in content_type and 'application/octet-stream' not in content_type:
            # Sometimes the content-type might not be set correctly, check by magic bytes
            if len(zip_response.content) < 4 or not zip_response.content.startswith(b'PK'):
                raise Exception(f"Expected ZIP file, got content-type: {content_type}, content preview: {zip_response.content[:100]}")
        
        # Validate ZIP file by magic bytes
        if len(zip_response.content) < 4:
            raise Exception("Response too short to be a valid ZIP file")
            
        if not zip_response.content.startswith(b'PK'):
            # Try to see if it's an error message
            try:
                error_preview = zip_response.content.decode('utf-8', errors='ignore')[:200]
                if any(keyword in error_preview.lower() for keyword in ['error', 'exception', 'not found', '<html>']):
                    raise Exception(f"Received error response instead of ZIP: {error_preview}")
            except:
                pass
            raise Exception("Response does not have ZIP file signature (PK)")
        
        print(f"✅ Valid ZIP file received ({len(zip_response.content)} bytes)")
        
        # Extract the file from the ZIP
        try:
            with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zip_file:
                # List files in the ZIP
                file_list = zip_file.namelist()
                print(f"📁 ZIP contains {len(file_list)} file(s): {file_list}")
                
                if not file_list:
                    raise Exception("ZIP file is empty")
                
                # Find the target file (by name or take the first one)
                target_file = None
                
                # Strategy 1: Exact filename match
                for zip_filename in file_list:
                    if zip_filename == filename:
                        target_file = zip_filename
                        print(f"✅ Found exact filename match: {target_file}")
                        break
                
                # Strategy 2: Partial filename match (ends with)
                if not target_file:
                    for zip_filename in file_list:
                        if zip_filename.endswith(filename) or filename.endswith(zip_filename):
                            target_file = zip_filename
                            print(f"✅ Found partial filename match: {target_file}")
                            break
                
                # Strategy 3: Same file extension
                if not target_file:
                    file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
                    for zip_filename in file_list:
                        if zip_filename.split('.')[-1].lower() == file_ext:
                            target_file = zip_filename
                            print(f"✅ Found file with same extension: {target_file}")
                            break
                
                # Strategy 4: Take the first file (fallback)
                if not target_file:
                    target_file = file_list[0]
                    print(f"⚠️ No exact match found, using first file: {target_file}")
                
                # Validate the target file in ZIP
                file_info = zip_file.getinfo(target_file)
                print(f"📄 Target file info: {target_file} (size: {file_info.file_size} bytes, compressed: {file_info.compress_size} bytes)")
                
                if file_info.file_size == 0:
                    raise Exception(f"Target file {target_file} is empty in ZIP")
                
                # Extract the file content
                file_content = zip_file.read(target_file)
                
                if len(file_content) == 0:
                    raise Exception(f"Extracted content from {target_file} is empty")
                
                print(f"✅ Successfully extracted {len(file_content)} bytes from ZIP file {target_file}")
                
                # Additional validation for known file types
                file_ext = target_file.split('.')[-1].lower() if '.' in target_file else ''
                
                if file_ext == 'xlsx' and not file_content.startswith(b'PK'):
                    raise Exception(f"Extracted XLSX file doesn't have valid format (should start with PK)")
                elif file_ext == 'pdf' and not file_content.startswith(b'%PDF'):
                    raise Exception(f"Extracted PDF file doesn't have valid format (should start with %PDF)")
                elif file_ext in ['txt', 'csv', 'json']:
                    # Try to decode text files to validate
                    try:
                        decoded = file_content.decode('utf-8', errors='ignore')
                        if len(decoded.strip()) == 0:
                            raise Exception(f"Extracted {file_ext} file appears to be empty or invalid")
                    except Exception as decode_error:
                        print(f"⚠️ Warning: Could not validate {file_ext} file: {decode_error}")
                
                return file_content
                
        except zipfile.BadZipFile as zip_error:
            print(f"❌ Invalid ZIP file: {zip_error}")
            # Try to see what we actually got
            content_preview = zip_response.content[:200]
            try:
                text_preview = content_preview.decode('utf-8', errors='ignore')
                print(f"Content preview: {text_preview}")
            except:
                print(f"Binary content preview: {content_preview}")
            raise Exception(f"Invalid ZIP file format: {zip_error}")
            
    except Exception as e:
        print(f"❌ Error downloading file as ZIP: {str(e)}")
        raise
def extract_csv_text(file_path):
    """Extract text from CSV files with smart delimiter detection"""
    try:
        # Try different separators and encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        separators = [',', ';', '\t', '|']
        
        df = None
        used_encoding = None
        used_separator = None
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep, nrows=1000)
                    if len(df.columns) > 1 and len(df) > 0:  # Valid data
                        used_encoding = encoding
                        used_separator = sep
                        break
                except:
                    continue
            if df is not None and len(df.columns) > 1:
                break
        
        if df is not None and len(df) > 0:
            text = f"CSV File Analysis:\n"
            text += f"Encoding: {used_encoding}, Separator: '{used_separator}'\n"
            text += f"Dimensions: {len(df)} rows × {len(df.columns)} columns\n"
            text += f"Columns: {', '.join(df.columns.astype(str))}\n\n"
            
            # Clean column names (remove extra whitespace)
            df.columns = df.columns.str.strip()
            
            text += "Sample Data:\n"
            text += df.head(15).to_string(index=False) + "\n\n"
            
            # Add summary statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                text += "Numeric Summary:\n"
                text += df[numeric_cols].describe().to_string() + "\n\n"
            
            # Show unique values for categorical columns (first few)
            categorical_cols = df.select_dtypes(include=['object']).columns[:3]
            for col in categorical_cols:
                unique_vals = df[col].value_counts().head(5)
                if len(unique_vals) > 0:
                    text += f"Top values in '{col}':\n{unique_vals.to_string()}\n\n"
            
            return text
        else:
            # Fallback to plain text
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                return f"CSV file (read as text):\n{content[:2000]}..." if len(content) > 2000 else content
                
    except Exception as e:
        # Ultimate fallback
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                return f"CSV parsing failed, read as text: {str(e)}\n\nContent:\n{content[:1000]}..."
        except:
            return f"CSV processing error: {str(e)}"

def extract_json_text(file_path):
    """Extract and format JSON data"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Pretty format JSON for AI understanding
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        
        # Add summary information
        text = f"JSON File Analysis:\n"
        text += f"Structure: {type(data).__name__}\n"
        
        if isinstance(data, dict):
            text += f"Top-level keys: {', '.join(list(data.keys())[:10])}\n"
        elif isinstance(data, list):
            text += f"Array length: {len(data)}\n"
            if len(data) > 0:
                text += f"First item type: {type(data[0]).__name__}\n"
        
        text += f"\nFormatted Content:\n{formatted[:3000]}"
        if len(formatted) > 3000:
            text += "\n... (truncated)"
        
        return text
        
    except json.JSONDecodeError as e:
        # Try to read as regular text if JSON parsing fails
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                return f"JSON parsing failed: {str(e)}\nFile content:\n{content[:1000]}..."
        except:
            return f"JSON processing error: {str(e)}"
    except Exception as e:
        return f"JSON file error: {str(e)}"

def extract_pptx_text(file_path):
    """Extract text from PowerPoint files"""
    try:
        from pptx import Presentation
        prs = Presentation(file_path)
        text = f"PowerPoint Presentation: {len(prs.slides)} slides\n\n"
        
        for i, slide in enumerate(prs.slides):
            text += f"--- Slide {i+1} ---\n"
            
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text += shape.text + "\n"
            
            text += "\n"
        
        return text
    except ImportError:
        return "ERROR: python-pptx not installed. Install with: pip install python-pptx"
    except Exception as e:
        return f"PowerPoint extraction error: {str(e)}"

def extract_xml_text(file_path):
    """Extract text from XML files"""
    try:
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        def extract_text_from_xml(element, level=0):
            result = ""
            indent = "  " * level
            
            # Add element name
            result += f"{indent}<{element.tag}>\n"
            
            # Add text content
            if element.text and element.text.strip():
                result += f"{indent}  {element.text.strip()}\n"
            
            # Process children
            for child in element:
                result += extract_text_from_xml(child, level + 1)
            
            return result
        
        text = f"XML File: Root element '{root.tag}'\n\n"
        text += extract_text_from_xml(root)
        
        return text[:5000] + ("..." if len(text) > 5000 else "")
        
    except Exception as e:
        # Fallback to text reading
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                return f"XML parsing failed: {str(e)}\nContent:\n{content[:1000]}..."
        except:
            return f"XML processing error: {str(e)}"

def extract_text_with_encoding_detection(file_path):
    """Extract text with automatic encoding detection"""
    try:
        # Try to detect encoding
        import chardet
        with open(file_path, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except ImportError:
        # Fallback without chardet
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # Ultimate fallback
        with open(file_path, "rb") as f:
            return f.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Text extraction error: {str(e)}"

def extract_text_from_file(file_path, file_type):
    """Enhanced text extraction supporting many file types - MAIN FUNCTION"""
    try:
        file_size = os.path.getsize(file_path)
        print(f"📄 Extracting from {file_path} (type: {file_type}, size: {file_size} bytes)")
        
        if file_size == 0:
            return "ERROR: File is empty"
        
        # Route to appropriate extraction function
        if file_type == "pdf":
            text = extract_pdf_text(file_path)
        elif file_type == "docx":
            text = extract_docx_text(file_path)
        elif file_type in ["xlsx", "xls"]:
            text = extract_excel_text(file_path, file_type)
        elif file_type == "csv":
            text = extract_csv_text(file_path)
        elif file_type == "json":
            text = extract_json_text(file_path)
        elif file_type == "pptx":
            text = extract_pptx_text(file_path)
        elif file_type == "xml":
            text = extract_xml_text(file_path)
        elif file_type == "txt":
            text = extract_text_with_encoding_detection(file_path)
        elif file_type in ["rtf"]:
            # Basic RTF handling
            content = extract_text_with_encoding_detection(file_path)
            # Strip RTF formatting codes
            text = re.sub(r'\{.*?\}', '', content)
            text = re.sub(r'\\[a-z]+\d*', '', text)
            text = ' '.join(text.split())
        else:
            # Unknown file type - try as text
            text = extract_text_with_encoding_detection(file_path)
            print(f"⚠️ Unknown file type {file_type}, treated as text")
        
        if not text or text.strip() == "":
            print(f"⚠️ No text extracted from {file_path}")
            return ""
        
        # Check for error messages in extracted text
        if text.startswith("ERROR:"):
            print(f"❌ {text}")
            return text
        
        print(f"✅ Successfully extracted {len(text)} characters from {file_path}")
        return text
        
    except Exception as e:
        error_msg = f"❌ Error extracting text from {file_path}: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return f"ERROR: {error_msg}"

# ================================
# FIREBASE AND AI INITIALIZATION
# ================================

# Initialize Firebase/Firestore
try:
    from firebase_admin import credentials, firestore, initialize_app
    from google.cloud import aiplatform
    from google.oauth2 import service_account
    
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "fire.json")
    if os.path.exists(cred_path):
        print(f"✅ Found credentials file at {cred_path}")
        
        firebase_cred = credentials.Certificate(cred_path)
        initialize_app(firebase_cred)
        db = firestore.client()
        FIREBASE_AVAILABLE = True
        print("✅ Successfully initialized Firebase")
        
        try:
            from vertexai import init as vertex_init
            from vertexai.generative_models import GenerativeModel, Part, SafetySetting
            
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "buildnblog-450618")
            gcp_credentials = service_account.Credentials.from_service_account_file(cred_path)
            aiplatform.init(project=project_id, location="us-central1", credentials=gcp_credentials)
            vertex_init(project=project_id, location="us-central1")
            VERTEX_AVAILABLE = True
            print("✅ Successfully initialized Vertex AI")
        except Exception as e:
            print(f"⚠️ Vertex AI initialization failed: {str(e)}")
    else:
        print(f"⚠️ Credentials file not found at {cred_path}. Firebase features will be disabled.")
except Exception as e:
    print(f"⚠️ Firebase initialization failed: {str(e)}")

# Initialize OpenAI client
try:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️ OPENAI_API_KEY environment variable not found. OpenAI embeddings will be disabled.")
    else:
        openai_client = OpenAI(api_key=openai_api_key)
        OPENAI_AVAILABLE = True
        print("✅ Successfully initialized OpenAI client")
except Exception as e:
    print(f"⚠️ OpenAI initialization failed: {str(e)}")

def get_openai_client():
    """Get the OpenAI client, initializing if needed"""
    global openai_client, OPENAI_AVAILABLE
    
    if not OPENAI_AVAILABLE:
        raise ValueError("OpenAI functionality is not available")
        
    if openai_client is None:
        try:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not found")
            
            openai_client = OpenAI(api_key=openai_api_key)
            print("✅ OpenAI client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing OpenAI client: {str(e)}")
            OPENAI_AVAILABLE = False
            raise
            
    return openai_client

# ================================
# UTILITY FUNCTIONS
# ================================

def update_upload_progress(upload_id, status, progress, stage, filename=""):
    """Update upload progress in memory with thread safety"""
    if not upload_id:
        return
        
    progress_data = {
        "upload_id": upload_id,
        "status": status,
        "progress": progress,
        "stage": stage,
        "filename": filename,
        "timestamp": time.time()
    }
    
    with progress_lock:
        upload_progress[upload_id] = progress_data
        print(f"Progress update for {upload_id}: {progress}% - {stage}")
    
    # Clean up completed uploads after a delay
    if status in ['Completed', 'error']:
        def cleanup():
            time.sleep(300)  # Wait 5 minutes before cleanup
            with progress_lock:
                if upload_id in upload_progress:
                    del upload_progress[upload_id]
                    print(f"Cleaned up completed upload: {upload_id}")
        
        cleanup_thread = Thread(target=cleanup)
        cleanup_thread.daemon = False
        cleanup_thread.start()

def update_backend_embedding_status(file_id, org_id, is_from_embedding):
    """Update the isFromEmbedding status in the backend database for both files and links"""
    try:
        BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://localhost:5000")
        
        # Method 1: Try files endpoint first (for file uploads)
        files_update_url = f"{BACKEND_API_URL}/api/files/updateFileSource"
        files_payload = {
            "fileId": file_id,
            "isFromEmbedding": is_from_embedding
        }
        
        try:
            files_response = requests.post(
                files_update_url,
                json=files_payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if files_response.status_code == 200:
                print(f"✅ Backend status updated for fileId: {file_id} -> {is_from_embedding}")
                return True
        except Exception as files_error:
            print(f"⚠️ Files endpoint failed: {files_error}")
        
        # Method 2: Try links endpoint (for website links)
        links_update_url = f"{BACKEND_API_URL}/api/links/{file_id}/embedding"
        links_payload = {
            "isFromEmbedding": is_from_embedding
        }
        
        try:
            links_response = requests.put(
                links_update_url,
                json=links_payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if links_response.status_code == 200:
                print(f"✅ Backend status updated for linkId: {file_id} -> {is_from_embedding}")
                return True
            else:
                print(f"⚠️ Links endpoint failed: {links_response.status_code}")
                print(f"Response: {links_response.text}")
                return False
                
        except Exception as links_error:
            print(f"⚠️ Links endpoint error: {links_error}")
            return False
            
    except Exception as e:
        print(f"⚠️ Error updating backend status: {str(e)}")
        return False

@retry(
    retry=retry_if_exception_type((Exception)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def embed_chunks(chunks, upload_id=None, org_id=None, filename=None):
    """Embed chunks with OpenAI API with retry logic"""
    try:
        client = get_openai_client()
        
        all_embeddings = []
        total = len(chunks)
        
        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_end = min(i + batch_size, len(chunks))
            
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            
            for j, item in enumerate(response.data):
                embedding = item.embedding
                all_embeddings.append(embedding)
            
            if upload_id:
                progress = min(75 + ((batch_end) / total) * 20, 95)
                update_upload_progress(upload_id, "Processing", progress, 
                                      f"Generating embeddings ({batch_end}/{total})")
            
            if i + batch_size < len(chunks):
                time.sleep(0.5)
        
        print(f"✅ Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
    except Exception as e:
        print(f"❌ Error generating embeddings: {str(e)}")
        raise

@retry(
    retry=retry_if_exception_type((Exception)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def embed_query(query):
    """Embed a single query with OpenAI API with retry logic"""
    try:
        client = get_openai_client()
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error embedding query: {str(e)}")
        raise

def parse_and_chunk(file_path, file_ext, chunk_size=50, max_chunks=1000):
    """Parse file content into chunks"""
    try:
        text = extract_text_from_file(file_path, file_ext)
        
        if not text or text.strip() == "":
            print(f"Warning: No text extracted from {file_path}")
            return []
        
        # Check for extraction errors
        if text.startswith("ERROR:"):
            print(f"Extraction error: {text}")
            return []
            
        words = text.split()
        
        if len(words) > max_chunks * chunk_size:
            words = words[:max_chunks * chunk_size]
            print(f"⚠️ File truncated to {max_chunks} chunks to avoid memory issues")
        
        chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]
        print(f"✅ Successfully chunked file into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"❌ Error in parse_and_chunk: {str(e)}")
        return []

def delete_collection(collection_ref, batch_size):
    """Enhanced helper function to delete all documents in a collection with counting"""
    if not FIREBASE_AVAILABLE:
        return 0
        
    deleted_count = 0
    
    try:
        docs = collection_ref.limit(batch_size).stream()
        batch_deleted = 0
        
        for doc in docs:
            doc.reference.delete()
            batch_deleted += 1
            deleted_count += 1
        
        if batch_deleted >= batch_size:
            deleted_count += delete_collection(collection_ref, batch_size)
            
    except Exception as e:
        print(f"❌ Error deleting collection batch: {str(e)}")
    
    return deleted_count

def generate_answer_with_gcp(query, context_chunks):
    """Generate answer using Google's Vertex AI"""
    if not VERTEX_AVAILABLE:
        return "Sorry, the AI generation service is currently unavailable."
        
    try:
        context_text = "\n\n".join(context_chunks)
        
        prompt = f"""
You are an intelligent assistant. Below is a set of information retrieved from various documents.

Context:
{context_text}

Question: {query}

Answer (based ONLY on the above context):
"""

        model = GenerativeModel("gemini-2.0-flash")

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
            stream=False
        )

        return responses.text.strip()
    except Exception as e:
        print(f"❌ Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"

def extract_questions_with_ai_direct(file_path, filename):
    """Extract questions directly from file using Vertex AI multimodal capabilities"""
    if not VERTEX_AVAILABLE:
        print("❌ Vertex AI not available")
        return []
        
    try:
        # Verify file exists and is readable
        if not os.path.exists(file_path):
            print(f"❌ File does not exist: {file_path}")
            return []
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"❌ File is empty: {file_path}")
            return []
            
        print(f"📄 Processing file: {filename} ({file_size} bytes)")
        
        # Get file extension to determine how to handle it
        file_ext = filename.split(".")[-1].lower()
        
        # Customize prompt based on file type
        if file_ext in ["png", "jpg", "jpeg", "gif", "webp"]:
            prompt = """
Analyze the uploaded image/visual content and extract meaningful questions that can be answered based on what you can see in the image.

Please extract the questions that someone might ask about this image. Focus on:
- Visual elements and content
- Text visible in the image (if any)
- Diagrams, charts, or data visualizations
- Objects, people, or scenes depicted
- Any processes or workflows shown
- Information that can be read or interpreted

For each question, provide:
1. A clear, specific question
2. A brief description explaining what the question aims to understand from the visual content

Format your response as a valid JSON array with objects containing "question" and "description" fields, like this:
[
  {{
    "question": "What does this chart show?",
    "description": "Understanding the main data or information presented in the visualization"
  }},
  {{
    "question": "What are the key steps in this process?",
    "description": "Identifying the workflow or procedure illustrated in the image"
  }}
]

IMPORTANT: Return ONLY the JSON array, no additional text or formatting.
""".format(filename=filename)
        else:
            prompt = """
Analyze the uploaded document and extract meaningful questions that can be answered based on the information provided.

Please extract exact questions that are there in document. 
Format your response as a valid JSON array with objects containing "question" and "description" fields, like this:
[
  {{
    "question": "What is the main purpose of this system?",
    "description": "Understanding the primary objective and goals of the system described in the document"
  }},
  {{
    "question": "How does the process work?",
    "description": "Detailed explanation of the workflow and steps involved in the process"
  }}
]

IMPORTANT: Return ONLY the JSON array, no additional text or formatting.
""".format(filename=filename)

        model = GenerativeModel("gemini-2.0-flash")
        
        # Prepare the content based on file type
        parts = [Part.from_text(prompt)]
        
        # Define MIME type mapping with proper Excel support
        mime_mapping = {
            "pdf": "application/pdf",
            "txt": "text/plain", 
            "csv": "text/csv",
            "json": "application/json",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg", 
            "gif": "image/gif",
            "webp": "image/webp",
        }
        
        # Files that Gemini can handle directly (Excel and Word files are NOT included)
        gemini_supported_files = ["pdf", "txt", "csv", "json", "png", "jpg", "jpeg", "gif", "webp"]
        
        if file_ext in gemini_supported_files:
            try:
                # Read file as bytes for direct upload
                with open(file_path, "rb") as f:
                    file_data = f.read()
                
                mime_type = mime_mapping.get(file_ext, "application/octet-stream")
                
                # Validate file size for Gemini (max ~10MB)
                if len(file_data) > 10 * 1024 * 1024:  # 10MB limit
                    print(f"⚠️ File too large for direct upload ({len(file_data)} bytes), falling back to text extraction")
                    if file_ext not in ["png", "jpg", "jpeg", "gif", "webp"]:
                        content = extract_text_from_file(file_path, file_ext)
                        if content and not content.startswith("ERROR:"):
                            if len(content) > 30000:
                                content = content[:30000] + "..."
                            parts.append(Part.from_text(f"\nDocument Content:\n{content}"))
                        else:
                            return []
                    else:
                        return []  # Can't process large image
                else:
                    # Add file part directly
                    file_part = Part.from_data(data=file_data, mime_type=mime_type)
                    parts.append(file_part)
                    
                    print(f"✅ Using direct multimodal upload for {filename} ({mime_type}, {len(file_data)} bytes)")
                
            except Exception as e:
                print(f"⚠️ Direct upload failed for {filename}, falling back to text extraction: {e}")
                # Fallback to text extraction for non-image files
                if file_ext not in ["png", "jpg", "jpeg", "gif", "webp"]:
                    content = extract_text_from_file(file_path, file_ext)
                    if content and not content.startswith("ERROR:"):
                        if len(content) > 30000:
                            content = content[:30000] + "..."
                        parts.append(Part.from_text(f"\nDocument Content:\n{content}"))
                    else:
                        return []
                else:
                    return []  # Can't process image if direct upload fails
        else:
            # For Excel, Word, and other files - extract text first (Gemini doesn't support these directly)
            print(f"📄 Extracting text from {filename} (file type: {file_ext})")
            content = extract_text_from_file(file_path, file_ext)
            if not content or content.startswith("ERROR:"):
                print(f"❌ No content extracted from {filename}: {content}")
                return []
            
            # Limit content length
            if len(content) > 30000:
                content = content[:30000] + "..."
            
            parts.append(Part.from_text(f"\nDocument Content:\n{content}"))
        
        # Generate response
        print(f"🤖 Sending request to Gemini for {filename}")
        response = model.generate_content(
            parts,
            generation_config={
                "max_output_tokens": 4096,
                "temperature": 0.3,
                "top_p": 0.8,
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
            stream=False
        )
        
        print(f"✅ Received response from Gemini for {filename}")
        
        # Parse the JSON response
        response_text = response.text.strip()
        print(f"📝 Raw response length: {len(response_text)} characters")
        
        # Clean up the response - remove any markdown formatting
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
        
        # Try to extract JSON from the response
        try:
            questions = json.loads(response_text)
            
            # Validate the structure
            if isinstance(questions, list):
                valid_questions = []
                for q in questions:
                    if isinstance(q, dict) and "question" in q and "description" in q:
                        valid_questions.append({
                            "question": str(q["question"]).strip(),
                            "description": str(q["description"]).strip()
                        })
                
                print(f"✅ Successfully parsed {len(valid_questions)} questions from {filename}")
                return valid_questions[:15]  # Limit to 15 questions
            else:
                print(f"⚠️ Invalid JSON structure from AI for {filename} (not a list)")
                return []
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error for {filename}: {e}")
            print(f"Raw response preview: {response_text[:500]}...")
            
            # Fallback: try to extract questions manually using regex
            questions = []
            
            # Try to find JSON objects in the text
            json_pattern = r'\{\s*"question":\s*"([^"]+)"\s*,\s*"description":\s*"([^"]+)"\s*\}'
            matches = re.findall(json_pattern, response_text)
            
            for question_text, description_text in matches:
                questions.append({
                    "question": question_text.strip(),
                    "description": description_text.strip()
                })
            
            if questions:
                print(f"✅ Extracted {len(questions)} questions using fallback regex for {filename}")
            else:
                print(f"❌ Could not extract any questions from {filename}")
                
            return questions[:15]
            
    except Exception as e:
        print(f"❌ Error extracting questions from {filename}: {str(e)}")
        traceback.print_exc()
        return []

# ================================
# FLASK ROUTES
# ================================

@app.route("/", methods=["GET"])
def index():
    """Root endpoint - simple health check with file support information"""
    supported, missing = check_file_processing_dependencies()
    
    return jsonify({
        "status": "online",
        "message": "Welcome to CareAI API - Enhanced Universal File Support",
        "version": "3.0.0",
        "features": [
            "Universal file type support",
            "Enhanced Excel/PDF/Word processing", 
            "Automatic encoding detection",
            "Smart CSV parsing",
            "Multimodal question extraction",
            "Real-time progress tracking",
            "Comprehensive error handling"
        ],
        "file_support": {
            "always_supported": supported["always"],
            "currently_available": supported["conditional"],
            "missing_dependencies": missing
        }
    })
# Add this function after postAgentResultsToBackend
def postQuestionsToBackend(rfpId, questions, auth_token):
    """Post extracted questions to the backend questions API"""
    if not rfpId or not questions:
        return

    try:
        backend_api_url = os.environ.get("BACKEND_API_URL", "http://localhost:8080")
        questions_url = f"{backend_api_url}/api/rfps/questions"
        
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        # Format questions for the API
        formatted_questions = []
        for q in questions:
            if isinstance(q, dict) and "question" in q and "description" in q:
                formatted_questions.append({
                    "question": q["question"],
                    "description": q["description"]
                })
        
        if not formatted_questions:
            print("⚠️ No valid questions to post to backend")
            return

        payload = {
            "rfpId": rfpId,
            "questions": formatted_questions
        }

        print(f"📝 Posting {len(formatted_questions)} questions to backend")
        
        response = requests.post(
            questions_url, 
            json=payload, 
            headers=headers, 
            timeout=30
        )

        if response.status_code in [200, 201]:
            print(f"✅ Successfully posted {len(formatted_questions)} questions to backend")
        else:
            print(f"⚠️ Failed to post questions: {response.status_code} - {response.text[:200]}")

    except Exception as e:
        print(f"❌ Error posting questions to backend: {str(e)}")
def postAgentResultsAndQuestions(rfpId, agentResults, auth_token,agent_id=None):
    """Post both agent results and questions to backend"""
    try:
        # Post agent results
        backend_api_url = os.environ.get("BACKEND_API_URL", "http://localhost:8080")
        agent_response_url = f"{backend_api_url}/api/rfps/{rfpId}/agent-response"
        
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"



        response = requests.post(
            agent_response_url,
            json={
                "agentId": agent_id,
                "questions": agentResults.get("questions", []),
            },
            headers=headers,
            timeout=30
        )

        if response.status_code in [200, 201]:
            print("✅ Agent results posted to backend successfully")
            
            # **NEW: Also post individual questions**
            questions = agentResults.get("questions", [])
            if questions and len(questions) > 0:
                postQuestionsToBackend(rfpId, questions, auth_token)
        else:
            print(f"⚠️ Failed to post agent results: {response.status_code} - {response.text[:200]}")

    except Exception as e:
        print(f"❌ Error posting agent results: {str(e)}")


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route("/status", methods=["GET"])
def service_status():
    """Detailed service status endpoint with file support info"""
    with progress_lock:
        active_uploads = len(upload_progress)
    
    supported, missing = check_file_processing_dependencies()
    
    status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "firebase": FIREBASE_AVAILABLE,
            "openai": OPENAI_AVAILABLE,
            "vertex_ai": VERTEX_AVAILABLE
        },
        "active_uploads": active_uploads,
        "version": "3.0.0",
        "file_processing": {
            "supported_types": supported["always"] + supported["conditional"],
            "missing_dependencies": missing,
            "categories": {
                "documents": ["pdf", "docx", "txt", "rtf"],
                "spreadsheets": ["xlsx", "xls", "csv"],
                "data": ["json", "xml"],
                "images": ["png", "jpg", "jpeg", "gif", "webp"],
                "presentations": ["pptx"]
            }
        },
        "capabilities": {
            "multimodal_analysis": VERTEX_AVAILABLE,
            "direct_file_upload": VERTEX_AVAILABLE,
            "image_processing": VERTEX_AVAILABLE,
            "document_embeddings": OPENAI_AVAILABLE and FIREBASE_AVAILABLE,
            "smart_encoding_detection": True,
            "comprehensive_error_handling": True
        }
    }
    
    return jsonify(status), 200

@app.route("/supported-files", methods=["GET"])
def get_supported_file_types():
    """Return detailed information about supported file types"""
    supported, missing = check_file_processing_dependencies()
    
    return jsonify({
        "supported_categories": {
            "documents": {
                "types": ["pdf", "docx", "txt", "rtf"],
                "description": "Text documents and reports"
            },
            "spreadsheets": {
                "types": ["xlsx", "xls", "csv"],
                "description": "Tabular data and calculations"
            },
            "data_formats": {
                "types": ["json", "xml"],
                "description": "Structured data formats"
            },
            "images": {
                "types": ["png", "jpg", "jpeg", "gif", "webp"],
                "description": "Visual content and diagrams"
            },
            "presentations": {
                "types": ["pptx"],
                "description": "Slide presentations"
            }
        },
        "currently_available": supported["always"] + supported["conditional"],
        "missing_dependencies": missing,
        "installation_commands": {
            "essential": "pip install openpyxl xlrd python-docx PyPDF2",
            "extended": "pip install python-pptx xmltodict chardet",
            "complete": "pip install openpyxl xlrd python-docx PyPDF2 python-pptx xmltodict chardet"
        },
        "dependency_status": {
            dep: dep not in [m.split()[0] for m in missing] 
            for dep in ["openpyxl", "xlrd", "python-docx", "PyPDF2", "python-pptx"]
        }
    })

@app.route("/extract-questions", methods=["POST", "OPTIONS"])
def extract_questions_from_files():
    """Extract questions from multiple uploaded files using AI (multimodal)"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not VERTEX_AVAILABLE:
        return jsonify({"error": "AI generation service is not available"}), 503
        
    try:
        org_id = request.args.get("orgId")
        extraction_id = request.args.get("extractionId", str(uuid.uuid4()))
        
        print(f"🤖 Starting multimodal question extraction with ID: {extraction_id} for org: {org_id}")
        
        if not org_id:
            return jsonify({"error": "orgId is required"}), 400

        # Check if files were uploaded and validate
        if not request.files:
            return jsonify({"error": "No files provided"}), 400

        uploaded_files = request.files.getlist('files')  # Get multiple files
        if not uploaded_files:
            return jsonify({"error": "No files found in request"}), 400
        
        # Check for empty files
        valid_files = []
        for file in uploaded_files:
            if file.filename and file.filename.strip():
                valid_files.append(file)
        
        if not valid_files:
            return jsonify({"error": "No valid files found (all files appear to be empty or unnamed)"}), 400

        print(f"📁 Processing {len(valid_files)} valid files")

        # Read all files into memory before starting async processing
        file_data_list = []
        for file in valid_files:
            try:
                file.seek(0)
                file_content = file.read()
                
                if len(file_content) == 0:
                    print(f"⚠️ Skipping empty file: {file.filename}")
                    continue
                    
                file_data_list.append({
                    'filename': file.filename,
                    'content': file_content,
                    'size': len(file_content)
                })
                print(f"✅ Read {file.filename} ({len(file_content)} bytes)")
                
            except Exception as e:
                print(f"❌ Error reading file {file.filename}: {str(e)}")
                return jsonify({"error": f"Failed to read file {file.filename}: {str(e)}"}), 400
            finally:
                try:
                    file.close()
                except:
                    pass
        
        if not file_data_list:
            return jsonify({"error": "No files could be read successfully"}), 400

        # Initialize progress tracking
        update_upload_progress(extraction_id, "Processing", 0, "Starting multimodal question extraction", "")
        
        def process_files_async():
            try:
                results = []
                total_files = len(file_data_list)
                
                # Get supported file types
                supported, missing = check_file_processing_dependencies()
                all_supported = supported["always"] + supported["conditional"] + ["png", "jpg", "jpeg", "gif", "webp"]
                
                for index, file_data in enumerate(file_data_list):
                    filename = file_data['filename']
                    file_content = file_data['content']
                    file_ext = filename.split(".")[-1].lower()
                    
                    # Validate file type
                    if file_ext not in all_supported:
                        results.append({
                            "filename": filename,
                            "status": "error",
                            "error": f"Unsupported file type: {file_ext}. Supported: {', '.join(sorted(all_supported))}",
                            "questions": []
                        })
                        continue
                    
                    try:
                        # Update progress
                        progress = (index / total_files) * 80
                        update_upload_progress(extraction_id, "Processing", progress, 
                                            f"Processing file {index + 1}/{total_files}: {filename}")
                        
                        # Save file content to temporary file
                        save_path = os.path.join(UPLOAD_FOLDER, f"{extraction_id}_{index}_{filename}")
                        with open(save_path, 'wb') as f:
                            f.write(file_content)
                        
                        update_upload_progress(extraction_id, "Processing", progress + 2, 
                                            f"File saved: {filename}")
                        
                        # Use direct file upload to AI
                        update_upload_progress(extraction_id, "Processing", progress + 5, 
                                            f"AI analyzing {filename}")
                        
                        questions = extract_questions_with_ai_direct(save_path, filename)
                        
                        if not questions:
                            results.append({
                                "filename": filename,
                                "status": "error", 
                                "error": "AI could not extract questions from this file",
                                "questions": []
                            })
                        else:
                            results.append({
                                "filename": filename,
                                "status": "success",
                                "file_type": file_ext,
                                "questions": questions,
                                "question_count": len(questions)
                            })
                            
                            print(f"✅ Extracted {len(questions)} questions from {filename}")
                        
                        # Clean up temporary file
                        try:
                            os.remove(save_path)
                        except Exception as e:
                            print(f"⚠️ Failed to remove temp file {save_path}: {e}")
                            
                    except Exception as file_error:
                        print(f"❌ Error processing file {filename}: {str(file_error)}")
                        traceback.print_exc()
                        results.append({
                            "filename": filename,
                            "status": "error",
                            "error": str(file_error),
                            "questions": []
                        })
                
                # Final processing - compile all questions
                update_upload_progress(extraction_id, "Processing", 90, "Compiling results")
                
                # Aggregate results
                all_questions = []
                successful_files = []
                failed_files = []
                
                for result in results:
                    if result["status"] == "success":
                        successful_files.append(result["filename"])
                        all_questions.extend(result["questions"])
                    else:
                        failed_files.append({
                            "filename": result["filename"],
                            "error": result.get("error", "Unknown error")
                        })
                
                # Store results in progress for retrieval
                final_results = {
                    "extraction_id": extraction_id,
                    "org_id": org_id,
                    "total_files": total_files,
                    "successful_files": len(successful_files),
                    "failed_files": len(failed_files),
                    "total_questions": len(all_questions),
                    "files_processed": results,
                    "all_questions": all_questions,
                    "file_details": {
                        "successful": successful_files,
                        "failed": failed_files
                    },
                    "processing_method": "enhanced_multimodal",
                    "supported_types": all_supported,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Update progress with final results
                update_upload_progress(extraction_id, "Completed", 100, "Enhanced question extraction completed", "")
                
                # Store results in progress data for retrieval
                with progress_lock:
                    if extraction_id in upload_progress:
                        upload_progress[extraction_id]["results"] = final_results
                
                print(f"✅ Enhanced question extraction completed. Total questions: {len(all_questions)}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Async processing error: {error_msg}")
                traceback.print_exc()
                update_upload_progress(extraction_id, "error", 0, f"Error: {error_msg}", "")
        
        # Start processing in background thread
        processing_thread = Thread(target=process_files_async)
        processing_thread.daemon = False
        processing_thread.start()
        
        return jsonify({
            "message": "Enhanced question extraction started. Check status with the extraction-status endpoint.",
            "extraction_id": extraction_id,
            "org_id": org_id,
            "files_count": len(file_data_list),
            "processing_method": "enhanced_multimodal",
            "supported_types": ["documents", "images", "spreadsheets", "presentations", "data_files"]
        }), 202

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Question Extraction Error: {error_msg}")
        traceback.print_exc()
        
        return jsonify({"error": "Internal server error", "details": error_msg}), 500

@app.route("/extraction-status", methods=["GET", "OPTIONS"])
def get_extraction_status():
    """Get question extraction status and results"""
    if request.method == "OPTIONS":
        return "", 200
        
    extraction_id = request.args.get("extractionId")
    
    if not extraction_id:
        return jsonify({"error": "extractionId is required"}), 400
    
    with progress_lock:
        exists = extraction_id in upload_progress
        status_data = upload_progress.get(extraction_id, None)
    
    if not exists:
        return jsonify({"exists": False}), 200
    
    response_data = {
        "exists": True,
        "status": status_data
    }
    
    # Include results if extraction is completed
    if status_data and status_data.get("status") == "Completed" and "results" in status_data:
        response_data["results"] = status_data["results"]
    
    return jsonify(response_data), 200

@app.route("/download-questions", methods=["GET", "OPTIONS"])
def download_questions():
    """Download extracted questions as JSON or text file"""
    if request.method == "OPTIONS":
        return "", 200
        
    try:
        extraction_id = request.args.get("extractionId")
        format_type = request.args.get("format", "json")  # json or txt
        
        if not extraction_id:
            return jsonify({"error": "extractionId is required"}), 400
        
        with progress_lock:
            status_data = upload_progress.get(extraction_id, None)
        
        if not status_data or "results" not in status_data:
            return jsonify({"error": "No results found for this extraction ID"}), 404
        
        results = status_data["results"]
        
        if format_type == "txt":
            # Generate text format
            content = f"Enhanced Question Extraction Results\n"
            content += f"=" * 50 + "\n"
            content += f"Extraction ID: {extraction_id}\n"
            content += f"Organization: {results['org_id']}\n"
            content += f"Processing Method: {results.get('processing_method', 'enhanced_multimodal')}\n"
            content += f"Total Files: {results['total_files']}\n"
            content += f"Successful Files: {results['successful_files']}\n"
            content += f"Total Questions: {results['total_questions']}\n"
            content += f"Timestamp: {results['timestamp']}\n\n"
            
            content += "=" * 50 + "\n"
            content += "ALL EXTRACTED QUESTIONS\n"
            content += "=" * 50 + "\n\n"
            
            for i, question_obj in enumerate(results['all_questions'], 1):
                content += f"{i}. {question_obj['question']}\n"
                content += f"   Description: {question_obj['description']}\n\n"
            
            content += "\n" + "=" * 50 + "\n"
            content += "FILE PROCESSING DETAILS\n"
            content += "=" * 50 + "\n\n"
            
            for file_result in results['files_processed']:
                content += f"File: {file_result['filename']}\n"
                content += f"Type: {file_result.get('file_type', 'unknown')}\n"
                content += f"Status: {file_result['status']}\n"
                if file_result['status'] == 'success':
                    content += f"Questions extracted: {file_result['question_count']}\n"
                    for q in file_result['questions']:
                        content += f"  - {q['question']}\n"
                        content += f"    {q['description']}\n"
                else:
                    content += f"Error: {file_result.get('error', 'Unknown error')}\n"
                content += "\n"
            
            return Response(
                content,
                mimetype="text/plain",
                headers={
                    "Content-Disposition": f"attachment; filename=questions_{extraction_id}.txt"
                }
            )
        else:
            # Return JSON format
            return Response(
                json.dumps(results, indent=2),
                mimetype="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=questions_{extraction_id}.json"
                }
            )
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Download Error: {error_msg}")
        return jsonify({"error": "Internal server error", "details": error_msg}), 500

@app.route("/upload", methods=["POST", "OPTIONS"])
def upload_file():
    """Upload and process a file using fileId for tracking"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase is not available"}), 503
        
    if not OPENAI_AVAILABLE:
        return jsonify({"error": "OpenAI embedding service is not available"}), 503
        
    save_path = None
    try:
        org_id = request.args.get("orgId")
        file_id = request.args.get("fileId")
        upload_id = request.args.get("uploadId", str(uuid.uuid4()))
        
        print(f"⏳ Starting enhanced upload with fileId: {file_id}, uploadId: {upload_id} for org: {org_id}")
        
        if not org_id:
            return jsonify({"error": "orgId is required"}), 400
            
        if not file_id:
            return jsonify({"error": "fileId is required"}), 400

        if "file" not in request.files:
            return jsonify({"error": "No file provided."}), 400

        file = request.files["file"]
        filename = file.filename
        file_ext = filename.split(".")[-1].lower()

        # Get currently supported types
        supported, missing = check_file_processing_dependencies()
        all_supported = supported["always"] + supported["conditional"] + ["png", "jpg", "jpeg", "gif", "webp"]
        
        if file_ext not in all_supported:
            return jsonify({
                "error": f"Unsupported file type: {file_ext}. Supported: {', '.join(sorted(all_supported))}",
                "missing_dependencies": missing,
                "install_command": f"pip install {' '.join(missing)}" if missing else None
            }), 400

        # Initialize progress
        update_upload_progress(upload_id, "Processing", 0, "Starting enhanced upload", filename)
        
        # Save file temporarily
        save_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{filename}")
        file.save(save_path)
        print(f"✅ File saved to {save_path}")
        
        # Process file in a separate thread
        def process_file_async():
            try:
                update_upload_progress(upload_id, "Processing", 25, "File saved", filename)
                
                # Extract text and create chunks (for embedding)
                update_upload_progress(upload_id, "Processing", 50, "Extracting text with enhanced processing", filename)
                chunks = parse_and_chunk(save_path, file_ext, chunk_size=50, max_chunks=500)
                
                if not chunks:
                    print(f"❌ No content extracted from {filename}")
                    update_upload_progress(upload_id, "error", 0, "No content extracted", filename)
                    # Update backend to mark as NOT having embeddings
                    update_backend_embedding_status(file_id, org_id, False)
                    return
                    
                # Generate embeddings
                update_upload_progress(upload_id, "Processing", 75, "Generating embeddings", filename)
                embeddings = embed_chunks(chunks, upload_id=upload_id, org_id=org_id, filename=filename)
                
                # Store in Firestore using fileId as document ID
                update_upload_progress(upload_id, "Processing", 90, "Storing embeddings", filename)
                
                file_doc_ref = db.collection("document_embeddings").document(f"org-{org_id}").collection("files").document(file_id)
                
                # Store file metadata
                file_doc_ref.set({
                    "filename": filename,
                    "file_id": file_id,
                    "upload_id": upload_id,
                    "file_type": file_ext,
                    "document_type": "global",
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "chunk_count": len(chunks),
                    "processing_version": "3.0.0"
                })
                
                # Store chunks in batches
                batch_size = 10
                total_chunks = len(chunks)
                
                for i in range(0, total_chunks, batch_size):
                    batch = db.batch()
                    end_idx = min(i + batch_size, total_chunks)
                    
                    for j in range(i, end_idx):
                        chunk_ref = file_doc_ref.collection("chunks").document(str(j))
                        batch.set(chunk_ref, {
                            "content": chunks[j],
                            "embedding": embeddings[j],
                            "index": j
                        })
                    
                    batch.commit()
                    del batch
                    import gc
                    gc.collect()
                    
                    progress = 90 + ((end_idx / total_chunks) * 10)
                    update_upload_progress(upload_id, "Processing", progress, 
                                        f"Storing embeddings ({end_idx}/{total_chunks})", filename)
                
                print(f"✅ Successfully processed file {filename} with fileId: {file_id}")
                update_upload_progress(upload_id, "Completed", 100, "Enhanced file processing completed", filename)
                
                # Update backend to mark file as having embeddings
                update_backend_embedding_status(file_id, org_id, True)
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Async processing error: {error_msg}")
                traceback.print_exc()
                update_upload_progress(upload_id, "error", 0, f"Error: {error_msg}", filename)
                # Update backend to mark as NOT having embeddings on error
                update_backend_embedding_status(file_id, org_id, False)
                
            finally:
                if save_path and os.path.exists(save_path):
                    try:
                        os.remove(save_path)
                        print(f"🧹 File {save_path} deleted.")
                    except Exception as e:
                        print(f"Error deleting file: {e}")
        
        # Start processing in background thread
        processing_thread = Thread(target=process_file_async)
        processing_thread.daemon = False
        processing_thread.start()
        
        return jsonify({
            "message": "Enhanced file upload started. Check status with the upload-status endpoint.",
            "file_id": file_id,
            "upload_id": upload_id,
            "file_type": file_ext,
            "processing_version": "3.0.0"
        }), 202

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Upload Error: {error_msg}")
        traceback.print_exc()
        
        if 'upload_id' in locals() and upload_id:
            update_upload_progress(upload_id, "error", 0, f"Error: {error_msg}", filename if 'filename' in locals() else "")
            
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
                print(f"🧹 File {save_path} deleted after error.")
            except Exception as cleanup_error:
                print(f"Error cleaning up file: {cleanup_error}")
                
        return jsonify({"error": "Internal server error", "details": error_msg}), 500

@app.route("/upload-status", methods=["GET", "OPTIONS"])
def get_upload_status():
    """Get current upload status"""
    if request.method == "OPTIONS":
        return "", 200
        
    upload_id = request.args.get("uploadId")
    
    if not upload_id:
        return jsonify({"error": "uploadId is required"}), 400
    
    with progress_lock:
        exists = upload_id in upload_progress
        status_data = upload_progress.get(upload_id, None)
    
    if not exists and status_data is None:
        time.sleep(0.1)
        with progress_lock:
            exists = upload_id in upload_progress
            status_data = upload_progress.get(upload_id, None)
    
    if not exists:
        return jsonify({"exists": False}), 200
    
    return jsonify({
        "exists": True,
        "status": status_data
    }), 200

@app.route("/update-embedding-status", methods=["POST", "OPTIONS"])
def update_embedding_status():
    """Update the isFromEmbedding flag for a file in the main database"""
    if request.method == "OPTIONS":
        return "", 200
        
    try:
        data = request.get_json(silent=True) or {}
        file_id = data.get("fileId")
        org_id = data.get("orgId")
        is_from_embedding = data.get("isFromEmbedding")
        
        if not file_id:
            return jsonify({"error": "fileId is required"}), 400
            
        if not org_id:
            return jsonify({"error": "orgId is required"}), 400
            
        if not isinstance(is_from_embedding, bool):
            return jsonify({"error": "isFromEmbedding must be a boolean"}), 400
        
        print(f"🔄 Manual status update for fileId: {file_id}, isFromEmbedding: {is_from_embedding}")
        
        # Update backend status
        success = update_backend_embedding_status(file_id, org_id, is_from_embedding)
        
        if success:
            return jsonify({
                "message": "Embedding status updated successfully",
                "fileId": file_id,
                "isFromEmbedding": is_from_embedding
            }), 200
        else:
            return jsonify({
                "error": "Failed to update backend",
                "fileId": file_id
            }), 500
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Update Status Error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": error_msg}), 500

@app.route("/reprocess-file", methods=["POST", "OPTIONS"])
def reprocess_file():
    """Reprocess a file to create embeddings"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase is not available"}), 503
        
    if not OPENAI_AVAILABLE:
        return jsonify({"error": "OpenAI embedding service is not available"}), 503
        
    try:
        data = request.get_json(silent=True) or {}
        file_id = data.get("fileId")
        org_id = data.get("orgId")
        file_url = data.get("fileUrl")  # URL to download the file
        filename = data.get("filename")
        
        if not all([file_id, org_id, file_url, filename]):
            return jsonify({"error": "fileId, orgId, fileUrl, and filename are required"}), 400
        
        print(f"🔄 Reprocessing file with enhanced processing: {filename} (fileId: {file_id})")
        
        # Generate a new upload ID for tracking
        upload_id = str(uuid.uuid4())
        
        def reprocess_async():
            save_path = None
            try:
                update_upload_progress(upload_id, "Processing", 10, "Downloading file", filename)
                
                # Download file from URL
                file_response = requests.get(file_url, timeout=30)
                if file_response.status_code != 200:
                    raise Exception(f"Failed to download file: {file_response.status_code}")
                
                # Save file temporarily
                file_ext = filename.split(".")[-1].lower()
                save_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{filename}")
                
                with open(save_path, 'wb') as f:
                    f.write(file_response.content)
                
                update_upload_progress(upload_id, "Processing", 30, "File downloaded", filename)
                
                # Extract text and create chunks
                update_upload_progress(upload_id, "Processing", 50, "Enhanced text extraction", filename)
                chunks = parse_and_chunk(save_path, file_ext, chunk_size=50, max_chunks=500)
                
                if not chunks:
                    print(f"❌ No content extracted from {filename}")
                    update_upload_progress(upload_id, "error", 0, "No content extracted", filename)
                    update_backend_embedding_status(file_id, org_id, False)
                    return
                
                # Generate embeddings
                update_upload_progress(upload_id, "Processing", 70, "Generating embeddings", filename)
                embeddings = embed_chunks(chunks, upload_id=upload_id, org_id=org_id, filename=filename)
                
                # Store in Firestore
                update_upload_progress(upload_id, "Processing", 90, "Storing embeddings", filename)
                
                file_doc_ref = db.collection("document_embeddings").document(f"org-{org_id}").collection("files").document(file_id)
                
                # Store file metadata
                file_doc_ref.set({
                    "filename": filename,
                    "file_id": file_id,
                    "upload_id": upload_id,
                    "file_type": file_ext,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "chunk_count": len(chunks),
                    "reprocessed": True,
                    "processing_version": "3.0.0"
                })
                
                # Store chunks in batches
                batch_size = 10
                total_chunks = len(chunks)
                
                for i in range(0, total_chunks, batch_size):
                    batch = db.batch()
                    end_idx = min(i + batch_size, total_chunks)
                    
                    for j in range(i, end_idx):
                        chunk_ref = file_doc_ref.collection("chunks").document(str(j))
                        batch.set(chunk_ref, {
                            "content": chunks[j],
                            "embedding": embeddings[j],
                            "index": j
                        })
                    
                    batch.commit()
                    del batch
                    import gc
                    gc.collect()
                
                print(f"✅ Successfully reprocessed file {filename} with fileId: {file_id}")
                update_upload_progress(upload_id, "Completed", 100, "Enhanced reprocessing completed", filename)
                
                # Update backend status
                update_backend_embedding_status(file_id, org_id, True)
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Reprocessing error: {error_msg}")
                update_upload_progress(upload_id, "error", 0, f"Error: {error_msg}", filename)
                update_backend_embedding_status(file_id, org_id, False)
                
            finally:
                if save_path and os.path.exists(save_path):
                    try:
                        os.remove(save_path)
                    except Exception as e:
                        print(f"Error deleting file: {e}")
        
        # Start reprocessing in background
        Thread(target=reprocess_async, daemon=False).start()
        
        return jsonify({
            "message": "Enhanced file reprocessing started",
            "fileId": file_id,
            "upload_id": upload_id,
            "processing_version": "3.0.0"
        }), 202
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Reprocess Error: {error_msg}")
        return jsonify({"error": "Internal server error", "details": error_msg}), 500

@app.route("/delete", methods=["DELETE", "OPTIONS"])
def delete_file_by_file_id():
    """Delete a file and its embeddings using fileId (primary method)"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase is not available"}), 503
        
    try:
        org_id = request.args.get("orgId")
        file_id = request.args.get("fileId")
        filename = request.args.get("filename")
        upload_id = request.args.get("uploadId")
        rfp_id = request.args.get("rfpId")
        if not org_id:
            return jsonify({"error": "orgId is required"}), 400
            
        if not file_id and not filename and not upload_id:
            return jsonify({"error": "fileId, uploadId, or filename is required"}), 400

        print(f"🗑️ Attempting to delete embeddings for org: {org_id}, fileId: {file_id}")

        files_ref = db.collection("document_embeddings").document(f"org-{org_id}").collection("files")
        
        deleted_count = 0
        deletion_details = []

        # Method 1: Direct lookup by fileId (most reliable)
        if file_id:
            try:
                file_doc_ref = files_ref.document(file_id)
                file_doc = file_doc_ref.get()
                
                if file_doc.exists:
                    file_data = file_doc.to_dict()
                    print(f"✅ Found file document by fileId: {file_id}")
                    
                    # Delete all chunks in the file
                    chunks_deleted = delete_collection(file_doc_ref.collection("chunks"), 100)
                    
                    # Delete the file document
                    file_doc_ref.delete()
                    deleted_count += 1
                    
                    deletion_details.append({
                        "method": "fileId_direct",
                        "fileId": file_id,
                        "filename": file_data.get("filename", "unknown"),
                        "file_type": file_data.get("file_type", "unknown"),
                        "chunks_deleted": chunks_deleted,
                        "processing_version": file_data.get("processing_version", "legacy"),
                        "success": True
                    })
                    
                    print(f"✅ Successfully deleted file and {chunks_deleted} chunks by fileId: {file_id}")
                    
                    # Update backend status
                    update_backend_embedding_status(file_id, org_id, False)
                    
                else:
                    print(f"⚠️ No file found with fileId: {file_id}")
                    
            except Exception as e:
                print(f"❌ Error deleting by fileId {file_id}: {str(e)}")
                deletion_details.append({
                    "method": "fileId_direct",
                    "fileId": file_id,
                    "success": False,
                    "error": str(e)
                })

        # Response with detailed information
        if deleted_count > 0:
            return jsonify({
                "message": f"Successfully deleted {deleted_count} file(s) and their embeddings",
                "deleted_count": deleted_count,
                "deletion_details": deletion_details,
                "org_id": org_id
            }), 200
        else:
            return jsonify({
                "message": "No matching files found to delete",
                "deleted_count": 0,
                "deletion_details": deletion_details,
                "org_id": org_id,
                "warning": "File may have already been deleted or never existed"
            }), 404

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Deletion Error: {error_msg}")
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error", 
            "details": error_msg
        }), 500

@app.route("/cleanup-orphaned", methods=["POST", "OPTIONS"])
def cleanup_orphaned_embeddings():
    """Clean up orphaned embeddings using fileIds from database"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase is not available"}), 503
        
    try:
        data = request.get_json(silent=True) or {}
        org_id = data.get("orgId")
        active_file_ids = data.get("activeFileIds", [])
        
        if not org_id:
            return jsonify({"error": "orgId is required"}), 400
            
        print(f"🧹 Starting enhanced cleanup for org: {org_id}")
        print(f"📋 Active fileIds to preserve: {len(active_file_ids)}")
        
        files_ref = db.collection("document_embeddings").document(f"org-{org_id}").collection("files")
        all_firestore_files = files_ref.stream()
        
        orphaned_files = []
        preserved_files = []
        
        for file_doc in all_firestore_files:
            doc_id = file_doc.id
            file_data = file_doc.to_dict()
            
            stored_file_id = file_data.get("file_id")
            
            is_active = (
                doc_id in active_file_ids or 
                stored_file_id in active_file_ids
            )
            
            if not is_active:
                print(f"🗑️ Found orphaned file: {doc_id} ({file_data.get('filename', 'unknown')})")
                
                chunks_deleted = delete_collection(file_doc.reference.collection("chunks"), 100)
                file_doc.reference.delete()
                
                orphaned_files.append({
                    "document_id": doc_id,
                    "file_id": stored_file_id,
                    "filename": file_data.get("filename", "unknown"),
                    "file_type": file_data.get("file_type", "unknown"),
                    "processing_version": file_data.get("processing_version", "legacy"),
                    "chunks_deleted": chunks_deleted
                })
                
                # Update backend status
                if stored_file_id:
                    update_backend_embedding_status(stored_file_id, org_id, False)
                    
            else:
                preserved_files.append({
                    "document_id": doc_id,
                    "file_id": stored_file_id,
                    "filename": file_data.get("filename", "unknown"),
                    "processing_version": file_data.get("processing_version", "legacy")
                })
        
        print(f"✅ Enhanced cleanup complete. Deleted {len(orphaned_files)} orphaned files, preserved {len(preserved_files)} active files")
        
        return jsonify({
            "message": f"Enhanced cleanup complete for org {org_id}",
            "orphaned_files_deleted": len(orphaned_files),
            "active_files_preserved": len(preserved_files),
            "deletion_details": orphaned_files,
            "cleanup_version": "3.0.0"
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Cleanup Error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": error_msg}), 500

@app.route("/files", methods=["GET", "OPTIONS"])
def list_files():
    """List all files for an organization with enhanced fileId information"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase is not available"}), 503
        
    try:
        org_id = request.args.get("orgId")
        
        if not org_id:
            return jsonify({"error": "orgId is required"}), 400
        
        files_ref = db.collection("document_embeddings").document(f"org-{org_id}").collection("files")
        files_docs = files_ref.stream()
        
        files = []
        for doc in files_docs:
            file_data = doc.to_dict()
            files.append({
                "document_id": doc.id,
                "file_id": file_data.get("file_id"),
                "upload_id": file_data.get("upload_id"),
                "filename": file_data.get("filename"),
                "file_type": file_data.get("file_type"),
                "created_at": file_data.get("created_at"),
                "chunk_count": file_data.get("chunk_count"),
                "reprocessed": file_data.get("reprocessed", False),
                "processing_version": file_data.get("processing_version", "legacy")
            })
        
        return jsonify({
            "org_id": org_id,
            "total_files": len(files),
            "files": files,
            "api_version": "3.0.0"
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ List Files Error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": error_msg}), 500

@app.route("/migrate-to-fileid", methods=["POST", "OPTIONS"])
def migrate_to_file_id():
    """Migrate existing uploadId-based documents to fileId-based system"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase is not available"}), 503
        
    try:
        data = request.get_json(silent=True) or {}
        org_id = data.get("orgId")
        file_mappings = data.get("fileMappings", [])
        
        if not org_id:
            return jsonify({"error": "orgId is required"}), 400
            
        print(f"🔄 Starting enhanced migration for org: {org_id}")
        
        files_ref = db.collection("document_embeddings").document(f"org-{org_id}").collection("files")
        migration_results = []
        
        for mapping in file_mappings:
            upload_id = mapping.get("uploadId")
            file_id = mapping.get("fileId")
            
            if not upload_id or not file_id:
                continue
                
            try:
                old_doc_ref = files_ref.document(upload_id)
                old_doc = old_doc_ref.get()
                
                if old_doc.exists:
                    old_data = old_doc.to_dict()
                    
                    new_doc_ref = files_ref.document(file_id)
                    
                    new_data = old_data.copy()
                    new_data["file_id"] = file_id
                    new_data["processing_version"] = "3.0.0"
                    new_data["migrated"] = True
                    
                    new_doc_ref.set(new_data)
                    
                    old_chunks = old_doc_ref.collection("chunks").stream()
                    batch = db.batch()
                    chunks_copied = 0
                    
                    for chunk_doc in old_chunks:
                        new_chunk_ref = new_doc_ref.collection("chunks").document(chunk_doc.id)
                        batch.set(new_chunk_ref, chunk_doc.to_dict())
                        chunks_copied += 1
                    
                    batch.commit()
                    
                    delete_collection(old_doc_ref.collection("chunks"), 100)
                    old_doc_ref.delete()
                    
                    # Update backend status
                    update_backend_embedding_status(file_id, org_id, True)
                    
                    migration_results.append({
                        "upload_id": upload_id,
                        "file_id": file_id,
                        "filename": old_data.get("filename"),
                        "chunks_migrated": chunks_copied,
                        "processing_version": "3.0.0",
                        "success": True
                    })
                    
                    print(f"✅ Migrated {upload_id} -> {file_id} ({chunks_copied} chunks)")
                    
            except Exception as e:
                migration_results.append({
                    "upload_id": upload_id,
                    "file_id": file_id,
                    "success": False,
                    "error": str(e)
                })
                print(f"❌ Failed to migrate {upload_id} -> {file_id}: {str(e)}")
        
        successful_migrations = len([r for r in migration_results if r["success"]])
        
        return jsonify({
            "message": f"Enhanced migration complete for org {org_id}",
            "total_attempted": len(file_mappings),
            "successful_migrations": successful_migrations,
            "migration_details": migration_results,
            "migration_version": "3.0.0"
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Migration Error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": error_msg}), 500

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat_with_doc():
    """Chat with documents using embeddings and AI"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase is not available"}), 503
        
    if not OPENAI_AVAILABLE:
        return jsonify({"error": "OpenAI embedding service is not available"}), 503
        
    if not VERTEX_AVAILABLE:
        return jsonify({"error": "AI generation service is not available"}), 503
        
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No valid JSON input found"}), 400

        query = data.get("query")
        org_id = data.get("orgId")
        file_ids = data.get("fileIds", [])  # Optional: search only specific files
        
        if not query or not org_id:
            return jsonify({"error": "Query and orgId are required."}), 400

        print(f"🔍 Processing enhanced chat query: '{query}' for org: {org_id}")

        # Get query embedding
        query_embedding = np.array(embed_query(query))

        # Get files for this organization
        org_files_ref = db.collection("document_embeddings").document(f"org-{org_id}").collection("files")
        
        if file_ids:
            # Search only specific files
            files = []
            for file_id in file_ids:
                doc = org_files_ref.document(file_id).get()
                if doc.exists:
                    files.append(doc)
        else:
            # Search all files
            files = org_files_ref.stream()
        
        retrieved_docs = []
        
        # Process each file
        for file_doc in files:
            file_data = file_doc.to_dict()
            
            # Get chunks for this file
            chunks_ref = file_doc.reference.collection("chunks")
            chunks = chunks_ref.stream()
            
            # Process each chunk
            for chunk_doc in chunks:
                chunk_data = chunk_doc.to_dict()
                
                # Convert to numpy array
                chunk_embedding = np.array(chunk_data["embedding"])
                
                # Calculate cosine similarity
                score = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                
                if score >= 0.2:  # Similarity threshold
                    retrieved_docs.append({
                        "content": chunk_data["content"], 
                        "score": float(score),
                        "filename": file_data.get("filename", "Unknown"),
                        "file_id": file_data.get("file_id", file_doc.id),
                        "file_type": file_data.get("file_type", "unknown"),
                        "processing_version": file_data.get("processing_version", "legacy")
                    })

        # Get top chunks by similarity
        top_chunks = sorted(retrieved_docs, key=lambda x: x["score"], reverse=True)[:5]

        if not top_chunks:
            # No relevant chunks found
            return jsonify({
                "query": query, 
                "retrieved_chunks": [], 
                "answer": "I couldn't find any relevant information in the uploaded documents to answer your question. Please make sure you have uploaded documents that contain information related to your query.",
                "source_files": [],
                "relevance_scores": [],
                "chat_type": "global"
            }), 200

        # Generate answer
        context_chunks = [doc["content"] for doc in top_chunks]
        answer = generate_answer_with_gcp(query, context_chunks)
        
        # Get unique source files with types
        source_files = []
        seen_files = set()
        for doc in top_chunks:
            file_key = f"{doc['filename']}_{doc['file_type']}"
            if file_key not in seen_files:
                source_files.append({
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_id": doc["file_id"],
                    "processing_version": doc["processing_version"]
                })
                seen_files.add(file_key)

        return jsonify({
            "query": query, 
            "retrieved_chunks": context_chunks, 
            "answer": answer,
            "source_files": source_files,
            "relevance_scores": [doc["score"] for doc in top_chunks],
            "chat_type": "global",
            "api_version": "3.0.0"
        }), 200

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Chat Error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": error_msg}), 500
# ================================
# RFP SUPPORT DOCUMENT ENDPOINTS
# ================================

@app.route("/upload-support-document", methods=["POST", "OPTIONS"])
def upload_support_document():
    """Upload and process a support document for a specific RFP"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase is not available"}), 503
        
    if not OPENAI_AVAILABLE:
        return jsonify({"error": "OpenAI embedding service is not available"}), 503
        
    save_path = None
    try:
        org_id = request.args.get("orgId")
        rfp_id = request.args.get("rfpId")
        file_id = request.args.get("fileId")
        upload_id = request.args.get("uploadId", str(uuid.uuid4()))
        
        print(f"📄 Starting RFP support document upload - RFP: {rfp_id}, fileId: {file_id}, uploadId: {upload_id}")
        
        if not org_id:
            return jsonify({"error": "orgId is required"}), 400
            
        if not rfp_id:
            return jsonify({"error": "rfpId is required"}), 400
            
        if not file_id:
            return jsonify({"error": "fileId is required"}), 400

        if "file" not in request.files:
            return jsonify({"error": "No file provided."}), 400

        file = request.files["file"]
        filename = file.filename
        file_ext = filename.split(".")[-1].lower()

        # Get currently supported types
        supported, missing = check_file_processing_dependencies()
        all_supported = supported["always"] + supported["conditional"] + ["png", "jpg", "jpeg", "gif", "webp"]
        
        if file_ext not in all_supported:
            return jsonify({
                "error": f"Unsupported file type: {file_ext}. Supported: {', '.join(sorted(all_supported))}",
                "missing_dependencies": missing,
                "install_command": f"pip install {' '.join(missing)}" if missing else None
            }), 400

        # Initialize progress
        update_upload_progress(upload_id, "Processing", 0, "Starting RFP support document upload", filename)
        
        # Save file temporarily
        save_path = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{filename}")
        file.save(save_path)
        print(f"✅ File saved to {save_path}")
        
        # Process file in a separate thread
        def process_support_file_async():
            try:
                update_upload_progress(upload_id, "Processing", 25, "File saved, extracting content", filename)
                
                # Extract text and create chunks
                chunks = parse_and_chunk(save_path, file_ext, chunk_size=50, max_chunks=500)
                
                if not chunks:
                    print(f"❌ No content extracted from {filename}")
                    update_upload_progress(upload_id, "error", 0, "No content extracted", filename)
                    update_backend_embedding_status(file_id, org_id, False)
                    return
                    
                # Generate embeddings
                update_upload_progress(upload_id, "Processing", 50, "Generating embeddings", filename)
                embeddings = embed_chunks(chunks, upload_id=upload_id, org_id=org_id, filename=filename)
                
                # Store in RFP-specific Firestore collection
                update_upload_progress(upload_id, "Processing", 80, "Storing RFP support document embeddings", filename)
                
                # Store in separate RFP support documents collection
                rfp_doc_ref = db.collection("rfp_support_embeddings").document(f"rfp-{rfp_id}").collection("files").document(file_id)
                
                # Store file metadata
                rfp_doc_ref.set({
                    "filename": filename,
                    "file_id": file_id,
                    "upload_id": upload_id,
                    "rfp_id": rfp_id,
                    "org_id": org_id,
                    "file_type": file_ext,
                    "document_type": "support",
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "chunk_count": len(chunks),
                    "processing_version": "4.0.0"
                })
                
                # Store chunks in batches
                batch_size = 10
                total_chunks = len(chunks)
                
                for i in range(0, total_chunks, batch_size):
                    batch = db.batch()
                    end_idx = min(i + batch_size, total_chunks)
                    
                    for j in range(i, end_idx):
                        chunk_ref = rfp_doc_ref.collection("chunks").document(str(j))
                        batch.set(chunk_ref, {
                            "content": chunks[j],
                            "embedding": embeddings[j],
                            "index": j
                        })
                    
                    batch.commit()
                    del batch
                    import gc
                    gc.collect()
                    
                    progress = 80 + ((end_idx / total_chunks) * 20)
                    update_upload_progress(upload_id, "Processing", progress, 
                                        f"Storing chunks ({end_idx}/{total_chunks})", filename)
                
                print(f"✅ Successfully processed RFP support document {filename} for RFP: {rfp_id}")
                update_upload_progress(upload_id, "Completed", 100, "RFP support document processing completed", filename)
                
                # Update backend to mark file as having embeddings
                update_backend_embedding_status(file_id, org_id, True)
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Async processing error: {error_msg}")
                traceback.print_exc()
                update_upload_progress(upload_id, "error", 0, f"Error: {error_msg}", filename)
                update_backend_embedding_status(file_id, org_id, False)
                
            finally:
                if save_path and os.path.exists(save_path):
                    try:
                        os.remove(save_path)
                        print(f"🧹 File {save_path} deleted.")
                    except Exception as e:
                        print(f"Error deleting file: {e}")
        
        # Start processing in background thread
        processing_thread = Thread(target=process_support_file_async)
        processing_thread.daemon = False
        processing_thread.start()
        
        return jsonify({
            "message": "RFP support document upload started. Check status with the upload-status endpoint.",
            "file_id": file_id,
            "upload_id": upload_id,
            "rfp_id": rfp_id,
            "file_type": file_ext,
            "document_type": "support",
            "processing_version": "4.0.0"
        }), 202

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Support Document Upload Error: {error_msg}")
        traceback.print_exc()
        
        if 'upload_id' in locals() and upload_id:
            update_upload_progress(upload_id, "error", 0, f"Error: {error_msg}", filename if 'filename' in locals() else "")
            
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
                print(f"🧹 File {save_path} deleted after error.")
            except Exception as cleanup_error:
                print(f"Error cleaning up file: {cleanup_error}")
                
        return jsonify({"error": "Internal server error", "details": error_msg}), 500
    
@app.route("/check-rfp-support-documents", methods=["GET", "OPTIONS"])
def check_rfp_support_documents():
    """Check if an RFP has support documents"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase is not available"}), 503
        
    try:
        rfp_id = request.args.get("rfpId")
        
        if not rfp_id:
            return jsonify({"error": "rfpId is required"}), 400
        
        # Check if RFP support documents exist
        rfp_files_ref = db.collection("rfp_support_embeddings").document(f"rfp-{rfp_id}").collection("files")
        files = list(rfp_files_ref.limit(1).stream())
        
        has_support_documents = len(files) > 0
        
        if has_support_documents:
            # Get all files for this RFP
            all_files = rfp_files_ref.stream()
            file_list = []
            
            for file_doc in all_files:
                file_data = file_doc.to_dict()
                file_list.append({
                    "file_id": file_data.get("file_id"),
                    "filename": file_data.get("filename"),
                    "file_type": file_data.get("file_type"),
                    "created_at": file_data.get("created_at"),
                    "chunk_count": file_data.get("chunk_count", 0)
                })
        else:
            file_list = []
        
        return jsonify({
            "rfp_id": rfp_id,
            "has_support_documents": has_support_documents,
            "document_count": len(file_list),
            "files": file_list
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Check RFP Support Documents Error: {error_msg}")
        return jsonify({"error": "Internal server error", "details": error_msg}), 500
@app.route("/chat-rfp", methods=["POST", "OPTIONS"])
def chat_with_rfp_documents():
    """Chat with RFP-specific support documents"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase is not available"}), 503
        
    if not OPENAI_AVAILABLE:
        return jsonify({"error": "OpenAI embedding service is not available"}), 503
        
    if not VERTEX_AVAILABLE:
        return jsonify({"error": "AI generation service is not available"}), 503
        
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No valid JSON input found"}), 400

        query = data.get("query")
        rfp_id = data.get("rfpId")
        
        if not query or not rfp_id:
            return jsonify({"error": "Query and rfpId are required."}), 400

        print(f"🤖 Processing RFP chat query: '{query}' for RFP: {rfp_id}")

        # Get query embedding
        query_embedding = np.array(embed_query(query))

        # Get RFP support documents
        rfp_files_ref = db.collection("rfp_support_embeddings").document(f"rfp-{rfp_id}").collection("files")
        files = rfp_files_ref.stream()
        
        retrieved_docs = []
        
        # Process each file
        for file_doc in files:
            file_data = file_doc.to_dict()
            
            # Get chunks for this file
            chunks_ref = file_doc.reference.collection("chunks")
            chunks = chunks_ref.stream()
            
            # Process each chunk
            for chunk_doc in chunks:
                chunk_data = chunk_doc.to_dict()
                
                # Convert to numpy array
                chunk_embedding = np.array(chunk_data["embedding"])
                
                # Calculate cosine similarity
                score = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                
                if score >= 0.2:  # Similarity threshold
                    retrieved_docs.append({
                        "content": chunk_data["content"], 
                        "score": float(score),
                        "filename": file_data.get("filename", "Unknown"),
                        "file_id": file_data.get("file_id", file_doc.id),
                        "file_type": file_data.get("file_type", "unknown"),
                        "document_type": "support"
                    })

        # Get top chunks by similarity
        top_chunks = sorted(retrieved_docs, key=lambda x: x["score"], reverse=True)[:5]

        if not top_chunks:
            return jsonify({
                "query": query, 
                "rfp_id": rfp_id,
                "retrieved_chunks": [], 
                "answer": "No relevant information found in the RFP support documents.",
                "source_files": [],
                "chat_type": "rfp_support"
            }), 200

        # Generate answer
        context_chunks = [doc["content"] for doc in top_chunks]
        answer = generate_answer_with_gcp(query, context_chunks)
        
        # Get unique source files
        source_files = []
        seen_files = set()
        for doc in top_chunks:
            file_key = f"{doc['filename']}_{doc['file_type']}"
            if file_key not in seen_files:
                source_files.append({
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_id": doc["file_id"],
                    "document_type": doc["document_type"]
                })
                seen_files.add(file_key)

        return jsonify({
            "query": query, 
            "rfp_id": rfp_id,
            "retrieved_chunks": context_chunks, 
            "answer": answer,
            "source_files": source_files,
            "relevance_scores": [doc["score"] for doc in top_chunks],
            "chat_type": "rfp_support"
        }), 200

    except Exception as e:
        error_msg = str(e)
        print(f"❌ RFP Chat Error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": error_msg}), 500
@app.route("/delete-rfp-support-document", methods=["DELETE", "OPTIONS"])
def delete_rfp_support_document():
    """Delete a specific RFP support document"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not FIREBASE_AVAILABLE:
        return jsonify({"error": "Firebase is not available"}), 503
        
    try:
        rfp_id = request.args.get("rfpId")
        file_id = request.args.get("fileId")

        if not rfp_id or not file_id:
            return jsonify({"error": "rfpId and fileId are required"}), 400

        print(f"🗑️ Deleting RFP support document - RFP: {rfp_id}, fileId: {file_id}")

        rfp_file_ref = db.collection("rfp_support_embeddings").document(f"rfp-{rfp_id}").collection("files").document(file_id)
        file_doc = rfp_file_ref.get()
        
        if file_doc.exists:
            file_data = file_doc.to_dict()
            
            # Delete all chunks
            chunks_deleted = delete_collection(rfp_file_ref.collection("chunks"), 100)
            
            # Delete the file document
            rfp_file_ref.delete()
            
            print(f"✅ Successfully deleted RFP support document and {chunks_deleted} chunks")
            
            # Update backend status
            update_backend_embedding_status(file_id, file_data.get("org_id"), False)
            
            return jsonify({
                "message": "Successfully deleted RFP support document",
                "rfp_id": rfp_id,
                "file_id": file_id,
                "filename": file_data.get("filename", "unknown"),
                "chunks_deleted": chunks_deleted
            }), 200
        else:
            return jsonify({
                "message": "RFP support document not found",
                "rfp_id": rfp_id,
                "file_id": file_id
            }), 404

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Delete RFP Support Document Error: {error_msg}")
        return jsonify({"error": "Internal server error", "details": error_msg}), 500
    
# Add this new endpoint to your Python Flask application

@app.route("/run-question-agent", methods=["POST", "OPTIONS"])
def run_question_agent():
    """Run Question Agent - Fetch project files and generate questions"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not VERTEX_AVAILABLE:
        return jsonify({"error": "AI generation service is not available"}), 503
        
    try:
        data = request.get_json(silent=True) or {}
        rfp_id = data.get("rfpId")
        org_id = data.get("orgId")
        agent_run_id = data.get("agentRunId", str(uuid.uuid4()))
        auth_token = data.get("authToken") 
        
        print(f"🤖 Starting Question Agent for RFP: {rfp_id}, Agent Run ID: {agent_run_id}")
        
        if not rfp_id:
            return jsonify({"error": "rfpId is required"}), 400
            
        if not org_id:
            return jsonify({"error": "orgId is required"}), 400

        # Initialize progress tracking for agent
        update_upload_progress(agent_run_id, "Processing", 0, "Starting Question Agent", "")
        
        def run_question_agent_async():
            try:
                # Step 1: Fetch project files from backend
                update_upload_progress(agent_run_id, "Processing", 10, "Fetching project files", "")
                
                backend_api_url = os.environ.get("BACKEND_API_URL", "http://localhost:8080")
                project_files_url = f"{backend_api_url}/api/rfps/project-files/{rfp_id}"
                headers = {"Content-Type": "application/json"}
                if auth_token:
                    headers["Authorization"] = f"Bearer {auth_token}"
                
                try:
                    print(f"🔍 Fetching project files from: {project_files_url}")
                    files_response = requests.get(project_files_url, headers=headers, timeout=30)
                    
                    if files_response.status_code != 200:
                        error_text = files_response.text[:300] if files_response.text else "No response"
                        print(f"❌ Failed to fetch project files: {files_response.status_code}")
                        print(f"Response: {error_text}")
                        raise Exception(f"Failed to fetch project files: {files_response.status_code} - {error_text}")
                    
                    files_data = files_response.json()
                    project_files = files_data.get("files", [])
                    
                    if not project_files:
                        print("⚠️ No project files found")
                        update_upload_progress(agent_run_id, "Completed", 100, "No project files found to process", "")
                        
                        # Store empty results
                        with progress_lock:
                            if agent_run_id in upload_progress:
                                upload_progress[agent_run_id]["agent_results"] = {
                                    "agent_run_id": agent_run_id,
                                    "agent_type": "question_generator",
                                    "rfp_id": rfp_id,
                                    "org_id": org_id,
                                    "total_files": 0,
                                    "successful_files": 0,
                                    "failed_files": 0,
                                    "total_questions": 0,
                                    "questions": [],
                                    "processed_files": [],
                                    "failed_files": [],
                                    "timestamp": datetime.now().isoformat(),
                                    "status": "completed"
                                }
                        return
                    
                    print(f"📁 Found {len(project_files)} project files to process")
                    update_upload_progress(agent_run_id, "Processing", 15, f"Found {len(project_files)} project files", "")
                    
                except Exception as fetch_error:
                    raise Exception(f"Error fetching project files: {str(fetch_error)}")
                
                # Step 2: Download and process each file using ZIP method
                all_questions = []
                processed_files = []
                failed_files = []
                
                total_files = len(project_files)
                
                for index, file_info in enumerate(project_files):
                    file_id = file_info.get("id")
                    filename = file_info.get("name") or file_info.get("fileName", f"file_{file_id}")
                    file_url = file_info.get("url")
                    file_type = file_info.get("fileType", "")
                    file_ext = filename.split(".")[-1].lower() if "." in filename else ""
                    
                    try:
                        # Update progress
                        progress = 15 + ((index / total_files) * 65)  # 15% to 80%
                        update_upload_progress(agent_run_id, "Processing", progress, 
                                            f"Processing file {index + 1}/{total_files}: {filename}")
                        
                        print(f"\n📄 Processing file {index + 1}/{total_files}:")
                        print(f"  ID: {file_id}")
                        print(f"  Name: {filename}")
                        print(f"  Type: {file_type}")
                        print(f"  Extension: {file_ext}")
                        
                        # Validate file info
                        if not file_id:
                            raise Exception("File ID is missing")
                        
                        if not filename or filename.strip() == "":
                            filename = f"file_{file_id}"
                            print(f"⚠️ Using generated filename: {filename}")
                        
                        # Check if file type is supported
                        supported, missing = check_file_processing_dependencies()
                        all_supported = supported["always"] + supported["conditional"] + ["png", "jpg", "jpeg", "gif", "webp"]
                        
                        if file_ext and file_ext not in all_supported:
                            error_msg = f"Unsupported file type: {file_ext}. Supported: {', '.join(sorted(all_supported))}"
                            print(f"⚠️ {error_msg}")
                            failed_files.append({
                                "filename": filename,
                                "file_id": file_id,
                                "error": error_msg
                            })
                            continue
                        
                        # Download file using ZIP method
                        update_upload_progress(agent_run_id, "Processing", progress + 1, 
                                            f"Downloading {filename} as ZIP")
                        
                        try:
                            file_content = download_single_file_as_zip(
                                backend_api_url, file_id, filename, headers
                            )
                            
                            if not file_content or len(file_content) == 0:
                                raise Exception("Downloaded file content is empty")
                                
                            print(f"✅ Successfully downloaded {filename} ({len(file_content)} bytes)")
                            
                        except Exception as download_error:
                            error_msg = f"Download failed: {str(download_error)}"
                            print(f"❌ {error_msg}")
                            failed_files.append({
                                "filename": filename,
                                "file_id": file_id,
                                "error": error_msg
                            })
                            continue
                        
                        # Save file temporarily
                        update_upload_progress(agent_run_id, "Processing", progress + 2, 
                                            f"Saving {filename} temporarily")
                        
                        temp_file_path = os.path.join(UPLOAD_FOLDER, f"{agent_run_id}_{index}_{filename}")
                        
                        try:
                            with open(temp_file_path, 'wb') as temp_file:
                                temp_file.write(file_content)
                            
                            # Verify saved file
                            saved_size = os.path.getsize(temp_file_path)
                            if saved_size != len(file_content):
                                raise Exception(f"File save verification failed: expected {len(file_content)}, got {saved_size}")
                                
                            print(f"💾 File saved temporarily: {temp_file_path} (verified: {saved_size} bytes)")
                            
                        except Exception as save_error:
                            error_msg = f"Failed to save file: {str(save_error)}"
                            print(f"❌ {error_msg}")
                            failed_files.append({
                                "filename": filename,
                                "file_id": file_id,
                                "error": error_msg
                            })
                            continue
                        
                        # Extract questions using AI
                        update_upload_progress(agent_run_id, "Processing", progress + 3, 
                                            f"AI analyzing {filename}")
                        
                        try:
                            questions = extract_questions_with_ai_direct(temp_file_path, filename)
                            
                            if questions and len(questions) > 0:
                                processed_files.append({
                                    "filename": filename,
                                    "file_id": file_id,
                                    "file_type": file_ext,
                                    "questions": questions,
                                    "question_count": len(questions)
                                })
                                all_questions.extend(questions)
                                print(f"✅ Extracted {len(questions)} questions from {filename}")
                            else:
                                error_msg = "AI could not extract questions from this file"
                                print(f"⚠️ {error_msg}")
                                failed_files.append({
                                    "filename": filename,
                                    "file_id": file_id,
                                    "error": error_msg
                                })
                                
                        except Exception as ai_error:
                            error_msg = f"AI processing failed: {str(ai_error)}"
                            print(f"❌ {error_msg}")
                            failed_files.append({
                                "filename": filename,
                                "file_id": file_id,
                                "error": error_msg
                            })
                        
                        # Clean up temporary file
                        try:
                            os.remove(temp_file_path)
                            print(f"🧹 Cleaned up temp file: {temp_file_path}")
                        except Exception as cleanup_error:
                            print(f"⚠️ Failed to remove temp file {temp_file_path}: {cleanup_error}")
                            
                    except Exception as file_error:
                        error_msg = str(file_error)
                        print(f"❌ Error processing file {filename}: {error_msg}")
                        traceback.print_exc()
                        failed_files.append({
                            "filename": filename,
                            "file_id": file_id,
                            "error": error_msg
                        })
                        
                        # Ensure temp file cleanup even on error
                        temp_file_path = os.path.join(UPLOAD_FOLDER, f"{agent_run_id}_{index}_{filename}")
                        if os.path.exists(temp_file_path):
                            try:
                                os.remove(temp_file_path)
                                print(f"🧹 Cleaned up temp file after error: {temp_file_path}")
                            except:
                                pass
                
                # Step 3: Compile and store results
                update_upload_progress(agent_run_id, "Processing", 85, "Compiling agent results")
                
                print(f"\n📊 Question Agent Results Summary:")
                print(f"  Total files: {total_files}")
                print(f"  Successfully processed: {len(processed_files)}")
                print(f"  Failed: {len(failed_files)}")
                print(f"  Total questions generated: {len(all_questions)}")
                
                # Store agent results
                agent_results = {
                    "agent_run_id": agent_run_id,
                    "agent_type": "question_generator",
                    "rfp_id": rfp_id,
                    "org_id": org_id,
                    "total_files": total_files,
                    "successful_files": len(processed_files),
                    "failed_files": len(failed_files),
                    "total_questions": len(all_questions),
                    "questions": all_questions,
                    "processed_files": processed_files,
                    "failed_files": failed_files,
                    "processing_method": "zip_extraction",
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
                
                # Update progress with completion
                update_upload_progress(agent_run_id, "Completed", 100, "Question Agent completed successfully")
                
                # Store results in progress for retrieval
                with progress_lock:
                    if agent_run_id in upload_progress:
                        upload_progress[agent_run_id]["agent_results"] = agent_results
                if len(all_questions) > 0:
                    postAgentResultsAndQuestions(rfp_id, agent_results, auth_token)
                
                print(f"✅ Question Agent completed successfully!")
                print(f"📈 Final Stats: {len(all_questions)} questions from {len(processed_files)}/{total_files} files")
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Question Agent error: {error_msg}")
                traceback.print_exc()
                update_upload_progress(agent_run_id, "error", 0, f"Agent Error: {error_msg}")
                
                # Store error results
                with progress_lock:
                    if agent_run_id in upload_progress:
                        upload_progress[agent_run_id]["agent_results"] = {
                            "agent_run_id": agent_run_id,
                            "agent_type": "question_generator",
                            "rfp_id": rfp_id,
                            "org_id": org_id,
                            "error": error_msg,
                            "timestamp": datetime.now().isoformat(),
                            "status": "failed"
                        }
                
                # Store results in progress for retrieval
                update_upload_progress(agent_run_id, "Completed", 100, "Question Agent completed successfully")
                
                with progress_lock:
                    if agent_run_id in upload_progress:
                        upload_progress[agent_run_id]["agent_results"] = agent_results
                
                print(f"✅ Question Agent completed. Generated {len(all_questions)} questions from {len(processed_files)} files")
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Question Agent error: {error_msg}")
                traceback.print_exc()
                update_upload_progress(agent_run_id, "error", 0, f"Agent Error: {error_msg}")
        
        # Start agent processing in background thread
        agent_thread = Thread(target=run_question_agent_async)
        agent_thread.daemon = False
        agent_thread.start()
        
        return jsonify({
            "message": "Question Agent started successfully",
            "agent_run_id": agent_run_id,
            "agent_type": "question_generator",
            "rfp_id": rfp_id,
            "status": "running"
        }), 202

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Question Agent Error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": error_msg}), 500

@app.route("/agent-status", methods=["GET", "OPTIONS"])
def get_agent_status():
    """Get agent run status and results"""
    if request.method == "OPTIONS":
        return "", 200
        
    agent_run_id = request.args.get("agentRunId")
    
    if not agent_run_id:
        return jsonify({"error": "agentRunId is required"}), 400
    
    with progress_lock:
        exists = agent_run_id in upload_progress
        status_data = upload_progress.get(agent_run_id, None)
    
    if not exists:
        return jsonify({"exists": False}), 200
    
    response_data = {
        "exists": True,
        "status": status_data
    }
    
    # Include agent results if completed
    if status_data and status_data.get("status") == "Completed" and "agent_results" in status_data:
        response_data["agent_results"] = status_data["agent_results"]
    
    return jsonify(response_data), 200

# Add this new endpoint after the existing /run-question-agent endpoint

@app.route("/run-question-agent-with-upload", methods=["POST", "OPTIONS"])
def run_question_agent_with_upload():
    """Run Question Agent with direct file upload - Upload files and generate questions"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not VERTEX_AVAILABLE:
        return jsonify({"error": "AI generation service is not available"}), 503
        
    try:
        rfp_id = request.form.get("rfpId")
        org_id = request.form.get("orgId") 
        agent_run_id = request.form.get("agentRunId", str(uuid.uuid4()))
        auth_token = request.form.get("authToken")
        agent_id = request.form.get("agentId")
        
        print(f"🤖 Starting Question Agent with Upload for RFP: {rfp_id}, Agent Run ID: {agent_run_id}")
        
        if not rfp_id:
            return jsonify({"error": "rfpId is required"}), 400
            
        if not org_id:
            return jsonify({"error": "orgId is required"}), 400
        if not agent_id:  # ADD THIS VALIDATION
                    return jsonify({"error": "agentId is required"}), 400
        # Check if files were uploaded
        if not request.files:
            return jsonify({"error": "No files provided"}), 400

        uploaded_files = request.files.getlist('files')
        if not uploaded_files:
            return jsonify({"error": "No files found in request"}), 400

        # Validate files
        valid_files = []
        for file in uploaded_files:
            if file.filename and file.filename.strip():
                valid_files.append(file)
        
        if not valid_files:
            return jsonify({"error": "No valid files found"}), 400

        print(f"📁 Processing {len(valid_files)} files for question extraction")

        # Read all files into memory
        file_data_list = []
        for file in valid_files:
            try:
                file.seek(0)
                file_content = file.read()
                
                if len(file_content) == 0:
                    print(f"⚠️ Skipping empty file: {file.filename}")
                    continue
                    
                file_data_list.append({
                    'filename': file.filename,
                    'content': file_content,
                    'size': len(file_content),
                    'file_object': file  # Keep reference for backend upload
                })
                print(f"✅ Read {file.filename} ({len(file_content)} bytes)")
                
            except Exception as e:
                print(f"❌ Error reading file {file.filename}: {str(e)}")
                return jsonify({"error": f"Failed to read file {file.filename}: {str(e)}"}), 400

        if not file_data_list:
            return jsonify({"error": "No files could be read successfully"}), 400

        # Initialize progress tracking
        update_upload_progress(agent_run_id, "Processing", 0, "Starting Question Agent with file upload", "")
        
        def process_files_with_upload_async():
            try:
                # Step 1: Upload files to backend uploadProjectFiles API
                update_upload_progress(agent_run_id, "Processing", 5, "Uploading files to backend", "")
                
                backend_api_url = os.environ.get("BACKEND_API_URL", "http://localhost:8080")
                upload_url = f"{backend_api_url}/api/rfps/{rfp_id}/project-files"
                
                # Prepare files for backend upload
                files_for_upload = []
                for file_data in file_data_list:
                    # Reset file pointer and create new file object for upload
                    file_content = file_data['content']
                    files_for_upload.append(
                        ('files', (file_data['filename'], io.BytesIO(file_content), 'application/octet-stream'))
                    )
                
                headers = {}
                if auth_token:
                    headers["Authorization"] = f"Bearer {auth_token}"
                
                try:
                    print(f"📤 Uploading {len(files_for_upload)} files to backend: {upload_url}")
                    backend_response = requests.post(
                        upload_url,
                        files=files_for_upload,
                        headers=headers,
                        timeout=60
                    )
                    
                    if backend_response.status_code in [200, 201]:
                        print(f"✅ Successfully uploaded files to backend")
                        update_upload_progress(agent_run_id, "Processing", 15, "Files uploaded to backend successfully", "")
                    else:
                        print(f"⚠️ Backend upload failed: {backend_response.status_code} - {backend_response.text[:200]}")
                        update_upload_progress(agent_run_id, "Processing", 15, "Backend upload failed, continuing with AI processing", "")
                        
                except Exception as upload_error:
                    print(f"⚠️ Backend upload error: {str(upload_error)}")
                    update_upload_progress(agent_run_id, "Processing", 15, "Backend upload failed, continuing with AI processing", "")

                # Step 2: Process files for question extraction
                update_upload_progress(agent_run_id, "Processing", 20, "Starting AI question extraction", "")
                
                all_questions = []
                processed_files = []
                failed_files = []
                total_files = len(file_data_list)
                
                # Get supported file types
                supported, missing = check_file_processing_dependencies()
                all_supported = supported["always"] + supported["conditional"] + ["png", "jpg", "jpeg", "gif", "webp"]
                
                for index, file_data in enumerate(file_data_list):
                    filename = file_data['filename']
                    file_content = file_data['content']
                    file_ext = filename.split(".")[-1].lower()
                    
                    # Validate file type
                    if file_ext not in all_supported:
                        failed_files.append({
                            "filename": filename,
                            "error": f"Unsupported file type: {file_ext}. Supported: {', '.join(sorted(all_supported))}"
                        })
                        continue
                    
                    try:
                        # Update progress
                        progress = 20 + ((index / total_files) * 60)  # 20% to 80%
                        update_upload_progress(agent_run_id, "Processing", progress, 
                                            f"AI processing file {index + 1}/{total_files}: {filename}")
                        
                        # Save file content to temporary file
                        save_path = os.path.join(UPLOAD_FOLDER, f"{agent_run_id}_{index}_{filename}")
                        with open(save_path, 'wb') as f:
                            f.write(file_content)
                        
                        # Extract questions using AI
                        questions = extract_questions_with_ai_direct(save_path, filename)
                        
                        if not questions:
                            failed_files.append({
                                "filename": filename,
                                "error": "AI could not extract questions from this file"
                            })
                        else:
                            processed_files.append({
                                "filename": filename,
                                "file_type": file_ext,
                                "questions": questions,
                                "question_count": len(questions)
                            })
                            all_questions.extend(questions)
                            print(f"✅ Extracted {len(questions)} questions from {filename}")
                        
                        # Clean up temporary file
                        try:
                            os.remove(save_path)
                        except Exception as e:
                            print(f"⚠️ Failed to remove temp file {save_path}: {e}")
                            
                    except Exception as file_error:
                        print(f"❌ Error processing file {filename}: {str(file_error)}")
                        failed_files.append({
                            "filename": filename,
                            "error": str(file_error)
                        })
                
                # Step 3: Compile and store results
                update_upload_progress(agent_run_id, "Processing", 85, "Compiling agent results")
                
                print(f"\n📊 Question Agent Results Summary:")
                print(f"  Total files: {total_files}")
                print(f"  Successfully processed: {len(processed_files)}")
                print(f"  Failed: {len(failed_files)}")
                print(f"  Total questions generated: {len(all_questions)}")
                
                # Store agent results (same format as before)
                agent_results = {
                    "agent_run_id": agent_run_id,
                    "agent_type": "question_generator",
                    "agent_id": agent_id,  # ADD THIS LINE
                    "rfp_id": rfp_id,
                    "org_id": org_id,
                    "total_files": total_files,
                    "successful_files": len(processed_files),
                    "failed_files": len(failed_files),
                    "total_questions": len(all_questions),
                    "questions": all_questions,
                    "processed_files": processed_files,
                    "failed_files": failed_files,
                    "processing_method": "direct_upload",
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
                
                # Update progress with completion
                update_upload_progress(agent_run_id, "Completed", 100, "Question Agent completed successfully")
                
                # Store results in progress for retrieval
                with progress_lock:
                    if agent_run_id in upload_progress:
                        upload_progress[agent_run_id]["agent_results"] = agent_results
                if len(all_questions) > 0:
                    postAgentResultsAndQuestions(rfp_id, agent_results, auth_token,agent_id)
                
                print(f"✅ Question Agent with upload completed successfully!")
                print(f"📈 Final Stats: {len(all_questions)} questions from {len(processed_files)}/{total_files} files")
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Question Agent error: {error_msg}")
                traceback.print_exc()
                update_upload_progress(agent_run_id, "error", 0, f"Agent Error: {error_msg}")
        
        # Start processing in background thread
        processing_thread = Thread(target=process_files_with_upload_async)
        processing_thread.daemon = False
        processing_thread.start()
        
        return jsonify({
            "message": "Question Agent with file upload started successfully",
            "agent_run_id": agent_run_id,
            "agent_type": "question_generator",
            "rfp_id": rfp_id,
            "files_count": len(file_data_list),
            "status": "running"
        }), 202

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Question Agent with Upload Error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": error_msg}), 500
    

@app.route("/text-operations", methods=["POST", "OPTIONS"])
def text_operations():
    """Perform AI operations on selected text (rewrite, summarize, expand, etc.)"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not VERTEX_AVAILABLE:
        return jsonify({"error": "AI generation service is not available"}), 503
        
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No valid JSON input found"}), 400

        selected_text = data.get("selectedText")
        operation = data.get("operation")  # rewrite, summarize, expand, improve, etc.
        context = data.get("context", "")  # Optional surrounding context
        file_type = data.get("fileType", "")  # Optional file type context
        
        if not selected_text or not operation:
            return jsonify({"error": "selectedText and operation are required"}), 400

        print(f"🔧 Text operation: {operation} on text: {selected_text[:100]}...")

        # Define operation prompts - UPDATED WITH ALL FRONTEND OPERATIONS
        operation_prompts = {
            "rewrite": f"""
Rewrite the following text to make it clearer and more professional while maintaining the same meaning:

Text: {selected_text}

Provide only the rewritten text without any additional explanations or formatting.
""",
            "expand": f"""
Expand the following text with more detail and explanation while maintaining the same tone and style:

Text: {selected_text}

Provide only the expanded text without any additional explanations or formatting.
""",
            "simplify": f"""
Simplify the following text to make it easier to understand while keeping the essential meaning:

Text: {selected_text}

Provide only the simplified text without any additional explanations or formatting.
""",
            "formalize": f"""
Make the following text more formal and professional in tone:

Text: {selected_text}

Provide only the formalized text without any additional explanations or formatting.
""",
            "casual": f"""
Make the following text more casual and conversational while maintaining professionalism:

Text: {selected_text}

Provide only the casual version without any additional explanations or formatting.
""",
            "shorten": f"""
Shorten the following text while keeping the key information and main message:

Text: {selected_text}

Provide only the shortened text without any additional explanations or formatting.
""",
            "grammar": f"""
Fix any grammar, spelling, and punctuation errors in the following text:

Text: {selected_text}

Provide only the corrected text without any additional explanations or formatting.
""",
            "professional": f"""
Make the following text more professional and business-appropriate:

Text: {selected_text}

Provide only the professional version without any additional explanations or formatting.
""",
            # Additional operations for completeness
            "summarize": f"""
Summarize the following text concisely while retaining the key points:

Text: {selected_text}

Provide only the summary without any additional explanations or formatting.
""",
            "improve": f"""
Improve the following text for better clarity, grammar, flow, and professionalism:

Text: {selected_text}

Provide only the improved text without any additional explanations or formatting.
""",
            "bullet_points": f"""
Convert the following text into clear, well-organized bullet points:

Text: {selected_text}

Provide only the bullet points without any additional explanations or formatting.
"""
        }

        if operation not in operation_prompts:
            supported_ops = list(operation_prompts.keys())
            return jsonify({
                "error": f"Unsupported operation: {operation}. Supported operations: {', '.join(supported_ops)}"
            }), 400

        prompt = operation_prompts[operation]

        # Add context if provided
        if context and context.strip():
            prompt += f"\n\nContext (for reference): {context[:500]}..."  # Limit context length

        # Add file type context if provided
        if file_type:
            prompt += f"\n\nNote: This text is from a {file_type.upper()} file."

        from vertexai.generative_models import GenerativeModel, Part, SafetySetting
        model = GenerativeModel("gemini-2.0-flash")

        response = model.generate_content(
            [Part.from_text(prompt)],
            generation_config={
                "max_output_tokens": 2048,
                "temperature": 0.3,
                "top_p": 0.8,
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
            stream=False
        )

        processed_text = response.text.strip()

        # Basic validation of the response
        if not processed_text:
            return jsonify({"error": "AI returned empty response"}), 500

        print(f"✅ Text operation '{operation}' completed successfully")

        return jsonify({
            "original_text": selected_text,
            "operation": operation,
            "processed_text": processed_text,
            "success": True,
            "character_count": {
                "original": len(selected_text),
                "processed": len(processed_text)
            }
        }), 200

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Text Operation Error: {error_msg}")
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error", 
            "details": error_msg,
            "success": False
        }), 500

# Add this endpoint to your existing Python Flask application after the /run-question-agent-with-upload endpoint

# Add this endpoint to your existing Python Flask application after the /run-question-agent-with-upload endpoint

@app.route("/run-proposal-narrative-agent", methods=["POST", "OPTIONS"])
def run_proposal_narrative_agent():
    """Run Proposal Narrative Writer Agent - Process RFP and generate proposal sections"""
    if request.method == "OPTIONS":
        return "", 200
        
    if not VERTEX_AVAILABLE:
        return jsonify({"error": "AI generation service is not available"}), 503
        
    try:
        rfp_id = request.form.get("rfpId")
        org_id = request.form.get("orgId") 
        agent_run_id = request.form.get("agentRunId", str(uuid.uuid4()))
        auth_token = request.form.get("authToken")
        
        print(f"🎯 Starting Proposal Narrative Agent for RFP: {rfp_id}, Agent Run ID: {agent_run_id}")
        
        if not rfp_id:
            return jsonify({"error": "rfpId is required"}), 400
            
        if not org_id:
            return jsonify({"error": "orgId is required"}), 400

        # Get project file (RFP document) - required
        project_file = request.files.get('projectFile')
        if not project_file or not project_file.filename:
            return jsonify({"error": "Project file (RFP document) is required"}), 400

        # Get support files - optional
        support_files = request.files.getlist('supportFiles')
        
        print(f"📄 Processing RFP document: {project_file.filename}")
        if support_files:
            print(f"📚 Processing {len(support_files)} support files for knowledge base")

        # Read all files into memory before starting async processing
        try:
            # Read project file content
            project_file.seek(0)
            project_content = project_file.read()
            project_file_data = {
                'filename': project_file.filename,
                'content': project_content,
                'size': len(project_content)
            }
            
            if len(project_file_data['content']) == 0:
                return jsonify({"error": "Project file is empty"}), 400
            
            # Read support files content
            support_files_data = []
            for support_file in support_files:
                if support_file.filename and support_file.filename.strip():
                    support_file.seek(0)
                    support_content = support_file.read()
                    
                    if len(support_content) > 0:
                        support_files_data.append({
                            'filename': support_file.filename,
                            'content': support_content,
                            'size': len(support_content)
                        })
                        print(f"✅ Read support file: {support_file.filename} ({len(support_content)} bytes)")
                    else:
                        print(f"⚠️ Skipping empty support file: {support_file.filename}")
            
            print(f"✅ Read project file: {project_file_data['filename']} ({len(project_file_data['content'])} bytes)")
            print(f"✅ Read {len(support_files_data)} support files")
            
        except Exception as e:
            print(f"❌ Error reading file contents: {str(e)}")
            return jsonify({"error": f"Failed to read file contents: {str(e)}"}), 400

        # Initialize progress tracking
        update_upload_progress(agent_run_id, "Processing", 0, "Starting Proposal Narrative Agent", "")
        
        def process_proposal_agent_async():
            try:
                # Step 1: Process support files for knowledge base (if any)
                support_results = []
                if support_files_data and len(support_files_data) > 0:
                    update_upload_progress(agent_run_id, "Processing", 10, "Processing support files for knowledge base", "")
                    
                    for idx, support_file_data in enumerate(support_files_data):
                        filename = support_file_data['filename']
                        file_content = support_file_data['content']
                        
                        try:
                            # Save support file temporarily
                            support_path = os.path.join(UPLOAD_FOLDER, f"{agent_run_id}_support_{idx}_{filename}")
                            with open(support_path, 'wb') as f:
                                f.write(file_content)
                            
                            # Upload to RFP-specific knowledge base
                            support_upload_id = f"support_{agent_run_id}_{idx}"
                            
                            # Process for embedding using RFP support document endpoint logic
                            file_ext = filename.split(".")[-1].lower()
                            chunks = parse_and_chunk(support_path, file_ext, chunk_size=50, max_chunks=500)
                            
                            if chunks:
                                embeddings = embed_chunks(chunks, upload_id=support_upload_id, org_id=org_id, filename=filename)
                                
                                # Store in RFP-specific Firestore collection
                                rfp_doc_ref = db.collection("rfp_support_embeddings").document(f"rfp-{rfp_id}").collection("files").document(support_upload_id)
                                
                                # Store file metadata
                                rfp_doc_ref.set({
                                    "filename": filename,
                                    "file_id": support_upload_id,
                                    "upload_id": support_upload_id,
                                    "rfp_id": rfp_id,
                                    "org_id": org_id,
                                    "file_type": file_ext,
                                    "document_type": "support",
                                    "agent_run_id": agent_run_id,
                                    "created_at": firestore.SERVER_TIMESTAMP,
                                    "chunk_count": len(chunks),
                                    "processing_version": "4.0.0"
                                })
                                
                                # Store chunks in batches
                                batch_size = 10
                                total_chunks = len(chunks)
                                
                                for i in range(0, total_chunks, batch_size):
                                    batch = db.batch()
                                    end_idx = min(i + batch_size, total_chunks)
                                    
                                    for j in range(i, end_idx):
                                        chunk_ref = rfp_doc_ref.collection("chunks").document(str(j))
                                        batch.set(chunk_ref, {
                                            "content": chunks[j],
                                            "embedding": embeddings[j],
                                            "index": j
                                        })
                                    
                                    batch.commit()
                                    del batch
                                    import gc
                                    gc.collect()
                                
                                support_results.append({
                                    "filename": filename,
                                    "status": "success",
                                    "chunks": len(chunks)
                                })
                                
                                print(f"✅ Support file processed for knowledge base: {filename}")
                            else:
                                support_results.append({
                                    "filename": filename,
                                    "status": "failed",
                                    "error": "No content extracted"
                                })
                            
                            # Clean up temp file
                            try:
                                os.remove(support_path)
                            except Exception as e:
                                print(f"⚠️ Failed to remove temp file {support_path}: {e}")
                                
                        except Exception as support_error:
                            print(f"❌ Error processing support file {filename}: {str(support_error)}")
                            support_results.append({
                                "filename": filename,
                                "status": "failed",
                                "error": str(support_error)
                            })
                
                # Step 2: Process main RFP document and extract proposal sections
                update_upload_progress(agent_run_id, "Processing", 40, "Analyzing RFP document and extracting sections", "")
                
                # Save project file temporarily using the content we read earlier
                project_filename = project_file_data['filename']
                project_content = project_file_data['content']
                project_path = os.path.join(UPLOAD_FOLDER, f"{agent_run_id}_project_{project_filename}")
                
                try:
                    # Write project file content to temporary file
                    with open(project_path, 'wb') as f:
                        f.write(project_content)
                    
                    print(f"✅ Project file saved temporarily: {project_path}")
                    
                    # Extract proposal sections and requirements from RFP using AI
                    proposal_sections = extract_proposal_sections_with_ai(project_path, project_filename)
                    
                    if not proposal_sections:
                        raise Exception("Could not extract proposal sections from RFP document")
                    
                    print(f"✅ Extracted {len(proposal_sections)} proposal sections from RFP")
                    
                    # Clean up project file
                    try:
                        os.remove(project_path)
                        print(f"🧹 Cleaned up project file: {project_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to remove temp file {project_path}: {e}")
                        
                except Exception as rfp_error:
                    raise Exception(f"Error processing RFP document: {str(rfp_error)}")
                
                # Step 3: Compile results
                update_upload_progress(agent_run_id, "Processing", 80, "Compiling proposal narrative results", "")
                
                # Create agent results
                agent_results = {
                    "agent_run_id": agent_run_id,
                    "agent_type": "proposal_narrative_writer",
                    "rfp_id": rfp_id,
                    "org_id": org_id,
                    "project_file": {
                        "filename": project_filename,
                        "status": "success"
                    },
                    "support_files_processed": len(support_results),
                    "support_files_success": len([r for r in support_results if r["status"] == "success"]),
                    "support_files": support_results,
                    "sections": proposal_sections,
                    "total_sections": len(proposal_sections),
                    "knowledge_base_enhanced": len([r for r in support_results if r["status"] == "success"]) > 0,
                    "processing_method": "proposal_narrative_generation",
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
                
                # Update progress with completion
                update_upload_progress(agent_run_id, "Completed", 100, "Proposal Narrative Agent completed successfully")
                
                # Store results in progress for retrieval
                with progress_lock:
                    if agent_run_id in upload_progress:
                        upload_progress[agent_run_id]["agent_results"] = agent_results
                
                # Post results to backend
                post_proposal_agent_results_to_backend(rfp_id, agent_results, auth_token)
                
                print(f"✅ Proposal Narrative Agent completed successfully!")
                print(f"📈 Final Stats: {len(proposal_sections)} sections extracted, {len([r for r in support_results if r['status'] == 'success'])} support files processed")
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Proposal Narrative Agent error: {error_msg}")
                traceback.print_exc()
                update_upload_progress(agent_run_id, "error", 0, f"Agent Error: {error_msg}")
                
                # Store error results
                with progress_lock:
                    if agent_run_id in upload_progress:
                        upload_progress[agent_run_id]["agent_results"] = {
                            "agent_run_id": agent_run_id,
                            "agent_type": "proposal_narrative_writer",
                            "rfp_id": rfp_id,
                            "org_id": org_id,
                            "error": error_msg,
                            "timestamp": datetime.now().isoformat(),
                            "status": "failed"
                        }
        
        # Start processing in background thread
        processing_thread = Thread(target=process_proposal_agent_async)
        processing_thread.daemon = False
        processing_thread.start()
        
        return jsonify({
            "message": "Proposal Narrative Agent started successfully",
            "agent_run_id": agent_run_id,
            "agent_type": "proposal_narrative_writer",
            "rfp_id": rfp_id,
            "project_file": project_file.filename,
            "support_files_count": len(support_files) if support_files else 0,
            "status": "running"
        }), 202

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Proposal Narrative Agent Error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": error_msg}), 500

def extract_proposal_sections_with_ai(file_path, filename):
    """Extract proposal sections and requirements from RFP document using Vertex AI"""
    if not VERTEX_AVAILABLE:
        print("❌ Vertex AI not available")
        return []
        
    try:
        # Verify file exists and is readable
        if not os.path.exists(file_path):
            print(f"❌ File does not exist: {file_path}")
            return []
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"❌ File is empty: {file_path}")
            return []
            
        print(f"📄 Processing RFP file: {filename} ({file_size} bytes)")
        
        # Get file extension to determine how to handle it
        file_ext = filename.split(".")[-1].lower()
        
        # Customize prompt for proposal section extraction
        prompt = f"""
Analyze the uploaded RFP document and extract all sections that require a proposal response. 

For each section that needs a response in the proposal, provide:
1. The section title/name
2. The type of section (e.g., "Technical Approach", "Methodology", "Qualifications", "Experience", "Timeline", "Budget", "Compliance", "Company Profile")
3. The specific requirements, questions, or criteria that need to be addressed

Focus on sections like:
- Technical approach and methodology
- Company qualifications and experience
- Project team and personnel
- Case studies and past performance
- Compliance and certifications
- Timeline and project management
- Budget and pricing (if mentioned)
- Company profile and capabilities
- Risk management
- Quality assurance
- Any specific questions posed to bidders

Format your response as a valid JSON array with objects containing "title", "type", and "content" fields:

[
  {{
    "title": "Technical Approach",
    "type": "Technical Approach",
    "content": "Describe your technical approach to implementing the solution, including methodology, tools, and processes."
  }},
  {{
    "title": "Company Experience", 
    "type": "Qualifications",
    "content": "Provide examples of similar projects completed in the last 5 years, including client references and outcomes achieved."
  }}
]

IMPORTANT: Return ONLY the JSON array, no additional text or formatting.
"""

        model = GenerativeModel("gemini-2.0-flash")
        
        # Prepare the content based on file type
        parts = [Part.from_text(prompt)]
        
        # Define MIME type mapping
        mime_mapping = {
            "pdf": "application/pdf",
            "txt": "text/plain", 
            "csv": "text/csv",
            "json": "application/json",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg", 
            "gif": "image/gif",
            "webp": "image/webp",
        }
        
        # Files that Gemini can handle directly
        gemini_supported_files = ["pdf", "txt", "csv", "json", "png", "jpg", "jpeg", "gif", "webp"]
        
        if file_ext in gemini_supported_files:
            try:
                # Read file as bytes for direct upload
                with open(file_path, "rb") as f:
                    file_data = f.read()
                
                mime_type = mime_mapping.get(file_ext, "application/octet-stream")
                
                # Validate file size for Gemini (max ~10MB)
                if len(file_data) > 10 * 1024 * 1024:  # 10MB limit
                    print(f"⚠️ File too large for direct upload ({len(file_data)} bytes), falling back to text extraction")
                    content = extract_text_from_file(file_path, file_ext)
                    if content and not content.startswith("ERROR:"):
                        if len(content) > 30000:
                            content = content[:30000] + "..."
                        parts.append(Part.from_text(f"\nRFP Document Content:\n{content}"))
                    else:
                        return []
                else:
                    # Add file part directly
                    file_part = Part.from_data(data=file_data, mime_type=mime_type)
                    parts.append(file_part)
                    
                    print(f"✅ Using direct multimodal upload for {filename} ({mime_type}, {len(file_data)} bytes)")
                
            except Exception as e:
                print(f"⚠️ Direct upload failed for {filename}, falling back to text extraction: {e}")
                content = extract_text_from_file(file_path, file_ext)
                if content and not content.startswith("ERROR:"):
                    if len(content) > 30000:
                        content = content[:30000] + "..."
                    parts.append(Part.from_text(f"\nRFP Document Content:\n{content}"))
                else:
                    return []
        else:
            # For Excel, Word, and other files - extract text first
            print(f"📄 Extracting text from {filename} (file type: {file_ext})")
            content = extract_text_from_file(file_path, file_ext)
            if not content or content.startswith("ERROR:"):
                print(f"❌ No content extracted from {filename}: {content}")
                return []
            
            # Limit content length
            if len(content) > 30000:
                content = content[:30000] + "..."
            
            parts.append(Part.from_text(f"\nRFP Document Content:\n{content}"))
        
        # Generate response
        print(f"🤖 Sending request to Gemini for proposal section extraction: {filename}")
        response = model.generate_content(
            parts,
            generation_config={
                "max_output_tokens": 4096,
                "temperature": 0.3,
                "top_p": 0.8,
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
            stream=False
        )
        
        print(f"✅ Received response from Gemini for {filename}")
        
        # Parse the JSON response
        response_text = response.text.strip()
        print(f"📝 Raw response length: {len(response_text)} characters")
        
        # Clean up the response - remove any markdown formatting
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
        
        # Try to extract JSON from the response
        try:
            sections = json.loads(response_text)
            
            # Validate the structure
            if isinstance(sections, list):
                valid_sections = []
                for section in sections:
                    if isinstance(section, dict) and "title" in section and "type" in section and "content" in section:
                        valid_sections.append({
                            "title": str(section["title"]).strip(),
                            "type": str(section["type"]).strip(),
                            "content": str(section["content"]).strip()
                        })
                
                print(f"✅ Successfully parsed {len(valid_sections)} proposal sections from {filename}")
                return valid_sections[:20]  # Limit to 20 sections
            else:
                print(f"⚠️ Invalid JSON structure from AI for {filename} (not a list)")
                return []
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error for {filename}: {e}")
            print(f"Raw response preview: {response_text[:500]}...")
            
            # Fallback: try to extract sections manually using regex
            sections = []
            
            # Try to find JSON objects in the text
            json_pattern = r'\{\s*"title":\s*"([^"]+)"\s*,\s*"type":\s*"([^"]+)"\s*,\s*"content":\s*"([^"]+)"\s*\}'
            matches = re.findall(json_pattern, response_text)
            
            for title, section_type, content in matches:
                sections.append({
                    "title": title.strip(),
                    "type": section_type.strip(),
                    "content": content.strip()
                })
            
            if sections:
                print(f"✅ Extracted {len(sections)} sections using fallback regex for {filename}")
            else:
                print(f"❌ Could not extract any sections from {filename}")
                
            return sections[:20]
            
    except Exception as e:
        print(f"❌ Error extracting proposal sections from {filename}: {str(e)}")
        traceback.print_exc()
        return []

def post_proposal_agent_results_to_backend(rfp_id, agent_results, auth_token):
    """Post proposal agent results to backend"""
    if not rfp_id or not agent_results:
        return

    try:
        backend_api_url = os.environ.get("BACKEND_API_URL", "http://localhost:8080")
        agent_response_url = f"{backend_api_url}/api/rfps/{rfp_id}/agent-response"
        
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        # Parse results into string format
        response_string = json.dumps({
            "summary": {
                "total_sections": agent_results.get("total_sections", 0),
                "support_files_processed": agent_results.get("support_files_processed", 0),
                "support_files_success": agent_results.get("support_files_success", 0),
                "knowledge_base_enhanced": agent_results.get("knowledge_base_enhanced", False),
            },
            "sections": agent_results.get("sections", []),
            "project_file": agent_results.get("project_file", {}),
            "support_files": agent_results.get("support_files", []),
            "timestamp": agent_results.get("timestamp"),
            "processing_method": agent_results.get("processing_method"),
        })

        response = requests.post(
            agent_response_url,
            json={
                "agent": "Proposal Narrative Writer",
                "response": response_string,
            },
            headers=headers,
            timeout=30
        )

        if response.status_code in [200, 201]:
            print("✅ Proposal agent results posted to backend successfully")
        else:
            print(f"⚠️ Failed to post proposal agent results: {response.status_code} - {response.text[:200]}")

    except Exception as e:
        print(f"❌ Error posting proposal agent results: {str(e)}")
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug_mode = os.environ.get("DEBUG", "0") == "1"
    
    print(f"🚀 CareAI API v2.3.0 - Enhanced with Multimodal Question Extraction")
    print(f"📋 Features: FileId management, Auto status sync, Multimodal question extraction, Direct file upload")
    print(f"🎯 Supported: Documents (PDF, DOCX, TXT, CSV, JSON, XLSX) + Images (PNG, JPG, GIF, WEBP)")
    
    if debug_mode:
        print(f"🔍 Starting Flask development server on port {port}")
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        threads = int(os.environ.get("WAITRESS_THREADS", "8"))
        print(f"🚀 Starting server with Waitress on port {port} with {threads} threads")
        serve(app, host="0.0.0.0", port=port, threads=threads)