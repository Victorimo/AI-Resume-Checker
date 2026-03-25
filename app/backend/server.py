from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response, FileResponse
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import shutil

# NLP Imports
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Document processing
import io

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Local File Storage Configuration
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initializing Sentence-BERT model (lazy loading)
sbert_model = None

def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sbert_model

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# The main app
app = FastAPI(title="AI Resume Analyzer")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ Models ============

class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class ResumeAnalysisRequest(BaseModel):
    resume_text: str
    job_description: Optional[str] = None

class JobRole(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    required_skills: List[str]
    preferred_skills: List[str] = []
    experience_years: int
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class JobRoleCreate(BaseModel):
    title: str
    description: str
    required_skills: List[str]
    preferred_skills: List[str] = []
    experience_years: int

class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    resume_text: str
    job_id: str
    relevance_score: float  # 0-1 ground truth
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class DatasetEntryCreate(BaseModel):
    resume_text: str
    job_id: str
    relevance_score: float

# ============ Local File Storage Functions ============

def save_file(file_id: str, filename: str, content: bytes) -> str:
    """Save file to local storage"""
    ext = filename.split(".")[-1].lower() if "." in filename else "bin"
    file_path = UPLOAD_DIR / f"{file_id}.{ext}"
    with open(file_path, "wb") as f:
        f.write(content)
    return str(file_path)

def get_file(file_path: str) -> bytes:
    """Get file from local storage"""
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    with open(path, "rb") as f:
        return f.read()

def delete_file(file_path: str) -> bool:
    """Delete file from local storage"""
    path = Path(file_path)
    if path.exists():
        path.unlink()
        return True
    return False

