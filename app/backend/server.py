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

# ============ NLP Functions ============

def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

def extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX bytes"""
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""

def preprocess_text(text: str) -> str:
    """Clean and preprocess text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\-\@\+]', '', text)
    return text.lower().strip()

def extract_skills(text: str) -> List[str]:
    """Extract skills from text using keyword matching"""
    common_skills = [
        'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node',
        'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ci/cd',
        'machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch',
        'data analysis', 'data science', 'statistics', 'pandas', 'numpy', 'scipy',
        'agile', 'scrum', 'project management', 'leadership', 'communication',
        'html', 'css', 'sass', 'tailwind', 'bootstrap', 'figma', 'ui/ux',
        'git', 'linux', 'bash', 'rest api', 'graphql', 'microservices',
        'c++', 'c#', 'go', 'rust', 'scala', 'kotlin', 'swift', 'r',
        'spark', 'hadoop', 'kafka', 'airflow', 'dbt', 'tableau', 'power bi',
        'security', 'devops', 'sre', 'automation', 'testing', 'qa',
        'product management', 'business analysis', 'stakeholder management'
    ]
    
    text_lower = text.lower()
    found_skills = []
    for skill in common_skills:
        if skill in text_lower:
            found_skills.append(skill)
    return list(set(found_skills))

def extract_experience_years(text: str) -> int:
    """Extract years of experience from text"""
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'experience[:\s]*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*(?:in|with)',
    ]
    
    text_lower = text.lower()
    max_years = 0
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                years = int(match)
                max_years = max(max_years, years)
            except:
                pass
    return max_years

def calculate_sbert_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity using Sentence-BERT"""
    model = get_sbert_model()
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)

def calculate_tfidf_similarity(text1: str, text2: str) -> float:
    """Calculate TF-IDF cosine similarity"""
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0

def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between word sets"""
    words1 = set(preprocess_text(text1).split())
    words2 = set(preprocess_text(text2).split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)

def calculate_skill_match(resume_skills: List[str], required_skills: List[str], preferred_skills: List[str]) -> Dict:
    """Calculate skill match scores"""
    resume_skills_set = set([s.lower() for s in resume_skills])
    required_set = set([s.lower() for s in required_skills])
    preferred_set = set([s.lower() for s in preferred_skills])
    
    matched_required = resume_skills_set.intersection(required_set)
    matched_preferred = resume_skills_set.intersection(preferred_set)
    missing_required = required_set - resume_skills_set
    
    required_match_ratio = len(matched_required) / len(required_set) if required_set else 1.0
    preferred_match_ratio = len(matched_preferred) / len(preferred_set) if preferred_set else 1.0
    
    return {
        "matched_required": list(matched_required),
        "matched_preferred": list(matched_preferred),
        "missing_required": list(missing_required),
        "required_match_ratio": required_match_ratio,
        "preferred_match_ratio": preferred_match_ratio
    }

def calculate_experience_match(resume_years: int, required_years: int) -> float:
    """Calculate experience match score"""
    if required_years == 0:
        return 1.0
    if resume_years >= required_years:
        return 1.0
    return resume_years / required_years

def calculate_hybrid_score(
    sbert_score: float,
    tfidf_score: float,
    jaccard_score: float,
    skill_match: Dict,
    experience_match: float
) -> Dict:
    """Calculate hybrid score combining all methods"""
    weights = {
        "sbert": 0.30,
        "tfidf": 0.15,
        "jaccard": 0.10,
        "skills": 0.30,
        "experience": 0.15
    }
    
    skill_score = (skill_match["required_match_ratio"] * 0.7 + 
                   skill_match["preferred_match_ratio"] * 0.3)
    
    hybrid_score = (
        weights["sbert"] * sbert_score +
        weights["tfidf"] * tfidf_score +
        weights["jaccard"] * jaccard_score +
        weights["skills"] * skill_score +
        weights["experience"] * experience_match
    )
    
    return {
        "hybrid_score": round(hybrid_score, 4),
        "component_scores": {
            "sbert": round(sbert_score, 4),
            "tfidf": round(tfidf_score, 4),
            "jaccard": round(jaccard_score, 4),
            "skills": round(skill_score, 4),
            "experience": round(experience_match, 4)
        },
        "weights": weights
    }

def suggest_career_paths(skills: List[str], experience_years: int) -> List[Dict]:
    """Suggest career paths based on skills and experience"""
    career_paths = {
        "software_engineer": {
            "title": "Software Engineer",
            "progression": ["Junior Developer", "Software Engineer", "Senior Software Engineer", "Staff Engineer", "Principal Engineer"],
            "required_skills": ["python", "java", "javascript", "git", "sql"],
            "growth_areas": ["system design", "architecture", "leadership", "cloud computing"]
        },
        "data_scientist": {
            "title": "Data Scientist",
            "progression": ["Junior Data Analyst", "Data Scientist", "Senior Data Scientist", "Lead Data Scientist", "Head of Data Science"],
            "required_skills": ["python", "machine learning", "statistics", "sql", "data analysis"],
            "growth_areas": ["deep learning", "mlops", "business acumen", "stakeholder management"]
        },
        "frontend_developer": {
            "title": "Frontend Developer",
            "progression": ["Junior Frontend Dev", "Frontend Developer", "Senior Frontend Dev", "Frontend Architect", "Head of Frontend"],
            "required_skills": ["javascript", "react", "css", "html", "typescript"],
            "growth_areas": ["performance optimization", "accessibility", "design systems", "leadership"]
        },
        "devops_engineer": {
            "title": "DevOps Engineer",
            "progression": ["Junior DevOps", "DevOps Engineer", "Senior DevOps", "DevOps Architect", "Head of Platform"],
            "required_skills": ["docker", "kubernetes", "aws", "linux", "ci/cd"],
            "growth_areas": ["security", "sre", "architecture", "team leadership"]
        },
        "product_manager": {
            "title": "Product Manager",
            "progression": ["Associate PM", "Product Manager", "Senior PM", "Group PM", "VP of Product"],
            "required_skills": ["product management", "agile", "stakeholder management", "data analysis"],
            "growth_areas": ["strategy", "leadership", "technical depth", "market analysis"]
        },
        "ml_engineer": {
            "title": "ML Engineer",
            "progression": ["Junior ML Engineer", "ML Engineer", "Senior ML Engineer", "ML Architect", "Head of ML"],
            "required_skills": ["python", "machine learning", "tensorflow", "pytorch", "mlops"],
            "growth_areas": ["deep learning", "production systems", "research", "leadership"]
        }
    }
    
    suggestions = []
    skills_lower = set([s.lower() for s in skills])
    
    for path_key, path_info in career_paths.items():
        path_skills = set([s.lower() for s in path_info["required_skills"]])
        match_count = len(skills_lower.intersection(path_skills))
        match_ratio = match_count / len(path_skills) if path_skills else 0
        
        if match_ratio >= 0.4:
            progression = path_info["progression"]
            if experience_years < 2:
                current_level = 0
            elif experience_years < 4:
                current_level = 1
            elif experience_years < 7:
                current_level = 2
            elif experience_years < 10:
                current_level = 3
            else:
                current_level = 4
            
            current_level = min(current_level, len(progression) - 1)
            
            suggestions.append({
                "path": path_info["title"],
                "match_percentage": round(match_ratio * 100, 1),
                "current_position": progression[current_level],
                "next_position": progression[min(current_level + 1, len(progression) - 1)] if current_level < len(progression) - 1 else "Peak Level",
                "progression": progression,
                "skills_to_develop": path_info["growth_areas"][:3],
                "matched_skills": list(skills_lower.intersection(path_skills)),
                "missing_skills": list(path_skills - skills_lower)
            })
    
    suggestions.sort(key=lambda x: x["match_percentage"], reverse=True)
    return suggestions[:5]
