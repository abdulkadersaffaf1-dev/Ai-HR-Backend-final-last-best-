"""
main.py — HR AI Agent FastAPI Backend
Run with: uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from typing import Any
import os
import joblib
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from models import (
    ScreenRequest, ScreenResponse, CategoryScore,
    MatchRequest, MatchResponse,
    RankRequest, RankResponse, CandidateScore,
    SkillGapRequest, SkillGapResponse, CourseRecommendation,
    FAQRequest, FAQResponse, FAQAlternative,
    InterviewResponse, InterviewQA,
    ReportResponse, TopJob,
    HealthResponse, CategoriesResponse,
)
from pipeline import (
    screen_resume,
    match_resume_to_job,
    rank_candidates,
    analyze_skill_gap,
    answer_faq,
    get_interview_questions,
    generate_report,
)
from utils import parse_skills

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Global state (loaded once at startup) ─────────────────────────────────────
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and datasets once at startup."""
    print("⏳ Loading ML models...")

    # Models
    from xgboost import XGBClassifier
    _m = XGBClassifier()
    _m.load_model(os.path.join(MODELS_DIR, "best_model.json"))
    state["model"] = _m
    state["tfidf"] = joblib.load(os.path.join(MODELS_DIR, "tfidf.pkl"))
    state["label_encoder"] = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    state["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Models loaded.")

    # Datasets
    print("⏳ Loading datasets...")
    state["resume_df"] = pd.read_csv(os.path.join(DATA_DIR, "Resume.csv"))
    state["jobs_df"] = pd.read_csv(os.path.join(DATA_DIR, "jobs_combined_extended.csv"))
    state["courses_df"] = pd.read_csv(os.path.join(DATA_DIR, "courses_combined.csv"))
    state["faq_df"] = pd.read_csv(os.path.join(DATA_DIR, "faq_clean.csv"))
    state["interview_df"] = pd.read_csv(os.path.join(DATA_DIR, "interview_clean.csv"))

    # Normalize columns
    state["resume_df"]["Category"] = state["resume_df"]["Category"].str.upper().str.strip()
    state["jobs_df"]["category"] = state["jobs_df"]["category"].str.upper().str.strip()
    state["courses_df"]["category"] = state["courses_df"]["category"].str.upper().str.strip()
    state["interview_df"]["category_clean"] = (
        state["interview_df"]["category"].str.upper().str.strip()
    )
    print("✅ Datasets loaded.")

    # Build known skills list from all job and course data
    all_skills: set[str] = set()
    for skills_raw in state["jobs_df"]["skills_clean"].dropna():
        all_skills.update(parse_skills(skills_raw))
    for skills_raw in state["courses_df"]["skills"].dropna():
        all_skills.update(parse_skills(skills_raw))
    state["all_known_skills"] = list(all_skills)

    # Embeddings are built lazily on first use (see get_resume_embeddings / get_faq_embeddings)
    state["resume_embeddings_cache"] = None
    state["faq_embeddings"] = None

    print("✅ All ready. Server is live.")

    yield  # App runs here

    print("Shutting down — clearing state.")
    state.clear()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HR AI Agent API",
    description="AI-powered HR recruitment automation backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Utility ────────────────────────────────────────────────────────────────────
def get_state(key: str):
    if key not in state:
        raise HTTPException(status_code=503, detail="Server is still initializing. Please retry.")
    return state[key]


def get_resume_embeddings():
    """Build resume embeddings on first use and cache them."""
    if state["resume_embeddings_cache"] is None:
        print("⏳ Building resume embeddings (first use, ~2 min)...")
        resume_texts = state["resume_df"]["Resume_str"].fillna("").tolist()
        state["resume_embeddings_cache"] = state["embedder"].encode(
            resume_texts, convert_to_tensor=True, batch_size=64, show_progress_bar=True
        )
        print("✅ Resume embeddings ready.")
    return state["resume_embeddings_cache"]


def get_faq_embeddings():
    """Build FAQ embeddings on first use and cache them."""
    if state["faq_embeddings"] is None:
        print("⏳ Building FAQ embeddings (first use)...")
        faq_questions = state["faq_df"]["Question"].fillna("").tolist()
        state["faq_embeddings"] = state["embedder"].encode(
            faq_questions, convert_to_tensor=True, batch_size=64, show_progress_bar=True
        )
        print("✅ FAQ embeddings ready.")
    return state["faq_embeddings"]


# ── 1. Health ──────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {"status": "ok"}


# ── 2. Categories ──────────────────────────────────────────────────────────────
@app.get("/categories", response_model=CategoriesResponse, tags=["System"])
def categories():
    le = get_state("label_encoder")
    return {"categories": sorted(le.classes_.tolist())}


# ── File text extractor ───────────────────────────────────────────────────────
def extract_text_from_file(file: UploadFile) -> str:
    content = file.file.read()
    filename = file.filename or ""
    if filename.lower().endswith(".pdf"):
        try:
            import io, pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            return " ".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            raise HTTPException(status_code=400, detail="Could not read PDF file.")
    else:
        return content.decode("utf-8", errors="ignore")


# ── 3. Resume Screening ────────────────────────────────────────────────────────
@app.post("/screen", response_model=ScreenResponse, tags=["Features"])
def screen(file: UploadFile = File(None), resume_text: str = Form(None)):
    """Upload a .pdf or .txt resume file, OR pass resume_text as a form field."""
    if file and file.filename:
        text = extract_text_from_file(file)
    elif resume_text:
        text = resume_text
    else:
        raise HTTPException(status_code=422, detail="Provide a file upload or resume_text.")
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from file.")
    result = screen_resume(text, get_state("tfidf"), get_state("model"), get_state("label_encoder"))
    return ScreenResponse(
        predicted_category=result["predicted_category"],
        confidence=result["confidence"],
        top3=[CategoryScore(**c) for c in result["top3"]],
    )


# ── 4. Job Matching ────────────────────────────────────────────────────────────
@app.post("/match", response_model=MatchResponse, tags=["Features"])
def match(job_title: str = Form(...), file: UploadFile = File(None), resume_text: str = Form(None)):
    """Upload a .pdf or .txt resume file + job_title, OR pass resume_text + job_title as form fields."""
    if file and file.filename:
        text = extract_text_from_file(file)
    elif resume_text:
        text = resume_text
    else:
        raise HTTPException(status_code=422, detail="Provide a file upload or resume_text.")
    result = match_resume_to_job(text, job_title, get_state("jobs_df"), get_state("embedder"), get_state("all_known_skills"))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No job found matching '{job_title}'.")
    return MatchResponse(**result)


# ── 5. Candidate Ranking ───────────────────────────────────────────────────────
@app.post("/rank", response_model=RankResponse, tags=["Features"])
def rank(req: RankRequest):
    result = rank_candidates(
        req.job_title,
        req.top_n,
        get_state("jobs_df"),
        get_state("resume_df"),
        get_resume_embeddings(),
        get_state("embedder"),
        get_state("all_known_skills"),
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"No job found matching '{req.job_title}'.")
    return RankResponse(
        job_title=result["job_title"],
        category=result["category"],
        candidates=[CandidateScore(**c) for c in result["candidates"]],
    )


# ── 6. Leaderboard (GET version of /rank) ─────────────────────────────────────
@app.get("/leaderboard", response_model=RankResponse, tags=["Features"])
def leaderboard(
    job_title: str = Query(..., description="Job title to search for"),
    top_n: int = Query(10, ge=1, le=100, description="Number of top candidates"),
):
    result = rank_candidates(
        job_title,
        top_n,
        get_state("jobs_df"),
        get_state("resume_df"),
        get_resume_embeddings(),
        get_state("embedder"),
        get_state("all_known_skills"),
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"No job found matching '{job_title}'.")
    return RankResponse(
        job_title=result["job_title"],
        category=result["category"],
        candidates=[CandidateScore(**c) for c in result["candidates"]],
    )


# ── 7. Skill Gap + Course Recommendations ─────────────────────────────────────
@app.post("/skill-gap", response_model=SkillGapResponse, tags=["Features"])
def skill_gap(req: SkillGapRequest):
    result = analyze_skill_gap(
        req.resume_text,
        req.job_title,
        get_state("jobs_df"),
        get_state("courses_df"),
        get_state("tfidf"),
        get_state("all_known_skills"),
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"No job found matching '{req.job_title}'.")
    return SkillGapResponse(
        matched_skills=result["matched_skills"],
        missing_skills=result["missing_skills"],
        recommended_courses=[CourseRecommendation(**c) for c in result["recommended_courses"]],
    )


# ── 8. FAQ Chatbot ─────────────────────────────────────────────────────────────
@app.post("/faq", response_model=FAQResponse, tags=["Features"])
def faq(req: FAQRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question cannot be empty.")
    result = answer_faq(
        req.question,
        get_state("faq_df"),
        get_faq_embeddings(),
        get_state("embedder"),
    )
    return FAQResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        alternatives=[FAQAlternative(**a) for a in result["alternatives"]],
    )


# ── 9. Interview Assistant ─────────────────────────────────────────────────────
@app.get("/interview/{category}", response_model=InterviewResponse, tags=["Features"])
def interview(category: str):
    result = get_interview_questions(category, get_state("interview_df"))
    if not result["questions"]:
        raise HTTPException(
            status_code=404,
            detail=f"No interview questions found for category '{category}'.",
        )
    return InterviewResponse(
        category=result["category"],
        questions=[InterviewQA(**q) for q in result["questions"]],
    )


# ── 10. Full Candidate Report ──────────────────────────────────────────────────
@app.get("/report/{resume_id}", response_model=ReportResponse, tags=["Features"])
def report(resume_id: int):
    result = generate_report(
        resume_id,
        get_state("resume_df"),
        get_state("jobs_df"),
        get_resume_embeddings(),
        get_state("embedder"),
        get_state("tfidf"),
        get_state("model"),
        get_state("label_encoder"),
        get_state("all_known_skills"),
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Resume ID {resume_id} not found.")
    return ReportResponse(
        resume_id=result["resume_id"],
        resume_text=result["resume_text"],
        predicted_category=result["predicted_category"],
        confidence=result["confidence"],
        top3=[CategoryScore(**c) for c in result["top3"]],
        top_jobs=[TopJob(**j) for j in result["top_jobs"]],
    )