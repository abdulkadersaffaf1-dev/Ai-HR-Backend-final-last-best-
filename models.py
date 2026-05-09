from pydantic import BaseModel
from typing import Optional


# ── Request Models ──────────────────────────────────────────────────────────

class ScreenRequest(BaseModel):
    resume_text: str

class MatchRequest(BaseModel):
    resume_text: str
    job_title: str

class RankRequest(BaseModel):
    job_title: str
    top_n: int = 10

class SkillGapRequest(BaseModel):
    resume_text: str
    job_title: str

class FAQRequest(BaseModel):
    question: str


# ── Response Models ─────────────────────────────────────────────────────────

class CategoryScore(BaseModel):
    category: str
    score: float

class ScreenResponse(BaseModel):
    predicted_category: str
    confidence: float
    top3: list[CategoryScore]

class MatchResponse(BaseModel):
    job_title: str
    category: str
    semantic_similarity: float
    skill_overlap_pct: float
    matched_skills: list[str]
    missing_skills: list[str]

class CandidateScore(BaseModel):
    resume_id: int
    category: str
    final_score: float
    semantic_sim: float
    percentile: float
    skill_pct: float
    category_fit: float

class RankResponse(BaseModel):
    job_title: str
    category: str
    candidates: list[CandidateScore]

class CourseRecommendation(BaseModel):
    skill: str
    course: str
    category: str

class SkillGapResponse(BaseModel):
    matched_skills: list[str]
    missing_skills: list[str]
    recommended_courses: list[CourseRecommendation]

class FAQAlternative(BaseModel):
    question: str
    answer: str
    score: float

class FAQResponse(BaseModel):
    answer: str
    confidence: float
    alternatives: list[FAQAlternative]

class InterviewQA(BaseModel):
    question: str
    answer_guideline: str

class InterviewResponse(BaseModel):
    category: str
    questions: list[InterviewQA]

class TopJob(BaseModel):
    title: str
    score: float

class ReportResponse(BaseModel):
    resume_id: int
    resume_text: str
    predicted_category: str
    confidence: float
    top3: list[CategoryScore]
    top_jobs: list[TopJob]

class HealthResponse(BaseModel):
    status: str

class CategoriesResponse(BaseModel):
    categories: list[str]
