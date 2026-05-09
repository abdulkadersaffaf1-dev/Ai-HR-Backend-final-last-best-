"""
pipeline.py — All ML logic functions.
All heavy objects (models, embeddings, dataframes) are passed in as arguments
so this module stays stateless and easy to test.
"""

import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sentence_transformers import util as st_util
import torch

from utils import industrial_clean, parse_skills, skill_overlap, extract_skills_from_text


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_best_job(title_query: str, jobs_df: pd.DataFrame) -> pd.Series | None:
    """Find the best matching job row for a free-text title query."""
    title_query_lower = title_query.lower().strip()

    # 1. Exact title match
    exact = jobs_df[jobs_df["title"].str.lower() == title_query_lower]
    if not exact.empty:
        return exact.iloc[0]

    # 2. Title contains full query
    partial = jobs_df[jobs_df["title"].str.lower().str.contains(
        re.escape(title_query_lower), na=False)]
    if not partial.empty:
        return partial.iloc[0]

    # 3. All query words appear in title (word-level match)
    query_words = [w for w in title_query_lower.split() if len(w) > 2]
    if query_words:
        mask = jobs_df["title"].str.lower().apply(
            lambda t: all(w in t for w in query_words)
        )
        word_match = jobs_df[mask]
        if not word_match.empty:
            return word_match.iloc[0]

    # 4. Any query word appears in title
    if query_words:
        mask = jobs_df["title"].str.lower().apply(
            lambda t: any(w in t for w in query_words)
        )
        any_match = jobs_df[mask]
        if not any_match.empty:
            return any_match.iloc[0]

    # 5. Category match
    cat_match = jobs_df[jobs_df["category"].str.lower().str.contains(
        re.escape(title_query_lower), na=False)]
    if not cat_match.empty:
        return cat_match.iloc[0]

    return None


def compute_semantic_sim(text_a: str, text_b: str, embedder) -> float:
    """Cosine similarity between two raw texts using sentence transformer."""
    emb_a = embedder.encode(text_a, convert_to_tensor=True)
    emb_b = embedder.encode(text_b, convert_to_tensor=True)
    return float(st_util.cos_sim(emb_a, emb_b)[0][0])


def compute_percentile(score: float, all_scores: np.ndarray) -> float:
    """Return percentile rank of score within all_scores (0.0–1.0)."""
    if len(all_scores) == 0:
        return 0.0
    return float(np.mean(all_scores <= score))


# ── Feature 1: Resume Screening ───────────────────────────────────────────────

def screen_resume(resume_text: str, tfidf, model, label_encoder) -> dict:
    """Clean → TF-IDF → XGBoost predict. Returns top-3 categories + confidence."""
    import xgboost as xgb
    cleaned = industrial_clean(resume_text)
    vec = tfidf.transform([cleaned])
    # Use booster directly to avoid predict_proba version mismatch
    booster = model.get_booster()
    dmat = xgb.DMatrix(vec)
    proba = booster.predict(dmat)[0]  # shape (24,)
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {"category": label_encoder.classes_[i], "score": round(float(proba[i]), 4)}
        for i in top3_idx
    ]
    return {
        "predicted_category": top3[0]["category"],
        "confidence": top3[0]["score"],
        "top3": top3,
    }


# ── Feature 2: Job Matching ───────────────────────────────────────────────────

def match_resume_to_job(
    resume_text: str,
    job_title: str,
    jobs_df: pd.DataFrame,
    embedder,
    all_known_skills: list[str],
) -> dict | None:
    """Semantic similarity + skill overlap between resume and best matching job."""
    job_row = find_best_job(job_title, jobs_df)
    if job_row is None:
        return None

    job_raw = str(job_row.get("job_text", ""))
    sem_sim = compute_semantic_sim(resume_text, job_raw, embedder)

    resume_skills = extract_skills_from_text(resume_text, all_known_skills)
    job_skills = parse_skills(job_row.get("skills_clean", ""))
    matched, missing, pct = skill_overlap(resume_skills, job_skills)

    return {
        "job_title": str(job_row.get("title", job_title)),
        "category": str(job_row.get("category", "")),
        "semantic_similarity": round(sem_sim, 4),
        "skill_overlap_pct": round(pct, 4),
        "matched_skills": matched,
        "missing_skills": missing,
    }


# ── Feature 3: Candidate Ranking ─────────────────────────────────────────────

def rank_candidates(
    job_title: str,
    top_n: int,
    jobs_df: pd.DataFrame,
    resume_df: pd.DataFrame,
    resume_embeddings_cache,  # tensor or numpy array
    embedder,
    all_known_skills: list[str],
) -> dict | None:
    """
    Hybrid scoring:
      final = 0.45*semantic_sim + 0.35*percentile + 0.15*skill_pct + 0.05*category_fit
    """
    job_row = find_best_job(job_title, jobs_df)
    if job_row is None:
        return None

    job_raw = str(job_row.get("job_text", ""))
    job_category = str(job_row.get("category", "")).upper()
    job_skills = parse_skills(job_row.get("skills_clean", ""))

    job_emb = embedder.encode(job_raw, convert_to_tensor=True)

    # Compute semantic similarities for all resumes at once
    if isinstance(resume_embeddings_cache, torch.Tensor):
        sims = st_util.cos_sim(resume_embeddings_cache, job_emb).cpu().numpy().flatten()
    else:
        sims = np.array([
            compute_semantic_sim(str(r), job_raw, embedder)
            for r in resume_df["Resume_str"].fillna("")
        ])

    candidates = []
    for idx, row in resume_df.iterrows():
        sem_sim = float(sims[idx if isinstance(idx, int) else resume_df.index.get_loc(idx)])
        resume_skills = extract_skills_from_text(str(row.get("Resume_str", "")), all_known_skills)
        _, _, skill_pct = skill_overlap(resume_skills, job_skills)
        category_fit = 1.0 if str(row.get("Category", "")).upper() == job_category else 0.0
        candidates.append({
            "resume_id": int(row.get("ID", idx)),
            "category": str(row.get("Category", "")),
            "sem_sim": sem_sim,
            "skill_pct": skill_pct,
            "category_fit": category_fit,
        })

    # Compute percentiles
    all_sims = np.array([c["sem_sim"] for c in candidates])
    for c in candidates:
        c["percentile"] = compute_percentile(c["sem_sim"], all_sims)
        c["final_score"] = round(
            0.45 * c["sem_sim"]
            + 0.35 * c["percentile"]
            + 0.15 * c["skill_pct"]
            + 0.05 * c["category_fit"],
            4,
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    top = candidates[:top_n]

    return {
        "job_title": str(job_row.get("title", job_title)),
        "category": job_category,
        "candidates": [
            {
                "resume_id": c["resume_id"],
                "category": c["category"],
                "final_score": c["final_score"],
                "semantic_sim": round(c["sem_sim"], 4),
                "percentile": round(c["percentile"], 4),
                "skill_pct": round(c["skill_pct"], 4),
                "category_fit": c["category_fit"],
            }
            for c in top
        ],
    }


# ── Feature 4: Skill Gap + Course Recommendation ─────────────────────────────

def analyze_skill_gap(
    resume_text: str,
    job_title: str,
    jobs_df: pd.DataFrame,
    courses_df: pd.DataFrame,
    tfidf,
    all_known_skills: list[str],
) -> dict | None:
    job_row = find_best_job(job_title, jobs_df)
    if job_row is None:
        return None

    job_skills = parse_skills(job_row.get("skills_clean", ""))
    resume_skills = extract_skills_from_text(resume_text, all_known_skills)
    matched, missing, _ = skill_overlap(resume_skills, job_skills)

    # Course recommendations for each missing skill
    recommended = []
    for skill in missing[:10]:  # cap at 10
        skill_vec = tfidf.transform([industrial_clean(skill)])
        if "clean_course" in courses_df.columns:
            course_vecs = tfidf.transform(courses_df["clean_course"].fillna(""))
        else:
            course_vecs = tfidf.transform(courses_df.get("course_text", courses_df.iloc[:, 0]).fillna("").apply(industrial_clean))

        sims = sk_cosine(skill_vec, course_vecs).flatten()
        best_idx = int(np.argmax(sims))
        best_row = courses_df.iloc[best_idx]
        course_name = str(
            best_row.get("course_name", best_row.get("title", best_row.get("course_text", "Unknown Course")))
        )[:120]
        recommended.append({
            "skill": skill,
            "course": course_name,
            "category": str(best_row.get("category", "")),
        })

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "recommended_courses": recommended,
    }


# ── Feature 5: HR FAQ Chatbot ─────────────────────────────────────────────────

def answer_faq(
    question: str,
    faq_df: pd.DataFrame,
    faq_embeddings,  # pre-encoded tensor
    embedder,
) -> dict:
    q_emb = embedder.encode(question, convert_to_tensor=True)
    sims = st_util.cos_sim(q_emb, faq_embeddings).cpu().numpy().flatten()
    top3_idx = np.argsort(sims)[::-1][:3]

    best = faq_df.iloc[top3_idx[0]]
    alts = []
    for i in top3_idx[1:]:
        row = faq_df.iloc[i]
        alts.append({
            "question": str(row.get("Question", "")),
            "answer": str(row.get("Answer", "")),
            "score": round(float(sims[i]), 4),
        })

    return {
        "answer": str(best.get("Answer", "")),
        "confidence": round(float(sims[top3_idx[0]]), 4),
        "alternatives": alts,
    }


# ── Feature 6: Interview Assistant ───────────────────────────────────────────

def get_interview_questions(category: str, interview_df: pd.DataFrame) -> dict:
    cat_upper = category.upper().strip()
    filtered = interview_df[
        interview_df["category_clean"].str.upper().str.strip() == cat_upper
    ]
    questions = []
    for _, row in filtered.iterrows():
        questions.append({
            "question": str(row.get("question", "")),
            "answer_guideline": str(row.get("answer_guideline", "")),
        })
    return {"category": category, "questions": questions}


# ── Feature 7: Full Report ────────────────────────────────────────────────────

def generate_report(
    resume_id: int,
    resume_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    resume_embeddings_cache,
    embedder,
    tfidf,
    model,
    label_encoder,
    all_known_skills: list[str],
) -> dict | None:
    row = resume_df[resume_df["ID"] == resume_id]
    if row.empty:
        return None
    row = row.iloc[0]
    resume_text = str(row.get("Resume_str", ""))

    screen_result = screen_resume(resume_text, tfidf, model, label_encoder)

    # Top 5 job matches
    top_jobs = []
    sample_jobs = jobs_df.drop_duplicates(subset=["title"]).head(50)
    job_embs = embedder.encode(
        sample_jobs["job_text"].fillna("").tolist(), convert_to_tensor=True
    )
    resume_emb = embedder.encode(resume_text, convert_to_tensor=True)
    sims = st_util.cos_sim(resume_emb, job_embs).cpu().numpy().flatten()
    top5_idx = np.argsort(sims)[::-1][:5]
    for i in top5_idx:
        jrow = sample_jobs.iloc[i]
        top_jobs.append({
            "title": str(jrow.get("title", "")),
            "score": round(float(sims[i]), 4),
        })

    return {
        "resume_id": resume_id,
        "resume_text": resume_text[:2000],
        "predicted_category": screen_result["predicted_category"],
        "confidence": screen_result["confidence"],
        "top3": screen_result["top3"],
        "top_jobs": top_jobs,
    }