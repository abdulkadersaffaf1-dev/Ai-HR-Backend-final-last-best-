import re
import contractions
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def industrial_clean(text: str) -> str:
    """Full NLP cleaning pipeline matching the notebook exactly."""
    if not text or (hasattr(text, '__class__') and text.__class__.__name__ == 'float'):
        return ""
    text = contractions.fix(str(text).lower())
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(
        r"http\S+|www\S+|\S+@\S+|(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",
        " ", text,
    )
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 1
    ]
    return " ".join(tokens)


def parse_skills(skills_raw) -> list[str]:
    """
    Parse skills from space-separated strings (as used in jobs_combined_extended.csv).
    Preserves slash-joined skills like tefl/tesol, ielts/toefl prep.
    Also handles comma/semicolon separated formats.
    """
    if not skills_raw:
        return []
    s = str(skills_raw).strip()
    s = re.sub(r"[\[\]'\"]", "", s)

    # Comma or semicolon separated
    if "," in s or ";" in s:
        parts = re.split(r"[,;]", s)
        return [p.strip().lower() for p in parts if p.strip()]

    # Space-separated — use known skill boundaries
    # Strategy: tokenize by space, then greedily merge tokens that form known patterns
    # Keep slash-joined tokens as single skills (e.g. tefl/tesol, ielts/toefl)
    tokens = s.lower().split()

    # Known 2-word skill pairs from your dataset
    TWO_WORD = {
        ("curriculum", "development"), ("classroom", "management"),
        ("differentiated", "instruction"), ("math", "proficiency"),
        ("student", "counseling"), ("child", "development"),
        ("lesson", "planning"), ("creative", "teaching"),
        ("ms", "office"), ("activity", "design"),
        ("iep", "development"), ("behavior", "management"),
        ("assistive", "technology"), ("machine", "learning"),
        ("deep", "learning"), ("data", "analysis"), ("data", "science"),
        ("project", "management"), ("product", "management"),
        ("user", "research"), ("user", "experience"), ("user", "interface"),
        ("adobe", "xd"), ("microsoft", "office"), ("microsoft", "excel"),
        ("power", "bi"), ("sql", "server"), ("node", "js"),
        ("human", "resources"), ("talent", "acquisition"),
        ("performance", "management"), ("customer", "service"),
        ("risk", "management"), ("financial", "analysis"),
        ("financial", "reporting"), ("business", "analysis"),
        ("social", "media"), ("email", "marketing"),
        ("content", "management"), ("software", "engineering"),
        ("web", "development"), ("mobile", "development"),
        ("lms", "student"),  # edge case
    }

    # 3-word skills
    THREE_WORD = {
        ("ielts", "toefl", "prep"),
        ("ielts/toefl", "prep", None),  # handle slash variant
    }

    skills = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        # Try 3-word match
        if i + 2 < len(tokens) and (tokens[i], tokens[i+1], tokens[i+2]) in THREE_WORD:
            skills.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
            i += 3
        # Try 2-word match
        elif i + 1 < len(tokens) and (tokens[i], tokens[i+1]) in TWO_WORD:
            skills.append(f"{tokens[i]} {tokens[i+1]}")
            i += 2
        else:
            # Single token — keep as-is (preserves tefl/tesol, ielts/toefl)
            if len(t) >= 2:
                skills.append(t)
            i += 1

    return list(dict.fromkeys(skills))  # deduplicate preserving order


def skill_overlap(resume_skills: list[str], job_skills: list[str]) -> tuple[list[str], list[str], float]:
    """Return matched skills, missing skills, and overlap percentage."""
    if not job_skills:
        return [], [], 0.0
    resume_set = set(resume_skills)
    job_set = set(job_skills)
    matched = list(resume_set & job_set)
    missing = list(job_set - resume_set)
    pct = len(matched) / len(job_set) if job_set else 0.0
    return matched, missing, pct


def extract_skills_from_text(text: str, all_known_skills: list[str]) -> list[str]:
    """Extract skills from raw resume text by matching against known skill list."""
    text_lower = text.lower()
    found = []
    for skill in all_known_skills:
        if skill.lower() in text_lower:
            found.append(skill.lower())
    return list(set(found))