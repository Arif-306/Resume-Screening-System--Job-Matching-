# match_utils.py
import fitz           # PyMuPDF
import docx2txt
import re
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')  # small + fast


# --------------------------------------------------------
# File se text extract karna (PDF/DOCX/TXT supported)
def extract_text_from_file(path):
    ext = path.lower().split('.')[-1]
    if ext == 'pdf':
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif ext in ('docx', 'doc'):
        return docx2txt.process(path)
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


# --------------------------------------------------------
# NLP preprocessing
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess(text):
    doc = nlp(text)
    tokens = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
    return " ".join(tokens)


# --------------------------------------------------------
# Skills loading
def load_skills_from_file(path):
    # one skill per line, stored in lowercase
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]


def extract_skills_by_list(text, skills_list):
    text_low = text.lower()
    found = []
    for s in skills_list:
        if s.lower() in text_low:
            found.append(s.lower())
    return sorted(set(found))


# --------------------------------------------------------
# Extract experience years
def extract_years_experience(text):
    matches = re.findall(r'(\d+)\s*\+?\s*(?:years|yrs|y)', text.lower())
    years = [int(m) for m in matches]
    return max(years) if years else 0


# --------------------------------------------------------
# Semantic similarity
def semantic_similarity_score(text1, text2):
    emb = model.encode([text1, text2])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return float((sim + 1) / 2)


# --------------------------------------------------------
# Compute final match
def compute_match(jd_text, resume_text, skills_list, weights=None, required_experience=0):
    if weights is None:
        weights = {'skills': 0.6, 'semantic': 0.3, 'experience': 0.1}

    jd_skills = set(extract_skills_by_list(jd_text, skills_list))
    res_skills = set(extract_skills_by_list(resume_text, skills_list))

    # skill overlap ratio
    skill_ratio = 0.0
    if len(jd_skills) > 0:
        skill_ratio = len(jd_skills & res_skills) / len(jd_skills)

    # semantic
    sem = semantic_similarity_score(jd_text, resume_text)

    # experience
    cand_years = extract_years_experience(resume_text)
    if required_experience <= 0:
        exp_score = 1.0
    else:
        exp_score = min(cand_years / required_experience, 1.0)

    total = (weights['skills'] * skill_ratio +
             weights['semantic'] * sem +
             weights['experience'] * exp_score)

    return {
        'score_percent': round(total * 100, 2),
        'skill_ratio': skill_ratio,
        'matched_skills': sorted(list(jd_skills & res_skills)),
        'missing_skills': sorted(list(jd_skills - res_skills)),
        'semantic': round(sem, 3),
        'experience_years': cand_years,
        'experience_match': round(exp_score, 3)
    }


# --------------------------------------------------------
# Example run
if __name__ == "__main__":
    jd = "Looking for Python developer with Django, REST APIs, Docker, and 2+ years experience."
    resume = extract_text_from_file("resumes/arif_resume.pdf")

    # ✅ Skills load karna ab file se hoga
    skills = load_skills_from_file("skills.txt")

    result = compute_match(jd, resume, skills, required_experience=2)

    # Sirf missing skills print karo
    if result['missing_skills']:
        print("❌ These skills are missing from the resume:")
        for skill in result['missing_skills']:
            print("-", skill)
    else:
        print("✅ All required skills are present in the resume!")
