import streamlit as st
from match_utils import extract_text_from_file, compute_match, load_skills_from_file

st.title("Resume Screening — Job Matching App")

# Job description input
jd = st.text_area("Paste Job Description here")

# Upload multiple resumes
uploaded_files = st.file_uploader("Upload resumes (PDF/DOCX)", accept_multiple_files=True)

# Load skills from file
skills = load_skills_from_file("skills.txt")  # skills.txt file create karna hoga

# Run matching button
if st.button("Run Matching") and jd and uploaded_files:
    for f in uploaded_files:
        # Save temporary file
        with open(f.name, "wb") as out:
            out.write(f.getbuffer())

        # Extract text
        text = extract_text_from_file(f.name)

        # Compute match
        res = compute_match(jd, text, skills, required_experience=2)

        # Show results
        st.write(f"**{f.name}** — Score: {res['score_percent']}%")
        st.write("Matched skills:", res['matched_skills'])
        st.write("Missing skills:", res['missing_skills'])
        st.write("---")
