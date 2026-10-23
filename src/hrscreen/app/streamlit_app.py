"""Streamlit demo for the viva: rank resumes for a job, browse the stored evaluation results,
and run the survey analysis. Start it with ``hrscreen app`` or ``streamlit run <this file>``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from hrscreen.data import load_jobs, make_synthetic_resumes
from hrscreen.parsing import parse_resume, redact_pii
from hrscreen.parsing.extract import SUPPORTED_SUFFIXES, extract_text
from hrscreen.ranking import embeddings_available, make_ranker
from hrscreen.survey import analyze, make_synthetic_responses, standardise_responses

RESULTS_DIR = Path(os.environ.get("HRSCREEN_RESULTS", "results"))

st.set_page_config(page_title="AI Candidate Screening", layout="wide")
st.title("AI-based candidate screening and ranking")
st.caption("Track A ranks resumes for a job; the other tabs show the evaluation, the fairness audit and the HR survey analysis.")

tab_rank, tab_bench, tab_fair, tab_survey = st.tabs(["Screen candidates", "Benchmark results", "Fairness audit", "Survey analysis"])


# ------------------------------------------------------------------ screen candidates
def read_uploads(files) -> pd.DataFrame:
    rows = []
    for f in files:
        try:
            rows.append({"resume_id": Path(f.name).stem, "text": extract_text(f.getvalue(), f.name)})
        except Exception as exc:  # a corrupt PDF should not break the whole batch
            st.warning(f"Could not read {f.name}: {exc}")
    return pd.DataFrame(rows, columns=["resume_id", "text"])


with tab_rank:
    jobs = load_jobs()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("1. Job description")
        choice = st.selectbox("Pick a job", [j.title for j in jobs] + ["Write my own"])
        job = next((j for j in jobs if j.title == choice), None)
        jd_text = st.text_area("Job description", value=job.text if job else "", height=220)
    with right:
        st.subheader("2. Resumes")
        uploads = st.file_uploader("Upload resumes", type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES], accept_multiple_files=True)
        use_demo = st.checkbox("No files? use SYNTHETIC demo resumes", value=not uploads)
        models = ["tfidf", "rules"] + (["embeddings"] if embeddings_available() else [])
        model_name = st.selectbox("Ranking model", models, help="tfidf = keyword weighting; rules = ATS-style skill coverage; embeddings = meaning-based")
        top_n = st.slider("Show top", 3, 50, 10)

    if st.button("Rank candidates", type="primary"):
        if not jd_text.strip():
            st.error("Enter a job description first.")
        else:
            resumes = read_uploads(uploads) if uploads else (make_synthetic_resumes(60, seed=7) if use_demo else pd.DataFrame())
            if resumes.empty:
                st.error("Upload at least one resume or tick the demo option.")
            else:
                if not uploads:
                    st.warning("Showing SYNTHETIC demo resumes - generated from templates, not real candidates.")
                texts = resumes["text"].tolist()
                ranker = make_ranker(model_name)
                ranker.fit(texts, [jd_text])
                resumes["score"] = ranker.score(jd_text, texts)
                jd_skills = set(parse_resume(jd_text).skills)
                parsed = [parse_resume(t) for t in texts]
                resumes["matched_skills"] = [", ".join(sorted(jd_skills & set(p.skills))) for p in parsed]
                resumes["missing_skills"] = [", ".join(sorted(jd_skills - set(p.skills))) for p in parsed]
                resumes["years_experience"] = [p.years_experience for p in parsed]
                resumes["education"] = [p.education_level for p in parsed]
                ranked = resumes.sort_values("score", ascending=False).reset_index(drop=True)
                ranked.insert(0, "rank", ranked.index + 1)
                shown = ranked.drop(columns=["text"] + [c for c in ("category", "is_synthetic") if c in ranked.columns])
                st.dataframe(shown.head(top_n), use_container_width=True, hide_index=True)
                st.download_button("Download full ranking (CSV)", shown.to_csv(index=False), "ranking.csv", "text/csv")
                st.caption("Emails, phone numbers and URLs are removed before scoring. The score is a similarity, not a verdict: "
                           "a human should review every shortlist.")
                with st.expander("Why is the top candidate on top?"):
                    best = ranked.iloc[0]
                    st.write(f"**{best['resume_id']}** matches {len(best['matched_skills'].split(', ')) if best['matched_skills'] else 0} "
                             f"of {len(jd_skills)} skills the job mentions. Missing: {best['missing_skills'] or 'none'}.")
                    st.text(redact_pii(best["text"])[:1200])


# ------------------------------------------------------------------ stored results
def show_run_folder(pattern: str, empty_hint: str, md_name: str):
    found = sorted(RESULTS_DIR.glob(pattern))
    if not found:
        st.info(empty_hint)
        return
    pick = st.selectbox("Result set", found, format_func=lambda p: str(p.parent.relative_to(RESULTS_DIR)), key=pattern)
    md = pick.parent / md_name
    if md.exists():
        st.markdown(md.read_text(encoding="utf-8"))


with tab_bench:
    show_run_folder("**/benchmark_summary.csv", "No benchmark yet. Run: `hrscreen benchmark --data resume-atlas`", "benchmark.md")

with tab_fair:
    show_run_folder("**/fairness_summary.csv", "No fairness audit yet. Run: `hrscreen fairness --data resume-atlas`", "fairness.md")


# ------------------------------------------------------------------ survey analysis
with tab_survey:
    st.write("Upload the Google Forms CSV export (question titles must match `hrscreen survey-form`).")
    csv = st.file_uploader("Survey responses (CSV)", type=["csv"], key="survey_csv")
    demo = st.checkbox("Use SYNTHETIC demo responses (to try the analysis)", value=csv is None)
    raw = pd.read_csv(csv) if csv is not None else (make_synthetic_responses(80, seed=1, form_style=True) if demo else None)
    if raw is not None:
        try:
            clean, warns = standardise_responses(raw)
            result = analyze(clean, warns)
        except ValueError as exc:
            st.error(str(exc))
        else:
            if result.synthetic:
                st.error("SYNTHETIC DATA - planted relationships for testing the pipeline. Not a finding.")
            st.metric("Valid responses", result.n)
            st.subheader("Scales")
            st.dataframe(result.descriptives.drop(columns=["name"]).round(3), hide_index=True, use_container_width=True)
            for name, reg in result.regressions.items():
                st.subheader(reg["description"])
                st.caption(f"`{reg['formula']}`  |  n = {reg['n']}  |  R2 = {reg['r2']:.3f}")
                st.dataframe(reg["table"].round(4), hide_index=True, use_container_width=True)
            st.subheader("Correlations")
            st.dataframe(result.correlation_table.round(4), hide_index=True, use_container_width=True)
            for w in result.warnings:
                st.warning(w)
