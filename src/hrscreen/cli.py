"""Command line interface: ``hrscreen <command>`` (or ``python -m hrscreen <command>``)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .data import load_jobs, load_resumes, make_synthetic_resumes
from .data.fetch import DATASETS, dataset_csv, fetch_dataset
from .data.jobs import relevance_from_recruiter_labels
from .ranking import MODEL_NAMES, EmbeddingRanker, embeddings_available, make_ranker


# ---------------------------------------------------------------- helpers

def resolve_data(source: str, sample: int | None, seed: int) -> tuple[pd.DataFrame, dict, str]:
    """Turn --data (a dataset name, 'synthetic', or a CSV path) into (resumes, cleaning report, description)."""
    if source == "synthetic":
        df = make_synthetic_resumes(sample or 600, seed=seed)
        return df, {"n_input": len(df), "n_output": len(df)}, "SYNTHETIC template resumes (pipeline test only, not a result)"
    if source in DATASETS:
        csv = dataset_csv(source)
        if not csv.exists():
            raise SystemExit(f"{csv} not found. Run:  hrscreen fetch-data --dataset {source}")
        spec = DATASETS[source]
        df, report = load_resumes(csv, text_col=spec["text_col"], label_col=spec["label_col"])
        desc = f"{source} ({spec['licence']})"
    else:
        df, report = load_resumes(source)
        desc = f"user file {source}"
    if sample and sample < len(df):
        frac = sample / len(df)
        if "category" in df.columns:
            df = df.groupby("category", group_keys=False).sample(frac=frac, random_state=seed)
        else:
            df = df.sample(sample, random_state=seed)
        df = df.reset_index(drop=True)
        desc += f", stratified subsample of {len(df)}"
    return df, report, desc


def make_embedder(models: list[str], cache: str | None) -> EmbeddingRanker | None:
    if not any("emb" in m for m in models):
        return None
    if not embeddings_available():
        raise SystemExit("Embedding models need sentence-transformers:  pip install 'hrscreen[embeddings]'")
    return EmbeddingRanker(cache_path=cache or "data/cache/embeddings_minilm.npz", verbose=True)


# ---------------------------------------------------------------- commands

def cmd_fetch(args):
    for name in ([args.dataset] if args.dataset else ["resume-atlas"]):
        out = fetch_dataset(name)
        print(f"{name}: {DATASETS[name]['note']}\n  licence: {DATASETS[name]['licence']}\n  saved to {out}")


def cmd_synth_resumes(args):
    df = make_synthetic_resumes(args.n, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC resumes to {args.out}")


def cmd_benchmark(args):
    from .evaluation import run_benchmark

    jobs = load_jobs(args.jobs)
    resumes, report, desc = resolve_data(args.data, args.sample, args.seed)
    models = [m.strip() for m in args.models.split(",")]
    relevance, label_source = None, "dataset category (proxy for a recruiter shortlist)"
    if args.recruiter_labels:
        resumes, relevance = relevance_from_recruiter_labels(args.recruiter_labels, resumes, jobs)
        label_source = f"recruiter ratings from {args.recruiter_labels}"
    result = run_benchmark(resumes, jobs, models, n_splits=args.folds, seed=args.seed,
                           embedder=make_embedder(models, args.embedding_cache), relevance=relevance,
                           label_source=label_source, manual_seconds_per_resume=args.manual_seconds)
    result.meta.update(data_description=desc, cleaning=report)
    out = result.save(args.out)
    print(result.to_markdown())
    print(f"Saved to {out}")


def cmd_fairness(args):
    from .fairness import run_fairness_audit

    jobs = load_jobs(args.jobs)
    resumes, report, desc = resolve_data(args.data, args.sample, args.seed)
    result = run_fairness_audit(resumes, jobs, args.model, n_splits=args.folds, seed=args.seed,
                                shortlist_frac=args.shortlist_frac,
                                embedder=make_embedder([args.model], args.embedding_cache),
                                extra_attributes=tuple(args.attribute or ()))
    result.meta["data_description"] = desc
    result.save(args.out)
    print(result.to_markdown())
    print(f"Saved to {args.out}")


def cmd_rank(args):
    jobs = {j.id: j for j in load_jobs()}
    if args.job:
        if args.job not in jobs:
            raise SystemExit(f"Unknown job '{args.job}'. Choose from {sorted(jobs)}")
        jd_text = jobs[args.job].text
    else:
        jd_text = Path(args.jd).read_text(encoding="utf-8")
    resumes, _ = load_resumes(args.resumes)
    ranker = make_ranker(args.model, embedder=make_embedder([args.model], None))
    if ranker.supervised:
        raise SystemExit("Supervised models need labelled training data; use tfidf, rules or embeddings here.")
    ranker.fit(resumes["text"].tolist(), [jd_text])
    resumes = resumes.assign(score=ranker.score(jd_text, resumes["text"].tolist())).sort_values("score", ascending=False)
    top = resumes.head(args.top)[["resume_id", "score"]]
    print(top.to_string(index=False))
    if args.out:
        resumes[["resume_id", "score"]].to_csv(args.out, index=False)


def cmd_survey_form(args):
    from .survey import questionnaire

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(questionnaire.to_markdown(), encoding="utf-8")
    pd.DataFrame(questionnaire.to_csv_rows()).to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print(f"Wrote {args.out} and {Path(args.out).with_suffix('.csv')}")


def cmd_survey_synth(args):
    from .survey import make_synthetic_responses

    df = make_synthetic_responses(args.n, args.seed, form_style=args.form_style)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} SYNTHETIC survey responses to {args.out} (pipeline test only, not a result)")


def cmd_survey_analyze(args):
    from .survey import analyze, load_form_csv

    clean, warnings = load_form_csv(args.csv)
    result = analyze(clean, warnings)
    result.save(args.out)
    print(result.to_markdown())
    print(f"Saved to {args.out}")


def cmd_app(args):
    app = Path(__file__).parent / "app" / "streamlit_app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *args.streamlit_args]))


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hrscreen", description="AI candidate screening + HR impact study toolkit.")
    p.add_argument("--version", action="version", version=f"hrscreen {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    data_help = "'synthetic', a dataset name (" + ", ".join(DATASETS) + "), or a path to a resume CSV"

    s = sub.add_parser("fetch-data", help="download a public resume dataset into data/raw/")
    s.add_argument("--dataset", choices=sorted(DATASETS), help="default: resume-atlas")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("synth-resumes", help="write SYNTHETIC resumes to a CSV (offline demos/tests)")
    s.add_argument("--n", type=int, default=600)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/synthetic/resumes.csv")
    s.set_defaults(func=cmd_synth_resumes)

    s = sub.add_parser("benchmark", help="cross-validated comparison of ranking models")
    s.add_argument("--data", default="resume-atlas", help=data_help)
    s.add_argument("--models", default="rules,tfidf,xgboost", help=f"comma list from {', '.join(MODEL_NAMES)}")
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--sample", type=int, help="use a stratified subsample of this many resumes")
    s.add_argument("--jobs", help="custom jobs JSON (same format as the bundled jobs.json)")
    s.add_argument("--recruiter-labels", help="CSV (resume_id, job_id, rating) of human recruiter ratings")
    s.add_argument("--manual-seconds", type=float, help="measured manual screening seconds per resume, to report time saved")
    s.add_argument("--embedding-cache", help="path of the .npz embedding cache")
    s.add_argument("--out", default="results/benchmark")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("fairness", help="bias audit of one model's shortlists")
    s.add_argument("--data", default="resume-atlas", help=data_help)
    s.add_argument("--model", default="tfidf", choices=MODEL_NAMES)
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--sample", type=int)
    s.add_argument("--jobs")
    s.add_argument("--shortlist-frac", type=float, default=0.10, help="share of each pool that is shortlisted")
    s.add_argument("--attribute", action="append", help="extra column of the resume CSV to audit (repeatable)")
    s.add_argument("--embedding-cache")
    s.add_argument("--out", default="results/fairness")
    s.set_defaults(func=cmd_fairness)

    s = sub.add_parser("rank", help="rank a folder/CSV of resumes for one job")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--jd", help="text file with a job description")
    g.add_argument("--job", help="id of a bundled job (e.g. data-science)")
    s.add_argument("--resumes", required=True, help="folder of PDF/DOCX/TXT files, or a CSV")
    s.add_argument("--model", default="tfidf", choices=MODEL_NAMES)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--out")
    s.set_defaults(func=cmd_rank)

    s = sub.add_parser("survey-form", help="write the questionnaire text for Google Forms")
    s.add_argument("--out", default="docs/questionnaire.md")
    s.set_defaults(func=cmd_survey_form)

    s = sub.add_parser("survey-synth", help="write SYNTHETIC survey responses (pipeline test only)")
    s.add_argument("--n", type=int, default=80)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--form-style", action="store_true", help="use question text as headers, like Google Forms")
    s.add_argument("--out", default="data/synthetic/survey_responses.csv")
    s.set_defaults(func=cmd_survey_synth)

    s = sub.add_parser("survey-analyze", help="analyse a Google Forms CSV export")
    s.add_argument("--csv", required=True)
    s.add_argument("--out", default="results/survey")
    s.set_defaults(func=cmd_survey_analyze)

    s = sub.add_parser("app", help="launch the Streamlit demo")
    s.add_argument("streamlit_args", nargs="*", help="extra arguments passed to `streamlit run`")
    s.set_defaults(func=cmd_app)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
