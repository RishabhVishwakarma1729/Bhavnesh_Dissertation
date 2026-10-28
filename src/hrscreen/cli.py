"""Command line interface: ``hrscreen <command>`` (or ``python -m hrscreen <command>``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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


def make_embedder(models: list[str], cache: str | None, device: str = "auto",
                  model_name: str | None = None) -> EmbeddingRanker | None:
    if not any("emb" in m for m in models):
        return None
    if not embeddings_available():
        raise SystemExit("Embedding models need sentence-transformers:  pip install 'hrscreen[embeddings]'")
    kwargs = {"model_name": model_name} if model_name else {}
    return EmbeddingRanker(cache_path=cache or "data/cache/embeddings_minilm.npz", verbose=True, device=device, **kwargs)


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


def cmd_export_splits(args):
    """Write the exact train/test split the benchmark uses (stratified k-fold), so it can be inspected and cited."""
    import sklearn

    from .evaluation import make_splits

    resumes, report, desc = resolve_data(args.data, None, args.seed)
    folds = make_splits(resumes, args.folds, args.seed)
    fold_of = np.full(len(resumes), -1)
    for k, (_, test_idx) in enumerate(folds):
        fold_of[test_idx] = k                       # each resume is in the TEST set of exactly one fold
    table = pd.DataFrame({"resume_id": resumes["resume_id"].to_numpy(), "fold": fold_of})
    if "category" in resumes.columns:
        table.insert(1, "category", resumes["category"].to_numpy())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "fold_assignments.csv", index=False)
    (out / "splits_meta.json").write_text(json.dumps({
        "data": desc, "cleaning": report, "n_resumes": len(resumes), "n_folds": args.folds, "seed": args.seed,
        "method": "StratifiedKFold(shuffle=True) on category (rare categories pooled), KFold when unlabelled",
        "scikit_learn": sklearn.__version__,
        "how_to_read": "fold k = the test set of run k; the other folds are its training set. resume_id R<row> is the 0-based row of the original CSV.",
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'fold_assignments.csv'}: {len(table)} resumes, test-set sizes per fold {np.bincount(fold_of).tolist()}")

    if args.materialize:
        root = Path(args.materialize)
        for k in range(args.folds):
            (root / f"fold_{k}").mkdir(parents=True, exist_ok=True)
            is_test = fold_of == k
            resumes.loc[~is_test].to_csv(root / f"fold_{k}" / "train.csv", index=False)
            resumes.loc[is_test].to_csv(root / f"fold_{k}" / "test.csv", index=False)
        print(f"Wrote fold_0 ... fold_{args.folds - 1} (train.csv + test.csv with the full text) under {root}")


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
                           embedder=make_embedder(models, args.embedding_cache, args.device, args.embedding_model),
                           relevance=relevance,
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
                                embedder=make_embedder([args.model], args.embedding_cache, args.device, args.embedding_model),
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


def cmd_web(args):
    import os

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hrscreen.web.settings")
    if args.debug:
        os.environ["HRSCREEN_DEBUG"] = "1"   # Django error pages with tracebacks
    try:
        import django  # noqa: F401
    except ImportError:
        raise SystemExit("The web UI needs Django:  pip install 'hrscreen[web]'")
    from django.core.management import execute_from_command_line

    execute_from_command_line(["hrscreen", "migrate", "--run-syncdb", "-v", "0"])   # creates the local SQLite file on first run
    print(f"hrscreen web UI: http://{args.host}:{args.port}/   (Ctrl+C to stop)", flush=True)
    execute_from_command_line(["hrscreen", "runserver", f"{args.host}:{args.port}", "--noreload"])


# ---------------------------------------------------------------- parser

def add_embedding_options(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--embedding-cache", help="path of the .npz embedding cache")
    sub.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                     help="where to run the embedding model (auto = GPU if one works)")
    sub.add_argument("--embedding-model", help="Hugging Face model id or a local model folder (default: all-MiniLM-L6-v2)")


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

    s = sub.add_parser("export-splits", help="write the exact train/test folds the benchmark uses")
    s.add_argument("--data", default="resume-atlas", help=data_help)
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="results/resume-atlas/splits", help="folder for fold_assignments.csv (small; safe to commit)")
    s.add_argument("--materialize", metavar="DIR", help="also write fold_k/train.csv and fold_k/test.csv with the full text (large)")
    s.set_defaults(func=cmd_export_splits)

    s = sub.add_parser("benchmark", help="cross-validated comparison of ranking models")
    s.add_argument("--data", default="resume-atlas", help=data_help)
    s.add_argument("--models", default="rules,tfidf,xgboost", help=f"comma list from {', '.join(MODEL_NAMES)}")
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--sample", type=int, help="use a stratified subsample of this many resumes")
    s.add_argument("--jobs", help="custom jobs JSON (same format as the bundled jobs.json)")
    s.add_argument("--recruiter-labels", help="CSV (resume_id, job_id, rating) of human recruiter ratings")
    s.add_argument("--manual-seconds", type=float, help="measured manual screening seconds per resume, to report time saved")
    add_embedding_options(s)
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
    add_embedding_options(s)
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

    s = sub.add_parser("web", help="launch the Django web UI")
    s.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 so resumes stay on this computer")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--debug", action="store_true", help="show error pages with tracebacks")
    s.set_defaults(func=cmd_web)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
