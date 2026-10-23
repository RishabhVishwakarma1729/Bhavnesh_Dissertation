"""HTTP views. Each one gathers input, calls a function from services.py, and renders a template."""

from __future__ import annotations

import csv

from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from hrscreen.survey import questionnaire as q

from . import services
from .models import ScreeningRun, SurveyRun


# ---------------------------------------------------------------------------------- home / about

@require_GET
def home(request):
    sets = services.result_sets("benchmark_summary.csv")
    return render(request, "hrscreen_web/home.html", {
        "nav": "home",
        "n_runs": ScreeningRun.objects.count(),
        "n_surveys": SurveyRun.objects.count(),
        "has_benchmark": bool(sets),
        "has_fairness": bool(services.result_sets("fairness_summary.csv")),
        "recent_runs": ScreeningRun.objects.all()[:4],
    })


@require_GET
def about(request):
    return render(request, "hrscreen_web/about.html", {"nav": "about"})


# ---------------------------------------------------------------------------------- screening

def _screen_context(request, **extra):
    return {"nav": "screen", "jobs": services.job_catalogue(), "models": services.available_models(),
            "max_resumes": settings.MAX_RESUMES, "max_embedding": settings.MAX_EMBEDDING_RESUMES,
            "max_mb": settings.MAX_RESUME_BYTES // (1024 * 1024), **extra}


@require_http_methods(["GET", "POST"])
def screen(request):
    if request.method == "GET":
        default_job = services.job_catalogue()[0]
        return render(request, "hrscreen_web/screen.html", _screen_context(
            request, form={"job": default_job["id"], "jd_text": default_job["text"], "model": "tfidf"}))

    form = {"job": request.POST.get("job", "custom"), "jd_text": request.POST.get("jd_text", "").strip(),
            "job_title": request.POST.get("job_title", "").strip(), "model": request.POST.get("model", "tfidf"),
            "demo": request.POST.get("demo") == "on"}
    catalogue = {j["id"]: j for j in services.job_catalogue()}
    if form["job"] in catalogue:
        base = catalogue[form["job"]]
        title = base["title"] + ("" if form["jd_text"] == base["text"] else " (edited)")
    else:
        title = form["job_title"] or "Custom job"
    files = request.FILES.getlist("resumes")
    try:
        if not files and not form["demo"]:
            raise services.UserError("Upload at least one resume, or tick the demo option to try it with synthetic resumes.")
        uploads = services.read_uploads(files) if not form["demo"] else []
        run = services.run_screening(job_title=title, jd_text=form["jd_text"], model=form["model"], uploads=uploads, demo=form["demo"])
    except services.UserError as exc:
        return render(request, "hrscreen_web/screen.html", _screen_context(request, form=form, error=str(exc)), status=400)
    messages.success(request, f"Ranked {run.ranked.count()} resumes.")
    return redirect("run_detail", run_id=run.id)


@require_GET
def run_list(request):
    return render(request, "hrscreen_web/run_list.html", {"nav": "runs", "runs": ScreeningRun.objects.all()})


@require_GET
def run_detail(request, run_id):
    run = get_object_or_404(ScreeningRun, id=run_id)
    ranked = list(run.ranked)
    try:
        top = max(1, int(request.GET.get("top", 25)))
    except ValueError:
        top = 25
    show_all = request.GET.get("top") == "all" or top >= len(ranked)
    shown = ranked if show_all else ranked[:top]
    best = max((c.score for c in ranked), default=0) or 1.0
    for c in shown:
        c.width = max(2, round(100 * max(c.score, 0) / best))
        c.n_matched = len(c.matched_skills)
    return render(request, "hrscreen_web/run_detail.html", {
        "nav": "runs", "run": run, "shown": shown, "n_ranked": len(ranked), "show_all": show_all, "top": top,
        "unreadable": list(run.unreadable), "n_jd_skills": len(shown[0].matched_skills) + len(shown[0].missing_skills) if shown else 0,
    })


@require_GET
def run_csv(request, run_id):
    run = get_object_or_404(ScreeningRun, id=run_id)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="ranking_{run.id}.csv"'
    writer = csv.writer(response)
    writer.writerow(["rank", "candidate", "score", "matched_skills", "missing_skills", "years_experience", "education"])
    for c in run.ranked:
        writer.writerow([c.rank, c.label, f"{c.score:.4f}", "; ".join(c.matched_skills), "; ".join(c.missing_skills),
                         "" if c.years_experience is None else c.years_experience, c.education_level])
    return response


@require_POST
def run_delete(request, run_id):
    get_object_or_404(ScreeningRun, id=run_id).delete()
    messages.success(request, "Run deleted.")
    return redirect("run_list")


# ---------------------------------------------------------------------------------- saved results

@require_GET
def benchmark(request):
    sets = services.result_sets("benchmark_summary.csv")
    chosen = services.pick_set(sets, request.GET.get("set"), "benchmark")
    ctx = {"nav": "benchmark", "sets": sets, "chosen": chosen}
    if chosen:
        ctx.update(services.load_benchmark(chosen))
    return render(request, "hrscreen_web/benchmark.html", ctx)


@require_GET
def fairness(request):
    audits = [services.load_fairness(s) for s in services.result_sets("fairness_summary.csv")]
    order = ["tfidf", "embeddings", "xgboost", "xgboost+emb"]          # simplest model first, as in the benchmark
    audits.sort(key=lambda a: (order.index(a["model"]) if a["model"] in order else len(order), a["model"]))
    glance = []
    for a in audits:
        by_attr = {r["attribute"]: r for r in a["summary"]}
        glance.append({"model": a["model"], "education": by_attr.get("education_level"), "experience": by_attr.get("experience_band"),
                       "name_audit": a["name_audit"], "name_swap": a["name_swap"]})
    return render(request, "hrscreen_web/fairness.html", {"nav": "fairness", "audits": audits, "glance": glance})


# ---------------------------------------------------------------------------------- survey

@require_http_methods(["GET", "POST"])
def survey(request):
    ctx = {"nav": "survey", "runs": SurveyRun.objects.all()[:10], "max_mb": settings.MAX_SURVEY_BYTES // (1024 * 1024)}
    if request.method == "GET":
        return render(request, "hrscreen_web/survey.html", ctx)
    try:
        if request.POST.get("demo") == "on":
            run = services.analyze_demo_survey()
        else:
            upload = request.FILES.get("responses")
            if not upload:
                raise services.UserError("Choose the Google Forms CSV export, or tick the demo option.")
            run = services.analyze_survey(upload.read(), upload.name)
    except services.UserError as exc:
        return render(request, "hrscreen_web/survey.html", {**ctx, "error": str(exc)}, status=400)
    messages.success(request, f"Analysed {run.n_responses} responses.")
    return redirect("survey_detail", run_id=run.id)


@require_GET
def survey_detail(request, run_id):
    run = get_object_or_404(SurveyRun, id=run_id)
    return render(request, "hrscreen_web/survey_detail.html", {"nav": "survey", "run": run, "r": run.results,
                                                                 "figures": list(services.SURVEY_FIGURES)})


@require_GET
def survey_figure(request, run_id, name):
    filename = services.SURVEY_FIGURES.get(name)
    path = services.survey_dir(run_id) / filename if filename else None
    if not path or not path.exists():
        raise Http404("No such figure")
    return FileResponse(open(path, "rb"), content_type="image/png")


@require_GET
def survey_file(request, run_id, kind):
    get_object_or_404(SurveyRun, id=run_id)
    files = {"report": ("survey_report.md", "text/markdown"), "scored": ("survey_scored.csv", "text/csv")}
    if kind not in files:
        raise Http404
    name, content_type = files[kind]
    path = services.survey_dir(run_id) / name
    if not path.exists():
        raise Http404
    return FileResponse(open(path, "rb"), content_type=content_type, as_attachment=True, filename=f"survey_{run_id}_{name}")


@require_POST
def survey_delete(request, run_id):
    run = get_object_or_404(SurveyRun, id=run_id)
    services.delete_survey_files(run.id)
    run.delete()
    messages.success(request, "Survey analysis deleted.")
    return redirect("survey")


# ---------------------------------------------------------------------------------- questionnaire

@require_GET
def questionnaire(request):
    sections = [{"code": code, "name": name, "items": [{"n": n, "text": it.text, "reverse": it.reverse}
                                                       for n, it in enumerate(q.items_of(code), 1)]}
                for code, name in q.CONSTRUCTS.items()]
    return render(request, "hrscreen_web/questionnaire.html", {
        "nav": "questionnaire", "context_questions": q.CONTEXT, "sections": sections, "likert": q.LIKERT_LABELS,
        "n_items": len(q.ITEMS), "markdown": q.to_markdown()})


@require_GET
def questionnaire_download(request, fmt):
    if fmt == "md":
        response = HttpResponse(q.to_markdown(), content_type="text/markdown")
        response["Content-Disposition"] = 'attachment; filename="questionnaire.md"'
        return response
    if fmt == "csv":
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="questionnaire.csv"'
        rows = q.to_csv_rows()
        writer = csv.DictWriter(response, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return response
    raise Http404
