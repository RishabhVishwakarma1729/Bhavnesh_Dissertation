from pathlib import Path

import pytest

pytest.importorskip("django")
pytest.importorskip("pytest_django")

from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from django.test import Client  # noqa: E402
from django.urls import reverse  # noqa: E402

from hrscreen.data import load_jobs, make_synthetic_resumes  # noqa: E402
from hrscreen.evaluation import run_benchmark  # noqa: E402
from hrscreen.fairness import run_fairness_audit  # noqa: E402
from hrscreen.web import services  # noqa: E402
from hrscreen.web.models import Candidate, ScreeningRun, SurveyRun  # noqa: E402

pytestmark = pytest.mark.django_db
JD = load_jobs()[0]  # Data Scientist
UNIQUE = "zebra-quokka-unique-sentence"
DS = f"Data scientist. Python, machine learning, pandas, scikit-learn, statistics, tableau, sql. {UNIQUE}. M.Tech. 5 years of experience."
CIVIL = "Civil engineer. AutoCAD, site supervision, quantity surveying, concrete, billing, primavera."
HR = "HR executive. Recruitment, payroll, onboarding, employee engagement, training and development."


def upload(name, body):
    return SimpleUploadedFile(name, body.encode("utf-8") if isinstance(body, str) else body)


def post_screen(client, files, **overrides):
    data = {"job": JD.id, "jd_text": JD.text, "model": "tfidf", "resumes": files, **overrides}
    return client.post(reverse("screen"), data)


# ------------------------------------------------------------------ pages

@pytest.mark.parametrize("name", ["home", "screen", "run_list", "benchmark", "fairness", "survey", "questionnaire", "about"])
def test_every_page_renders(client, name):
    response = client.get(reverse(name))
    assert response.status_code == 200
    assert b"<title>" in response.content and b"hrscreen" in response.content


def test_navigation_marks_the_current_page(client):
    html = client.get(reverse("questionnaire")).content.decode()
    nav = html[html.index("<nav"):html.index("</nav>")]      # the stylesheet also mentions the attribute, so look only at the nav
    assert nav.count('aria-current="page"') == 1 and 'Questionnaire</a>' in nav
    current_link = nav[nav.index('aria-current="page"'):].split("</a>")[0]
    assert "Questionnaire" in current_link


def test_screen_form_is_prefilled_and_offers_models(client):
    html = client.get(reverse("screen")).content.decode()
    assert "Data Scientist" in html and "TF-IDF similarity" in html and "Keyword rules" in html
    assert JD.text[:40] in html                     # works without JavaScript


# ------------------------------------------------------------------ screening

def test_screening_ranks_the_best_match_first(client):
    response = post_screen(client, [upload("civil.txt", CIVIL), upload("ds.txt", DS), upload("hr.txt", HR)])
    assert response.status_code == 302
    run = ScreeningRun.objects.get()
    assert [c.label for c in run.ranked] == ["ds.txt", "civil.txt", "hr.txt"] or run.ranked[0].label == "ds.txt"
    top = run.ranked[0]
    assert top.rank == 1 and "python" in top.matched_skills and top.education_level == "master" and top.years_experience == 5.0
    page = client.get(response["Location"])
    assert page.status_code == 200 and b"ds.txt" in page.content and b"Skills in common" in page.content


def test_resume_text_is_never_stored(client):
    post_screen(client, [upload("ds.txt", DS), upload("hr.txt", HR)])
    stored = []
    for run in ScreeningRun.objects.all():
        stored += [run.job_title, run.jd_text]
    for c in Candidate.objects.all():
        stored += [c.label, c.error, c.education_level, str(c.matched_skills), str(c.missing_skills)]
    assert UNIQUE not in " ".join(stored)


def test_screening_needs_files_or_demo(client):
    response = client.post(reverse("screen"), {"job": JD.id, "jd_text": JD.text, "model": "tfidf"})
    assert response.status_code == 400 and b"Upload at least one resume" in response.content
    assert ScreeningRun.objects.count() == 0


def test_screening_rejects_a_trivial_job_description(client):
    response = post_screen(client, [upload("a.txt", DS)], jd_text="data")
    assert response.status_code == 400 and b"too short" in response.content


def test_unreadable_files_are_reported_not_fatal(client):
    post_screen(client, [upload("ds.txt", DS), upload("virus.exe", b"MZ"), upload("empty.txt", "   ")])
    run = ScreeningRun.objects.get()
    reasons = {c.label: c.error for c in run.unreadable}
    assert "Unsupported" in reasons["virus.exe"] and "No text" in reasons["empty.txt"]
    assert run.ranked.count() == 1
    assert b"Files that could not be ranked" in Client().get(reverse("run_detail", args=[run.id])).content


def test_all_unreadable_is_an_error(client):
    response = post_screen(client, [upload("virus.exe", b"MZ")])
    assert response.status_code == 400 and b"None of the files could be read" in response.content


def test_demo_run_is_clearly_flagged(client):
    response = client.post(reverse("screen"), {"job": JD.id, "jd_text": JD.text, "model": "rules", "demo": "on"})
    run = ScreeningRun.objects.get()
    assert run.is_demo and run.ranked.count() == 60
    html = client.get(response["Location"]).content.decode()
    assert "Synthetic demo" in html and "synthetic demo" in client.get(reverse("run_list")).content.decode()


def test_edited_job_description_is_labelled(client):
    post_screen(client, [upload("ds.txt", DS)], jd_text=JD.text + " Also must know Spark and Airflow in depth.")
    assert ScreeningRun.objects.get().job_title.endswith("(edited)")


def test_custom_job_uses_the_given_title(client):
    post_screen(client, [upload("ds.txt", DS)], job="custom", job_title="Analytics Lead")
    assert ScreeningRun.objects.get().job_title == "Analytics Lead"


def test_too_many_files_is_rejected(client, settings):
    settings.MAX_RESUMES = 2
    response = post_screen(client, [upload(f"r{i}.txt", DS) for i in range(3)])
    assert response.status_code == 400 and b"Too many files" in response.content


def test_oversized_file_is_listed_as_unreadable(client, settings):
    settings.MAX_RESUME_BYTES = 100
    post_screen(client, [upload("big.txt", "x" * 500), upload("ok.txt", "tiny " * 5)])
    assert "larger than" in ScreeningRun.objects.get().unreadable[0].error


def test_run_detail_top_parameter(client):
    client.post(reverse("screen"), {"job": JD.id, "jd_text": JD.text, "model": "tfidf", "demo": "on"})
    run = ScreeningRun.objects.get()
    assert client.get(reverse("run_detail", args=[run.id]) + "?top=5").content.decode().count('class="num rank"') == 5
    assert client.get(reverse("run_detail", args=[run.id]) + "?top=all").content.decode().count('class="num rank"') == 60
    assert client.get(reverse("run_detail", args=[run.id]) + "?top=banana").status_code == 200


def test_csv_download_and_delete(client):
    post_screen(client, [upload("ds.txt", DS), upload("hr.txt", HR)])
    run = ScreeningRun.objects.get()
    csv_response = client.get(reverse("run_csv", args=[run.id]))
    assert csv_response["Content-Type"].startswith("text/csv") and b"rank,candidate,score" in csv_response.content
    assert client.get(reverse("run_delete", args=[run.id])).status_code == 405        # deleting needs POST
    assert client.post(reverse("run_delete", args=[run.id])).status_code == 302
    assert ScreeningRun.objects.count() == 0 and Candidate.objects.count() == 0       # candidates go with the run
    assert client.get(reverse("run_detail", args=[run.id])).status_code == 404


def test_csrf_is_enforced():
    strict = Client(enforce_csrf_checks=True)
    assert strict.post(reverse("screen"), {"job": JD.id}).status_code == 403


def test_html_in_labels_is_escaped(client):
    hostile = "<img src=x onerror=alert(1)>.txt"           # no slash: a slash would be stripped as a path separator
    post_screen(client, [upload(hostile, DS)])
    run = ScreeningRun.objects.get()
    assert run.ranked[0].label == hostile                  # stored as-is, neutralised on output
    html = client.get(reverse("run_detail", args=[run.id])).content.decode()
    assert "<img src=x" not in html and "&lt;img src=x onerror=alert(1)&gt;.txt" in html


# ------------------------------------------------------------------ saved results

@pytest.fixture
def results_dir(tmp_path, settings):
    """A results folder holding a small SYNTHETIC benchmark and fairness audit."""
    settings.RESULTS_DIR = tmp_path
    resumes, jobs = make_synthetic_resumes(240, seed=3), load_jobs()
    run_benchmark(resumes, jobs, ["rules", "tfidf"], n_splits=3, timing_sample=20, progress=lambda *_: None).save(tmp_path / "demo" / "benchmark")
    run_fairness_audit(resumes, jobs, "tfidf", n_splits=3, name_audit_size=40, n_boot=20, progress=lambda *_: None).save(tmp_path / "demo" / "fairness_tfidf")
    return tmp_path


def test_empty_states_when_no_results(client, tmp_path, settings):
    settings.RESULTS_DIR = tmp_path
    assert b"No benchmark results found" in client.get(reverse("benchmark")).content
    assert b"No fairness audits found" in client.get(reverse("fairness")).content


def test_benchmark_page_shows_models_holm_and_caveats(client, results_dir):
    html = client.get(reverse("benchmark")).content.decode()
    for needle in ("rules", "tfidf", "p adjusted", "Ground truth", "Rank correlation", "nDCG@10 by job"):
        assert needle in html
    assert "Spearman" in html                      # the caveat explaining why rank correlation is left out


def test_benchmark_set_parameter_cannot_escape_the_results_folder(client, results_dir):
    for hostile in ("../../etc/passwd", "..\\..\\secret", "/etc/passwd", "demo/../../x"):
        assert client.get(reverse("benchmark"), {"set": hostile}).status_code == 200   # falls back to a real set


def test_fairness_page_shows_groups_and_name_audit(client, results_dir):
    html = client.get(reverse("fairness")).content.decode()
    for needle in ("At a glance", "equal-opportunity", "Name-swap test", "education", "experience", "4/5"):
        assert needle in html


@pytest.mark.skipif(not (Path(__file__).parents[1] / "results" / "resume-atlas" / "benchmark").exists(), reason="committed results absent")
def test_committed_real_results_render(client, settings):
    settings.RESULTS_DIR = Path(__file__).parents[1] / "results"
    bench = client.get(reverse("benchmark"), {"set": "resume-atlas/benchmark"}).content.decode()
    assert "xgboost+emb" in bench and "11901" in bench and "not significant" in bench
    fair = client.get(reverse("fairness")).content.decode()
    assert "embeddings" in fair and "Name-swap test" in fair


# ------------------------------------------------------------------ survey

def test_demo_survey_analysis_is_flagged_and_has_figures(client, tmp_path, settings):
    settings.DATA_DIR = tmp_path
    response = client.post(reverse("survey"), {"demo": "on"})
    assert response.status_code == 302
    run = SurveyRun.objects.get()
    assert run.is_synthetic and run.n_responses == 80
    page = client.get(response["Location"]).content.decode()
    assert "SYNTHETIC DATA" in page and "Cronbach" in page and "Perceived efficiency" in page
    for name in services.SURVEY_FIGURES:
        figure = client.get(reverse("survey_figure", args=[run.id, name]))
        assert figure.status_code == 200 and figure["Content-Type"] == "image/png"
    assert client.get(reverse("survey_file", args=[run.id, "report"])).status_code == 200
    assert client.get(reverse("survey_file", args=[run.id, "scored"])).status_code == 200


def test_uploaded_google_forms_csv_is_analysed(client, tmp_path, settings):
    settings.DATA_DIR = tmp_path
    from hrscreen.survey import make_synthetic_responses

    csv = make_synthetic_responses(70, seed=9, form_style=True).drop(columns=["is_synthetic"]).to_csv(index=False)
    response = client.post(reverse("survey"), {"responses": upload("export.csv", csv)})
    assert response.status_code == 302
    run = SurveyRun.objects.get()
    assert not run.is_synthetic and run.source_name == "export.csv"
    assert b"SYNTHETIC DATA" not in client.get(response["Location"]).content   # real-looking data is not stamped


def test_survey_errors_are_friendly(client, tmp_path, settings):
    settings.DATA_DIR = tmp_path
    assert b"Choose the Google Forms CSV" in client.post(reverse("survey"), {}).content
    bad = client.post(reverse("survey"), {"responses": upload("x.csv", "a,b\n1,2\n")})
    assert bad.status_code == 400 and b"not found" in bad.content
    empty = client.post(reverse("survey"), {"responses": upload("x.csv", "a,b\n")})
    assert empty.status_code == 400
    garbage = client.post(reverse("survey"), {"responses": upload("x.csv", b"\x00\x01\x02")})
    assert garbage.status_code == 400
    assert SurveyRun.objects.count() == 0


def test_survey_figure_and_file_routes_only_serve_known_names(client, tmp_path, settings):
    settings.DATA_DIR = tmp_path
    client.post(reverse("survey"), {"demo": "on"})
    run = SurveyRun.objects.get()
    assert client.get(f"/survey/{run.id}/figure/passwd.png").status_code == 404
    assert client.get(reverse("survey_file", args=[run.id, "secrets"])).status_code == 404
    assert client.get(reverse("survey_figure", args=[9999, "constructs"])).status_code == 404


def test_deleting_a_survey_removes_its_files(client, tmp_path, settings):
    settings.DATA_DIR = tmp_path
    client.post(reverse("survey"), {"demo": "on"})
    run = SurveyRun.objects.get()
    folder = services.survey_dir(run.id)
    assert folder.exists()
    assert client.post(reverse("survey_delete", args=[run.id])).status_code == 302
    assert not folder.exists() and SurveyRun.objects.count() == 0


# ------------------------------------------------------------------ questionnaire

def test_questionnaire_page_and_downloads(client):
    html = client.get(reverse("questionnaire")).content.decode()
    assert "reverse-worded" in html and "Linear scale" in html
    md = client.get(reverse("questionnaire_download", args=["md"]))
    assert md["Content-Disposition"].startswith("attachment") and b"HR Perception Survey" in md.content
    assert b"PE1" in client.get(reverse("questionnaire_download", args=["csv"])).content
    assert client.get(reverse("questionnaire_download", args=["exe"])).status_code == 404


# ------------------------------------------------------------------ CLI

def test_web_command_is_registered():
    from hrscreen.cli import build_parser

    args = build_parser().parse_args(["web", "--port", "8123", "--debug"])
    assert args.port == 8123 and args.debug and args.host == "127.0.0.1"


# ------------------------------------------------------------------ fairness glance table

def test_name_swap_lines_are_parsed_into_figures():
    lines = ["Resumes tested: 400",
             "Mean score difference (female-coded minus male-coded), in SDs: -0.0191 (Wilcoxon p = 2.3e-22)",
             "Shortlist decisions changed by the name alone: 0.67% of resume-job pairs (13 favoured the female-coded name, 27 the male-coded name)"]
    assert services.parse_name_swap(lines) == {"diff": "-0.0191", "p": "2.3e-22", "flip_pct": "0.67", "female": 13, "male": 27}
    assert services.parse_name_swap(["unexpected wording"]) is None


def test_fairness_glance_shows_compact_name_swap_and_orders_models(client, settings):
    settings.RESULTS_DIR = Path(__file__).parents[1] / "results"
    if not (settings.RESULTS_DIR / "resume-atlas" / "fairness_tfidf").exists():
        pytest.skip("committed results absent")
    html = client.get(reverse("fairness")).content.decode()
    glance = html[html.index("At a glance"):html.index("Four-fifths rule:")]
    assert "of decisions flip" in glance and "favour female-coded" in glance
    assert glance.index("tfidf") < glance.index("embeddings") < glance.index("xgboost+emb")   # simplest model first
    assert "0.795" in glance                                                                    # three decimals, so a near-miss is visible


# ------------------------------------------------------------------ sample files for the upload forms

BEST_SAMPLE_FOR_JOB = {
    "data-science": "priya_nair_data_scientist.pdf", "python-developer": "rohan_mehta_python_developer.docx",
    "java-developer": "ananya_iyer_java_developer.txt", "devops-engineer": "karan_malhotra_devops_engineer.pdf",
    "web-designing": "sneha_kulkarni_web_designer.txt", "hr": "farah_khan_hr_executive.docx",
    "sales": "vikram_singh_sales_executive.txt", "business-analyst": "deepa_reddy_business_analyst.docx",
    "civil-engineer": "arjun_das_civil_engineer.txt",
}


def sample_uploads():
    return [upload(s["name"], (services.SAMPLES_DIR / s["name"]).read_bytes()) for s in services.sample_resumes()]


def test_sample_resumes_are_shipped_in_three_formats_with_one_unreadable():
    names = [s["name"] for s in services.sample_resumes()]
    assert len(names) == 10 and {n.rsplit(".", 1)[1] for n in names} == {"pdf", "docx", "txt"}
    read = services.read_uploads(sample_uploads())
    unreadable = [u for u in read if u.error]
    assert [u.label for u in unreadable] == ["blank_scan_unreadable.pdf"] and "No text" in unreadable[0].error
    assert len([u for u in read if not u.error]) == 9


@pytest.mark.parametrize("job_id,expected", sorted(BEST_SAMPLE_FOR_JOB.items()))
def test_each_sample_resume_is_the_best_match_for_its_own_job(client, job_id, expected):
    """The samples are only useful for testing if the ranking they produce makes sense."""
    jd = next(j for j in services.job_catalogue() if j["id"] == job_id)
    response = client.post(reverse("screen"), {"job": job_id, "jd_text": jd["text"], "model": "tfidf", "resumes": sample_uploads()})
    assert response.status_code == 302
    run = ScreeningRun.objects.get()
    assert run.ranked[0].label == expected, [(c.label, round(c.score, 3)) for c in run.ranked[:3]]
    assert run.unreadable.count() == 1                                   # the blank PDF is reported, not fatal


def test_sample_endpoints_serve_only_the_shipped_files(client):
    for s in services.sample_resumes():
        response = client.get(reverse("sample_resume", args=[s["name"]]))
        assert response.status_code == 200 and b"".join(response.streaming_content) == (services.SAMPLES_DIR / s["name"]).read_bytes()
    assert client.get(reverse("sample_resume", args=["priya_nair_data_scientist.pdf"]))["Content-Type"] == "application/pdf"
    for hostile in ("..%2F..%2Fsettings.py", "%2e%2e/%2e%2e/secret_key.txt", r"..\..\x", "nope.pdf", ".gitignore", "samples"):
        assert client.get("/samples/resumes/" + hostile).status_code == 404, hostile


def test_sample_zip_and_sample_survey_downloads(client):
    import io
    import zipfile

    zipped = client.get(reverse("sample_resumes_zip"))
    assert zipped["Content-Disposition"].startswith("attachment")
    assert sorted(zipfile.ZipFile(io.BytesIO(zipped.content)).namelist()) == sorted(s["name"] for s in services.sample_resumes())
    survey_csv = client.get(reverse("sample_survey"))
    assert survey_csv["Content-Type"].startswith("text/csv") and "SYNTHETIC" in survey_csv["Content-Disposition"]
    assert survey_csv.content.count(b"\n") == 81 and b"is_synthetic" in survey_csv.content.splitlines()[0]


def test_sample_survey_goes_through_the_normal_upload_and_stays_flagged(client, tmp_path, settings):
    settings.DATA_DIR = tmp_path
    response = client.post(reverse("survey"), {"responses": upload("sample_survey_responses_SYNTHETIC.csv", services.sample_survey_csv())})
    assert response.status_code == 302
    run = SurveyRun.objects.get()
    assert run.is_synthetic and run.source_name == "sample_survey_responses_SYNTHETIC.csv" and run.n_responses == 80
    assert b"SYNTHETIC DATA" in client.get(response["Location"]).content


def test_attached_files_win_over_the_demo_checkbox(client, tmp_path, settings):
    post_screen(client, [upload("ds.txt", DS), upload("hr.txt", HR)], demo="on")
    run = ScreeningRun.objects.get()
    assert not run.is_demo and run.ranked.count() == 2                  # the two attached files, not 60 synthetic ones
    settings.DATA_DIR = tmp_path
    client.post(reverse("survey"), {"demo": "on", "responses": upload("mine.csv", services.sample_survey_csv())})
    assert SurveyRun.objects.get().source_name == "mine.csv"


def test_upload_pages_offer_the_sample_controls(client):
    screen = client.get(reverse("screen")).content.decode()
    assert 'id="load-samples"' in screen and "Load 10 sample resumes" in screen and reverse("sample_resumes_zip") in screen
    assert "samples-data" in screen and "priya_nair_data_scientist.pdf" in screen
    survey = client.get(reverse("survey")).content.decode()
    assert 'id="load-sample-csv"' in survey and reverse("sample_survey") in survey and "SYNTHETIC" in survey
